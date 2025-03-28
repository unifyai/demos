import json
import os
import re
import threading

import cv2
import numpy as np
import pdfplumber
import unify
import wget
from pdf2image import convert_from_path
from pypdf import PdfReader, PdfWriter

unify.CLIENT_LOGGING = True
from helpers import (
    VALID_NUMERALS,
    build_response_format,
    encode_image,
    is_invalid_question_order,
    parse_key,
    prune_invalid_leading_alphanumeric_questions,
    update_str_in_table,
)
from prompts import *

url = (
    "https://www.ocr.org.uk/Images/169000-foundation-tier-sample-assessment"
    "-materials.pdf"
)
this_dir = os.path.dirname(__file__)
pdfs_dir = os.path.join(this_dir, "pdfs")
os.makedirs(pdfs_dir, exist_ok=True)
fname = url.split("/")[-1]
pdf_path = os.path.join(pdfs_dir, fname)
if not os.path.exists(pdf_path):
    fname = wget.download(url, out=pdfs_dir)
pdf_path = os.path.join(pdfs_dir, fname)
pdf_dir = os.path.join(pdfs_dir, fname[:-4])
os.makedirs(pdf_dir, exist_ok=True)
reader = PdfReader(pdf_path)


def prune_page_number(pg_text: str, page_num: int, question_num: int):
    if page_num == question_num:
        return pg_text
    # remove page number to avoid any confusion with question number
    split = pg_text.split(str(question_num))
    if len(split) == 1:
        return pg_text
    pre_q = split[0]
    post_q = split[1:]
    pre_q = pre_q.replace(str(page_num), "")
    return str(question_num).join([pre_q] + post_q)


def parse_pdf_into_papers_and_markschemes():
    paper_cover_text = (
        "completetheboxesabovewithyourname,centrenumberandcandidatenumber."
    )
    markscheme_cover_text = "markscheme"
    paper_cover_pages = list()
    markscheme_cover_pages = list()
    looking_for_paper = True

    for i, page in enumerate(reader.pages):
        text_stripped = page.extract_text().lower().replace(" ", "")
        if looking_for_paper and paper_cover_text in text_stripped:
            paper_cover_pages.append(i)
            looking_for_paper = False
        elif not looking_for_paper and markscheme_cover_text in text_stripped:
            markscheme_cover_pages.append(i)
            looking_for_paper = True

    left_pointer = paper_cover_pages.pop(0)
    count = 1
    while paper_cover_pages or markscheme_cover_pages:
        count_dir = os.path.join(pdf_dir, str(count))
        os.makedirs(count_dir, exist_ok=True)
        # paper
        writer = PdfWriter()
        right_pointer = markscheme_cover_pages.pop(0)
        [writer.add_page(pg) for pg in reader.pages[left_pointer:right_pointer]]
        paper_fpath = os.path.join(count_dir, "paper.pdf")
        writer.write(paper_fpath)
        left_pointer = right_pointer
        # mark-scheme
        writer = PdfWriter()
        if paper_cover_pages:
            right_pointer = paper_cover_pages.pop(0)
        else:
            right_pointer = len(reader.pages) + 1
        [writer.add_page(pg) for pg in reader.pages[left_pointer:right_pointer]]
        markscheme_fpath = os.path.join(count_dir, "markscheme.pdf")
        writer.write(markscheme_fpath)
        left_pointer = right_pointer
        # count
        count += 1


def _fill_missing_questions_n_pages(questions_to_pages):
    prev_question_num = 0
    prev_pages = list()
    new_questions_to_pages = questions_to_pages.copy()
    all_alphanum = list(questions_to_pages.keys())
    all_pages = list(questions_to_pages.values())
    for i, (question_alphanum, pages) in enumerate(questions_to_pages.items()):
        question_num = int(question_alphanum.split(".")[0])
        if question_num not in (prev_question_num, prev_question_num + 1):
            min_pg = min(all_pages[i - 1])
            max_pg = max(pages)
            union_of_pages = list(range(min_pg, max_pg + 1))
            for q_num in range(prev_question_num + 1, question_num):
                # we somehow missed the entire question, let's assume there are no
                # sub-question components (a, b, c etc.)
                new_questions_to_pages[str(q_num)] = union_of_pages
        elif prev_pages and pages[0] > prev_pages[-1] + 1:
            prev_alphanum = all_alphanum[i - 1]
            new_questions_to_pages[prev_alphanum] = list(
                range(prev_pages[0], pages[0] + 1),
            )
            new_questions_to_pages[question_alphanum] = list(
                range(prev_pages[-1], pages[-1] + 1),
            )
        prev_question_num = question_num
        prev_pages = pages
    return dict(sorted(new_questions_to_pages.items()))


def parse_paper(paper_num):
    question_detector = unify.Unify(
        "o1@openai",
        cache=True,
        system_message=QUESTION_DETECTION,
        stateful=True,
    )
    diagram_detector = unify.Unify("o1@openai", cache=True)

    paper_dir = os.path.join(pdf_dir, str(paper_num), "paper")
    os.makedirs(paper_dir, exist_ok=True)
    paper_path = paper_dir + ".pdf"

    reader = pdfplumber.open(paper_path)
    questions = dict()

    all_images = [
        np.asarray(img.getdata())
        .reshape(img.size[1], img.size[0], 3)
        .astype(
            np.uint8,
        )
        for img in convert_from_path(paper_path)
    ]

    diagram_images = dict()

    def parse_question_detector(response):
        parsed = (
            response.lower()
            .split(
                "answer:",
            )[-1]
            .replace(
                "\n",
                "",
            )
            .replace(
                "`",
                "",
            )
            .replace(
                " ",
                "",
            )
            .split(
                ",",
            )
        )
        return [p for p in parsed if p != ""]

    def parse_into_pages():
        diagram_detector.set_system_message(DIAGRAM_DETECTION_ON_PAGE)
        question_to_pages = dict()
        latest_num = 0
        latest_char = "`"
        for page_num, page in enumerate(reader.pages):
            page_num += 1
            text = page.extract_text().split("OCR  2024  J560/0")[-1][2:]
            text = prune_page_number(text, page_num, latest_num + 1)
            # detect diagrams on page
            img = all_images[page_num - 1]
            diagram_response = diagram_detector.generate(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": text,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,"
                                    f"{encode_image(img)}",
                                },
                            },
                        ],
                    },
                ],
            )
            contains_diagram = "yes" in diagram_response.split("\n")[-1].strip().lower()
            img_dir = os.path.join(paper_dir, "imgs")
            os.makedirs(img_dir, exist_ok=True)
            fname = f"page{page_num}.png"
            if not os.path.exists(fname):
                cv2.imwrite(os.path.join(img_dir, fname), img)
            if contains_diagram:
                diagram_images[page_num] = img
            question_detector.set_messages([])  # clear previous chat
            question_detector.set_system_message(
                QUESTION_DETECTION.replace(
                    "{n-1}",
                    str(latest_num),
                )
                .replace(
                    "{n0}",
                    str(latest_num + 1),
                )
                .replace(
                    "{n1}",
                    str(latest_num + 2),
                )
                .replace(
                    "{n2}",
                    str(latest_num + 3),
                )
                .replace(
                    "{c0}",
                    chr(ord(latest_char) + 1),
                )
                .replace(
                    "{c1}",
                    chr(ord(latest_char) + 2),
                )
                .replace(
                    "{c2}",
                    chr(ord(latest_char) + 3),
                )
                .replace(
                    "{explanation}",
                    (
                        f"did not have any alpha sub-questions, so we should see a "
                        f"numeric question first on this page, possibly followed by `a`"
                        if latest_char == "`"
                        else f"ended with question {latest_num} {latest_char}"
                    ),
                ),
            )
            response = question_detector.generate(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": text,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,"
                                    f"{encode_image(img)}",
                                },
                            },
                        ],
                    },
                ],
            )
            assert len(question_detector.messages) == 2
            detected_qs = parse_question_detector(response)
            if not all(
                v.isdigit() or (len(v) == 1 and v.isalpha()) or v in VALID_NUMERALS
                for v in detected_qs
            ):
                continue
            invalid_sequence = is_invalid_question_order(
                detected_qs,
                str(latest_num + 1),
                chr(ord(latest_char) + 1),
            )
            count = 0
            attempts = 3
            while invalid_sequence and count < attempts:
                response = question_detector.generate(
                    f"The previous response {detected_qs} was invalid. "
                    f"The next question *number* must be {latest_num + 1}, "
                    "and if the *first* item on the page is a letter (before any "
                    f"numbers) then it must be {chr(ord(latest_char) + 1)} (as an "
                    f"overflow of question {latest_num} from the previous page). "
                    "Finally, any letters immediately after a new question number "
                    "*must* begin with `a` and then ascend alphabetically one character "
                    f"at a time. Your answer of {detected_qs} does not adhere to "
                    f"these rules. Perhaps you mistook the page number {page_num} for "
                    "the question number, and the letter refers to a prior question? "
                    "Other similar mistakes might be possible. Please have another "
                    "think and provide an updated answer.",
                )
                detected_qs = parse_question_detector(response)
                if not all(
                    v.isdigit() or (len(v) == 1 and v.isalpha()) or v in VALID_NUMERALS
                    for v in detected_qs
                ):
                    detected_qs = response
                    count += 1
                    assert len(question_detector.messages) == 2 + count * 2
                    continue
                invalid_sequence = is_invalid_question_order(
                    detected_qs,
                    str(latest_num + 1),
                    chr(ord(latest_char) + 1),
                )
                count += 1
                assert len(question_detector.messages) == 2 + count * 2
            assert (
                not invalid_sequence
            ), f"Still an invalid sequence {detected_qs} after {attempts} attempts"
            num = latest_num
            char = latest_char
            for i, item in enumerate(detected_qs):
                if item in VALID_NUMERALS:
                    question_to_pages[f"{num}.{char}.{item}"] = [page_num]
                elif item.isalpha():
                    if (
                        len(detected_qs) == i + 1
                        or (
                            detected_qs[i + 1] not in VALID_NUMERALS
                            and detected_qs[i + 1].isalpha()
                        )
                        or detected_qs[i + 1].isdigit()
                    ):
                        question_to_pages[f"{num}.{item}"] = [page_num]
                    char = item
                elif item.isdigit():
                    if len(detected_qs) == i + 1 or detected_qs[i + 1].isdigit():
                        question_to_pages[item] = [page_num]
                    num = int(item)
                else:
                    raise ValueError(f"Invalid type for question: {item}")
            latest_num = num
            latest_char = "`" if detected_qs[-1].isnumeric() else char
        return _fill_missing_questions_n_pages(question_to_pages), latest_num

    question_to_pages, num_questions = parse_into_pages()
    question_to_pages = dict(
        sorted(question_to_pages.items(), key=lambda item: parse_key(item[0])),
    )
    with open(os.path.join(paper_dir, "question_to_pages.json"), "w+") as file:
        file.write(json.dumps(question_to_pages, indent=4))

    json_file_lock = threading.Lock()

    def parse_question(question_num: int):
        question_parser = unify.Unify("o1@openai", cache=True)
        question_component_parser = unify.Unify(
            "o1@openai",
            system_message=QUESTION_COMPONENT_PARSER,
            cache=True,
        )
        text_only_detector = unify.Unify(
            "o1@openai",
            cache=True,
            system_message=TEXT_ONLY_DETECTION,
        )
        sub_questions = [
            ".".join(k.split(".")[1:])
            for k, v in question_to_pages.items()
            if k.startswith(str(question_num) + ".")
        ]
        pages = [
            v
            for k, v in question_to_pages.items()
            if (k == str(question_num) or k.startswith(str(question_num) + "."))
        ]
        pages = list(dict.fromkeys([item for sublist in pages for item in sublist]))
        current_text = "".join([reader.pages[pg - 1].extract_text() for pg in pages])
        imgs = [all_images[pg - 1] for pg in pages]
        question_parser.set_system_message(
            QUESTION_PARSER.replace(
                "{question_number}",
                str(question_num),
            )
            .replace(
                "{preceding}",
                str(question_num - 1),
            )
            .replace(
                "{subsequent}",
                str(question_num + 1),
            ),
        )
        question_parsed = question_parser.generate(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": current_text,
                        },
                    ]
                    + [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,"
                                f"{encode_image(img)}",
                            },
                        }
                        for img in imgs
                    ],
                },
            ],
        )
        question_component_parser.set_response_format(
            build_response_format(question_num, sub_questions),
        )
        question_comp_parsed = question_component_parser.generate(question_parsed)
        question_comp_parsed = json.loads(question_comp_parsed)
        response = text_only_detector.generate(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(question_parsed, indent=4),
                        },
                    ]
                    + [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,"
                                f"{encode_image(img)}",
                            },
                        }
                        for img in imgs
                    ],
                },
            ],
        )
        text_only = "yes" in response.split("\n")[-1].lower()
        questions[question_num] = {
            "question": question_parsed,
            "question-components": (
                question_comp_parsed
                if sub_questions
                else question_comp_parsed[str(question_num)]
            ),
            "text-only": text_only,
            "pages": pages,
            "correctly_parsed": True,
        }
        parsed = json.dumps(dict(sorted(questions.items())), indent=4)
        json_file_lock.acquire()
        with open(os.path.join(paper_dir, "parsed.json"), "w+") as file:
            file.write(parsed)
        json_file_lock.release()

        # image
        imgs = [diagram_images[pg] for pg in pages if pg in diagram_images]
        if not imgs:
            return
        diagram_detector.set_system_message(
            DIAGRAM_DETECTION_IN_QUESTION.replace(
                "{question_number}",
                str(question_num),
            )
            .replace(
                "{preceding}",
                str(question_num - 1),
            )
            .replace(
                "{subsequent}",
                str(question_num + 1),
            ),
        )
        diagram_response = diagram_detector.generate(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": current_text,
                        },
                    ]
                    + [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encode_image(img)}",
                            },
                        }
                        for img in imgs
                    ],
                },
            ],
        )
        contains_diagram = "yes" in diagram_response.split("\n")[-1].strip().lower()
        # incrementally save to file
        if not contains_diagram:
            return
        img_dir = os.path.join(paper_dir, "imgs")
        os.makedirs(img_dir, exist_ok=True)
        fnames = [f"page{pg}.png" for pg in pages if pg in diagram_images]
        questions[question_num]["images"] = fnames
        for fname, img in zip(fnames, imgs):
            if not os.path.exists(fname):
                cv2.imwrite(os.path.join(img_dir, fname), img)

    unify.map(parse_question, list(range(1, num_questions + 1)))
    reader.close()


def parse_markscheme(paper_num, question_to_subquestions, subquestions):
    qna_detector = unify.Unify(
        "o1@openai",
        cache=True,
        system_message=QUESTION_ANSWER_DETECTION,
    )
    diagram_detector = unify.Unify("o1@openai", cache=True)

    markscheme_dir = os.path.join(pdf_dir, str(paper_num), "markscheme")
    os.makedirs(markscheme_dir, exist_ok=True)
    markscheme_path = markscheme_dir + ".pdf"

    reader = pdfplumber.open(markscheme_path)
    questions = dict()

    all_images = [
        np.asarray(img.getdata())
        .reshape(img.size[1], img.size[0], 3)
        .astype(
            np.uint8,
        )
        for img in convert_from_path(markscheme_path)
    ]

    diagram_images = dict()

    def parse_question_detector(response):
        parsed = (
            response.lower()
            .split(
                "answer:",
            )[-1]
            .replace(
                "\n",
                "",
            )
            .replace(
                "`",
                "",
            )
            .replace(
                " ",
                "",
            )
            .split(
                ",",
            )
        )
        # remove accidental page numbers
        result = list()
        for item in reversed(parsed):
            if item.isdigit() and item in result:
                continue
            result.append(item)
        result.reverse()
        return [c for c in result if c != ""]

    def parse_into_pages():
        diagram_detector.set_system_message(
            DIAGRAM_DETECTION_ON_PAGE.replace("questions", "questions or answers"),
        )
        question_to_pages = dict()
        all_detected_qs = list()
        latest_num = 0
        latest_char = "`"
        for page_num, page in enumerate(reader.pages):
            page_num += 1
            text = page.extract_text().split("OCR  2024  J560/0")[-1][2:]
            # remove assessment objectives
            text = re.sub(r"\d+\s+AO[123]\.\w+", "", text)
            table = page.extract_table()
            if table:
                table = update_str_in_table(
                    table,
                    lambda x: re.sub(r"\d+\s+AO[123]\.\w+", "", x),
                )
                text = (
                    f"**Pure text representation:**\n{text}\n\n"
                    f"**Extracted table:**\n{json.dumps(table, indent=4)}"
                )

            # detect diagrams on page
            img = all_images[page_num - 1]
            diagram_response = diagram_detector.generate(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": text,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,"
                                    f"{encode_image(img)}",
                                },
                            },
                        ],
                    },
                ],
            )
            contains_diagram = "yes" in diagram_response.split("\n")[-1].strip().lower()
            img_dir = os.path.join(markscheme_dir, "imgs")
            os.makedirs(img_dir, exist_ok=True)
            fname = f"page{page_num}.png"
            if not os.path.exists(fname):
                cv2.imwrite(os.path.join(img_dir, fname), img)
            if contains_diagram:
                diagram_images[page_num] = img
            qna_detector.set_system_message(
                QUESTION_ANSWER_DETECTION.replace(
                    "{detected_so_far}",
                    (
                        "the full set of questions detected so far up to and including "
                        f"the last page are: {all_detected_qs}"
                        if all_detected_qs
                        else "this is the first page in the markscheme"
                    ),
                )
                .replace(
                    "{i0}",
                    str(subquestions[0]),
                )
                .replace(
                    "{i1}",
                    str(subquestions[1] if len(subquestions) > 1 else ""),
                )
                .replace(
                    "{i2}",
                    str(subquestions[2] if len(subquestions) > 2 else ""),
                )
                .replace(
                    "{n0}",
                    str(latest_num + 1),
                )
                .replace(
                    "{n1}",
                    str(latest_num + 2),
                )
                .replace(
                    "{n2}",
                    str(latest_num + 3),
                )
                .replace(
                    "{full_question_structure}",
                    json.dumps(question_to_subquestions, indent=4),
                )
                .replace(
                    "{subquestions}",
                    ", ".join([str(sq) for sq in subquestions]),
                ),
            )
            response = qna_detector.generate(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": text,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,"
                                    f"{encode_image(img)}",
                                },
                            },
                        ],
                    },
                ],
            )
            detected_qs = parse_question_detector(response)
            if not all(
                v.isdigit() or (len(v) == 1 and v.isalpha()) or v in VALID_NUMERALS
                for v in detected_qs
            ):
                continue
            assert detected_qs
            detected_qs = prune_invalid_leading_alphanumeric_questions(detected_qs)
            paper_subquestions = [str(sq) for sq in subquestions[0 : len(detected_qs)]]
            # ToDo: maybe turn this assertion into repeated LLM calls until they match
            assert paper_subquestions == detected_qs, (
                f"Subquestions parsed from paper {paper_subquestions} do not match "
                f"those parsed from the markscheme {detected_qs} for page {page_num}"
            )
            [subquestions.pop(0) for _ in range(len(detected_qs))]
            num = latest_num
            char = latest_char
            for i, item in enumerate(detected_qs):
                if item in VALID_NUMERALS:
                    question_to_pages[f"{num}.{char}.{item}"] = [page_num]
                elif item.isalpha():
                    if (
                        len(detected_qs) == i + 1
                        or (
                            detected_qs[i + 1] not in VALID_NUMERALS
                            and detected_qs[i + 1].isalpha()
                        )
                        or detected_qs[i + 1].isdigit()
                    ):
                        question_to_pages[f"{num}.{item}"] = [page_num]
                    char = item
                elif item.isdigit():
                    if len(detected_qs) == i + 1 or detected_qs[i + 1].isdigit():
                        question_to_pages[item] = [page_num]
                    num = int(item)
                else:
                    raise ValueError(f"Invalid type for question: {item}")
            latest_num = num
            latest_char = "`" if detected_qs[-1].isnumeric() else char
            all_detected_qs += detected_qs
            if not subquestions:
                break
        return _fill_missing_questions_n_pages(question_to_pages), latest_num

    question_to_pages, num_questions = parse_into_pages()
    question_to_pages = dict(
        sorted(question_to_pages.items(), key=lambda item: parse_key(item[0])),
    )
    with open(os.path.join(markscheme_dir, "question_to_pages.json"), "w+") as file:
        file.write(json.dumps(question_to_pages, indent=4))

    json_file_lock = threading.Lock()

    def parse_question(question_num: int):
        question_answer_parser = unify.Unify("o1@openai", cache=True)
        mark_breakdown_detector = unify.Unify("o1@openai", cache=True)
        sub_questions = [
            ".".join(k.split(".")[1:])
            for k, v in question_to_pages.items()
            if k.startswith(str(question_num) + ".")
        ]
        pages = [
            v
            for k, v in question_to_pages.items()
            if (k == str(question_num) or k.startswith(str(question_num) + "."))
        ]
        pages = list(dict.fromkeys([item for sublist in pages for item in sublist]))
        current_text = ""
        for pg in pages:
            pg_text = reader.pages[pg - 1].extract_text()
            pg_text = prune_page_number(pg_text, pg, question_num)
            current_text += pg_text
        imgs = [all_images[pg - 1] for pg in pages]
        if sub_questions:
            sub_questions_expr = "sub-questions: " + ", ".join(sub_questions)
            fields_expr = "**all corresponding fields**"
        else:
            sub_questions_expr = "only the question number"
            fields_expr = "the field"
        question_answer_parser.set_system_message(
            QUESTION_ANSWER_PARSER.replace(
                "{question_number}",
                str(question_num),
            )
            .replace(
                "{preceding}",
                str(question_num - 1),
            )
            .replace(
                "{subsequent}",
                str(question_num + 1),
            )
            .replace(
                "{sub-questions}",
                sub_questions_expr,
            )
            .replace(
                "{field(s)}",
                fields_expr,
            ),
        )
        question_answer_parser.set_response_format(
            build_response_format(question_num, sub_questions),
        )
        qna = question_answer_parser.generate(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": current_text,
                        },
                    ]
                    + [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,"
                                f"{encode_image(img)}",
                            },
                        }
                        for img in imgs
                    ],
                },
            ],
        )
        qna = json.loads(qna)
        mark_breakdown_detector.set_system_message(
            (
                MARK_BREAKDOWN_DETECTION
                if sub_questions
                else MARK_BREAKDOWN_DETECTION_NO_SUBQS
            ),
        )
        mark_breakdown_detector.set_response_format(
            build_response_format(question_num, sub_questions + ["total"], int),
        )
        mark_breakdown = mark_breakdown_detector.generate(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(qna, indent=4),
                        },
                    ]
                    + [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,"
                                f"{encode_image(img)}",
                            },
                        }
                        for img in imgs
                    ],
                },
            ],
        )
        mark_breakdown = json.loads(mark_breakdown)
        total_marks = mark_breakdown["total"]
        sum_of_marks = sum([v for k, v in mark_breakdown.items() if k != "total"])
        assert not sub_questions or total_marks == sum_of_marks, (
            "total number of marks must equal the sum of each question component, "
            f"but found {total_marks} and {sum_of_marks} respectively."
        )
        questions[question_num] = {
            "markscheme-components": qna if sub_questions else qna[str(question_num)],
            "mark-breakdown": mark_breakdown,
            "pages": pages,
            "correctly_parsed": True,
        }
        parsed = json.dumps(dict(sorted(questions.items())), indent=4)
        json_file_lock.acquire()
        with open(os.path.join(markscheme_dir, "parsed.json"), "w+") as file:
            file.write(parsed)
        json_file_lock.release()

        # image
        imgs = [diagram_images[pg] for pg in pages if pg in diagram_images]
        if not imgs:
            return
        diagram_detector.set_system_message(
            DIAGRAM_DETECTION_IN_QUESTION.replace(
                "{question_number}",
                str(question_num),
            )
            .replace(
                "{preceding}",
                str(question_num - 1),
            )
            .replace(
                "{subsequent}",
                str(question_num + 1),
            ),
        )
        diagram_response = diagram_detector.generate(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": current_text,
                        },
                    ]
                    + [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encode_image(img)}",
                            },
                        }
                        for img in imgs
                    ],
                },
            ],
        )
        contains_diagram = "yes" in diagram_response.split("\n")[-1].strip().lower()
        # incrementally save to file
        if not contains_diagram:
            return
        img_dir = os.path.join(markscheme_dir, "imgs")
        os.makedirs(img_dir, exist_ok=True)
        fnames = [f"page{pg}.png" for pg in pages if pg in diagram_images]
        questions[question_num]["images"] = fnames
        for fname, img in zip(fnames, imgs):
            if not os.path.exists(fname):
                cv2.imwrite(os.path.join(img_dir, fname), img)

    unify.map(parse_question, list(range(1, num_questions + 1)))
    reader.close()


if __name__ == "__main__":
    parse_pdf_into_papers_and_markschemes()
    subdirs = sorted(os.listdir(pdf_dir))

    def _parse(subdir: str):

        # paper
        target_paper_fpath = os.path.join(pdf_dir, subdir, "paper/parsed.json")
        if not os.path.exists(target_paper_fpath):
            parse_paper(int(subdir))

        target_q_to_pages_fpath = os.path.join(
            pdf_dir,
            subdir,
            "paper/question_to_pages.json",
        )
        with open(target_q_to_pages_fpath) as f:
            target_q_to_pages = json.load(f)
        question_to_subquestions = dict()
        subquestions = list()
        next_char = "a"
        for k in target_q_to_pages.keys():
            if "." not in k:
                question_to_subquestions[int(k)] = list()
                subquestions.append(int(k))
            else:
                k_split = k.split(".")
                q_num = k_split[0]
                q_num = int(q_num)
                if q_num not in question_to_subquestions:
                    subquestions.append(q_num)
                    next_char = "a"
                    question_to_subquestions[q_num] = list()
                letter = k_split[1]
                if letter == next_char:
                    subquestions.append(letter)
                    next_char = chr(ord(letter) + 1)
                if len(k_split) == 3:
                    numeral = k_split[2]
                    subquestions.append(numeral)
                question_to_subquestions[q_num].append(".".join(k_split[1:]))

        # markscheme
        target_markscheme_fpath = os.path.join(
            pdf_dir,
            subdir,
            "markscheme/parsed.json",
        )
        if not os.path.exists(target_markscheme_fpath):
            parse_markscheme(int(subdir), question_to_subquestions, subquestions)

    unify.map(_parse, subdirs)
