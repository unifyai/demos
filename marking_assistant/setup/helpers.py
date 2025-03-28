import base64
import json
import os
from typing import Callable, List

import cv2
from pydantic import create_model

VALID_NUMERALS = ("i", "ii", "iii", "iv", "v", "vi")


def build_response_format(question_num, sub_questions, dtype=str):
    response_keys = sub_questions if sub_questions else [str(question_num)]
    response_fields = dict(zip(response_keys, [(dtype, ...)] * len(response_keys)))
    return create_model("Response", **response_fields)


def is_invalid_question_order(detected_qs, valid_num, valid_char):
    valid_numeral = 0
    for i, item in enumerate(detected_qs):
        if item in VALID_NUMERALS:
            if item != VALID_NUMERALS[valid_numeral]:
                return True
            valid_numeral += 1
        elif item.isalpha():
            if item != valid_char:
                return True
            valid_char = chr(ord(valid_char) + 1)
            valid_numeral = 0
        elif item.isdigit():
            if item != valid_num:
                return True
            valid_num = str(int(valid_num) + 1)
            valid_char = "a"
            valid_numeral = 0
        else:
            raise ValueError(f"Invalid type for question: {item}")
    return False


def update_str_in_table(table: List, fn: Callable):
    new_table = list()
    for i, item in enumerate(table):
        if isinstance(item, str):
            new_table.append(fn(item))
        elif isinstance(item, list):
            new_table.append(update_str_in_table(item, fn))
        elif item is None:
            new_table.append("")
        else:
            raise ValueError(
                "Expected nested list, but found {} of type {}".format(fn, type(item)),
            )
    return new_table


def prune_invalid_leading_alphanumeric_questions(detected_qs):
    ret = list()
    next_num = None
    next_char = None
    next_numeral = None
    for q in reversed(detected_qs):
        if q in VALID_NUMERALS:
            if not next_numeral or q == next_numeral:
                ret.append(q)
            else:
                break
            next_numeral_idx = VALID_NUMERALS.index(q) - 1
            next_numeral = (
                VALID_NUMERALS[next_numeral_idx] if next_numeral_idx >= 0 else None
            )
        elif q.isalpha():
            if not next_char or q == next_char:
                ret.append(q)
            else:
                break
            next_char = chr(ord(q) - 1) if q != "a" else None
        elif q.isnumeric():
            q = int(q)
            if not next_num or q == next_num:
                ret.append(str(q))
            else:
                break
            next_num = q - 1
            if next_num == 0:
                break
    return list(reversed(ret))


def parse_key(k: str):
    """
    Splits the string on the first dot.
    - The part before the dot is turned into an integer (if there's no dot, the whole key is an integer).
    - The remainder (if any) is kept as a suffix for secondary sorting.
    """
    parts = k.split(".", 1)
    num = int(parts[0])  # integer portion
    suffix = parts[1] if len(parts) > 1 else ""  # suffix after the first dot, if any
    return num, suffix


def encode_image(image_path):
    _, buffer = cv2.imencode(".jpg", image_path)
    return base64.b64encode(buffer).decode("utf-8")


def load_questions_and_answers():
    questions_and_answers = dict()
    pdfs_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "pdfs")
    for subject_dir in os.listdir(pdfs_dir):
        if subject_dir.endswith(".pdf"):
            continue
        subject = subject_dir.replace("_", " ")
        subject_dir_abs = os.path.join(pdfs_dir, subject_dir)
        for paper_dir in os.listdir(subject_dir_abs):
            if paper_dir.endswith(".pdf"):
                continue
            paper_id = paper_dir.replace("_", " ")
            paper_dir_abs = os.path.join(subject_dir_abs, paper_dir)
            with open(os.path.join(paper_dir_abs, "markscheme/parsed.json")) as f:
                markscheme = json.load(f)
            with open(os.path.join(paper_dir_abs, "paper/parsed.json")) as f:
                questions = json.load(f)
            for question_num, question in questions.items():
                if not question["text-only"]:
                    continue
                ans_n_marks = markscheme[question_num]
                markscheme_components, mark_breakdown = (
                    ans_n_marks["markscheme-components"],
                    ans_n_marks["mark-breakdown"],
                )
                question_imgs = question.get("images")
                if question_imgs:
                    question_imgs = [
                        os.path.join(
                            os.path.join(paper_dir_abs, "paper/imgs"),
                            fname,
                        )
                        for fname in question_imgs
                    ]
                markscheme_imgs = ans_n_marks.get("images")
                if markscheme_imgs:
                    markscheme_imgs = [
                        os.path.join(
                            os.path.join(paper_dir_abs, "markscheme/imgs"),
                            fname,
                        )
                        for fname in markscheme_imgs
                    ]
                correctly_parsed = (
                    question["correctly_parsed"] and ans_n_marks["correctly_parsed"]
                )
                questions_and_answers[question["question"]] = {
                    "subject": subject,
                    "paper_id": paper_id,
                    "question_num": question_num,
                    "question_components": question["question-components"],
                    "question_pages": question["pages"],
                    "markscheme_pages": ans_n_marks["pages"],
                    "answer": markscheme_components,
                    "marks": mark_breakdown,
                    "question_imgs": question_imgs,
                    "markscheme_imgs": markscheme_imgs,
                    "correctly_parsed": correctly_parsed,
                }
        return questions_and_answers
