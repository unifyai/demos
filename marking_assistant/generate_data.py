import os
import json
import cv2
import argparse
from pydantic import BaseModel, create_model

import unify

unify.CLIENT_LOGGING = True
from prompts import *
from helpers import load_questions_and_answers, encode_image


parser = argparse.ArgumentParser()
parser.add_argument(
    "--labelled",
    help="Generate synthetic labelled data",
    action="store_true",
)
parser.add_argument(
    "--usage",
    help="Generate synthetic usage data",
    action="store_true",
)
args = parser.parse_args()
mode = "usage" if args.usage else "labelled"

this_dir = os.path.dirname(__file__)
pdfs_dir = os.path.join(this_dir, "pdfs")


class AnswerForTargetMarks(BaseModel):
    marks: int
    answer: str
    rationale: str


def generate_question(question, data, idx):
    subject_dir = os.path.join(pdfs_dir, data["subject"].replace(" ", "_"))
    paper_dir = os.path.join(subject_dir, data["paper_id"].replace(" ", "_"))
    paper_imgs_dir = os.path.join(paper_dir, "paper/imgs")
    assert os.path.exists(paper_imgs_dir)
    img_fpaths = [
        os.path.join(paper_imgs_dir, f"page{pg}.png") for pg in data["question_pages"]
    ]
    imgs = [cv2.imread(fpath, -1) for fpath in img_fpaths]
    to_write = dict()
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
    fname = f"{mode}_data_{idx}"
    data_path = os.path.join(data_dir, fname)
    generation_client = unify.Unify("o1@openai", cache=True)
    targets = dict()
    for target in range(data["marks"]["total"] + 1):
        generation_client.set_system_message(
            GENERATE_RESPONSE_PROMPT.replace(
                "{target}",
                str(target),
            )
            .replace(
                "{num_marks}",
                str(data["marks"]["total"]),
            )
            .replace(
                "{question}",
                question,
            )
            .replace(
                "{question_num}",
                str(data["question_num"]),
            )
            .replace(
                "{markscheme}",
                json.dumps(data["answer"], indent=4),
            )
            .replace(
                "{mark_breakdown}",
                json.dumps(data["marks"], indent=4),
            ),
        )
        sub_questions = [k for k in data["marks"] if k != "total"]
        if sub_questions:
            response_keys = sub_questions
            response_fields = dict(
                zip(response_keys, [(AnswerForTargetMarks, ...)] * len(response_keys)),
            )
            response_format = create_model("Response", **response_fields)
        else:
            response_format = AnswerForTargetMarks
        generation_client.set_response_format(response_format)
        response = generation_client.generate(
            messages=[
                {
                    "role": "user",
                    "content": [
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
        response = json.loads(response)
        if sub_questions:
            sum_of_marks = sum([v["marks"] for k, v in response.items()])
        else:
            sum_of_marks = response["marks"]
        assert sum_of_marks == target, (
            "The sum of marks awarded across sub-questions "
            f"{json.dumps(response, indent=4)} is not equal "
            f"to the target {target}"
        )
        targets[target] = response
    targets["subject"] = data["subject"]
    targets["paper_id"] = data["paper_id"]
    targets["question_num"] = int(data["question_num"])
    targets["question_components"] = data["question_components"]
    targets["markscheme"] = data["answer"]
    targets["available_marks"] = int(data["marks"]["total"])
    targets["mark_breakdown"] = data["marks"]
    targets["question_pages"] = [int(pg) for pg in data["question_pages"]]
    targets["markscheme_pages"] = [int(pg) for pg in data["markscheme_pages"]]
    if data["question_imgs"]:
        targets["question_imgs"] = [fp.split("/")[-1] for fp in data["question_imgs"]]
    else:
        targets["question_imgs"] = None
    if data["markscheme_imgs"]:
        targets["markscheme_imgs"] = [
            fp.split("/")[-1] for fp in data["markscheme_imgs"]
        ]
    else:
        targets["markscheme_imgs"] = None
    # incremental file writing
    to_write[question] = targets
    with open(data_path, "w+") as f:
        f.write(json.dumps(to_write, indent=4))


def combine_data():
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
    if os.path.exists(os.path.join(data_dir, f"{mode}_data.json")):
        os.remove(os.path.join(data_dir, f"{mode}_data.json"))
    data_paths = [
        os.path.join(data_dir, fname)
        for fname in os.listdir(data_dir)
        if fname.startswith(f"{mode}_data_")
    ]
    data = {}
    for data_path in data_paths:
        with open(data_path, "r") as f:
            this_data = json.load(f)
        data = {**data, **this_data}
        os.remove(data_path)
    with open(os.path.join(data_dir, f"{mode}_data.json"), "w+") as f:
        f.write(json.dumps(data, indent=4))


def main():
    qna = load_questions_and_answers()
    args = [
        (question, dct, i)
        for i, (question, dct) in enumerate(qna.items())
        if dct["correctly_parsed"]
    ]
    unify.map(generate_question, args)
    combine_data()


if __name__ == "__main__":
    os.makedirs(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "data"),
        exist_ok=True,
    )
    assert not (
        args.labelled and args.usage
    ), "Please specify either --labelled or --usage, not both."
    assert args.labelled or args.usage, "Please specify one of --labelled or --usage."
    main()
