import base64
import os
import json
import cv2
import random
random.seed(0)

import unify
unify.set_seed(0)
unify.activate("EdTech")

this_dir = os.path.dirname(os.path.abspath(__file__))


# Download Data #
# --------------#

# Functions

def encode_image(image):
    _, buffer = cv2.imencode(".jpg", image)
    return base64.b64encode(buffer).decode("utf-8")


def create_dataset():
    logs = list()
    example_id = 0
    for question, data in labelled_data.items():
        data = data.copy()
        data["available_marks_total"] = data["available_marks"]
        data["available_marks"] = data["mark_breakdown"]
        del data["mark_breakdown"]
        if not isinstance(data["question_components"], dict):
            data["question_components"] = {"_": data["question_components"]}
        data["sub_questions"] = data["question_components"]
        del data["question_components"]
        if not isinstance(data["markscheme"], dict):
            data["markscheme"] = {"_": data["markscheme"]}
        subject_dir = os.path.join("data/parsed", data["subject"].replace(" ", "_"))
        paper_dir = os.path.join(subject_dir, data["paper_id"].replace(" ", "_"))
        q_imgs_dir = os.path.join(paper_dir, "paper/imgs")
        q_img_fpaths = [f"{q_imgs_dir}/page{pg}.png" for pg in data["question_pages"]]
        question_imgs = [encode_image(cv2.imread(fpath, -1)) for fpath in q_img_fpaths]
        m_imgs_dir = os.path.join(paper_dir, "markscheme/imgs")
        m_img_fpaths = [f"{m_imgs_dir}/page{pg}.png" for pg in data["markscheme_pages"]]
        markscheme_imgs = [
            encode_image(cv2.imread(fpath, -1)) for fpath in m_img_fpaths
        ]
        for mark, ans_n_rat in data.items():
            if not all(c.isdigit() for c in mark):
                continue
            mark_int = int(mark)
            if "answer" in ans_n_rat:
                student_answer = {"_": ans_n_rat["answer"]}
                correct_marks = {
                    "_": {
                        "marks": ans_n_rat["marks"],
                        "rationale": ans_n_rat["rationale"],
                    },
                }
            else:
                student_answer = {k: v["answer"] for k, v in ans_n_rat.items()}
                correct_marks = {
                    k: {
                        "marks": v["marks"],
                        "rationale": v["rationale"],
                    }
                    for k, v in ans_n_rat.items()
                }
                correct_marks_breakdown = {k: v["marks"] for k, v in ans_n_rat.items()}
            per_question_breakdown = {
                k: {
                    "sub_question": q,
                    "available_marks": am,
                    "student_answer": sa,
                    "markscheme": ms,
                    "correct_marks": cm,
                }
                for (k, q), am, sa, ms, cm in zip(
                    data["sub_questions"].items(),
                    data["available_marks"].values(),
                    student_answer.values(),
                    data["markscheme"].values(),
                    correct_marks.values(),
                )
            }
            per_question_breakdown["question"] = question
            logs.append(
                {
                    **{k: v for k, v in data.items() if not k.isdigit()},
                    **{
                        "example_id": example_id,
                        "question": question,
                        "student_answer": student_answer,
                        "correct_marks": correct_marks,
                        "correct_marks_total": mark_int,
                        "per_question_breakdown": per_question_breakdown,
                        "question_pages": question_imgs,
                        "markscheme_pages": markscheme_imgs,
                    },
                },
            )
            example_id += 1
    return logs


# Execute

with open(os.path.join(this_dir, "data", "labelled_data.json"), "r") as f:
    labelled_data = json.load(f)

data = create_dataset()
dataset = unify.Dataset(data, name="TestSet").sync()
