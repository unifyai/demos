import os
import json
import random
random.seed(0)

import unify
unify.set_seed(0)
unify.activate("EdTech")

this_dir = os.path.dirname(os.path.abspath(__file__))


# Download Data #
# --------------#

# Functions

pdf_url = "https://raw.githubusercontent.com/unifyai/demos/refs/heads/main/ai_tutor/data/parsed"

def reformat_data():
    logs = list()
    example_id = 0
    for question, data in labelled_data.items():
        data = data.copy()
        del data["question_imgs"]
        del data["markscheme_imgs"]
        data["question_pages"] = [
            f"{pdf_url}/{data['subject'].replace(' ', '_')}/{data['paper_id'].replace(' ', '_')}/paper/pdfs/page{p}.pdf"
            for p in data["question_pages"]
        ]
        data["markscheme_pages"] = [
            f"{pdf_url}/{data['subject'].replace(' ', '_')}/{data['paper_id'].replace(' ', '_')}/markscheme/pdfs/page{p}.pdf"
            for p in data["markscheme_pages"]
        ]
        data["available_marks_total"] = data["available_marks"]
        data["available_marks"] = data["mark_breakdown"]
        del data["mark_breakdown"]
        if not isinstance(data["question_components"], dict):
            data["question_components"] = {"_": data["question_components"]}
        data["sub_questions"] = data["question_components"]
        del data["question_components"]
        if not isinstance(data["markscheme"], dict):
            data["markscheme"] = {"_": data["markscheme"]}
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
                    },
                },
            )
            example_id += 1
    return logs


# Generate Dataset Json #
# ----------------------#

with open(os.path.join(this_dir, "data", "labelled_data.json"), "r") as f:
    labelled_data = json.load(f)

data = reformat_data()
with open(os.path.join(this_dir, "data", "dataset.json"), "w+") as f:
    json.dump(data, f, indent=4)
