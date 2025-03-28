import os
import wget
import json
import unify


unify.activate("MarkingAssistant")
unify.set_context("Evals")


agent = unify.Unify("o3-mini@openai", traced=True, cache="read-only")


if os.path.exists(".cache.json"):
    os.remove(".cache.json")
wget.download(
    "https://raw.githubusercontent.com/"
    "unifyai/demos/refs/heads/main/"
    "marking_assistant/.cache.json"
)


test_set_10 = unify.download_dataset("TestSet10")


system_message = """
Your task is to award a suitable number of marks for a student's answer to a question, from 0 up to a maximum of {available_marks_total} marks.

The question is:

{question}


Their answer to this question is:

{answer}


As the very final part of your response, simply provide the number of marks on a *new line*, without any additional formatting. For example:

3
"""


@unify.traced
def call_agent(system_msg, question, answer, available_marks_total):
    local_agent = agent.copy()
    local_agent.set_system_message(
        system_msg.replace(
            "{question}", question
        ).replace(
            "{answer}", json.dumps(answer, indent=4)
        ).replace(
            "{available_marks_total}", str(available_marks_total)
        )
    )
    return local_agent.generate()


@unify.log
def evaluate(
    question,
    student_answer,
    available_marks_total,
    correct_marks_total,
    _system_message,
):
    pred_marks = call_agent(
        _system_message, question, student_answer,
        available_marks_total
    )
    _pred_marks_split = pred_marks.split("\n")
    pred_marks_total, diff_total, error_total = None, None, None
    for _substr in reversed(_pred_marks_split):
        _extracted = "".join([c for c in _substr if c.isdigit()])
        if _extracted != "":
          pred_marks_total = int(_extracted)
          diff_total = correct_marks_total - pred_marks_total
          error_total = abs(diff_total)
          break
    pred_marks = {"_": {"marks": pred_marks_total, "rationale": pred_marks}}
    return


with unify.Experiment("simple_agent", overwrite=True), unify.Params(
    system_message=system_message,
    dataset="TestSet10",
    source=unify.get_source()
):
    unify.map(
        evaluate,
        [
             dict(**d.entries, _system_message=system_message)
             for d in test_set_10
        ],
        name="Evals"
    )
