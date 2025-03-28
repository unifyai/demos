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


general_guidelines = """----

1.
M marks are for using a correct method and are not lost for purely numerical errors.
A marks are for an accurate answer and depend on preceding M (method) marks. Therefore M0 A1 cannot be awarded.
B marks are independent of M (method) marks and are for a correct final answer, a partially correct answer, or a correct intermediate stage.
SC marks are for special cases that are worthy of some credit.

2.
Unless the answer and marks columns of the mark scheme specify M and A marks etc, or the mark scheme is ‘banded’, then if the correct answer is clearly given and is not from wrong working full marks should be awarded.

Do not award the marks if the answer was obtained from an incorrect method, i.e. incorrect working is seen and the correct answer clearly follows from it.

3.
Where follow through (FT) is indicated in the mark scheme, marks can be awarded where the candidate’s work follows correctly from a previous answer whether or not it was correct.

Figures or expressions that are being followed through are sometimes encompassed by single quotation marks after the word their for clarity, e.g. FT 180 × (their ‘37’ + 16), or FT 300 – (their ‘52 + 72’). Answers to part questions which are being followed through are indicated by e.g. FT 3 × their (a).

For questions with FT available you must ensure that you refer back to the relevant previous answer. You may find it easier to mark these questions candidate by candidate rather than question by question.

4.
Where dependent (dep) marks are indicated in the mark scheme, you must check that the candidate has met all the criteria specified for the mark to be awarded.

5.
The following abbreviations are commonly found in GCSE Mathematics mark schemes.
- **figs 237**, for example, means any answer with only these digits. You should ignore leading or trailing zeros and any decimal point e.g. 237000, 2.37, 2.370, 0.00237 would be acceptable but 23070 or 2374 would not.
- **isw** means **ignore subsequent working** after correct answer obtained and applies as a default.
- **nfww** means not from wrong working.
- **oe** means **or equivalent**.
- **rot** means **rounded or truncated**.
- **seen** means that you should award the mark if that number/expression is seen anywhere in the answer space, including the answer line, even if it is not in the method leading to the final answer
- **soi** means seen or implied.

6.
In questions with no final answer line, make no deductions for wrong work after an acceptable answer (ie **isw**) unless the mark scheme says otherwise, indicated by the instruction ‘mark final answer’.

7.
In questions with a final answer line following working space:

(i)If the correct answer is seen in the body of working and the answer given on the answer line is a clear transcription error allow full marks unless the mark scheme says ‘mark final answer’. Place the annotation ✓ next to the correct answer.

(ii)If the correct answer is seen in the body of working but the answer line is blank, allow full marks. Place the annotation ✓ next to the correct answer.

(iii)If the correct answer is seen in the body of working but a completely different answer is seen on the answer line, then accuracy marks for the answer are lost. Method marks could still be awarded. Use the M0, M1, M2 annotations as appropriate and place the annotation  next to the wrong answer.

8.
In questions with a final answer line:

(i)If one answer is provided on the answer line, mark the method that leads to that answer.

(ii)If more than one answer is provided on the answer line and there is a single method provided, award method marks only.

(iii)If more than one answer is provided on the answer line and there is more than one method provided, award zero marks for the question unless the candidate has clearly indicated which method is to be marked.

9.
In questions with no final answer line:

(i)If a single response is provided, mark as usual.

(ii)If more than one response is provided, award zero marks for the question unless the candidate has clearly indicated which response is to be marked.

10.
When the data of a question is consistently misread in such a way as not to alter the nature or difficulty of the question, please follow the candidate’s work and allow follow through for **A** and **B** marks. Deduct 1 mark from any **A** or **B** marks earned and record this by using the MR annotation. **M** marks are not deducted for misreads.

11.
Unless the question asks for an answer to a specific degree of accuracy, always mark at the greatest number of significant figures even if this is rounded or truncated on the answer line. For example, an answer in the mark scheme is 15 75, which is seen in the working. The candidate then rounds or truncates this to 15.8, 15 or 16 on the answer line. Allow full marks for the 15.75.

12.
Ranges of answers given in the mark scheme are always inclusive.

13.
For methods not provided for in the mark scheme give as far as possible equivalent marks for equivalent work.

14.
Anything in the mark scheme which is in square brackets […] is not required for the mark to be earned, but if present it must be correct.

----"""


system_message = """
Your task is to award a suitable number of marks for a student's answer to a question, from 0 up to a maximum of {available_marks_total} marks.

The general marking guidelines (relevant for all questions) are as follows:

{general_guidelines}

The question you need to mark is:

{question}


The markscheme for this specific question is:

{markscheme}


The student's answer to this question (which you need to marked) is:

{answer}


As the very final part of your response, simply provide the number of marks on a *new line*, without any additional formatting. For example:

3
"""


system_message = system_message.replace(
    "{general_guidelines}", general_guidelines
)


@unify.traced
def call_agent(system_msg, question, markscheme, answer, available_marks_total):
    local_agent = agent.copy()
    local_agent.set_system_message(
        system_msg.replace(
            "{question}", question
        ).replace(
            "{markscheme}", json.dumps(markscheme, indent=4)
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
    markscheme,
    correct_marks_total,
    _system_message,
):
    pred_marks = call_agent(
        _system_message, question, markscheme, student_answer,
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


with unify.Experiment("add_marking_guidelines", overwrite=True), unify.Params(
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
