import json
import os
import re
import textwrap

import unify
import wget
from pydantic import BaseModel, create_model

unify.activate("MarkingAssistant")
unify.set_context("Evals")


agent = unify.Unify("o3-mini@openai", traced=True, cache="read-only")


if os.path.exists(".cache.json"):
    os.remove(".cache.json")
wget.download(
    "https://raw.githubusercontent.com/"
    "unifyai/demos/refs/heads/main/"
    "marking_assistant/.cache.json",
)


test_set_10 = unify.download_dataset("TestSet10")


def pretty_print_dict(d, indent=0):
    output = ""
    for key, value in d.items():
        if key != "_":
            output += " " * indent + str(key) + ":\n"
        if isinstance(value, dict):
            output += pretty_print_dict(value, indent=indent + (4*int(key!="_")))
        else:
            for line in str(value).splitlines():
                output += " " * (indent + 4) + line + "\n"
    return output


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

general_guidelines = (
    general_guidelines.rstrip("-")
    + """15.
When students are explaining something in their answer, then their explanation must make *exactly* the same point(s) as are made in the markscheme. The wording can be slightly different, but the underlying observations/reasons must be *identical*, unless otherwise stated *explicitly* in the markscheme.

----
"""
)


subq_system_message = """
Your task is to award a suitable number of marks for a student's answer to question {subq}, from 0 up to a maximum of {available_marks} marks.

The general marking guidelines (relevant for all questions) are as follows:

{general_guidelines}


The *overall* question is:

{question}

{prior_context}

The specific question you need to mark is:

{subquestion}


Their answer to this specific question is:

{answer}


The markscheme for this specific question is:

{markscheme}

{mark_types_explanation}

{mark_observations}

{output_response_explanation}
""".replace(
    "{general_guidelines}",
    general_guidelines
)


mark_system_message = """
Your task is to determine whether mark {mark} should be awarded for the following student's answer to question {subq}, based on the provided markscheme.

The general marking guidelines (relevant for all questions) are as follows:

{general_guidelines}


The *overall* question is:

{question}

{prior_context}

The specific question you need to mark is:

{subquestion}


Their answer to this specific question is:

{answer}


The markscheme for this specific question, with the mark in question {mark} expressed in bold and with a prepending `(to consider!)`, is as follows:

{markscheme}

{mark_types_explanation}

You should populate the `thoughts` field with your thoughts on the whether the specific mark identified within the markscheme should be awarded for the student's answer. The mark might be irrelevant given the student's approach or answer, in which case just respond `False` for the `should_award` field, and explain this in the `thoughts` field. Please think carefully about your decision for the mark, considering the general guidelines.
""".replace(
    "{general_guidelines}",
    general_guidelines
)


prior_context_exp = """
All of the *preceeding* sub-questions, their specific markschemes and the student's answers are as follows:
"""


output_response_explanation = "You should populate the `reasoning` field with your general thoughts on each individual mark identified in the markscheme, and also a decision as to whether each of these mark should be awarded. These marks are not necessarily cumulative with regards to the marks to award, and some may be irrelevant given the student's approach or answer, in which case just respond `False` for the `should_award` field. Finally, you should put the total number of marks to award in the `marks` field."


class MarksAndReasoning(BaseModel):
    reasoning: str
    marks: int


class ThoughtsAndAwardDecision(BaseModel):
    thoughts: str
    should_award: bool


@unify.traced(name="create_per_mark_reasoning_format_{mark_types}")
def create_per_mark_reasoning_format(mark_types):
    response_fields = dict(
        zip(
            mark_types + ["overall_thoughts"], [(ThoughtsAndAwardDecision, ...)] * len(mark_types) + [(str, ...)]
        )
    )
    return create_model('PerMarkReasoning', **response_fields)


@unify.traced(name="create_marks_and_reasoning_format_{mark_types}")
def create_marks_and_reasoning_format(mark_types):
    return create_model(
        'MarksAndReasoning',
        reasoning=(create_per_mark_reasoning_format(mark_types), ...),
        marks=(int, ...)
    )


@unify.traced(name="create_response_format_{mark_types}")
def create_response_format(response_keys, mark_types):
    if response_keys:
        response_fields = dict(
            zip(
                response_keys,
                [
                    (create_marks_and_reasoning_format(mark_types[key]), ...)
                    for key in response_keys
                ]
            )
        )
        return create_model('Response', **response_fields)
    else:
        return create_marks_and_reasoning_format(mark_types["_"])


@unify.traced(name="parse_marks_from_markscheme{subquestion}")
def parse_marks_from_markscheme(subquestion: str, markscheme: str):
    extracted_marks = re.findall(r'(?:SC|M|A|B)\d+', markscheme)
    if not extracted_marks:
        return []
    marks_n_context = list()
    for i, mark in enumerate(extracted_marks):
        index = markscheme.find(mark)
        chunk = markscheme[0:index]
        if i > 0:
            prev_mark = extracted_marks[i-1]
            marks_n_context[i-1][1] += chunk
        markscheme = markscheme[index:]
        marks_n_context.append([mark, chunk])
    marks_n_context[-1][1] += markscheme
    return marks_n_context


mark_types = {
    "M": "M{num} ({num_marks}) should be awarded if a correct method is used, and should not be lost for purely numerical errors.",
    "A": "A{num} ({num_marks}) should be awarded for an accurate answer, and this depends on preceding M (method) marks. If preceding M (method marks are not awarded, then A{num} cannot be awarded).",
    "B": "B{num} ({num_marks}) should be awarded for the correct final answer, a partially correct answer, or a correct intermediate stage (depending on how this is expressed and explained below). B{num} is independent of M (method) marks.",
    "SC": "SC{num} ({num_marks}) should be awarded for the special cases explained below, which are worthy of some credit.",
}


@unify.traced(name="update_markscheme{subquestion}")
def update_markscheme(subquestion: str, markscheme: str):
    m_marks = sorted(list(set(re.findall(r"M\d+", markscheme))))
    a_marks = sorted(list(set(re.findall(r"A\d+", markscheme))))
    b_marks = sorted(list(set(re.findall(r"B\d+", markscheme))))
    sc_marks = sorted(list(set(re.findall(r"SC\d+", markscheme))))
    if not any(m_marks + a_marks + b_marks + sc_marks):
        return markscheme
    markscheme = "{mark_types}With this in mind, marks should be awarded as follows:\n" + markscheme
    for marks in (m_marks, a_marks, b_marks, sc_marks):
        for mark in marks:
            key = "".join(c for c in mark if not c.isdigit())
            num_marks = int("".join(c for c in mark if c.isdigit()))
            explanation = mark_types[key]
            explanation = explanation.replace(
                "{num}",
                str(num_marks),
            ).replace(
                "{num_marks}",
                "1 mark" if num_marks == 1 else f"{num_marks} marks",
            )
            markscheme = markscheme.replace(
                "{mark_types}",
                explanation + "\n{mark_types}",
            )
    markscheme = markscheme.replace(
        "{mark_types}",
        "",
    )
    return markscheme


@unify.traced(name="extract_mark_type_explanation_{marks_to_consider}")
def extract_mark_type_explanation(markscheme: str, marks_to_consider=None):
    m_marks = sorted(list(set(re.findall(r'M\d+', markscheme))))
    a_marks = sorted(list(set(re.findall(r'A\d+', markscheme))))
    b_marks = sorted(list(set(re.findall(r'B\d+', markscheme))))
    sc_marks = sorted(list(set(re.findall(r'SC\d+', markscheme))))
    if not any(m_marks + a_marks + b_marks + sc_marks):
        return ""
    full_exp = """As a recap, the general guidelines for each of these mark types are as follows:

{mark_types_explanation}"""
    for marks in (m_marks, a_marks, b_marks, sc_marks):
        for mark in marks:
            if marks_to_consider and mark not in marks_to_consider:
                continue
            key = "".join(c for c in mark if not c.isdigit())
            num_marks = int("".join(c for c in mark if c.isdigit()))
            exp = mark_types[key]
            exp = exp.replace(
                "{num}", str(num_marks)
            ).replace(
                "{num_marks}", "1 mark" if num_marks == 1 else f"{num_marks} marks"
            )
            full_exp = full_exp.replace(
                "{mark_types_explanation}",
                key + ":/n" + exp + "\n\n{mark_types_explanation}"
            )
    return full_exp.replace("{mark_types_explanation}", "")


@unify.traced(name="call_subq_agent_{subq}")
def call_subq_agent(example_id, subq, subq_agent, markscheme, mark_sys_msg):
    parsed_markscheme = parse_marks_from_markscheme(f"_{subq}" if subq != "_" else "", markscheme)
    mark_agents = [
        [k, agent.copy()] for k in
        [itm[0] for itm in parsed_markscheme]
    ]
    [
        agnt.set_response_format(ThoughtsAndAwardDecision)
        for _, agnt in mark_agents
    ]
    for i, (k, v) in enumerate(parsed_markscheme):
        mark_agents[i][1].set_system_message(
            mark_sys_msg.replace(
                "{mark}", k
            ).replace(
                "{markscheme}", markscheme
            ).replace(
                v, v.replace(k, f"**{k}** (to consider!)")
            ).replace(
                "{mark_types_explanation}",
                extract_mark_type_explanation(markscheme, [k])
            )
        )
    if mark_agents:
        explanation = "An expert marker has already taken a look at the student's answer, and they have made the following observations for each of the candidate marks mentioned in the markscheme. You should pay special attention to these observations."
        vals = unify.map(
            lambda a: json.loads(a.generate()),
            [agnt for _, agnt in mark_agents],
            name=f"Evals[{example_id}]->SubQAgent[{subq}]->MarkAgent"
        )
        keys = list()
        for k, _ in mark_agents:
            if k not in keys:
                keys.append(k)
                continue
            keys.append(
                k + f"({len([ky for ky in keys if k in ky])})"
            )
        mark_obs_dict = dict(zip(keys, vals))
        mark_observations = explanation + "\n\n" + pretty_print_dict(
            mark_obs_dict, indent=4
        )
    else:
        mark_observations = ""
    subq_agent.set_system_message(
        subq_agent.system_message.replace(
            "{mark_observations}",
            mark_observations
        )
    )
    ret = subq_agent.generate()
    if "```" in ret:
        ret = ret.split("```")[-2].lstrip("json")
    ret = json.loads(ret)
    if not mark_agents:
        return ret
    ret["reasoning"] = {
        **mark_obs_dict,
        "overall_thoughts": ret["reasoning"]
    }
    return ret


@unify.traced
def call_agent(
    example_id,
    subq_system_message,
    mark_system_message,
    question_num,
    question,
    sub_questions,
    markscheme,
    answer,
    available_marks
):
    subq_agents = {k: agent.copy() for k in markscheme.keys()}
    with_subqs = len(markscheme) > 1
    response_formats = {
        k: MarksAndReasoning for k, v in markscheme.items()
    }
    [
        agnt.set_response_format(rf)
        for agnt, rf in zip(
            subq_agents.values(), response_formats.values()
        )
    ]
    mark_sys_msgs = list()
    for i, k in enumerate(markscheme.keys()):
        subq_agents[k].set_system_message(
            subq_system_message.replace(
                "{subq}", k.replace("_", str(question_num))
            ).replace(
                "{question}", question,
            ).replace(
                "{subquestion}", sub_questions[k]
            ).replace(
                "{markscheme}", markscheme[k]
            ).replace(
                "{mark_types_explanation}",
                extract_mark_type_explanation(markscheme[k])
            ).replace(
                "{answer}", answer[k]
            ).replace(
                "{available_marks}",
                str(available_marks[k.replace("_", "total")])
            ).replace(
                "{output_response_explanation}",
                output_response_explanation
            ).replace(
            "{prior_context}", (prior_context_exp + pretty_print_dict(
              {
                  k: {
                      "sub-question": sub_questions[k],
                      "markscheme": markscheme[k],
                      "answer": answer[k]
                  } for k in list(sub_questions.keys())[0:i]
              },
              indent=4
            )) if with_subqs and i > 0 else ""
          )
        )
        mark_sys_msgs.append(
            mark_system_message.replace(
                "{subq}", k.replace("_", str(question_num))
            ).replace(
                "{question}", textwrap.indent(question, " " * 4),
            ).replace(
                "{subquestion}", textwrap.indent(sub_questions[k], " " * 4),
            ).replace(
                "{answer}", textwrap.indent(answer[k], " " * 4),
            ).replace(
            "{prior_context}", (prior_context_exp + pretty_print_dict(
              {
                  k: {
                      "sub-question": sub_questions[k],
                      "markscheme": markscheme[k],
                      "answer": answer[k]
                  } for k in list(sub_questions.keys())[0:i]
              },
              indent=4
            )) if with_subqs and i > 0 else ""
          )
        )
    rets = unify.map(
        lambda *a: call_subq_agent(example_id, *a),
        list(sub_questions.keys()),
        list(subq_agents.values()),
        list(markscheme.values()),
        mark_sys_msgs,
        from_args=True,
        name=f"Evals[{example_id}]->SubQAgent"
    )
    return dict(zip(markscheme.keys(), rets))


@unify.log
def evaluate(
    example_id,
    question_num,
    question,
    sub_questions,
    student_answer,
    available_marks,
    markscheme,
    correct_marks,
    per_question_breakdown,
    _subq_system_message,
    _mark_system_message
):
    pred_marks = call_agent(
        example_id,
        _subq_system_message,
        _mark_system_message,
        question_num,
        question,
        sub_questions,
        markscheme,
        student_answer,
        available_marks
    )
    pred_marks_total = sum([v["marks"] for v in pred_marks.values()])
    diff = {
        k: vcor["marks"] - vpred["marks"] for (k, vcor), (_, vpred) in
        zip(correct_marks.items(), pred_marks.items())
    }
    error = {k: abs(v) for k, v in diff.items()}
    diff_total = sum(diff.values())
    error_total = sum(error.values())
    per_question_breakdown = {
        k: {
            **per_question_breakdown[k],
            "predicted_marks": pm,
            "diff": d
        } for (k, pqb), pm, d in zip(
            per_question_breakdown.items(),
            pred_marks.values(),
            diff.values()
        )
    }
    return error


with unify.Experiment(
    "queries_per_mark",
    overwrite=True,
), unify.Params(
    subq_system_message=subq_system_message,
    mark_system_message=mark_system_message,
    dataset="TestSet10",
    source=unify.get_source(),
):
    unify.map(
        evaluate,
        [dict(**d.entries, _subq_system_message=subq_system_message, _mark_system_message=mark_system_message) for d in test_set_10],
        name="Evals",
    )
