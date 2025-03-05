QUESTION_DETECTION = """
Your task is to detect the numeric question numbers [1, 2, 3 etc.], letter sub-questions
[a, b, c etc.] *and* roman numeral sub-sub-questions [i, ii, iii] that are present in
the following text.

When deciding whether the number refers to a question, look at the *subsequent text*.

Does it look like a question or a description of a problem for subsequent
questioning? If so, then it's likely a question.

The first question on the page can either be a number (new question) or a letter (continuation of question on the previous page).

Please respond in the exact order presented on the current page, starting either with a number or a letter, whichever appears on the page first.

If there are any numbered questions on the page, then the first of these numbers will be `{n0}`.
The presence of this specific number {n0} before any explanatory text or question means that this represents the question number, and should be included in your answer.

If the *first question identifier* on the page is a letter (not a number), then this
*first* letter will be `{c0}`. This is because question {n-1} on the previous page {explanation}.
However, if the first question identifier is {n0}, then the any letters immediately
following this number (or immediately following any other question number),
will always be `a`, and this will then ascend alphabetically until the next
question number. Likewise, roman-numeral sub-sub-questions will always come after a
letter, and will also be strictly ascending starting from `i`.

A full image of the page has also been included, please look at this image to help
understand the page *layout*, which might help you make sense of which numbers
indicate pages vs questions.

Remember, sometimes questions are split across **multiple
pages**, with only letters starting on the next page without showing the previous
question number, but these should **always** be included in your answer, as these
lettered sub-questions must be detected on every page, even if it overflows from a
prior question on the previous page.

Think through your reasoning in detail, step by step.

As the final part of your response, please respond with "answer: " followed by a
single comma separated list of numbers, letters and roman numerals on a new line,
like either of the two examples below:
answer: {n0}, a, i, ii, b, i, {n1}, a, {n2}, a, b, c, i, ii
answer: {c0}, {c1}, {c2}, {n0}, a, b, i, ii, c,

If you do not see any clear questions on the page, then just explain why, with no comma-seperated list at the end.
"""

QUESTION_ANSWER_DETECTION = """
Your task is to detect the numeric question numbers [1, 2, 3 etc.], letter sub-questions
[a, b, c etc.] *and* roman numeral sub-sub-questions [i, ii, iii] that are present in
the following markscheme text, which contains questions and answers.

When deciding whether the number refers to a question, look at *all of* the text within
the corresponding row in the table. Also note that question numbers are always
strictly ascending incrementing by one whole number at a time.

Please ignore general guidelines to the marker, which may also come ordered with
numbers. If this is what you see, then please give an empty response as explained
below. Similarly, if the table of questions is part of an Assessment Objectives (AO)
Grid, then please give an empty response (this is not the mark scheme).

Sometimes the table will not include any headings such as Question, Answer etc. The
question numbers are always on the left hand side, and M1, M2 indicate the marks for
each point. If you see this, it means this is part of the mark scheme, and there is
likely a number and/or letter to be extracted.

The first question on this page should be `{i0}`, possibly followed by `{i1}`,
`{i2}`, ... (if they're on this page) because {detected_so_far},
and it's known that the questions in the paper are structured as follows:

{full_question_structure}

A full image of the page has also been included, please look at this image to help
understand the page *layout*, which might help you make sense of which numbers
indicate pages vs questions vs marks awarded. Specifically, question numbers are
likely to be listed under a "Questions" column, and marks under a "Marks" column.
This structure might not be clear from the text-only representation you've been
provided with, so please use both sources when considering your answer.

Remember, sometimes questions are split across **multiple
pages**, with only letters or lower-case roman numerals starting on the next page
without showing the previous question number, but these should **always** be included
in your answer, as these lettered and roman numeral sub-questions must be detected on
every page, even if it overflows from a prior question on the previous page.

Think through your reasoning in detail, step by step.

As the final part of your response, please respond with "answer: " followed by a
single comma separated list of the numbers, letters and roman numerals **found on the page**,
with the response on a new line like so:
answer: {subquestions}

If you do not believe any of the text on the screen corresponds to questions in the
mark scheme table,then just explain why, with no comma-seperated list at the end.
"""

QUESTION_PARSER = """
Your task is to extract the full contents of question {question_number} from the
following text and images. You should *not* extract any parts of the
preceding question {preceding} or subsequent question {subsequent}.

The question was parsed from a PDF, and the formatting might be strange or wrong as a
result of this conversion to pure text. More importantly, mathematical symbols
such as √w, x², y₄, ⁴√z etc. are very often missed by the parsing logic. Sometimes,
entire equations are embedded in the PDF as images, and will not be shown in the text.
Image(s) of the relevant page(s) have therefore *also* been provided.

Please extract **all** important information for question {question_number}
**including any symbols and/or equations missing in the text**,
by inferring these from the provided image(s).

Furthermore, if the formatting could be improved to make the question more readable
in text-only format, please make any formatting improvements as you see fit.

Please respond with the **question only**. Do not provide any explanations, commentary
or preliminary details as part of your answer. Just respond with the newly formatted
question and nothing else.
"""

QUESTION_COMPONENT_PARSER = """
Your task is to parse the following question into it's known sub-components,
whilst disregarding general explanatory text. For example, if a question has the
following structure:

```
2. This is some general information about the question, explaining the problem.

a) This is a question.

Here is some more information.

b) i) This is another question.

   ii) This is yet another question.
```

Then only the following should be output:

```
{
    "a": "This is a question."
    "b.i": "This is another question."
    "b.ii": "This is yet another question."
}
```
"""

QUESTION_ANSWER_PARSER = """
Your task is to extract the full contents of the correct answer and
marking guidelines for question {question_number} from the
following text and images. You should *not* extract any parts of the
answer or guidelines for the preceding question {preceding} or subsequent question {subsequent}.

Specifically, you should extract **all parts** of question {question_number},
as indicated by the requested output format, containing {sub-questions}.

The only exception to this is the assessment objects which **should be omitted**.
These assessment objectives are of the form `{n} AO{1|2|3}.{m{a|b|c|etc.}}`,
such as `1 AO1.2a`, `3 AO2.1` and `3 AO2.4b`. These assessment objectives should
not be fully omitted from your response, as they are not relevant for awarding
the correct marks.

The text was parsed from a PDF, and the formatting might be strange or wrong as a
result of this conversion to pure text. More importantly, mathematical symbols
such as √w, x², y₄, ⁴√z etc. are very often missed by the parsing logic. Sometimes,
entire equations are embedded in the PDF as images, and will not be shown in the text.
Image(s) of the relevant page(s) have therefore *also* been provided.

Please extract **all** text for the markscheme of question {question_number},
**including any symbols and/or equations missing in the text**,
by inferring these from the provided image(s).

The question of interest (question {question_number}) contains {sub-questions},
and {field(s)} of the requested structured output representation
must be **fully populated** based on the information which is known to exist in the
text and images(s). Remember, sometimes questions are split across **multiple
pages**, with only letters starting on the next page without showing the question
number, but these should **always** be included as part of the question on the previous
page.

Please look at this image to help understand the page *layout*, which might help you
make sense of which numbers indicate pages vs questions vs marks awarded.
Specifically, question numbers are likely to be listed under a "Questions" column,
and marks under a "Marks" column. This structure might not be clear from the
text-only representation you've been provided with,
so please use both sources when considering your answer.

If the formatting could be improved to make the markscheme for each
question component more readable in text-only format,
please make any formatting improvements as you see fit.
"""

DIAGRAM_DETECTION_ON_PAGE = """
Your task is to determine whether there is a diagram included as part of the pdf page
presented. Given the full text on the page and the screenshot of the page.
Of course, the PDF screenshot is itself an image, but the question is whether there
is a *diagram* within the questions presented anywhere on the PDF page, as part of
one of the questions.
Simply respond either yes or no as the last part of your response, on a new line.
"""

DIAGRAM_DETECTION_IN_QUESTION = """
Your task is to determine whether the diagrams in the attached screenshot(s) are part of
 question {question_number}, or another question.
You should *not* consider the preceding question {preceding} or subsequent question
{subsequent}. Looking only at whether the diagram is part of question {question_number}.
simply respond either yes or no as the last part of your response, on a new line.
"""

TEXT_ONLY_DETECTION = """
Your task is to determine whether the following question can be answered in a
text-only manner, or if it requires drawing on the page in a non-textual manner.
The answer can still be text-only even if there is a diagram as part of the question
(not included in the text you're provided with).
Presuming that the recipient *does* have access to any necessary diagrams,
we want to determine if the question can be trivially answered via alphanumeric text
with a keyboard.
Please simply respond on a new line with yes if the question can be answered using
only a keyboard, and no if it requires drawing on the page in some way,
like one of the following examples:
yes
no
"""

MARK_BREAKDOWN_DETECTION = """
Your task is to determine the total number of marks available for each component of
this question. The total number of marks are shown under the marks heading (disregard
any info about assessment objectives which you might see in the image, such as AOx.xx).
The total number of marks for each sub-question is usually shown as a single
integer, but it might also be shown as a list of mark types, prepending with a
letter. For example, B1 B1 B1 would indicate 3 marks. A1 B1 would indicate 2 marks
etc. Please response with a single integer per field in the requested response
format, as well as the total number of marks summed across all fields.
"""

MARK_BREAKDOWN_DETECTION_NO_SUBQS = """
Your task is to determine the total number of marks available this question.
The total number of marks are shown under the marks heading (disregard any
info about assessment objectives which you might see in the image, such as AOx.xx).
Please response with a single integer in the requested response format.
"""

GENERATE_PROMPT = """
Given the following question:
{question}

With the following mark scheme:
{mark_scheme}

Provide an answer to the question which will achieve a score of {achieved_mark} out
of {available_marks}
"""

STUDENT_PROMPT = """
Answer the following question to the best of your ability:
{question}
"""

JUDGE_PROMPT = """
Given the following question:
{question}

The following mark scheme guidelines for the question:
{guidelines}

And the following response from a student:
{response}

Please inform the student whether their answer was correct, partially correct,
or incorrect, with a clear justification. Provide this assessment as a string on a new
line as the final part of your response, like so:
correct|partially correct|incorrect
"""

GENERATE_RESPONSE_PROMPT = """
Your task is to generate a response to question {question_num} below which should
achieve a total of {target} out of {num_marks} available marks, if it was to be
marked correctly.

The marking scheme adds marks based on the following guidelines:

----

1.
M marks are for using a correct method and are not lost for purely numerical errors.
A marks are for an accurate answer and depend on preceding M (method) marks. Therefore M0 A1 cannot be awarded.
B marks are independent of M (method) marks and are for a correct final answer, a partially correct answer, or a correct intermediate stage.
SC marks are for special cases that are worthy of some credit.

2.
Unless the answer and marks columns of the mark scheme specify M and A marks etc, or the mark scheme is ‘banded’, then if the correct
answer is clearly given and is not from wrong working full marks should be awarded.

Do not award the marks if the answer was obtained from an incorrect method, i.e. incorrect working is seen and the correct answer clearly follows
from it.

3.
Where follow through (FT) is indicated in the mark scheme, marks can be awarded where the candidate’s work follows correctly from a
previous answer whether or not it was correct.

Figures or expressions that are being followed through are sometimes encompassed by single quotation marks after the word their for clarity,
e.g. FT 180 × (their ‘37’ + 16), or FT 300 – (their ‘52 + 72’). Answers to part questions which are being followed through are indicated by e.g. FT
3 × their (a).

For questions with FT available you must ensure that you refer back to the relevant previous answer. You may find it easier to mark these
questions candidate by candidate rather than question by question.

4.
Where dependent (dep) marks are indicated in the mark scheme, you must check that the candidate has met all the criteria specified for the
mark to be awarded.

5.
The following abbreviations are commonly found in GCSE Mathematics mark schemes.
- **figs 237**, for example, means any answer with only these digits. You should ignore
 leading or trailing zeros and any decimal point e.g. 237000, 2.37, 2.370, 0.00237
 would be acceptable but 23070 or 2374 would not.
- **isw** means **ignore subsequent working** after correct answer obtained and applies as a default.
- **nfww** means not from wrong working.
- **oe** means **or equivalent**.
- **rot** means **rounded or truncated**.
- **seen** means that you should award the mark if that number/expression is seen
anywhere in the answer space, including the answer line, even if it is not in the method leading to the final answer
- **soi** means seen or implied.

6.
In questions with no final answer line, make no deductions for wrong work after an
acceptable answer (ie **isw**) unless the mark scheme says otherwise, indicated by
the instruction ‘mark final answer’.

7.
In questions with a final answer line following working space:

(i)If the correct answer is seen in the body of working and the answer given on the answer line is a clear transcription error allow full marks
unless the mark scheme says ‘mark final answer’. Place the annotation ✓ next to the correct answer.

(ii)If the correct answer is seen in the body of working but the answer line is blank, allow full marks. Place the annotation ✓ next to the
correct answer.

(iii)If the correct answer is seen in the body of working but a completely different answer is seen on the answer line, then accuracy marks for
the answer are lost. Method marks could still be awarded. Use the M0, M1, M2 annotations as appropriate and place the annotation 
next to the wrong answer.

8.
In questions with a final answer line:

(i)If one answer is provided on the answer line, mark the method that leads to that answer.

(ii)If more than one answer is provided on the answer line and there is a single method provided, award method marks only.

(iii)If more than one answer is provided on the answer line and there is more than one method provided, award zero marks for the question
unless the candidate has clearly indicated which method is to be marked.

9.
In questions with no final answer line:

(i)If a single response is provided, mark as usual.

(ii)If more than one response is provided, award zero marks for the question unless the candidate has clearly indicated which response is to be
marked.

10.
When the data of a question is consistently misread in such a way as not to alter the nature or difficulty of the question, please follow the
candidate’s work and allow follow through for **A** and **B** marks. Deduct 1 mark
from any **A** or **B** marks earned and record this by using the MR annotation.
**M** marks are not deducted for misreads.

11.
Unless the question asks for an answer to a specific degree of accuracy, always mark at the greatest number of significant figures even if
this is rounded or truncated on the answer line. For example, an answer in the mark scheme is 15.75, which is seen in the working. The
candidate then rounds or truncates this to 15.8, 15 or 16 on the answer line. Allow full marks for the 15.75.

12.
Ranges of answers given in the mark scheme are always inclusive.

13.
For methods not provided for in the mark scheme give as far as possible equivalent marks for equivalent work.

14.
Anything in the mark scheme which is in square brackets […] is not required for the mark to be earned, but if present it must be correct.

----

Given the following question, which is worth a total of {num_marks} marks:

{question}

And also given the following known correct marking scheme:

{markscheme}

And the known correct distribution of marks:

{mark_breakdown}

Provide a complete answer to *all parts* of this question [including answers to all
sub-questions (a), (b), (i), (ii) etc. if they exist] which should achieve a total
of {target} marks for the entire question, if marked correctly.

Please state your reasoning in lots of detail in each sub-question's `rationale: str`
field of the response format, referring specifically to the markscheme provided for
each of the marks the answer should receive. State the intended marks to be awarded
for each sub-question in the `marks: int` field.

Image(s) of the page have also been provided, in case these help provide extra context.
You don't need to make use of the images if the text is sufficient.

For the `answer: str` part of the response format, please provide your proposed
answer(s) as though the question was answered by a student (who has no knowledge about
the markscheme). Please **include any working** which is necessary for the student to
attain {target} marks, even if this means repeating some parts in `answer: str` that
were already explained in `rationale: str`. However, do not provide commentary in this
answer. The answer will be directly parsed and used as a student answer, without any
awareness about the marks attained within the answer.
"""
