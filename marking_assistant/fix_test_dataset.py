import json
import os
import random

random.seed(0)

import unify

unify.set_seed(0)
unify.activate("EdTech")

paper_id_dict = {
    "J560 01 Paper 1 (Foundation Tier) Sample Question Paper": [1, 2, 5, 13, 14, 16],
    "J560 02 Paper 2 (Foundation Tier) Sample Question Paper": [
        3,
        4,
        6,
        7,
        9,
        10,
        12,
        15,
        17,
        19,
    ],
    "J560 03 Paper 3 (Foundation Tier) Sample Question Paper": [8, 11, 18, 20],
}

question_num_dict = {
    1: [1],
    11: [2, 6, 10],
    9: [3, 12, 20],
    18: [4],
    8: [5, 8],
    10: [7, 17],
    15: [9],
    17: [11, 19],
    14: [13],
    2: [14],
    7: [15],
    19: [16, 18],
}

correct_marks_total_dict = {
    5: [1, 7, 8, 9],
    0: [2, 18, 20],
    1: [3, 4, 13, 14, 15, 16],
    2: [5, 6],
    10: [10],
    6: [11, 19],
    4: [12],
    7: [17],
}

this_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(this_dir, "data", "test_set.json"), "r") as f:
    data = json.load(f)

for i in range(1, 21):
    paper_id = [k for k, v in paper_id_dict.items() if i in v]
    assert len(paper_id) == 1
    paper_id = paper_id[0]
    question_num = [k for k, v in question_num_dict.items() if i in v]
    assert len(question_num) == 1
    question_num = question_num[0]
    correct_marks_total = [k for k, v in correct_marks_total_dict.items() if i in v]
    assert len(correct_marks_total) == 1
    correct_marks_total = correct_marks_total[0]
    index = [
        idx
        for idx, d in enumerate(data)
        if d["paper_id"] == paper_id
        and d["question_num"] == question_num
        and d["correct_marks_total"] == correct_marks_total
    ]
    assert len(index) == 1
    index = index[0]
    data.insert(0, data.pop(index))

with open(os.path.join(this_dir, "data", "test_set.json"), "w+") as f:
    json.dump(data, f, indent=4)
