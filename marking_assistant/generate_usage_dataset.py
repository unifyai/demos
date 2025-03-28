import datetime
import json
import math
import os
import random
import shutil

import unify
import wget

unify.activate("MarkingAssistant")

this_dir = os.path.dirname(os.path.abspath(__file__))

CORRELATIONS = {
    # How usage frequency depends on age.
    # usage_prob ~ exp( usage_age_factor * (age - 40) )
    # => if age > 40, usage increases exponentially; if age < 40, < 1 => less usage
    "usage_age_factor": 0.02,
    # Time-of-day usage distribution
    # We'll define a normal distribution for "peak usage hour" with mean:
    #   mean_hour = time_base_hour + time_age_factor * (age - time_age_center)
    "time_age_center": 40.0,
    "time_age_factor": -0.3,
    # Negative => older => earlier usage (midday); younger => evening
    "time_base_hour": 16.0,  # 16:00 is the average usage hour for someone at age=40
    "time_hour_spread": 3.0,  # std dev in hours
    # Score distribution by age
    # We use a logistic transform for "prob of a high score" that depends on:
    #   logistic( score_age_slope * (age - score_age_center) + gender_boost )
    "score_age_center": 25.0,
    "score_age_slope": 0.3,
    # Gender-based shift in the logistic for scoring
    "score_gender_boost_male": 0.2,
    "score_gender_boost_female": -0.1,
}


def compute_usage_weight(age):
    """
    People over 40 use the platform more, with a continuous,
    exponential weighting that depends on age - 40.
    """
    factor = CORRELATIONS["usage_age_factor"]
    # Weighted by exp( factor * (age - 40) )
    return math.exp(factor * (age - 40))


def parse_date_of_birth(dob_str):
    """
    Expects dob_str in format YYYY-MM-DD or similar.
    Adjust as needed if your data is different.
    Returns integer age (approx).
    """
    # Very naive approach, just split on '-'
    # In real code, consider using datetime.strptime
    parts = dob_str.split("-")
    if len(parts) < 1:
        return 30  # fallback if something goes wrong
    birth_year = int(parts[0])
    current_year = datetime.datetime.now().year
    return current_year - birth_year


def weighted_random_choice(students):
    """
    Pick a student with probability proportional to the usage_weight
    computed from their age. This ensures older folks get chosen more often.
    """
    # 1) Compute each student's usage weight
    weights = []
    for student in students:
        age = parse_date_of_birth(student["date_of_birth"])
        w = compute_usage_weight(age)
        weights.append(w)

    total_weight = sum(weights)
    if total_weight <= 0:
        # fallback to uniform random
        return random.choice(students)

    # 2) Sample from that distribution
    r = random.random() * total_weight
    cumulative = 0.0
    for student, w in zip(students, weights):
        cumulative += w
        if r < cumulative:
            return student

    # fallback
    return students[-1]


def sample_usage_time(age):
    """
    Return a datetime (or just hour) representing the usage time.
    We define a normal distribution with:
      mean_hour = time_base_hour + time_age_factor * (age - time_age_center)
      stdev = time_hour_spread
    """
    mean_hour = CORRELATIONS["time_base_hour"] + CORRELATIONS["time_age_factor"] * (
        age - CORRELATIONS["time_age_center"]
    )
    stdev = CORRELATIONS["time_hour_spread"]

    # Sample from normal distribution
    sampled_hour = random.gauss(mean_hour, stdev)

    # Wrap to [0, 24)
    if sampled_hour < 0:
        sampled_hour = 24 + (sampled_hour % 24)
    else:
        sampled_hour = sampled_hour % 24

    # Convert to an actual datetime (today’s date)
    today = datetime.datetime.now().date()
    hour_int = int(sampled_hour)
    minute_int = int((sampled_hour - hour_int) * 60)
    usage_dt = datetime.datetime(
        year=today.year,
        month=today.month,
        day=today.day,
        hour=hour_int,
        minute=minute_int,
        second=random.randint(0, 59),
    )
    return usage_dt


def logistic(x):
    """Classic logistic function."""
    return 1.0 / (1.0 + math.exp(-x))


def sample_score(age, gender, available_marks):
    """
    We use a logistic function to model the probability of obtaining
    a 'high' mark. Then we sample from 0..available_marks accordingly.

    For simplicity, we define "score index" from 0..available_marks,
    pick one according to a distribution that skews higher as logistic(...) -> bigger.

    You can implement more nuanced distributions if you wish.
    """
    # logistic argument
    slope = CORRELATIONS["score_age_slope"]
    center = CORRELATIONS["score_age_center"]

    # Transform gender to a numeric offset
    # Avoid a discrete "if" to handle male vs female;
    # for example, a dictionary of multipliers:
    #   male => +0.2, female => -0.1
    gender_map = {
        "male": CORRELATIONS["score_gender_boost_male"],
        "female": CORRELATIONS["score_gender_boost_female"],
    }
    gender_boost = gender_map.get(gender.lower(), 0.0)

    # logistic argument
    logit_val = slope * (age - center) + gender_boost
    high_score_prob = logistic(logit_val)

    # We'll treat 'high_score_prob' as the chance to pick the top half of the marks range
    # Then, for the lower half, we distribute the remaining probability.
    # This is just one approach; you can invent more nuanced ones if you like.

    # Let mid = available_marks // 2
    # Probability of score > mid is high_score_prob
    # Probability of score <= mid is (1 - high_score_prob)
    # We'll sample a discrete number in [0, available_marks].

    mid = available_marks // 2
    # Are we above or below mid?
    if random.random() < high_score_prob:
        # pick from mid+1..available_marks uniformly
        return random.randint(mid + 1, available_marks)
    else:
        return random.randint(0, mid)


fpath = os.path.join(this_dir, "data/labelled_data.json")
if not os.path.exists(fpath):
    wget.download(
        "https://raw.githubusercontent.com/unifyai/demos/refs/heads/main/ai_tutor/data/labelled_data.json",
    )
    shutil.move("labelled_data.json", fpath)
with open(fpath, "r") as f:
    usage_data = json.load(f)

fpath = os.path.join(this_dir, "data/students.json")
if not os.path.exists(fpath):
    wget.download(
        "https://raw.githubusercontent.com/unifyai/demos/refs/heads/main/ai_tutor/data/students.json",
    )
    shutil.move("students.json", fpath)
with open(fpath, "r") as f:
    student_data = json.load(f)

questions = list(usage_data.keys())
img_dir = "parsed/GCSE_(9–1)_Mathematics"


dataset = []

for _ in range(10000):

    # 1) Pick a student *probabilistically* based on usage weights
    student = weighted_random_choice(student_data)
    student_age = parse_date_of_birth(student["date_of_birth"])

    # 2) Pick a question (uniform random, or do your own weighting)
    question = random.choice(questions)
    question_dict = usage_data[question]

    # 3) Determine a numeric score to aim for
    available_marks = int(question_dict["available_marks"])
    gender = student["gender"]
    score = sample_score(student_age, gender, available_marks)

    # Standardize markscheme dtype
    if not isinstance(question_dict["markscheme"], dict):
        question_dict["markscheme"] = {"_": question_dict["markscheme"]}

    # 4) Pick the answer string from the question corresponding to the score
    ans_n_rat = question_dict[str(score)]
    if "answer" in ans_n_rat:
        provided_answer = {"_": ans_n_rat["answer"]}
        correct_marks_rationale = {"_": ans_n_rat["rationale"]}
        correct_marks_breakdown = {"_": ans_n_rat["marks"]}
    else:
        provided_answer = {k: v["answer"] for k, v in ans_n_rat.items()}
        correct_marks_rationale = {k: v["rationale"] for k, v in ans_n_rat.items()}
        correct_marks_breakdown = {k: v["marks"] for k, v in ans_n_rat.items()}

    # 5) Simulate the time of usage
    usage_timestamp = sample_usage_time(student_age)

    # 8) Log the event
    data = {
        "student/timestamp": str(usage_timestamp),
        "student/first_name": student["first_name"],
        "student/last_name": student["last_name"],
        "student/email": student["email"],
        "student/gender": student["gender"],
        "student/date_of_birth": student["date_of_birth"],
        "question/subject": question_dict["subject"],
        "question/paper_id": question_dict["paper_id"],
        "question/question_num": question_dict["question_num"],
        "question/question": question,
        "question/provided_answer": provided_answer,
        "question/markscheme": question_dict["markscheme"],
        "question/available_marks": question_dict["available_marks"],
        "question/chosen_score": score,
    }

    dataset.append(data)

with open(os.path.join(this_dir, "data/usage_data.json"), "w+") as f:
    json.dump(dataset, f, indent=4)
