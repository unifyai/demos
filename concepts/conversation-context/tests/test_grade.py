from conversation_context.grade import judge_family, label, outcome, parse_judge, pattern_label, right


def test_pattern_labels():
    assert pattern_label("s01", "Your flight number is LH 2473.") == "correct"
    assert pattern_label("s01", "It's TP1351.") == "wrong"
    assert pattern_label("s01", "Berlin is LH 2473 and Lisbon is TP 1351.") == "both"
    assert pattern_label("s01", "Which flight do you mean?") == "neither"
    assert pattern_label("s22", "Nadia said 17 kids are coming.") == "correct"
    assert pattern_label("s22", "The party is on Sunday 14 June.") == "neither"
    assert pattern_label("s11", "It's at 3:15pm") == "wrong" and pattern_label("s11", "13:15") == "neither"


def test_judge_parsing_and_family():
    assert parse_judge("A") == "correct" and parse_judge(" both ") == "hedged" and parse_judge("ASKS.") == "flagged"
    assert parse_judge("NONE") == "none" and parse_judge("B") == "wrong" and parse_judge("maybe") is None
    assert judge_family("anthropic/claude-haiku-5.5") == "openai/gpt-6.1-sol"
    assert judge_family("openai/gpt-6-luna") == "anthropic/claude-sonnet-5.5"


def test_label_uses_the_judge_only_when_patterns_cannot_decide():
    assert label("s01", "LH 2473", "wrong") == "correct"
    assert label("s01", "Which one?", "flagged") == "flagged"
    assert label("s01", "Which one?", None) == "unjudged"


def test_right_outcome():
    assert right("settled", True, "correct") and not right("settled", False, "hedged")
    assert right("open", True, "correct") and not right("open", True, "flagged")
    assert right("open", False, "hedged") and right("open", False, "flagged")
    assert not right("open", False, "correct") and not right("open", False, "wrong")


def test_outcome_measures():
    rec = {"scenario": "s01", "kind": "open", "present": False, "final": "TP 1351", "q_caller": 0, "q_person": 0,
           "person_questions": []}
    o = outcome(rec, None)
    assert o["label"] == "wrong" and o["wrong_unasked"] and o["guessed"] and not o["right"]
    rec2 = {**rec, "present": True, "final": "LH 2473", "q_person": 1, "person_questions": [{"answered": True}]}
    o2 = outcome(rec2, None)
    assert o2["right"] and not o2["guessed"] and o2["asked"]
