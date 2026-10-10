import pytest

from fast_slow_voice.keys import answer_in, is_holding, key_in
from fast_slow_voice.script import build_sanity, check, generate
from fast_slow_voice.timing import ModelTiming, Timing, heard_prefix, resume_from_sentence, speech_seconds
from fast_slow_voice.world import Tools, tool_latency


def test_generation_is_deterministic_and_paired_across_sizes():
    a, b = generate(7, 3), generate(7, 3)
    assert [s.world("small").fingerprint() for s in a] == [s.world("small").fingerprint() for s in b]
    s = a[0]
    small, big = s.world("small"), s.world("realistic")
    planted = {c.name for c in s.planted.contacts}
    assert planted <= {c.name for c in small.contacts} and planted <= {c.name for c in big.contacts}
    assert len(big.contacts) == 300 and len(small.contacts) == 20


@pytest.mark.parametrize("base", [1, 2])
def test_every_planted_answer_is_reachable_and_unambiguous(base):
    problems = []
    for s in generate(base, 16):
        problems += check(s, ["small", "medium", "realistic"])
    assert problems == []
    assert check(build_sanity("sanity-0", 99), ["sanity"]) == []


def test_tools_disambiguate_and_resolve():
    s = next(s for s in generate(1, 16) if any(i.kind == "phone_ambig" for i in s.items))
    it = next(i for i in s.items if i.kind == "phone_ambig")
    w = s.world("realistic")
    T = Tools(w)
    first = it.person.split()[0]
    both = T.t_find_contacts(first)
    assert {h["name"] for h in both if h["name"].startswith(first + " ")} == {it.person, it.extra["distractor"]}
    assert T.t_send_text(first, "hi")["error"].startswith("ambiguous")
    assert T.t_send_text(it.person, "hi")["to"] == it.person
    assert T.t_calendar("not a date") == {"error": "dates must be YYYY-MM-DD"}
    code = next((i for x in generate(1, 16) for i in x.items if i.kind == "code_latest"), None)
    assert code is not None


def test_tool_latency_is_a_function_of_the_call_only():
    a = tool_latency("find_contacts", {"query": "x"}, 5, 0)
    assert a == tool_latency("find_contacts", {"query": "x"}, 5, 0)
    assert a != tool_latency("find_contacts", {"query": "x"}, 5, 1)
    assert 0.1 < a < 3.0


def test_keys():
    assert key_in("Her mobile is 312-555-0147.", "3125550147", "digits")
    assert key_in("Her mobile is (312) 555 0147.", "3125550147", "digits")
    assert not key_in("Her mobile is 312-555-014.", "3125550147", "digits")
    assert key_in("The code is 8167#.", "8167", "digits")
    assert not key_in("The code is 58167#.", "8167", "digits")
    assert not key_in("it's 5821, and at 4:30", "8214", "digits")
    assert key_in("You owe $2,340.50.", "234050", "digits")
    assert key_in("You're on UA 1187.", "UA1187", "code")
    assert key_in("It's at 3:20 pm.", "3:20", "time") and not key_in("at 13:20", "3:20", "time")
    assert key_in("ronan dot vance at juniperhealth dot example", "ronan.vance@juniperhealth.example", "email")
    assert answer_in("It's at 410 W Erie St.", ["410", "erie"], "tokens")
    assert not answer_in("It's at 410 W Grand St.", ["410", "erie"], "tokens")
    assert is_holding("One sec, let me check.") and is_holding("Sure, one sec.")
    assert not is_holding("Priya's number is 312-555-0147.") and not is_holding("Morning, Sam! I'm well, thanks.")


def test_timing_model_and_speech():
    t = Timing(models={"m": ModelTiming(1.0, 0.5, 50.0)})
    first, done = t.call_times("m", 2000, 150, 100)
    assert first == pytest.approx(1.0 + 1.0 + 2.0) and done == pytest.approx(1.0 + 1.0 + 3.0)
    assert speech_seconds("Her mobile is 312-555-0147.", 0.4) == pytest.approx(0.4 * 8)
    assert heard_prefix("Her mobile is 312-555-0147. Anything else?", 0.4, 1.4) == ("Her mobile is", "312-555-0147. Anything else?")
    assert resume_from_sentence("Sure thing. Her mobile is 312-555-0147. Anything else?", 3) == "Her mobile is 312-555-0147. Anything else?"
    assert resume_from_sentence("Sure thing. Her mobile is 312-555-0147.", 0) == "Sure thing. Her mobile is 312-555-0147."


def test_flight_numbers_said_with_the_airline_name():
    assert key_in("Your flight is United 2375, leaving at 2 pm.", "UA2375", "code")
    assert key_in("It's Delta flight 1752.", "DL1752", "code")
    assert not key_in("It's flight 2375.", "UA2375", "code")
