import asyncio
import copy

import pytest
from helpers import FAST, FAST_THINK, SLOW, FakeLLM, Oracle, instruction_of, make_timing

from fast_slow_voice.arms import ARMS, Arm, Models, Relay, SlowSpeaks, VoiceLoop, Worker, make_arm, parse_label
from fast_slow_voice.metrics import call_metrics
from fast_slow_voice.script import Trigger, Turn, build_sanity, generate
from fast_slow_voice.sim import Sim
from fast_slow_voice.timing import speech_seconds

MODELS = Models(FAST, SLOW, FAST_THINK)


def custom_script(turns: list[tuple[str, str, Trigger]]):
    """A sanity script with its turns replaced: (text, kind, trigger)."""
    s = copy.deepcopy(build_sanity("t-0", 5))
    s.turns = [Turn(f"t{i + 1}", text, kind, trig, kind != "ack", "x") for i, (text, kind, trig) in enumerate(turns)]
    return s


def run(script, arm, llm, timing=None, size="sanity"):
    sim = Sim(script, script.world(size), arm, timing or make_timing(), llm, run_seed=0, tag="test", meta={})
    if hasattr(llm.policy, "bind"):
        llm.policy.bind(sim)
    asyncio.run(sim.run())
    return sim


def no_llm(*a):
    raise AssertionError("no model call expected")


class Scripted(Arm):
    """Says fixed lines when Sam's turns are heard, without any model."""

    def __init__(self, lines: dict):
        super().__init__(MODELS)
        self.lines = lines

    def on_person(self, sim, pt):
        for text, kind in self.lines.get(pt.turn.id, []):
            sim.say(text, "single", kind)


def test_barge_in_cuts_the_line_and_drops_queued_voice_speech_but_not_verbatim():
    long = " ".join(["word"] * 20)
    script = custom_script([("Hello there Ava", "greeting", Trigger("at", delay=0.5)),
                            ("Stop, sorry", "aside", Trigger("at", delay=2.0)),
                            ("Bye now", "closing", Trigger("after_any", gap=0.5, patience=30))])
    arm = Scripted({"t1": [(long, "voice"), ("queued voice line", "voice"), ("verbatim line", "verbatim")]})
    sim = run(script, arm, FakeLLM(no_llm))
    t1, t2 = sim.persons[0], sim.persons[1]
    assert t1.t_start == pytest.approx(0.5) and t1.t_end == pytest.approx(0.5 + speech_seconds("Hello there Ava", 0.35))
    u_long, u_voice, u_verb = sim.utts[0], sim.utts[1], sim.utts[2]
    assert u_long.t_start == pytest.approx(t1.t_heard)  # starts when the turn is heard
    assert t2.barged and t2.t_start == pytest.approx(t1.t_end + 2.0)
    assert u_long.cut and u_long.t_end == pytest.approx(t2.t_start + 0.2)
    n_heard = int((u_long.t_end - u_long.t_start) / 0.4 + 1e-9)
    assert u_long.heard == " ".join(["word"] * n_heard)
    assert u_voice.dropped and u_voice.t_start is None
    assert not u_verb.dropped and u_verb.t_start == pytest.approx(t2.t_heard)
    assert len(sim.persons) == 3


def test_calls_are_resolved_before_later_events():
    """A call's speech lands at its virtual time even though Sam's next turn was
    already scheduled; the simulator awaits calls started before an event."""
    script = custom_script([("Hi Ava", "greeting", Trigger("at", delay=0.5)),
                            ("Anyway", "aside", Trigger("at", delay=4.0))])

    def policy(model, messages, kw, meta):
        return ("Hello Sam", [], 2, 0)

    sim = run(script, make_arm("fast_only", MODELS), FakeLLM(policy), timing=make_timing(fast_a=1.0))
    t1 = sim.persons[0]
    u = sim.utts[0]
    assert u.text == "Hello Sam" and u.t_start == pytest.approx(t1.t_heard + 1.0 + 0.25)
    assert sim.persons[1].t_start == pytest.approx(t1.t_end + 4.0) and not sim.persons[1].barged


def test_voice_loop_cancels_a_call_overtaken_before_its_first_token():
    script = custom_script([("Hi Ava", "greeting", Trigger("at", delay=0.5)),
                            ("And another thing", "aside", Trigger("at", delay=0.2)),
                            ("Bye", "closing", Trigger("after_any", gap=0.5, patience=30))])
    n = {"k": 0}

    def policy(model, messages, kw, meta):
        n["k"] += 1
        return (f"reply {n['k']}", [], 2, 0)

    sim = run(script, make_arm("slow_only", MODELS), FakeLLM(policy), timing=make_timing(slow_a=3.0))
    calls = list(sim.calls.values())
    assert calls[0].cancelled and not calls[1].cancelled
    said = [u.text for u in sim.utts if u.t_start is not None]
    assert "reply 1" not in said and said[0] == "reply 2"
    assert "Sam spoke" in calls[1].meta["events"][0] and len(calls[1].meta["events"]) == 2


def test_worker_never_cancels_and_batches_what_arrives_meanwhile():
    script = custom_script([("Hi Ava", "greeting", Trigger("at", delay=0.5)),
                            ("How are things", "aside", Trigger("at", delay=0.5)),
                            ("Bye", "closing", Trigger("after_any", gap=0.5, patience=40))])

    def policy(model, messages, kw, meta):
        return ("", [], 5, 0) if meta["role"] == "slow" else ("Hi!", [], 2, 0)

    sim = run(script, Relay(MODELS), FakeLLM(policy), timing=make_timing(slow_a=4.0))
    slow = [c for c in sim.calls.values() if c.role == "slow"]
    assert not any(c.cancelled for c in slow)
    assert slow[0].meta["triggers"] == [["person", "t1"]]
    # t2 (and the goodbye, which also came while the first call ran) wait for the next call
    assert slow[1].meta["triggers"][0] == ["person", "t2"]
    assert slow[1].t0 == pytest.approx(slow[0].t_done)
    assert sim.persons[1].t_heard < slow[0].t_done


def test_relay_slow_text_becomes_a_note_the_fast_model_passes_on():
    script = custom_script([("What's the code?", "greeting", Trigger("at", delay=0.5)),
                            ("Bye", "closing", Trigger("after_any", gap=0.5, patience=40))])

    def policy(model, messages, kw, meta):
        if meta["role"] == "slow":
            return ("code: 4417#", [], 5, 0)
        if "note from the back office arrived" in instruction_of(messages):
            return ("The code is 4417#.", [], 4, 0)
        return ("Let me check on that.", [], 4, 0)

    sim = run(script, Relay(MODELS), FakeLLM(policy))
    assert [n.text for n in sim.notes][0] == "code: 4417#"
    said = [u.text for u in sim.utts if u.t_start is not None]
    assert said[:2] == ["Let me check on that.", "The code is 4417#."]


def test_parse_label():
    assert parse_label("HOLD: Sure, one sec.") == ("HOLD", "Sure, one sec.")
    assert parse_label("SMALL_TALK: Morning!") == ("SMALL_TALK", "Morning!")
    assert parse_label("**REPLAY**") == ("REPLAY", "")
    assert parse_label("NONE") == ("NONE", "")
    assert parse_label("Small talk: hi") == ("SMALL_TALK", "hi")
    assert parse_label("Hello there") == ("UNLABELLED", "Hello there")
    assert parse_label("HOLD: Sure, one sec.\n\nHOLD: Sure, one sec.") == ("HOLD", "Sure, one sec.")
    assert parse_label("Sure.\nHOLD: One sec.") == ("HOLD", "One sec.")


def test_slow_speaks_small_talk_cancels_the_slow_run_and_hold_keeps_it():
    script = custom_script([("Morning Ava!", "greeting", Trigger("at", delay=0.5)),
                            ("What's the code?", "aside", Trigger("after_any", gap=0.5, patience=20)),
                            ("Bye", "closing", Trigger("after_any", gap=0.5, patience=40))])

    def policy(model, messages, kw, meta):
        instr = instruction_of(messages)
        if meta["role"] == "fast":
            if "Morning Ava" in instr:
                return ("SMALL_TALK: Morning, Sam!", [], 6, 0)
            if "Bye" in instr:
                return ("SMALL_TALK: Bye!", [], 4, 0)
            return ("HOLD: One sec.", [], 6, 0)
        if "(t1)" in instr:
            return ("Good morning from the slow model.", [], 8, 0)
        if "(t2)" in instr:
            return ("The code is 4417#.", [], 8, 0)
        return ("", [], 4, 0)

    sim = run(script, SlowSpeaks(MODELS), FakeLLM(policy))
    said = [u.text for u in sim.utts if u.t_start is not None]
    assert "Good morning from the slow model." not in said
    assert said[:3] == ["Morning, Sam!", "One sec.", "The code is 4417#."]
    slow = [c for c in sim.calls.values() if c.role == "slow"]
    assert slow[0].cancelled and not slow[1].cancelled
    fast = [c for c in sim.calls.values() if c.role == "fast"]
    assert fast[0].meta["slow_run"] == "cancelled"


def test_slow_speaks_replay_resumes_from_the_sentence_that_was_cut():
    script = custom_script([("What's her number?", "greeting", Trigger("at", delay=0.5)),
                            ("Sorry, one sec. Okay, go on.", "interrupt_resume",
                             Trigger("barge_answer", delay=1.2, patience=30, items=["q1"])),
                            ("Bye", "closing", Trigger("after_any", gap=0.5, patience=40))])
    gold = script.item("q1").gold
    line = f"Sure thing. Her mobile is {gold}. Anything else?"

    def policy(model, messages, kw, meta):
        instr = instruction_of(messages)
        if meta["role"] == "fast":
            if "go on" in instr:
                assert "Not heard:" in instr
                return ("REPLAY", [], 2, 0)
            return ("HOLD: One sec.", [], 6, 0) if "number" in instr else ("SMALL_TALK: Bye!", [], 4, 0)
        if "(t1)" in instr:
            return (line, [], 20, 0)
        return ("", [], 4, 0)

    sim = run(script, SlowSpeaks(MODELS), FakeLLM(policy))
    cut = next(u for u in sim.utts if u.text == line)
    assert cut.cut and cut.heard == "Sure thing. Her"
    replay = next(u for u in sim.utts if u.source == "replay")
    assert replay.text == f"Sorry, as I was saying. Her mobile is {gold}. Anything else?"
    assert replay.t_start is not None and not replay.cut


def test_slow_filler_plays_a_canned_filler_only_while_the_slow_model_is_working():
    script = custom_script([("What's the code?", "greeting", Trigger("at", delay=0.5)),
                            ("Bye", "closing", Trigger("after_substance", gap=0.5, patience=40))])

    def policy(model, messages, kw, meta):
        return ("The code is 4417#.", [], 8, 0) if "What's the code" in str(messages) and "Bye" not in instruction_of(messages) else ("Bye!", [], 2, 0)

    sim = run(script, make_arm("slow_filler", MODELS), FakeLLM(policy), timing=make_timing(slow_a=3.0))
    said = [(u.source, u.text) for u in sim.utts if u.t_start is not None]
    assert said[0] == ("filler", "One moment.") and said[1] == ("single", "The code is 4417#.")
    t1 = sim.persons[0]
    assert sim.utts[0].t_start == pytest.approx(t1.t_end + 1.2)
    # a quick model leaves no room for the filler
    sim2 = run(script, make_arm("slow_filler", MODELS), FakeLLM(policy), timing=make_timing(slow_a=0.3))
    assert all(u.source != "filler" for u in sim2.utts)


@pytest.mark.parametrize("arm", ARMS)
def test_every_arm_with_a_perfect_model_delivers_every_answer(arm):
    for s in generate(3, 3):
        sim = run(s, make_arm(arm, MODELS), FakeLLM(Oracle(arm)), size="small")
        rec = sim.record()
        m = call_metrics(rec, s, sim.world)
        assert m["turns_spoken"] == m["turns_total"], arm
        assert m["n_correct"] == m["n_wanted"], (arm, [r for r in m["items"] if not r["delivered"]])
        assert m["invented"] == [] and m["flags"] == []
        assert all(a["sent_right"] and a["confirmed"] and not a["premature_claim"] for a in m["actions"])
        assert all(t["ttfw"] is not None for t in m["ttfw"] if t["kind"] in ("greeting", "question", "closing"))


def test_silence_markers_and_spoken_text_cleaning():
    from fast_slow_voice.sim import clean_spoken, is_placeholder
    for t in ["[silent]", "<br>", "(no text)", "Nothing new.", "[silent] Sam is only thanking me", "",
              "Sam has said goodbye, so there's nothing more to say.\n\n[silent]",
              "[\u0455ilent]", "[\u10dasilent]", "[\u0561\u0571\u0561\u0575\u0576silent]", "[\u0aa7 silent]"]:  # garbled by GPT-6 Luna
        assert is_placeholder(t), t
    for t in ["No problem.", "Okay.", "Hi Sam", "Sure, one sec. (silently checking)", "The code is 4417#."]:
        assert not is_placeholder(t), t
    assert clean_spoken("Sam is wrapping up.\n\nAva: Bye, Sam!") == "Bye, Sam!"
    assert clean_spoken("Ava (voice): One sec.") == "One sec."


def test_made_up_tools_are_not_run_and_empty_notes_are_not_sent():
    script = custom_script([("Hi Ava", "greeting", Trigger("at", delay=0.5)),
                            ("Bye", "closing", Trigger("after_any", gap=0.5, patience=20))])

    def policy(model, messages, kw, meta):
        if meta["role"] == "slow":
            return ("<br>", [("noop", {})], 5, 0)
        return ("Hi!" if "Bye" not in instruction_of(messages) else "Bye!", [], 2, 0)

    sim = run(script, Relay(MODELS), FakeLLM(policy))
    assert sim.tool_calls == [] and sim.notes == []
    assert all(c.meta.get("ignored_tools") == ["noop"] for c in sim.calls.values() if c.role == "slow")


def test_sam_does_not_wait_for_a_reply_to_an_aside_once_the_answer_is_in():
    s = copy.deepcopy(build_sanity("t-1", 6))
    q1 = s.item("q1")
    s.turns = [Turn("t1", q1.text, "question", Trigger("at", delay=0.5), True, "x", asks=["q1"]),
               Turn("t2", "Ugh, traffic.", "aside", Trigger("after_any", gap=0.6, patience=3.0), False, "x"),
               Turn("t3", "Got it, thanks.", "ack", Trigger("after_answer", gap=0.5, patience=25.0, items=["q1"]), False, "x")]

    def policy(model, messages, kw, meta):
        instr = instruction_of(messages)
        return (f"It's {q1.gold}.", [], 5, 0) if "0.5s" in instr else ("[silent]", [], 2, 0)

    sim = run(s, make_arm("fast_only", MODELS), FakeLLM(policy))
    t2, t3 = sim.persons[1], sim.persons[2]
    assert t3.t_start == pytest.approx(t2.t_end + 2.0)


def test_slow_speaks_replay_is_skipped_when_the_slow_model_has_spoken_again():
    script = custom_script([("What's her number?", "greeting", Trigger("at", delay=0.5)),
                            ("Sorry, one sec. Okay, go on.", "interrupt_resume",
                             Trigger("barge_answer", delay=1.2, patience=30, items=["q1"])),
                            ("Bye", "closing", Trigger("after_any", gap=0.5, patience=40))])
    gold = script.item("q1").gold
    line = f"Sure thing. Her mobile is {gold}. Anything else?"

    def policy(model, messages, kw, meta):
        instr = instruction_of(messages)
        if meta["role"] == "fast":
            if "go on" in instr:
                return ("REPLAY", [], 2, 0)
            return ("HOLD: One sec.", [], 6, 0) if "number" in instr else ("SMALL_TALK: Bye!", [], 4, 0)
        if "(t1)" in instr:
            return (line, [], 20, 0)
        if "(t2)" in instr:  # the slow model answers the interruption itself, before the replay is decided
            return (f"Her mobile is {gold}.", [], 8, 0)
        return ("", [], 4, 0)

    # a slow model much quicker than the fast one: its new line is already being said when the replay is decided
    sim = run(script, SlowSpeaks(MODELS), FakeLLM(policy), timing=make_timing(fast_a=1.5, slow_a=0.3))
    assert all(u.source != "replay" for u in sim.utts)
    fast = [c for c in sim.calls.values() if c.role == "fast" and c.meta.get("label") == "REPLAY"]
    assert fast and fast[0].meta.get("replay", "").startswith("skipped")
    assert [u.text for u in sim.utts if u.t_start is not None and gold in u.text and u.t_start > sim.persons[1].t_end] \
        == [f"Her mobile is {gold}."]


def test_slow_speaks_replay_is_not_lost_to_a_waiting_line_of_the_run_it_cancels():
    """Found in the cheap tiers (slow model as quick as the fast one): the slow
    model's line for Sam's turn was waiting (first token in, text-to-speech not
    yet started) when the fast model's REPLAY label arrived; the replay was
    skipped because a slow line was waiting, then the label cancelled that run
    and dropped its line, so Sam heard nothing. The waiting line of the run the
    label cancels must not stop the replay."""
    script = custom_script([("What's her number?", "greeting", Trigger("at", delay=0.5)),
                            ("Sorry, one sec. Okay, go on.", "interrupt_resume",
                             Trigger("barge_answer", delay=1.2, patience=30, items=["q1"])),
                            ("Bye", "closing", Trigger("after_any", gap=0.5, patience=40))])
    gold = script.item("q1").gold
    line = f"Sure thing. Her mobile is {gold}. Anything else?"

    def policy(model, messages, kw, meta):
        instr = instruction_of(messages)
        if meta["role"] == "fast":
            if "go on" in instr:
                return ("REPLAY", [], 2, 0)
            return ("HOLD: One sec.", [], 6, 0) if "number" in instr else ("SMALL_TALK: Bye!", [], 4, 0)
        if "(t1)" in instr:
            return (line, [], 20, 0)
        if "(t2)" in instr:  # first token 0.4 s in, still writing when the label lands at 0.54 s
            return (f"Her mobile is {gold}.", [], 40, 0)
        return ("", [], 4, 0)

    sim = run(script, SlowSpeaks(MODELS), FakeLLM(policy), timing=make_timing(fast_a=0.5, slow_a=0.4))
    t2 = sim.persons[1]
    waiting = next(u for u in sim.utts if u.text == f"Her mobile is {gold}.")
    assert waiting.dropped and waiting.t_start is None
    heard_again = [u for u in sim.utts if u.t_start is not None and u.t_start > t2.t_end and gold in u.text]
    assert [u.source for u in heard_again] == ["replay"]
    assert heard_again[0].text == f"Sorry, as I was saying. Her mobile is {gold}. Anything else?"
    fast = next(c for c in sim.calls.values() if c.role == "fast" and c.meta.get("label") == "REPLAY")
    assert fast.meta["slow_run"] == "cancelled" and "replay" not in fast.meta
