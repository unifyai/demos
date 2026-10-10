"""The baselines (flat, blocking, background), with the model stubbed."""

from __future__ import annotations

import asyncio

from conftest import FakeLLM, last_text, make_arm, say, tc, until

from separate_loops.arms import CONV_BACKGROUND, CONV_BLOCKING, CONV_HANDLE, FLAT_SYSTEM, TALK, TODAY, WORK, WORK_HANDLE, WORK_NOREACH


def test_shared_instructions_are_verbatim_in_every_arm():
    for s in (FLAT_SYSTEM, CONV_HANDLE, CONV_BACKGROUND, CONV_BLOCKING):
        assert s.startswith(TODAY + "\n\n" + TALK)
    for s in (FLAT_SYSTEM, WORK_HANDLE, WORK_NOREACH):
        assert WORK in s and s.startswith(TODAY)


def test_flat_message_during_a_model_call_discards_that_decision():
    def flat(m):
        if "pause" in last_text(m):
            return say("Paused.")
        return say("", tc("messaging_send_message", to="Priya", text="Hi"))

    llm = FakeLLM(lambda loop, m, t: flat(m), delay=0.08)
    arm, run, env, inst = make_arm("flat", llm)

    async def go():
        main = asyncio.create_task(arm.main())
        arm.deliver({"id": 0, "text": "Message Priya."})
        await asyncio.sleep(0.03)
        arm.deliver({"id": 1, "text": "Wait, pause."})
        await until(lambda: run.agent_msgs)
        main.cancel()
        await asyncio.gather(main, return_exceptions=True)

    asyncio.run(go())
    assert env.log == [] and run.calls[0]["discarded"] and run.agent_msgs[0]["text"] == "Paused."
    assert run.agent_msgs[0]["seen"] == 1


def test_flat_messages_between_steps_follow_the_tool_results_and_the_turn_ends_without_tools():
    contacts = []

    def flat(m):
        sent = sum(1 for x in m if x["role"] == "tool")
        if "how's it going" in last_text(m).lower() and m[-1]["role"] == "user":
            return say("One sent so far.", tc("messaging_send_message", to=contacts[1], text="Hi"))
        if sent < 2:
            return say("", tc("messaging_send_message", to=contacts[sent], text="Hi"))
        return say("Both sent.")

    llm = FakeLLM(lambda loop, m, t: flat(m), delay=0.005)
    arm, run, env, inst = make_arm("flat", llm)
    contacts[:] = [c["name"] for c in inst["sites"]["messaging"]["contacts"][:2]]

    async def go():
        main = asyncio.create_task(arm.main())
        arm.deliver({"id": 0, "text": "Message two people."})
        await until(lambda: any(a["t_start"] is not None for a in env.log))
        arm.deliver({"id": 1, "text": "How's it going?"})
        await until(lambda: any(m["text"] == "Both sent." for m in run.agent_msgs))
        n = len(llm.calls)
        await asyncio.sleep(0.05)
        assert len(llm.calls) == n  # the turn ended; nothing runs until the next message
        arm.deliver({"id": 2, "text": "Thanks!"})
        await until(lambda: len(llm.calls) > n)
        main.cancel()
        await asyncio.gather(main, return_exceptions=True)

    asyncio.run(go())
    ctx = arm.ctx
    i_tool = next(i for i, x in enumerate(ctx) if x["role"] == "tool")
    i_q = next(i for i, x in enumerate(ctx) if x["role"] == "user" and "How's it going" in x["content"])
    assert ctx[i_tool - 1]["role"] == "assistant" and i_q == i_tool + 1
    assert [m["text"] for m in run.agent_msgs][:2] == ["One sent so far.", "Both sent."]


def test_blocking_messages_wait_until_run_task_returns():
    def conv(m):
        last = m[-1]
        if last["role"] == "tool":
            return say("Done. You asked: " + (last_text(m) if m[-1]["role"] == "user" else "nothing"))
        if "Could you invite" in last_text(m):
            return say("", tc("run_task", description="Invite them."))
        return say("Answering: " + last_text(m))

    def work(m):
        n = sum(1 for x in m if x["role"] == "tool")
        return say("", tc("messaging_list_group", group="book club")) if n < 3 else say("Invited everyone.")

    llm = FakeLLM(lambda loop, m, t: conv(m) if loop == "conv" else work(m), delays={"conv": 0.005, "work": 0.03})
    arm, run, env, inst = make_arm("blocking", llm)

    async def go():
        main = asyncio.create_task(arm.main())
        arm.deliver({"id": 0, "text": inst["request"]})
        await until(lambda: len(env.log) >= 1)
        arm.deliver({"id": 1, "text": "How's it going?"})
        n_conv = len([c for c in llm.calls if c["loop"] == "conv"])
        await until(lambda: run.tasks and run.tasks[0]["state"] == "finished")
        assert len([c for c in llm.calls if c["loop"] == "conv"]) == n_conv  # no conversation call while it ran
        await until(lambda: len(run.agent_msgs) >= 1)
        main.cancel()
        await asyncio.gather(main, *arm.aio_tasks(), return_exceptions=True)

    asyncio.run(go())
    after = [c for c in llm.calls if c["loop"] == "conv"][-1]["messages"]
    i_result = next(i for i, x in enumerate(after) if x["role"] == "tool")
    assert "Invited everyone." in after[i_result]["content"]
    assert after[i_result + 1]["role"] == "user" and "How's it going?" in after[i_result + 1]["content"]
    assert [t["function"]["name"] for t in arm.tools] == ["run_task"]


def test_background_starts_at_once_reports_progress_can_stop_and_cannot_steer():
    def conv(m):
        last = m[-1]
        if last["role"] == "tool":
            return say("OK.")
        if "Could you invite" in last_text(m):
            return say("", tc("start_task", description="Invite them."))
        if "How's it going" in last_text(m):
            return say("", tc("task_status", task_id="t1"))
        return say("Noted.")

    def work(m):
        return say("", tc("messaging_list_group", group="book club"))

    llm = FakeLLM(lambda loop, m, t: conv(m) if loop == "conv" else work(m), delays={"conv": 0.005, "work": 0.01})
    arm, run, env, inst = make_arm("background", llm)

    async def go():
        main = asyncio.create_task(arm.main())
        arm.deliver({"id": 0, "text": inst["request"]})
        await until(lambda: len(env.log) >= 2)
        arm.deliver({"id": 1, "text": "How's it going?"})
        await until(lambda: any(x["role"] == "tool" and x["content"].startswith("Task t1: running") for x in arm.ctx))
        arm.tasks["t1"].stop()
        await until(lambda: any("has stopped" in str(x.get("content")) for x in arm.ctx))
        main.cancel()
        await asyncio.gather(main, *arm.aio_tasks(), return_exceptions=True)

    asyncio.run(go())
    assert [t["function"]["name"] for t in arm.tools] == ["start_task", "task_status", "stop_task"]
    work_tools = next(c["tools"] for c in llm.calls if c["loop"] == "work:t1")
    assert "ask_person" not in work_tools and "messaging_send_message" in work_tools
    assert llm.calls[0]["messages"][0]["content"] == CONV_BACKGROUND
    assert next(c for c in llm.calls if c["loop"] == "work:t1")["messages"][0]["content"] == WORK_NOREACH


def test_every_arm_offers_the_work_the_same_site_tools():
    seen = {}
    for arm_name in ("flat", "blocking", "background", "handle"):
        def policy(loop, m, t, arm_name=arm_name):
            if loop in ("flat",) or loop.startswith("work"):
                seen[arm_name] = sorted(x for x in t if x not in ("ask_person", "reply_to_person"))
                return say("Done.")
            if m[-1]["role"] == "tool":
                return say("OK.")
            return say("", tc("run_task" if arm_name == "blocking" else "start_task", description="x"))

        llm = FakeLLM(policy, delay=0.001)
        arm, run, env, inst = make_arm(arm_name, llm)

        async def go():
            main = asyncio.create_task(arm.main())
            arm.deliver({"id": 0, "text": "go"})
            await until(lambda: arm_name in seen)
            main.cancel()
            await asyncio.gather(main, *arm.aio_tasks(), return_exceptions=True)

        asyncio.run(go())
    assert len({tuple(v) for v in seen.values()}) == 1 and len(seen["flat"]) == 18


def test_background_stop_mid_batch_starts_nothing_more():
    contacts = []

    def work(m):
        if not any(x["role"] == "assistant" for x in m):
            return say("", *[tc("messaging_send_message", to=c, text="Hi") for c in contacts])
        return say("Done.")

    llm = FakeLLM(lambda loop, m, t: work(m), delay=0.002)
    arm, run, env, inst = make_arm("background", llm)
    contacts[:] = [c["name"] for c in inst["sites"]["messaging"]["contacts"][:4]]

    async def go():
        await arm.dispatch(tc("start_task", description="x"))
        await until(lambda: any(a["t_start"] is not None for a in env.log))
        arm.tasks["t1"].stop()
        await arm.tasks["t1"].done.wait()

    asyncio.run(go())
    assert len([a for a in env.log if a["t_start"] is not None]) == 1
    assert run.tasks[0]["state"] == "stopped"



def test_flat_replies_with_a_tool_and_carries_on():
    contacts = []

    def flat(m):
        sent = sum(1 for x in m if x["role"] == "tool" and '"sent": true' in x["content"])
        if m[-1]["role"] == "user" and "How's it going" in m[-1]["content"]:
            return say("", tc("reply_to_person", text=f"{sent} sent so far."), tc("messaging_send_message", to=contacts[sent], text="Hi"))
        if sent < 2:
            return say("", tc("messaging_send_message", to=contacts[sent], text="Hi"))
        return say("Both sent.")

    llm = FakeLLM(lambda loop, m, t: flat(m), delay=0.005)
    arm, run, env, inst = make_arm("flat", llm)
    contacts[:] = [c["name"] for c in inst["sites"]["messaging"]["contacts"][:2]]

    async def go():
        main = asyncio.create_task(arm.main())
        arm.deliver({"id": 0, "text": "Message two people."})
        await until(lambda: any(a["t_end"] is not None for a in env.log))
        arm.deliver({"id": 1, "text": "How's it going?"})
        await until(lambda: any(m["text"] == "Both sent." for m in run.agent_msgs))
        main.cancel()
        await asyncio.gather(main, return_exceptions=True)

    asyncio.run(go())
    texts = [m["text"] for m in run.agent_msgs]
    assert texts[0].endswith("sent so far.") and texts[-1] == "Both sent."
    assert len([a for a in env.log if a["t_start"] is not None]) == 2
    assert "reply_to_person" in llm.calls[0]["tools"]
