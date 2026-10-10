"""Run one instance under one arm and return the run's record."""

from __future__ import annotations

import asyncio
import time
import traceback
from typing import Any

from .arms import ARMS
from .core import Run
from .person import Person
from .sites import Clock, Env

# A cap on each reply, reasoning included. 2000 truncated 2.5% of Haiku's replies with large batches of
# tool calls (fault 6); the cap is far above any reply now, and every call's finish reason is recorded.
AGENT_KW = {"max_tokens": 16000}


async def run_one(inst: dict[str, Any], arm: str, model: str, llm: Any, *, run_id: str, speed: float = 1.0,
                  person_model: str | None = None, cap_s: float | None = None,
                  reasoning: str | None = None) -> dict[str, Any]:
    clock = Clock()
    env = Env(inst, clock, speed=speed)
    kw = dict(AGENT_KW) | ({"reasoning": {"effort": reasoning}} if reasoning else {})
    run = Run(run_id, model, llm, clock, model_kw=kw)
    env.listeners.append(lambda ev, rec: run.touch())
    agent = ARMS[arm](run, env)
    person = Person(run, env, inst, agent.deliver, speed=speed, **({"model": person_model} if person_model else {}))
    t_agent = asyncio.create_task(agent.main())
    t_person = asyncio.create_task(person.main())
    cap = (cap_s if cap_s is not None else inst["cap_s"]) * speed
    idle = inst["idle_s"] * speed
    wall0 = time.time()
    reason, error = None, None
    while reason is None:
        await asyncio.sleep(min(0.2, 0.2 * speed * 5))
        now = clock.now()
        if t_agent.done():
            reason = "crash"
            exc = t_agent.exception()
            error = "".join(traceback.format_exception(exc)) if exc else "the agent's loop ended"
        elif t_person.done() and t_person.exception():
            reason = "crash"
            error = "person: " + "".join(traceback.format_exception(t_person.exception()))
        elif now > cap:
            reason = "cap"
        elif (person.script_done and not person.busy and run.inflight == 0 and env.inflight == 0
              and now - run.last_activity >= idle):
            reason = "idle"
    t_end = clock.now()
    pending = [t_agent, t_person, *agent.aio_tasks()]
    for t in pending:
        t.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    return {
        "run_id": run_id, "instance": inst["id"], "condition": inst["condition"], "family": inst["family"],
        "seed": inst["seed"], "arm": arm, "model": model, "model_kw": kw, "wall_start": wall0,
        "end": {"t": t_end, "reason": reason, "error": error},
        "person": run.person_msgs, "agent": run.agent_msgs, "calls": run.calls, "tasks": run.tasks,
        "actions": env.log, "state": env.state(), "responder": person.log, "errors": run.errors,
        "contexts": run.contexts,
    }
