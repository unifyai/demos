"""Stack probe: one tiny tool-using request per model, to check tool calls,
the reply format, cost reporting and prompt caching. Costs a fraction of a cent.

    ../../bin/with-openrouter-key uv run python scripts/probe.py anthropic/claude-haiku-5.5 ...
"""

import asyncio
import json
import sys
from pathlib import Path

from demos_common import LLM

from execution_coordinates.agent import handle, system_prompt
from execution_coordinates.executor import Executor
from execution_coordinates.tools import Coordinate, Ctx

RUNS = Path(__file__).resolve().parents[1] / "runs"


async def main(models):
    async with LLM(RUNS / "ledger.jsonl", "execution-coordinates", max_concurrency=4) as llm:
        for model in models:
            ex = Executor(Path("/tmp/execcoord-probe") / model.replace("/", "_"))
            try:
                # a long filler history, so the cached prefix is over every provider's minimum
                filler = [{"role": "user", "content": "Background notes, ignore: " + ("ledger rows and rates. " * 400)},
                          {"role": "assistant", "content": "Noted."}]
                sys_p = system_prompt(1, 1, "threads")
                for attempt in range(2):
                    oc = await handle(llm, model, sys_p, filler, "[Probe · Dan] What is 6 * 7? Compute it with "
                                      "run_python in a session called probe, then answer.", Coordinate(ex),
                                      Ctx("probe", f"probe#{attempt}"), max_calls=6, meta={"phase": "probe"})
                    print(model, "attempt", attempt, "stopped", oc.stopped, "answer", oc.answer)
                    for c in oc.calls:
                        print("   call", json.dumps(c))
                    for t in oc.tools:
                        print("   tool", t["name"], json.dumps(t.get("args"))[:200])
                print("   assistant keys:", sorted({k for m in oc.messages if m["role"] == "assistant" for k in m}))
            finally:
                await ex.close()


asyncio.run(main(sys.argv[1:]))
