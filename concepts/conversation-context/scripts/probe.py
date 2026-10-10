"""Probe models once: tool calling, reasoning tokens, cost per token. Not an experiment."""
import asyncio, json, sys
from demos_common import LLM

TOOLS = [{"type": "function", "function": {"name": "search", "description": "Search the records.",
          "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}]

async def main(models):
    async with LLM("runs/ledger.jsonl", "probe", max_concurrency=4) as llm:
        for m in models:
            msgs = [{"role": "system", "content": "You look things up for Maya. Use search(query)."},
                    {"role": "user", "content": "What's the flight number for my Berlin trip?"}]
            r = await llm.chat(m, msgs, tools=TOOLS, max_tokens=8000)
            print(m, "| served", r.model, "| tool_calls", json.dumps(r.tool_calls)[:160], "| text", r.text[:80].replace("\n"," "),
                  "| p", r.prompt_tokens, "c", r.completion_tokens, "reason", r.reasoning_tokens, "cost", r.cost, "lat", round(r.latency_s, 2))

asyncio.run(main(sys.argv[1:]))
