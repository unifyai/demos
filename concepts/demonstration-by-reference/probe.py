"""Measure the constraints the plan depends on, before any experiment run.

  limits  How many images one request may carry, per model (tiny images, so cheap).
  tokens  Prompt tokens one frame costs at each resolution and encoding, per model.
  payload Whether a request of N real frames goes through, and its latency and cost.

Every call goes through demos_common.LLM and lands in runs/ledger.jsonl tagged
"probe". Results are appended to runs/probes.jsonl.

  ../../bin/with-openrouter-key uv run python probe.py limits --models anthropic/claude-haiku-5.5 openai/gpt-6-luna
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import time
from pathlib import Path

from PIL import Image, ImageDraw

from demos_common import LLM, LLMError, image_part

HERE = Path(__file__).parent
RUNS = HERE / "runs"


def tiny_png(i: int) -> bytes:
    img = Image.new("RGB", (48, 48), ((i * 37) % 256, (i * 91) % 256, (i * 53) % 256))
    ImageDraw.Draw(img).text((4, 18), str(i % 1000), fill=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def log(row: dict) -> None:
    RUNS.mkdir(exist_ok=True)
    with open(RUNS / "probes.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row))


PROVIDERS = {"anthropic": ["anthropic"], "openai": ["openai"], "google": ["google-vertex", "google-ai-studio"]}


def provider_for(model: str) -> dict:
    # Pin the first-party provider so latency and price are one endpoint's.
    return {"order": PROVIDERS[model.split("/")[0]], "allow_fallbacks": False}


async def limits(llm: LLM, models: list[str], counts: list[int]) -> None:
    for model in models:
        for n in counts:
            content: list[dict] = [{"type": "text", "text": f"There are {n} small images below. Reply with the single word OK."}]
            content += [image_part(tiny_png(i)) for i in range(n)]
            t0 = time.time()
            try:
                r = await llm.chat(model, [{"role": "user", "content": content}], tag="probe", max_tokens=200,
                                   provider=provider_for(model), meta={"probe": "limits", "n": n})
                log({"probe": "limits", "model": model, "n_images": n, "ok": True, "served": r.model,
                     "prompt_tokens": r.prompt_tokens, "cost": r.cost, "latency_s": round(time.time() - t0, 2), "text": r.text[:40]})
            except LLMError as e:
                log({"probe": "limits", "model": model, "n_images": n, "ok": False, "error": str(e)[:400]})


async def tokens(llm: LLM, models: list[str], frames: list[Path]) -> None:
    for model in models:
        base = await llm.chat(model, [{"role": "user", "content": [{"type": "text", "text": "Reply OK."}]}], tag="probe",
                              max_tokens=200, provider=provider_for(model), meta={"probe": "tokens", "frame": None})
        for fp in frames:
            data = fp.read_bytes()
            mime = "image/png" if fp.suffix == ".png" else ("image/webp" if fp.suffix == ".webp" else "image/jpeg")
            content = [{"type": "text", "text": "Reply OK."}, image_part(data, mime)]
            r = await llm.chat(model, [{"role": "user", "content": content}], tag="probe", max_tokens=200,
                               provider=provider_for(model), meta={"probe": "tokens", "frame": fp.name})
            w, h = Image.open(fp).size
            log({"probe": "tokens", "model": model, "frame": fp.name, "size": [w, h], "bytes": len(data),
                 "image_tokens": r.prompt_tokens - base.prompt_tokens, "served": r.model, "cost": r.cost})


async def payload(llm: LLM, models: list[str], frames: list[Path], n: int) -> None:
    for model in models:
        content: list[dict] = [{"type": "text", "text": f"{n} screenshots follow. Reply with the single word OK."}]
        total = 0
        for i in range(n):
            fp = frames[i % len(frames)]
            data = fp.read_bytes()
            total += len(data)
            mime = "image/png" if fp.suffix == ".png" else ("image/webp" if fp.suffix == ".webp" else "image/jpeg")
            content.append({"type": "text", "text": f"frame {i}"})
            content.append(image_part(data, mime))
        t0 = time.time()
        try:
            r = await llm.chat(model, [{"role": "user", "content": content}], tag="probe", max_tokens=200,
                               provider=provider_for(model), meta={"probe": "payload", "n": n})
            log({"probe": "payload", "model": model, "n_frames": n, "bytes": total, "ok": True, "served": r.model,
                 "prompt_tokens": r.prompt_tokens, "cost": r.cost, "latency_s": round(time.time() - t0, 2)})
        except LLMError as e:
            log({"probe": "payload", "model": model, "n_frames": n, "bytes": total, "ok": False, "error": str(e)[:400]})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["limits", "tokens", "payload"])
    ap.add_argument("--models", nargs="+", default=["anthropic/claude-haiku-5.5", "openai/gpt-6-luna"])
    ap.add_argument("--counts", nargs="+", type=int, default=[99, 101, 200, 500, 600, 1000])
    ap.add_argument("--frames", nargs="*", type=Path, default=[])
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()

    async def go() -> None:
        async with LLM(RUNS / "ledger.jsonl", tag="probe", max_concurrency=4, max_retries=2) as llm:
            if a.what == "limits":
                await limits(llm, a.models, a.counts)
            elif a.what == "tokens":
                await tokens(llm, a.models, a.frames)
            else:
                await payload(llm, a.models, a.frames, a.n)

    asyncio.run(go())


if __name__ == "__main__":
    main()
