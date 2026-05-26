"""End-to-end verification: call the actual Gradio handlers in app.run_task
with realistic inputs, then *assert* output structure + non-empty content.

Difference vs diagnose.py: this exercises the same code path the UI hits
(handler → call_* → usage_rows → return tuple), not just the bare HF API.

Pass: process exit 0. Fail: exit 1 with details printed.
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent))
import app  # noqa: E402

EX = app.EXAMPLES_DIR
HF = (os.getenv("HF_TOKEN") or "").strip()


def bullet(ok: bool, label: str, detail: str) -> str:
    return f"  {'OK ' if ok else 'X  '} {label:<22}  {detail}"


def verify(task: str, model: str, system: str, prompt: str,
           image: Image.Image | None, audio_path: str | None,
           max_tokens: int = 64, temperature: float = 0.0) -> tuple[bool, list[str]]:
    print(f"\n=== {task}  ({model}) ===")
    t0 = time.perf_counter()
    text_out, img_out, usage, raw, ms = app.run_task(
        task, model, system, prompt, image, audio_path, max_tokens, temperature, HF,
    )
    elapsed = int((time.perf_counter() - t0) * 1000)

    checks: list[tuple[bool, str, str]] = []

    # error gate
    if isinstance(raw, dict) and "error" in raw:
        checks.append((False, "handler error", raw["error"][:200]))
        return False, [bullet(o, l, d) for o, l, d in checks]

    # latency sanity
    checks.append((ms > 0, "latency>0", f"{ms} ms (wallclock {elapsed} ms)"))

    if task in {"text-to-text", "image-text-to-text", "speech-to-text"}:
        checks.append((bool(text_out) and len(text_out.strip()) > 0,
                       "text_out non-empty",
                       f"{len(text_out)} chars: {text_out[:80].replace(chr(10),' ')!r}"))

    if task in {"text-to-text", "image-text-to-text"}:
        # usage table must report real tokens (not estimates)
        prompt_row = next((r for r in usage if r[0] == "prompt"), None)
        total_row = next((r for r in usage if r[0] == "total"), None)
        src = prompt_row[2] if prompt_row else "?"
        ptok = prompt_row[1] if prompt_row else 0
        ctok = next((r[1] for r in usage if r[0] == "completion"), 0)
        ttok = total_row[1] if total_row else 0
        checks.append((src == "api",
                       "usage source",
                       f"src={src} (must be 'api' from API response)"))
        checks.append((ptok > 0 and ctok > 0,
                       "tokens >0",
                       f"prompt={ptok} completion={ctok} total={ttok}"))
        # raw must serialize properly with usage field
        has_usage = isinstance(raw, dict) and isinstance(raw.get("usage"), dict)
        checks.append((has_usage,
                       "raw.usage present",
                       f"keys={sorted((raw.get('usage') or {}).keys()) if has_usage else 'missing'}"))

    if task == "text-to-image":
        checks.append((img_out is not None,
                       "image returned",
                       f"type={type(img_out).__name__}"))
        if img_out is not None:
            checks.append((img_out.size[0] > 0 and img_out.size[1] > 0,
                           "image dims",
                           f"size={img_out.size} mode={img_out.mode}"))

    if task == "speech-to-text":
        # output already checked above; also confirm transcript actually says something
        words = (text_out or "").strip().split()
        checks.append((len(words) >= 2,
                       "transcript >=2 words",
                       f"words={len(words)}: {' '.join(words[:8])}"))

    return all(c[0] for c in checks), [bullet(o, l, d) for o, l, d in checks]


def main() -> int:
    if not HF:
        print("HF_TOKEN missing in .env"); return 1

    pizza = Image.open(EX / "sample_shapes.jpg")
    speech = str(EX / "sample_speech.ogg")

    cases = [
        ("text-to-text", "meta-llama/Llama-3.1-8B-Instruct",
         "", "Reply with exactly: ok", None, None, 32, 0.0),
        ("image-text-to-text", "Qwen/Qwen3-VL-30B-A3B-Instruct",
         "", "one line: what is in the image?", pizza, None, 60, 0.0),
        ("text-to-image", "black-forest-labs/FLUX.1-schnell",
         "", "a small red apple, white background", None, None, 0, 0.0),
        ("speech-to-text", "openai/whisper-large-v3-turbo",
         "", "", None, speech, 0, 0.0),
    ]

    results: list[tuple[str, bool, list[str]]] = []
    for c in cases:
        ok, lines = verify(*c)
        for ln in lines: print(ln)
        results.append((c[0], ok, lines))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    pass_n = sum(1 for _, ok, _ in results if ok)
    for name, ok, _ in results:
        print(f"  {'OK ' if ok else 'X  '} {name}")
    print(f"\n  {pass_n}/{len(results)} passed")
    return 0 if pass_n == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
