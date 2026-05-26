"""Standalone diagnostic — tests each task path against HF Inference Providers
with the configured HF_TOKEN. Prints a result table with exact errors.

Run: python diagnose.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from huggingface_hub import InferenceClient, get_token, whoami
from huggingface_hub.utils import HfHubHTTPError
from PIL import Image

load_dotenv()
ROOT = Path(__file__).resolve().parent
EX = ROOT / "examples"

HF_TOKEN = (os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN") or "").strip()


def banner(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def fmt_err(e: BaseException) -> str:
    msg = f"{type(e).__name__}: {e}"
    if isinstance(e, HfHubHTTPError):
        resp = getattr(e, "response", None)
        if resp is not None:
            try:
                msg += f"\n  status={resp.status_code}\n  body={resp.text[:400]}"
            except Exception:
                pass
    return msg


def check_token() -> dict[str, object]:
    banner("1. HF_TOKEN identity & permissions")
    info: dict[str, object] = {"token_present": bool(HF_TOKEN), "token_prefix": HF_TOKEN[:8] if HF_TOKEN else None}
    if not HF_TOKEN:
        print("  X  HF_TOKEN missing in .env")
        return info
    print(f"  token prefix: {HF_TOKEN[:8]}...")

    try:
        who = whoami(token=HF_TOKEN)
        info["whoami"] = {k: who.get(k) for k in ("name", "type", "email") if isinstance(who, dict) and k in who}
        if isinstance(who, dict):
            print(f"  user: {who.get('name')}   type: {who.get('type')}")
            # token scope hints
            auth = who.get("auth") or {}
            access_token = auth.get("accessToken") if isinstance(auth, dict) else None
            if isinstance(access_token, dict):
                fine = access_token.get("fineGrained")
                if isinstance(fine, dict):
                    scopes = fine.get("scoped") or []
                    perms = sorted({p for s in scopes for p in (s.get("permissions") or [])})
                    info["fine_grained_permissions"] = perms
                    print(f"  fine-grained permissions: {perms}")
                role = access_token.get("role")
                if role:
                    info["role"] = role
                    print(f"  classic token role: {role}")
    except Exception as e:
        info["whoami_error"] = fmt_err(e)
        print(f"  X  whoami failed → {fmt_err(e)}")

    # Probe inference-providers permission via a tiny chat call to a known-cheap model
    try:
        client = InferenceClient(token=HF_TOKEN, provider="auto")
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=4, temperature=0.0,
        )
        ms = int((time.perf_counter() - t0) * 1000)
        info["chat_probe"] = "ok"
        info["chat_probe_ms"] = ms
        print(f"  OK chat probe (Llama-3.1-8B) in {ms} ms")
    except Exception as e:
        info["chat_probe_error"] = fmt_err(e)
        print(f"  X  chat probe failed → {fmt_err(e)}")
    return info


def test_task(name: str, fn) -> dict[str, object]:
    banner(f"{name}")
    t0 = time.perf_counter()
    try:
        result = fn()
        ms = int((time.perf_counter() - t0) * 1000)
        print(f"  OK in {ms} ms  →  {result}")
        return {"task": name, "status": "ok", "latency_ms": ms, "info": str(result)[:300]}
    except Exception as e:
        ms = int((time.perf_counter() - t0) * 1000)
        err = fmt_err(e)
        print(f"  X  in {ms} ms")
        for line in err.splitlines():
            print(f"     {line}")
        return {"task": name, "status": "fail", "latency_ms": ms, "error": err}


def t_text_to_text():
    client = InferenceClient(token=HF_TOKEN, provider="auto")
    r = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[{"role": "user", "content": "say hi in 3 words"}],
        max_tokens=16, temperature=0.0,
    )
    return r.choices[0].message.content[:80]


def t_image_text_to_text():
    import base64
    from io import BytesIO
    img = Image.open(EX / "sample_shapes.jpg")
    img.thumbnail((512, 512))
    buf = BytesIO(); img.convert("RGB").save(buf, format="PNG")
    data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    client = InferenceClient(token=HF_TOKEN, provider="auto")
    r = client.chat.completions.create(
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": "what is in the image? one line."},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]}],
        max_tokens=40, temperature=0.0,
    )
    return r.choices[0].message.content[:120]


def t_text_to_image():
    client = InferenceClient(token=HF_TOKEN, provider="auto")
    img = client.text_to_image(prompt="a red apple on a white plate, photorealistic", model="black-forest-labs/FLUX.1-schnell")
    return f"PIL image {img.size} mode={img.mode}"


def t_text_to_speech():
    client = InferenceClient(token=HF_TOKEN, provider="auto")
    audio = client.text_to_speech("Hello, this is a test.", model="facebook/mms-tts-eng")
    return f"audio bytes: {len(audio)}"


def t_speech_to_text():
    client = InferenceClient(token=HF_TOKEN, provider="auto")
    audio_path = EX / "sample_speech.ogg"
    with open(audio_path, "rb") as f:
        audio = f.read()
    result = client.automatic_speech_recognition(audio, model="openai/whisper-large-v3-turbo")
    return f"{result}"


def main() -> int:
    print("HF Inference Providers — Demo diagnostic")
    token_info = check_token()
    results = [
        {"task": "_token", "info": token_info},
        test_task("2. text-to-text  (Llama-3.1-8B-Instruct)", t_text_to_text),
        test_task("3. image-text-to-text  (Qwen2.5-VL-7B)", t_image_text_to_text),
        test_task("4. text-to-image  (FLUX.1-schnell)", t_text_to_image),
        test_task("5. text-to-speech  (mms-tts-eng)", t_text_to_speech),
        test_task("6. speech-to-text  (whisper-large-v3-turbo)", t_speech_to_text),
    ]
    banner("SUMMARY")
    for r in results[1:]:
        flag = "OK " if r.get("status") == "ok" else "X  "
        print(f"  {flag}{r['task']}  ({r.get('latency_ms', '-')} ms)")
    (ROOT / "diagnose_report.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n  full report → diagnose_report.json")
    failed = sum(1 for r in results[1:] if r.get("status") == "fail")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
