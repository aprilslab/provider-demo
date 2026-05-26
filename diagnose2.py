"""Test alternative models/calling-patterns for the failing tasks."""
from __future__ import annotations
import os, time, traceback, base64
from io import BytesIO
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from huggingface_hub.utils import HfHubHTTPError
from PIL import Image

load_dotenv()
ROOT = Path(__file__).resolve().parent
EX = ROOT / "examples"
HF = (os.getenv("HF_TOKEN") or "").strip()


def fmt_err(e):
    msg = f"{type(e).__name__}: {e}"
    if isinstance(e, HfHubHTTPError):
        resp = getattr(e, "response", None)
        if resp is not None:
            try:
                msg += f"\n    status={resp.status_code}\n    body={resp.text[:500]}"
            except Exception:
                pass
    return msg


def try_(label, fn):
    print(f"\n--- {label}")
    t = time.perf_counter()
    try:
        r = fn()
        ms = int((time.perf_counter() - t) * 1000)
        print(f"   OK in {ms}ms  →  {str(r)[:200]}")
        return True
    except Exception as e:
        ms = int((time.perf_counter() - t) * 1000)
        print(f"   X  in {ms}ms")
        for ln in fmt_err(e).splitlines():
            print(f"     {ln}")
        return False


def image_to_data_url():
    img = Image.open(EX / "sample_shapes.jpg")
    img.thumbnail((512, 512))
    buf = BytesIO(); img.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def vlm_call(model):
    def fn():
        c = InferenceClient(token=HF, provider="auto")
        r = c.chat.completions.create(
            model=model,
            messages=[{"role":"user","content":[
                {"type":"text","text":"one line: what is in the image?"},
                {"type":"image_url","image_url":{"url": image_to_data_url()}},
            ]}],
            max_tokens=40, temperature=0.0,
        )
        return r.choices[0].message.content
    return fn


def tts_call(model, text="Hello world"):
    def fn():
        c = InferenceClient(token=HF, provider="auto")
        b = c.text_to_speech(text, model=model)
        return f"bytes={len(b)}"
    return fn


def asr_call(model, mode):
    def fn():
        c = InferenceClient(token=HF, provider="auto")
        path = str(EX / "sample_speech.ogg")
        if mode == "path":
            return c.automatic_speech_recognition(path, model=model)
        if mode == "bytes":
            with open(path, "rb") as f:
                return c.automatic_speech_recognition(f.read(), model=model)
    return fn


print("=" * 70); print("VLM alternative models"); print("=" * 70)
for m in [
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "google/gemma-3-4b-it",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen/Qwen2-VL-7B-Instruct",
]:
    try_(f"VLM {m}", vlm_call(m))

print("\n" + "=" * 70); print("TTS alternative models"); print("=" * 70)
for m in [
    "facebook/mms-tts-eng",
    "microsoft/speecht5_tts",
    "suno/bark",
    "ResembleAI/chatterbox",
]:
    try_(f"TTS {m}", tts_call(m))

print("\n" + "=" * 70); print("ASR calling-patterns"); print("=" * 70)
for m in [
    "openai/whisper-large-v3-turbo",
    "openai/whisper-large-v3",
]:
    for mode in ("path", "bytes"):
        try_(f"ASR {m} via {mode}", asr_call(m, mode))
