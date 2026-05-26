from __future__ import annotations

import base64
import json
import os
import time
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import gradio as gr
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from huggingface_hub.utils import HfHubHTTPError
from PIL import Image

load_dotenv()

ROOT = Path(__file__).resolve().parent
EXAMPLES_DIR = ROOT / "examples"
EXAMPLES_DIR.mkdir(exist_ok=True)

# Binary samples are NOT committed (HF Spaces git rejects binaries without Xet/LFS).
# Fetched from Wikimedia Commons (PD/CC) on first run.
SAMPLE_SOURCES: dict[str, str] = {
    "sample_shapes.jpg":
        "https://commons.wikimedia.org/wiki/Special:FilePath/Eq_it-na_pizza-margherita_sep2005_sml.jpg?width=720",
    "sample_chart.png":
        "https://commons.wikimedia.org/wiki/Special:FilePath/Bar_graphs.png",
    "sample_receipt.jpg":
        "https://commons.wikimedia.org/wiki/Special:FilePath/SK%20Korea%20tour%20%E9%A6%96%E7%88%BE%20supermarket%20official%20receipt%20computer%20print%20out%20in%20Korean%20language%20only%20July-2013.JPG?width=700",
    "sample_speech.ogg":
        "https://commons.wikimedia.org/wiki/Special:FilePath/Recording_of_speaker_of_British_English_(Received_Pronunciation).ogg",
}


def ensure_samples() -> None:
    """Download missing example files from Wikimedia Commons.

    Idempotent — only fetches what's not already on disk. Skips silently
    on network failure so the rest of the app can still start (failing
    sample alone shows '확인 불가' message in UI)."""
    for name, url in SAMPLE_SOURCES.items():
        p = EXAMPLES_DIR / name
        if p.exists() and p.stat().st_size > 0:
            continue
        try:
            req = Request(url, headers={"User-Agent": "hf-inference-providers-demo/1.0 (educational)"})
            with urlopen(req, timeout=30) as r:
                data = r.read()
            p.write_bytes(data)
            print(f"[ensure_samples] fetched {name} ({len(data)} bytes)")
        except Exception as e:
            print(f"[ensure_samples] failed {name}: {type(e).__name__}: {e}")


ensure_samples()

INTRO_MD = """
# HF Inference Providers Demo

**Inference provider** = 학습된 모델을 HTTP API로 호스팅하는 외부 서비스.
직접 GPU 띄울 필요 없이 **모델 이름 + 입력**만 보내면 결과를 받는다.

본 데모는 **Hugging Face Inference Providers** 한 곳에서 4가지 task를 호출해 본다.
- 같은 클라이언트(`huggingface_hub.InferenceClient`)지만 task마다 호출 메서드와 입출력 modality가 다름
- task별로 input/output 컴포넌트가 달라진다는 점을 직접 비교
"""

PROVIDER_ROWS = [
    ["HF Inference Providers", "text · VLM · text-to-image · TTS · ASR · video", "HF credits · 무료 한도 포함", "한 token으로 다수 provider 라우팅, 오픈모델 풍부", "모델별 provider 가용성 변동", "https://huggingface.co/docs/inference-providers"],
    ["OpenRouter", "text chat · VLM", "모델별 token 단가 명시", "300+ 모델을 동일 chat schema로 호출", "이미지/오디오 생성 직접 지원 없음", "https://openrouter.ai/models"],
    ["Google Gemini API", "text · VLM · audio · video · document", "free tier + 사용량", "단일 API로 멀티모달 전체 커버", "Google API key 별도 발급", "https://ai.google.dev/"],
    ["Replicate", "image · video · audio · text", "초당 GPU 시간 과금", "creative 생성 모델 카탈로그", "모델별 schema/콜드스타트 차이", "https://replicate.com/explore"],
    ["Together AI", "text · VLM · embedding", "token 단가 (오픈모델 중심)", "오픈모델 추론 속도/가격 경쟁력", "closed 모델 없음", "https://www.together.ai/"],
    ["Groq", "text · VLM 일부", "free tier + 사용량", "LPU 기반 초저지연 추론", "모델 카탈로그 작음", "https://groq.com/"],
    ["Fal AI", "image · video · audio", "초당 호출 과금", "creative 생성 속도/UX", "text 모델 부족", "https://fal.ai/models"],
]

TASKS = [
    "text-to-text",
    "image-text-to-text",
    "text-to-image",
    "speech-to-text",
]

TASK_LABEL = {
    "text-to-text": "Text → Text (chat)",
    "image-text-to-text": "Image + Text → Text (VLM)",
    "text-to-image": "Text → Image",
    "speech-to-text": "Audio → Text (ASR)",
}

TASK_MODELS: dict[str, list[str]] = {
    "text-to-text": [
        "meta-llama/Llama-3.1-8B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
    ],
    "image-text-to-text": [
        "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "Qwen/Qwen3-VL-235B-A22B-Instruct",
        "google/gemma-3-27b-it",
        "zai-org/GLM-4.5V",
    ],
    "text-to-image": [
        "black-forest-labs/FLUX.1-schnell",
        "black-forest-labs/FLUX.1-dev",
        "stabilityai/stable-diffusion-xl-base-1.0",
    ],
    "speech-to-text": [
        "openai/whisper-large-v3-turbo",
        "openai/whisper-large-v3",
    ],
}

# Hint shown when a HF Inference Providers call returns model_not_supported.
PROVIDER_ENABLE_HINT = (
    "Provider 활성화 필요: https://huggingface.co/settings/inference-providers 에서 해당 task의 "
    "provider(Together · Fireworks · Replicate · Fal · Cerebras 등)를 켜야 합니다."
)

SAMPLES: dict[str, list[tuple[str, str, str | None]]] = {
    "text-to-text": [
        ("이미지 생성 프롬프트 만들기",
         "다음 한국어 설명을 FLUX·SDXL 같은 이미지 생성 모델이 잘 따라하는 영어 프롬프트로 변환해줘.\n"
         "- 주제, 스타일, 구도, 조명, 색감을 명시\n"
         "- 부정 프롬프트(negative prompt) 한 줄도 함께\n"
         "- 최종 결과를 코드블록으로 감싸줘\n\n"
         "한국어 설명:\n"
         "수업 발표용 포스터 헤더. 깨끗한 흰 배경에 노트북, 이미지/오디오/문서 아이콘이 있고 제목 들어갈 공간이 비어있는 일러스트."
         ,
         None),
        ("프롬프트 개선기 (vague → specific)",
         "사용자가 LLM에게 줄 프롬프트가 모호하다. 더 좋은 프롬프트로 다시 써줘.\n"
         "출력 형식:\n"
         "1. 모호한 부분 지적 (불릿)\n"
         "2. 개선된 프롬프트 (코드블록)\n"
         "3. 왜 더 좋은지 한 줄 이유\n\n"
         "원본 프롬프트:\n"
         "\"우리 회사 소개글 좀 써줘\""
         ,
         None),
        ("개념을 다른 청자에게 설명",
         "다음 주제를 두 가지 청자에게 각각 다른 깊이로 설명해줘. 비유를 적극 활용.\n"
         "- (a) 초등학교 6학년 — 3문장\n"
         "- (b) 컴퓨터공학 전공 1학년 — 5문장\n\n"
         "주제: GPU와 CPU의 차이"
         ,
         None),
    ],
    "image-text-to-text": [
        ("사진 보고 레시피 생성",
         "사진 속 음식을 한국 가정에서 만들 수 있는 레시피로 작성해줘.\n"
         "1. 보이는 토핑/재료 (관찰)\n"
         "2. 4인분 재료와 분량\n"
         "3. 조리 순서 (번호 매김, 5~7단계)\n"
         "4. 실패하지 않는 핵심 팁 1개",
         "sample_shapes.jpg"),
        ("차트 읽기",
         "이 차트의 축 이름과 가장 큰 값/가장 작은 값을 추출해줘. 추세를 한 줄로 요약.",
         "sample_chart.png"),
        ("영수증 정보 추출",
         "영수증에서 날짜, 총 금액, 상호명, 결제수단을 표로 정리해줘. 보이지 않는 항목은 '확인 불가'.",
         "sample_receipt.jpg"),
    ],
    "text-to-image": [
        ("로고풍 일러스트",
         "minimalist flat logo of a robot holding a paintbrush, white background, vector style, clean lines",
         None),
        ("발표 포스터 헤더",
         "clean classroom poster header illustration, laptop with image audio document icons, white background, soft pastel colors, empty space for title",
         None),
    ],
    "speech-to-text": [
        ("RP English — North Wind and the Sun (36s)",
         "",
         "sample_speech.ogg"),
    ],
}

DEFAULT_SYSTEM = "You are a helpful assistant. Answer in Korean unless the user requests otherwise."


# ---------- Auth ----------

def get_key(override: str) -> str:
    override = (override or "").strip()
    if override:
        return override
    return os.getenv("HF_TOKEN", "").strip() or os.getenv("HUGGINGFACEHUB_API_TOKEN", "").strip()


# ---------- Encoding ----------

def pil_to_data_url(image: Image.Image, max_side: int = 1024) -> str:
    img = image.copy()
    img.thumbnail((max_side, max_side))
    buf = BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def to_serializable(obj: Any) -> Any:
    from dataclasses import is_dataclass, asdict
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if is_dataclass(obj) and not isinstance(obj, type):
        try:
            return asdict(obj)
        except Exception:
            pass
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if hasattr(obj, "__dict__"):
        return {k: to_serializable(v) for k, v in vars(obj).items()}
    return str(obj)


# ---------- HF calls per task ----------

def call_text_to_text(model: str, system: str, prompt: str, max_tokens: int, temperature: float, key: str):
    client = InferenceClient(token=key, provider="auto")
    messages: list[dict[str, Any]] = []
    if (system or "").strip():
        messages.append({"role": "system", "content": system.strip()})
    messages.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(
        model=model, messages=messages,
        max_tokens=int(max_tokens), temperature=float(temperature),
    )
    raw = to_serializable(resp)
    out = ((raw.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    return out, raw


def call_image_text_to_text(model: str, system: str, prompt: str, image: Image.Image, max_tokens: int, temperature: float, key: str):
    client = InferenceClient(token=key, provider="auto")
    messages: list[dict[str, Any]] = []
    if (system or "").strip():
        messages.append({"role": "system", "content": system.strip()})
    messages.append({"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": pil_to_data_url(image)}},
    ]})
    resp = client.chat.completions.create(
        model=model, messages=messages,
        max_tokens=int(max_tokens), temperature=float(temperature),
    )
    raw = to_serializable(resp)
    out = ((raw.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    return out, raw


def call_text_to_image(model: str, prompt: str, key: str) -> Image.Image:
    client = InferenceClient(token=key, provider="auto")
    return client.text_to_image(prompt=prompt, model=model)


def call_speech_to_text(model: str, audio_path: str, key: str) -> tuple[str, dict[str, Any]]:
    client = InferenceClient(token=key, provider="auto")
    # pass file path directly so the client infers MIME from the extension
    resp = client.automatic_speech_recognition(audio_path, model=model)
    raw = to_serializable(resp)
    if isinstance(raw, dict):
        text = raw.get("text") or ""
    elif isinstance(raw, str):
        text = raw
    else:
        text = str(raw)
    return text, raw if isinstance(raw, dict) else {"text": text}


# ---------- Token usage ----------

def usage_rows(prompt: str, output: str, usage: dict[str, Any] | None) -> list[list[Any]]:
    if usage:
        pt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        ct = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        tt = int(usage.get("total_tokens") or pt + ct)
        src = "api"
    else:
        pt = max(1, round(len(prompt or "") / 3)) if prompt else 0
        ct = max(1, round(len(output or "") / 3)) if output else 0
        tt = pt + ct
        src = "estimate"
    return [["prompt", pt, src], ["completion", ct, src], ["total", tt, src]]


# ---------- Code snippets ----------

def make_code(task: str, model: str, system: str, prompt: str, max_tokens: int, temperature: float) -> str:
    if not model:
        return "# 모델을 선택하면 코드가 생성됩니다."
    sys_q = json.dumps(system or "", ensure_ascii=False)
    pr_q = json.dumps(prompt or "", ensure_ascii=False)
    mt = int(max_tokens)
    temp = float(temperature)
    header = (
        "from huggingface_hub import InferenceClient\n"
        "\n"
        'client = InferenceClient(token="hf_...", provider="auto")\n\n'
    )
    if task == "text-to-text":
        return header + (
            "response = client.chat.completions.create(\n"
            f'    model="{model}",\n'
            "    messages=[\n"
            f"        {{'role': 'system', 'content': {sys_q}}},\n"
            f"        {{'role': 'user', 'content': {pr_q}}},\n"
            "    ],\n"
            f"    max_tokens={mt}, temperature={temp},\n"
            ")\n"
            "print(response.choices[0].message.content)\n"
        )
    if task == "image-text-to-text":
        return header + (
            "# PIL 이미지를 base64 data URL로 인코딩 후 image_url에 넣음\n"
            "response = client.chat.completions.create(\n"
            f'    model="{model}",\n'
            "    messages=[\n"
            f"        {{'role': 'system', 'content': {sys_q}}},\n"
            "        {'role': 'user', 'content': [\n"
            f"            {{'type': 'text', 'text': {pr_q}}},\n"
            "            {'type': 'image_url', 'image_url': {'url': data_url}},\n"
            "        ]},\n"
            "    ],\n"
            f"    max_tokens={mt}, temperature={temp},\n"
            ")\n"
            "print(response.choices[0].message.content)\n"
        )
    if task == "text-to-image":
        return header + (
            f'image = client.text_to_image(prompt={pr_q}, model="{model}")\n'
            'image.save("out.png")\n'
        )
    if task == "speech-to-text":
        return header + (
            'with open("input.ogg", "rb") as f:\n'
            "    audio = f.read()\n"
            f'result = client.automatic_speech_recognition(audio, model="{model}")\n'
            "print(result.text if hasattr(result, 'text') else result)\n"
        )
    return "# unsupported task"


# ---------- Runner ----------

def _empty_outputs(err: str | None = None):
    err_obj = {"error": err} if err else None
    return ("", None, usage_rows("", "", None), err_obj, 0)


def run_task(task: str, model: str, system: str, prompt: str,
             image: Image.Image | None, audio_path: str | None,
             max_tokens: int, temperature: float, override_key: str,
             progress=gr.Progress(track_tqdm=False)):
    progress(0.05, desc=f"준비 중 · task={task}")
    key = get_key(override_key)
    if not key:
        return _empty_outputs("HF_TOKEN 환경변수 또는 API key 입력이 필요합니다.")
    if not model:
        return _empty_outputs("모델을 선택해 주세요.")

    needs_prompt = task in {"text-to-text", "image-text-to-text", "text-to-image"}
    if needs_prompt and not (prompt or "").strip():
        return _empty_outputs("프롬프트를 입력해 주세요.")
    if task == "image-text-to-text" and image is None:
        return _empty_outputs("이미지를 업로드하거나 샘플을 선택해 주세요.")
    if task == "speech-to-text" and not audio_path:
        return _empty_outputs("오디오 파일을 업로드하거나 샘플을 선택해 주세요.")

    def _is_transient(text: str, status: int | None) -> bool:
        markers = ("memory_limit_exceeded", "service_overloaded",
                   "service temporarily", "overloaded", "rate limit", "too many requests")
        if any(m in (text or "").lower() for m in markers):
            return True
        return status in (429, 502, 503, 504)

    def _exec_with_retry(do):
        last_exc: BaseException | None = None
        for attempt in range(2):
            try:
                if attempt == 0:
                    progress(0.2, desc=f"provider 호출 중 · {model}")
                else:
                    progress(0.4, desc=f"재시도 중 (provider 일시 과부하) · {model}")
                return do()
            except HfHubHTTPError as e:
                last_exc = e
                body = getattr(e, "response", None)
                body_text = ""
                status = None
                if body is not None:
                    try:
                        body_text = body.text
                        status = body.status_code
                    except Exception:
                        body_text = str(e)
                if attempt == 0 and _is_transient(body_text, status):
                    progress(0.3, desc="provider 과부하 · 2.5초 후 재시도")
                    time.sleep(2.5)
                    continue
                raise
        raise last_exc  # type: ignore

    t0 = time.perf_counter()
    try:
        if task == "text-to-text":
            out, raw = _exec_with_retry(lambda: call_text_to_text(model, system, prompt, max_tokens, temperature, key))
            progress(1.0, desc="완료")
            ms = int((time.perf_counter() - t0) * 1000)
            return out, None, usage_rows(prompt, out, raw.get("usage")), raw, ms

        if task == "image-text-to-text":
            out, raw = _exec_with_retry(lambda: call_image_text_to_text(model, system, prompt, image, max_tokens, temperature, key))
            progress(1.0, desc="완료")
            ms = int((time.perf_counter() - t0) * 1000)
            return out, None, usage_rows(prompt, out, raw.get("usage")), raw, ms

        if task == "text-to-image":
            progress(0.5, desc=f"이미지 생성 중 (FLUX/SDXL은 7-15초 소요) · {model}")
            img = _exec_with_retry(lambda: call_text_to_image(model, prompt, key))
            progress(1.0, desc="완료")
            ms = int((time.perf_counter() - t0) * 1000)
            return "", img, usage_rows(prompt, "", None), {"latency_ms": ms, "model": model, "size": img.size}, ms

        if task == "speech-to-text":
            progress(0.5, desc=f"오디오 인식 중 · {model}")
            text, raw = _exec_with_retry(lambda: call_speech_to_text(model, audio_path, key))
            progress(1.0, desc="완료")
            ms = int((time.perf_counter() - t0) * 1000)
            return text, None, usage_rows("", text, None), raw, ms

        return _empty_outputs(f"unsupported task: {task}")

    except HfHubHTTPError as e:
        body = getattr(e, "response", None)
        body_text = ""
        status = None
        if body is not None:
            try:
                body_text = body.text[:1200]
                status = body.status_code
            except Exception:
                body_text = str(e)
        else:
            body_text = str(e)
        msg = f"HF API 오류: {body_text}"
        transient_markers = ("memory_limit_exceeded", "service_overloaded",
                              "service temporarily", "503", "overloaded",
                              "rate limit", "429", "too many requests")
        if any(m in body_text.lower() for m in transient_markers) or status in (429, 502, 503, 504):
            msg = (
                "Provider 일시 과부하 / rate limit입니다.\n"
                "- 잠시 (10-30초) 기다린 후 Run 다시 시도\n"
                "- 같은 task의 다른 모델로 교체 후 시도 (provider auto 라우팅이 바뀜)\n\n"
                f"원본 응답: {body_text[:400]}"
            )
        elif "model_not_supported" in body_text or "is not supported by any provider" in body_text:
            msg += f"\n\n→ {PROVIDER_ENABLE_HINT}"
        elif "Pay-as-you" in body_text or "billing" in body_text.lower():
            msg = (
                "Provider pay-as-you-go 결제 활성화 필요.\n"
                "- https://huggingface.co/settings/billing 에서 해당 provider 결제 활성화\n"
                "- 또는 결제 불필요한 다른 모델 선택\n\n"
                f"원본 응답: {body_text[:400]}"
            )
        return _empty_outputs(msg)
    except StopIteration:
        return _empty_outputs(
            f"이 task에 대한 provider가 등록되어 있지 않습니다 (모델 '{model}'을 서빙하는 provider 없음).\n\n→ {PROVIDER_ENABLE_HINT}"
        )
    except HTTPError as e:
        try:
            body = e.read().decode("utf-8")
        except Exception:
            body = str(e)
        return _empty_outputs(f"HTTP {e.code}: {body[:600]}")
    except URLError as e:
        return _empty_outputs(f"네트워크 오류: {e.reason}")
    except Exception as e:
        return _empty_outputs(f"{type(e).__name__}: {e}")


# ---------- UI helpers ----------

def update_models(task: str):
    items = TASK_MODELS.get(task, [])
    return gr.update(choices=items, value=items[0] if items else None)


def update_samples(task: str):
    titles = [s[0] for s in SAMPLES.get(task, [])]
    return gr.update(choices=titles, value=titles[0] if titles else None)


TASK_IO = {
    "text-to-text":       {"text_in": True,  "image_in": False, "audio_in": False, "text_out": True,  "image_out": False, "system": True,  "gen": True},
    "image-text-to-text": {"text_in": True,  "image_in": True,  "audio_in": False, "text_out": True,  "image_out": False, "system": True,  "gen": True},
    "text-to-image":      {"text_in": True,  "image_in": False, "audio_in": False, "text_out": False, "image_out": True,  "system": False, "gen": False},
    "speech-to-text":     {"text_in": False, "image_in": False, "audio_in": True,  "text_out": True,  "image_out": False, "system": False, "gen": False},
}


def load_sample(task: str, title: str):
    text_val = ""
    image_val = None
    audio_val = None
    for s in SAMPLES.get(task, []):
        if s[0] != title:
            continue
        text_val = s[1]
        attached = s[2]
        if attached:
            p = EXAMPLES_DIR / attached
            if p.exists():
                if task == "image-text-to-text":
                    image_val = Image.open(p)
                elif task == "speech-to-text":
                    audio_val = str(p)
        break
    return text_val, image_val, audio_val


def code_refresh(task, model, system, prompt, max_tokens, temperature):
    return make_code(task, model or "", system or "", prompt or "", max_tokens, temperature)


# ---------- Health check (probe each task once) ----------

def _assert_text(text_out: str) -> tuple[bool, str]:
    n = len((text_out or "").strip())
    return n > 0, f"text len={n}"


def _assert_tokens(usage: list[list[Any]]) -> tuple[bool, str]:
    pt = next((r[1] for r in usage if r[0] == "prompt"), 0)
    ct = next((r[1] for r in usage if r[0] == "completion"), 0)
    src = next((r[2] for r in usage if r[0] == "prompt"), "?")
    return (src == "api" and pt > 0 and ct > 0,
            f"src={src} prompt={pt} completion={ct}")


def _assert_image(img: Image.Image | None) -> tuple[bool, str]:
    if img is None:
        return False, "no image"
    return (img.size[0] > 0 and img.size[1] > 0), f"{img.size} {img.mode}"


def run_health_check(override_key: str):
    key = get_key(override_key)
    if not key:
        return [["token", "X FAIL", "HF_TOKEN 환경변수 또는 입력 필요"]], "HF_TOKEN 누락"

    summary_lines: list[str] = []
    rows: list[list[Any]] = []

    # 1. token identity & scopes
    try:
        from huggingface_hub import whoami
        who = whoami(token=key)
        user = who.get("name") if isinstance(who, dict) else "?"
        auth = (who.get("auth") or {}) if isinstance(who, dict) else {}
        at = auth.get("accessToken") or {}
        fine = at.get("fineGrained") or {}
        scopes = fine.get("scoped") or []
        perms = sorted({p for s in scopes for p in (s.get("permissions") or [])})
        scope_text = ", ".join(perms) if perms else (at.get("role") or "unknown")
        summary_lines.append(f"사용자: **{user}**  ·  토큰 권한: `{scope_text}`")
        ok = "inference.serverless.write" in perms or at.get("role") in {"write", "fineGrained"}
        rows.append(["token / whoami", "OK" if ok else "WARN",
                     f"user={user}  perms={scope_text}"
                     + ("" if ok else "  (inference.serverless.write 권한 권장)")])
    except Exception as e:
        summary_lines.append(f"whoami 실패: `{type(e).__name__}: {e}`")
        rows.append(["token / whoami", "X FAIL", f"{type(e).__name__}: {e}"])

    # 2. per-task probes calling the actual run_task handler with assertions
    pizza = Image.open(EXAMPLES_DIR / "sample_shapes.jpg") if (EXAMPLES_DIR / "sample_shapes.jpg").exists() else None
    speech = str(EXAMPLES_DIR / "sample_speech.ogg") if (EXAMPLES_DIR / "sample_speech.ogg").exists() else None

    cases = [
        ("text-to-text", "meta-llama/Llama-3.1-8B-Instruct",
         "", "Reply with exactly: ok", None, None, 16, 0.0,
         lambda t, i, u: [_assert_text(t), _assert_tokens(u)]),
        ("image-text-to-text", TASK_MODELS["image-text-to-text"][0],
         "", "one line: what is in the image?", pizza, None, 60, 0.0,
         lambda t, i, u: [_assert_text(t), _assert_tokens(u)]),
        ("text-to-image", "black-forest-labs/FLUX.1-schnell",
         "", "a small red apple, white background", None, None, 0, 0.0,
         lambda t, i, u: [_assert_image(i)]),
        ("speech-to-text", "openai/whisper-large-v3-turbo",
         "", "", None, speech, 0, 0.0,
         lambda t, i, u: [_assert_text(t)]),
    ]

    ok_cnt = 0
    for task, model, system, prompt, img, audio_path, mt, tp, asserts in cases:
        t0 = time.perf_counter()
        text_out, img_out, usage, raw, _ms = run_task(
            task, model, system, prompt, img, audio_path, mt, tp, key,
        )
        elapsed = int((time.perf_counter() - t0) * 1000)

        if isinstance(raw, dict) and "error" in raw:
            err = raw["error"][:300].replace("\n", " ")
            rows.append([task, f"X · {elapsed} ms", err])
            continue

        a = asserts(text_out, img_out, usage)
        passed = all(p[0] for p in a)
        detail = " · ".join(p[1] for p in a)
        rows.append([task, f"{'OK' if passed else 'WARN'} · {elapsed} ms", f"{model}  →  {detail}"])
        if passed:
            ok_cnt += 1

    n = len(cases)
    summary = "\n".join(summary_lines) + f"\n\n결과: **{ok_cnt} / {n} task PASS** (응답 + 검증 모두 통과)"
    if ok_cnt < n:
        summary += f"\n\n실패한 task → {PROVIDER_ENABLE_HINT}"
        summary += "\n- 결제 활성화: https://huggingface.co/settings/billing — 일부 모델(예: TTS chatterbox)은 fal-ai pay-as-you-go 필요"
    return rows, summary


# ---------- Compare (text-to-text / image-text-to-text only) ----------

def compare_run(task: str, models_csv: str, system: str, prompt: str,
                image: Image.Image | None, max_tokens: int, temperature: float, override_key: str):
    key = get_key(override_key)
    if not key:
        return [["", "error", "HF_TOKEN 또는 API key 필요"]]
    if not (prompt or "").strip():
        return [["", "error", "프롬프트 비어있음"]]
    if task == "image-text-to-text" and image is None:
        return [["", "error", "VLM에는 이미지 필요"]]
    models = [m.strip() for m in (models_csv or "").split(",") if m.strip()]
    if not models:
        return [["", "error", "모델 ID 1개 이상 입력"]]

    rows: list[list[Any]] = []
    for m in models:
        t0 = time.perf_counter()
        try:
            if task == "text-to-text":
                out, raw = call_text_to_text(m, system, prompt, max_tokens, temperature, key)
            else:
                out, raw = call_image_text_to_text(m, system, prompt, image, max_tokens, temperature, key)
            ms = int((time.perf_counter() - t0) * 1000)
            tt = (raw.get("usage") or {}).get("total_tokens") or "-"
            preview = out if len(out) <= 600 else out[:600] + "..."
            rows.append([m, f"{ms} ms · {tt} tok", preview])
        except Exception as e:
            ms = int((time.perf_counter() - t0) * 1000)
            rows.append([m, f"{ms} ms · error", f"{type(e).__name__}: {e}"])
    return rows


# ---------- UI ----------

TASK_CHOICES = list(TASKS)
TASK_HELP_MD = "\n".join(f"- `{t}` — {TASK_LABEL[t]}" for t in TASKS)

with gr.Blocks(title="HF Inference Providers Demo") as demo:
    gr.Markdown(INTRO_MD)

    with gr.Tabs():
        # ===== Tab 1 =====
        with gr.Tab("1. Providers"):
            gr.Markdown("### Provider 비교 카탈로그\n각 provider의 지원 modality · 가격 모델 · 강점 / 한계 요약.")
            gr.Dataframe(
                headers=["Provider", "Modality", "가격 모델", "강점", "한계", "공식 링크"],
                value=PROVIDER_ROWS, interactive=False, wrap=True,
            )
            gr.Markdown(
                "### 본 데모가 호출하는 provider\n"
                "**Hugging Face Inference Providers** 한 곳. `huggingface_hub.InferenceClient`로 4가지 task 메서드 사용.\n"
                "\n"
                "| Task | InferenceClient method | 입력 | 출력 |\n"
                "|---|---|---|---|\n"
                "| text-to-text | `chat.completions.create` | text | text |\n"
                "| image-text-to-text | `chat.completions.create` (image_url) | text + image | text |\n"
                "| text-to-image | `text_to_image` | text | PIL image |\n"
                "| speech-to-text | `automatic_speech_recognition` | audio bytes | text |\n"
                "\n"
                "### API key\n"
                "`HF_TOKEN`을 `.env` 또는 환경변수로 설정하거나, Playground 상단에 일회성 입력.\n"
                "발급: https://huggingface.co/settings/tokens (Inference Providers 호출 권한 필요)\n"
            )

        # ===== Tab 2 =====
        with gr.Tab("2. Playground"):
            gr.Markdown("Task와 모델을 고른 뒤 샘플을 선택하거나 직접 입력해 Run.\n\n" + TASK_HELP_MD)

            with gr.Row():
                pg_task = gr.Radio(choices=TASK_CHOICES, value="text-to-text", label="Task")
            with gr.Row():
                pg_model = gr.Dropdown(label="Model", interactive=True)
                pg_sample = gr.Dropdown(label="Sample preset", interactive=True)
            pg_key = gr.Textbox(label="HF_TOKEN override (선택)", type="password", placeholder=".env 사용 시 비워둠")

            with gr.Accordion("System prompt", open=False, visible=TASK_IO["text-to-text"]["system"]) as pg_system_acc:
                pg_system = gr.Textbox(label="System", lines=2, value=DEFAULT_SYSTEM)
            with gr.Group(visible=TASK_IO["text-to-text"]["text_in"]) as pg_prompt_group:
                pg_prompt = gr.Textbox(label="Text input", lines=6)
            with gr.Group(visible=TASK_IO["text-to-text"]["image_in"]) as pg_image_group:
                pg_image = gr.Image(type="pil", label="Image input", sources=["upload", "clipboard"])
            with gr.Group(visible=TASK_IO["text-to-text"]["audio_in"]) as pg_audio_in_group:
                pg_audio_in = gr.Audio(type="filepath", label="Audio input", sources=["upload", "microphone"])

            with gr.Row(visible=TASK_IO["text-to-text"]["gen"]) as pg_gen_row:
                pg_max = gr.Slider(16, 2048, value=512, step=16, label="Max output tokens")
                pg_temp = gr.Slider(0, 2, value=0.4, step=0.1, label="Temperature")

            pg_run = gr.Button("Run", variant="primary")

            with gr.Row():
                pg_latency = gr.Number(label="Latency (ms)", value=0, interactive=False)
                pg_usage = gr.Dataframe(headers=["Part", "Tokens", "Source"], value=usage_rows("", "", None), interactive=False)

            with gr.Group(visible=TASK_IO["text-to-text"]["text_out"]) as pg_text_out_group:
                pg_text_out = gr.Textbox(label="Text output", lines=10)
            with gr.Group(visible=TASK_IO["text-to-text"]["image_out"]) as pg_img_out_group:
                pg_img_out = gr.Image(label="Image output", type="pil")

            with gr.Accordion("Raw response", open=False):
                pg_raw = gr.JSON()
            with gr.Accordion("호출 코드 (복사용)", open=False):
                pg_code = gr.Code(language="python", lines=20)

            def _refresh_all(task: str):
                io = TASK_IO.get(task, TASK_IO["text-to-text"])
                models = TASK_MODELS.get(task, [])
                samples = SAMPLES.get(task, [])
                titles = [s[0] for s in samples]

                # preload first sample's content so the prompt/image/audio area
                # doesn't visibly empty-then-fill on task switch
                preset_text, preset_image, preset_audio = "", None, None
                if samples:
                    s = samples[0]
                    preset_text = s[1]
                    attached = s[2]
                    if attached:
                        p = EXAMPLES_DIR / attached
                        if p.exists():
                            if task == "image-text-to-text":
                                preset_image = Image.open(p)
                            elif task == "speech-to-text":
                                preset_audio = str(p)

                return (
                    gr.update(choices=models, value=models[0] if models else None),       # pg_model
                    gr.update(choices=titles, value=titles[0] if titles else None),       # pg_sample
                    gr.update(visible=io["text_in"]),                                     # pg_prompt_group
                    preset_text,                                                           # pg_prompt
                    gr.update(visible=io["image_in"]),                                    # pg_image_group
                    preset_image,                                                          # pg_image
                    gr.update(visible=io["audio_in"]),                                    # pg_audio_in_group
                    preset_audio,                                                          # pg_audio_in
                    gr.update(visible=io["system"]),                                      # pg_system_acc
                    gr.update(visible=io["gen"]),                                         # pg_gen_row
                    gr.update(visible=io["text_out"]),                                    # pg_text_out_group
                    "",                                                                    # pg_text_out
                    gr.update(visible=io["image_out"]),                                   # pg_img_out_group
                    None,                                                                  # pg_img_out
                    usage_rows("", "", None),                                              # pg_usage
                    None,                                                                  # pg_raw
                    0,                                                                     # pg_latency
                )

            refresh_outputs = [
                pg_model, pg_sample,
                pg_prompt_group, pg_prompt,
                pg_image_group, pg_image,
                pg_audio_in_group, pg_audio_in,
                pg_system_acc, pg_gen_row,
                pg_text_out_group, pg_text_out,
                pg_img_out_group, pg_img_out,
                pg_usage, pg_raw, pg_latency,
            ]

            pg_task.change(_refresh_all, inputs=pg_task, outputs=refresh_outputs)
            pg_sample.change(load_sample, inputs=[pg_task, pg_sample], outputs=[pg_prompt, pg_image, pg_audio_in])

            for c in (pg_task, pg_model, pg_system, pg_prompt, pg_max, pg_temp):
                c.change(code_refresh,
                         inputs=[pg_task, pg_model, pg_system, pg_prompt, pg_max, pg_temp],
                         outputs=pg_code)

            pg_run.click(run_task,
                         inputs=[pg_task, pg_model, pg_system, pg_prompt, pg_image, pg_audio_in, pg_max, pg_temp, pg_key],
                         outputs=[pg_text_out, pg_img_out, pg_usage, pg_raw, pg_latency],
                         show_progress="full")

            demo.load(_refresh_all, inputs=pg_task, outputs=refresh_outputs)

        # ===== Tab 3 =====
        with gr.Tab("3. Health check"):
            gr.Markdown(
                "### 토큰 권한 + task별 호출 점검\n"
                "현재 `HF_TOKEN`으로 5가지 task를 각 1회씩 작은 입력으로 호출하고 결과/실패 사유를 표로 보여준다.\n"
                "실패하면 가장 흔한 원인은 다음 두 가지:\n"
                "1. 토큰 권한 부족 → 토큰 발급 시 `Make calls to Inference Providers` (`inference.serverless.write`) 체크\n"
                "2. 해당 task의 provider 미활성화 → https://huggingface.co/settings/inference-providers 에서 켜기"
            )
            hc_key = gr.Textbox(label="HF_TOKEN override (선택)", type="password")
            hc_run = gr.Button("Run health check", variant="primary")
            hc_summary = gr.Markdown()
            hc_table = gr.Dataframe(headers=["항목", "상태", "결과/에러"], wrap=True, interactive=False)
            hc_run.click(run_health_check, inputs=hc_key, outputs=[hc_table, hc_summary])

        # ===== Tab 4 =====
        with gr.Tab("4. Compare"):
            gr.Markdown("### 같은 입력 → 여러 모델 비교\nchat 계열(text-to-text · image-text-to-text)만 지원. 쉼표로 모델 ID 나열.")
            cmp_task = gr.Radio(
                choices=["text-to-text", "image-text-to-text"],
                value="text-to-text", label="Task",
            )
            cmp_key = gr.Textbox(label="HF_TOKEN override (선택)", type="password")
            cmp_suggest = gr.Textbox(label="추천 모델 (복사해서 아래에 사용)", interactive=False, lines=2)
            cmp_models = gr.Textbox(label="비교할 모델 ID (쉼표 구분)", lines=2)
            cmp_system = gr.Textbox(label="System", lines=2, value=DEFAULT_SYSTEM)
            cmp_prompt = gr.Textbox(label="Prompt", lines=4, value="다음 한국어 문장을 영어로 번역해줘:\n\"내일 회의 시간을 오후 3시로 옮길 수 있을까요?\"")
            cmp_image = gr.Image(type="pil", label="Image input  (image-text-to-text 에만 사용)", sources=["upload", "clipboard"])
            with gr.Row():
                cmp_max = gr.Slider(16, 1024, value=320, step=16, label="Max tokens")
                cmp_temp = gr.Slider(0, 2, value=0.4, step=0.1, label="Temperature")
            cmp_run = gr.Button("Compare", variant="primary")
            cmp_out = gr.Dataframe(headers=["Model", "Latency · Tokens", "Output (preview)"], wrap=True, interactive=False)

            def _refresh_cmp(task):
                return ", ".join(TASK_MODELS.get(task, []))

            cmp_task.change(_refresh_cmp, inputs=cmp_task, outputs=cmp_suggest)
            demo.load(_refresh_cmp, inputs=cmp_task, outputs=cmp_suggest)

            cmp_run.click(compare_run,
                          inputs=[cmp_task, cmp_models, cmp_system, cmp_prompt, cmp_image, cmp_max, cmp_temp, cmp_key],
                          outputs=cmp_out)


if __name__ == "__main__":
    is_space = bool(os.getenv("SPACE_ID"))
    demo.launch(
        server_name="0.0.0.0" if is_space else "127.0.0.1",
        server_port=int(os.getenv("PORT", 7860)),
        ssr_mode=False,
        theme=gr.themes.Soft(),
    )
