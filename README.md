---
title: HF Inference Providers Demo
emoji: 🤗
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# HF Inference Providers Demo

`Hugging Face Inference Providers`를 단일 진입점으로 5가지 task를 호출해 보는 Gradio 데모.

## 탭 구성

1. **Providers** — 주요 inference provider 카탈로그 비교 표
2. **Playground** — Task × Model 단일 호출
3. **Compare** — chat 계열에서 모델 2-4개 동시 호출, 결과/레이턴시/토큰 비교

## 지원 task

| Task | InferenceClient method | 입력 | 출력 |
|---|---|---|---|
| `text-to-text` | `chat.completions.create` | text | text |
| `image-text-to-text` (VLM) | `chat.completions.create` (`image_url` content) | text + image | text |
| `text-to-image` | `text_to_image` | text | PIL image |
| `text-to-speech` (TTS) | `text_to_speech` | text | audio bytes (flac/wav) |
| `speech-to-text` (ASR) | `automatic_speech_recognition` | audio bytes | text |

Task를 바꾸면 input/output 컴포넌트가 자동으로 토글된다.

## 예시 파일 (`examples/`)

| 파일 | 용도 |
|---|---|
| `sample_shapes.jpg` | VLM 기본 (피자 사진) |
| `sample_chart.png` | VLM 차트 읽기 |
| `sample_receipt.jpg` | VLM 영수증 정보 추출 (한국어) |
| `sample_speech.ogg` | ASR 입력 (eSpeak 영문 합성 음성) |

모두 위키미디어 커먼즈 공개 라이선스.

## 환경변수

```dotenv
HF_TOKEN=hf_your_token_here
```

- 발급: https://huggingface.co/settings/tokens — **Inference Providers** 호출 권한 필요 (text-to-image / TTS 등 일부 모델은 credit 필요)
- UI 상단의 override 필드에 일회성 입력도 가능

## 로컬 실행

```powershell
cd 00_for_me/src/hf_model_token_demo
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python app.py
```

브라우저: http://localhost:7860

## 배포 (GitHub → HF Space CI/CD)

토큰 2개를 분리해서 사용한다.

| 토큰 | 발급 권한 | 등록 위치 | 용도 |
|---|---|---|---|
| `HF_DEPLOY_TOKEN` | **Write access to contents/settings of selected repos** → 이 Space 1개만 선택 | GitHub repo → Settings → Secrets and variables → Actions | GH Actions가 Space repo에 push |
| `HF_TOKEN` | **Make calls to Inference Providers** (`inference.serverless.write`) | HF Space → Settings → Variables and secrets | 앱 런타임에서 모델 호출 |

### 1. HF Space 생성 (한 번)
- https://huggingface.co/new-space
- Owner = `jaeiJJ`, Space name = `hf-inference-providers-demo`, SDK = **Docker**

### 2. 토큰 등록
- https://huggingface.co/settings/tokens 에서 **두 개 별도 발급**
  - deploy token: `Write access to contents/settings of selected repos` → 위 Space 선택
  - inference token: `Make calls to Inference Providers`
- GitHub repo Secret: `HF_DEPLOY_TOKEN` ← deploy token
- HF Space Secret: `HF_TOKEN` ← inference token

### 3. push
```bash
git push origin main
```
GitHub Actions(`sync-to-hf.yml`)가 자동으로 Space repo에 force push → Space 빌드 트리거.
