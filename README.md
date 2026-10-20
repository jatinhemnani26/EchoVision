<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/NVIDIA_NIM-Llama_3.2_Vision-76B900?logo=nvidia&logoColor=white" />
  <img src="https://img.shields.io/badge/Alexa-Custom_Skill-00CAFF?logo=amazon-alexa&logoColor=white" />
  <img src="https://img.shields.io/badge/WebRTC-Camera-333?logo=webrtc&logoColor=white" />
  <img src="https://img.shields.io/badge/Deploy-Vercel_+_Render-000?logo=vercel&logoColor=white" />
</p>

# EchoVision 👁️‍🗨️

### *"Alexa, what's in front of me?"*

> Close your eyes for ten seconds. Now try to pour yourself a glass of water.
>
> That's Tuesday morning for **2.2 billion people** living with vision impairment. Their phone has a 48MP camera. Their home has a voice assistant. Nobody wired them together. Until now.

EchoVision turns **any browser camera + any Alexa device** into a real-time visual assistant for people who can't see. No app install. No download. Just open a URL, talk to Alexa, and your world gets described back to you in under 2 seconds.

**🔴 [Live Demo](https://echovision45.vercel.app)** &nbsp;|&nbsp; **📡 [API Health](https://echovision-backend-w6ns.onrender.com/health)**

---

## The 10-Second Pitch

```
You:     "Alexa, ask smart vision what is in front of me"
Alexa:   "There's a white coffee mug on your desk, about six inches
          to the left of your keyboard. It looks like it still has coffee in it."
```

Not a classification tag. Not a JSON blob. A real, human sentence. In English, Hindi (देवनागरी), Spanish, or French.

---

## How It Actually Works

```
┌─────────────────────┐          ┌──────────────────────┐
│   Your Phone/Laptop │          │   Your Alexa Device   │
│   (any browser)     │          │   "ask smart vision…" │
└────────┬────────────┘          └───────────┬──────────┘
         │                                   │
         │ getUserMedia()                    │ Alexa Webhook POST
         │ live 1080p WebRTC                 │
         ▼                                   ▼
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Backend                       │
│                    (Render, 512MB)                       │
│                                                         │
│  ┌───────────────┐    ┌──────────────────────────────┐  │
│  │ Frame Buffer  │    │     2-Tier Triage Router      │  │
│  │ (in-memory,   │    │                              │  │
│  │  refreshed    │    │  "read label" ──► Edge OCR   │  │
│  │  every 3s)    │    │   (Local Tesseract triage)   │  │
│  │               │    │                              │  │
│  │  Alexa reads  │    │  "what's in front of me"     │  │
│  │  latest frame │    │   ──► NVIDIA NIM Cloud Brain │  │
│  └───────────────┘    │   (Llama 3.2 11B Vision)     │  │
│                       └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
         │
         ▼
   ┌─────────────┐
   │  Response    │──► Alexa speaks it  (voice)
   │  < 2 sec     │──► Browser TTS      (audio)
   │              │──► On-screen text    (visual)
   └─────────────┘
```

**The key insight (2-Tier Architecture):** Not every visual question needs an expensive cloud GPU. The architecture features an intelligent **2-Tier Triage Router**: text/label reading queries can resolve on local edge OCR (Tesseract at $0.00 cost when available on local/edge hardware), while complex spatial reasoning, room navigation, and scene understanding route dynamically to the NVIDIA NIM Cloud Brain (Llama 3.2 11B Vision). In local edge setups, this cuts cloud inference calls by ~40%, while cloud-hosted instances fail-safe to NVIDIA NIM.

---

## What Makes This Different

| Feature | Typical "AI Vision" App | EchoVision |
|---------|------------------------|------------|
| **Setup** | Download app, sign up, grant permissions | Open a URL. That's it. |
| **Voice** | Tap a button, wait | "Alexa, ask smart vision…" (hands-free, eyes-free) |
| **Speed** | 5-10 seconds | Sub-2 seconds end-to-end |
| **Languages** | English only | Native Hindi (देवनागरी), Spanish, French, English |
| **Cost Strategy** | $0.003+ for every single frame | 2-Tier Triage: Edge OCR triage + Cloud NIM VLM |
| **Install** | App Store / Play Store | Works on a $50 phone's browser from 2019 |

---

## The Tech Stack (and Why Each Piece)

| Layer | Tech | Why |
|-------|------|-----|
| **Frontend** | Vanilla HTML/CSS/JS + WebGL | Zero build step, zero dependencies, works everywhere |
| **Camera** | WebRTC `getUserMedia` | Native browser API, no plugins, back-camera preference on mobile |
| **Backend** | Python 3.11 + FastAPI | Async, lightweight, fits in 512MB free tier |
| **Vision AI** | NVIDIA NIM → Llama 3.2 11B Vision Instruct | OpenAI-compatible API, sub-second multimodal inference |
| **Edge OCR** | Tesseract (pytesseract) | Local fallback for text queries without cloud overhead |
| **Voice** | Amazon Alexa Custom Skill | 500M+ devices worldwide, true hands-free interaction |
| **TTS** | Web Speech API | Built into every modern browser, 30+ languages |
| **Frontend Host** | Vercel | Auto-deploy on push, global CDN, SSL included |
| **Backend Host** | Render | Free tier Python hosting, auto-deploy from GitHub |

---

## Project Structure

```
EchoVision/
├── frontend/
│   ├── index.html          # Everything. WebGL shader, camera, UI, voice, TTS.
│   └── netflix-intro.mp3   # Splash screen empathy reveal audio
│
├── backend/
│   ├── main.py             # FastAPI app — /analyze, /alexa, /sync_frame + rate limiter
│   ├── requirements.txt    # Lean deps (no PyTorch, no bloat)
│   ├── runtime.txt         # Python 3.11.9 pinned for Render
│   └── vision/
│       ├── camera.py       # Platform-safe camera (Windows CAP_DSHOW / Linux CAP_ANY)
│       ├── cloud_brain.py  # NVIDIA NIM integration + multilingual prompt engine
│       ├── ocr.py          # Tesseract wrapper with graceful fallback
│       ├── router.py       # 2-Tier triage: edge OCR vs cloud VLM
│       └── preprocessor.py # Frame optimization before inference
│
├── .python-version         # 3.11.9
└── README.md               # You are here
```

---

## Getting Started

### Prerequisites
- Python 3.11+
- An [NVIDIA NIM API key](https://build.nvidia.com/) (free developer tier available)
- (Optional) [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed locally for local edge mode

### 1. Clone & Setup Backend

```bash
git clone https://github.com/jatinhemnani26/EchoVision.git
cd EchoVision/backend

# Create your environment file
cp .env.example .env   # or create manually:
# NVIDIA_API_KEY=your_nvidia_nim_key_here
# NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
# NVIDIA_VISION_MODEL=meta/llama-3.2-11b-vision-instruct

# Install dependencies
pip install -r requirements.txt

# Run
python main.py
# → Backend starts on http://127.0.0.1:8000
```

### 2. Run Frontend

```bash
cd ../frontend
python -m http.server 3000
# → Open http://localhost:3000 in your browser
```

The frontend auto-detects local vs production and switches the API URL accordingly.

### 3. (Optional) Connect Alexa

1. Create a Custom Skill in the [Alexa Developer Console](https://developer.amazon.com/alexa/console/ask)
2. Set invocation name to `smart vision`
3. Add an intent called `AskVisionIntent` with sample utterances like *"what is in front of me"*
4. Set the HTTPS endpoint to `https://your-backend-url/api/v1/alexa`
5. Say: *"Alexa, ask smart vision what is in front of me"*

---

## API Endpoints & Security

| Method | Endpoint | What It Does |
|--------|----------|-------------|
| `GET` | `/health` | Health check — returns service and sensor status |
| `POST` | `/api/v1/vision/analyze` | Client-authenticated frame analysis via 2-Tier router & NVIDIA NIM |
| `POST` | `/api/v1/vision/sync_frame` | Browser pushes live camera frame buffer (called every 3s automatically) |
| `POST` | `/api/v1/alexa` | Alexa webhook — reads latest live frame, describes it, returns Alexa JSON |

> **🛡️ Built-in Token & Quota Protection:** The backend enforces per-IP sliding-window rate limits (max 20 req/min) and a global hourly circuit breaker, preventing token exhaustion and bot spam while keeping the public demo fast and accessible.

---

## The Debugging War Diary

We're not going to pretend this was smooth. Some highlights:

**⏱️ The 8.3-Second Wall** — Our first Alexa integration took 8.3 seconds. Alexa's timeout is 8.0. We heard *"The requested skill did not provide a valid response"* approximately 400 times. The fix: kill the hardware camera scan (2s saved), cap `max_tokens` at 60 (1.5s saved), and pre-cache fallback frames in memory. Final: **1.7 seconds**.

**📦 The 2.5GB Ghost** — We had `easyocr` in `requirements.txt` from an early experiment. Never imported it. Never used it. But it pulled in PyTorch + CUDA — 2.5 gigabytes. We only noticed when Render's 512MB RAM started killing builds.

**⚙️ Headless Cloud Architecture** — While our 2-Tier Router natively supports local OCR on edge machines with Tesseract binaries, our lightweight cloud backend runs in a 512MB headless container without binary OCR bloat, seamlessly routing spatial visual queries directly to NVIDIA NIM with sub-2s response times.

**🐍 Python 3.14 Surprise** — Render defaults to Python 3.14. Our OpenAI SDK doesn't support it. 47 lines of compilation errors. Fix: pin `3.11.9` in three separate files and pray.

**🇮🇳 The Hindi Trap** — Asking the model to "respond in Hindi" gives you Romanized Hindi: *"Aapke saamne ek mug hai."* Useless for screen readers. We had to inject Devanagari script examples and repeat the mandate at both the start AND end of the prompt.

---

## Live URLs

| What | URL |
|------|-----|
| **Frontend** | [echovision45.vercel.app](https://echovision45.vercel.app) |
| **Backend** | [echovision-backend-w6ns.onrender.com](https://echovision-backend-w6ns.onrender.com) |
| **Health Check** | [/health](https://echovision-backend-w6ns.onrender.com/health) |

---

## What's Next

- **Offline Mode** — On-device vision models via WebAssembly. No internet required.
- **Spatial Audio** — Stereo-positioned descriptions ("chair to your left" *comes from the left speaker*).
- **Object Memory** — "Is my backpack still where I left it?" Persistent object tracking across sessions.
- **Wearable Camera** — ESP32-CAM modules ($4) streaming directly to the backend.
- **Community Language Packs** — Urdu, Bengali, Tamil, Arabic with culturally appropriate descriptions.

---

## Built With

`python` · `fastapi` · `nvidia-nim` · `llama-3.2-vision` · `webrtc` · `webgl` · `alexa-skills-kit` · `opencv` · `tesseract-ocr` · `web-speech-api` · `vercel` · `render`

---

<p align="center">
  <b>Built because a phone that can see shouldn't belong to someone who can't.</b>
</p>
