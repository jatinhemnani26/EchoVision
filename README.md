# EchoVision

EchoVision is an advanced multimodal AI perception platform designed to bring visual context and ambient intelligence to various form factors. This repository contains both the cinematic frontend UI and the backend intelligence engine.

## Architecture

- **Frontend**: A cinematic, highly optimized static web application featuring raw typography, hardware-grade tactile interactions, and a purely atmospheric interface. (No generic AI chat interfaces)
- **Backend**: A FastAPI-based intelligence engine leveraging NVIDIA APIs for visual context analysis and AI logic.

## Folder Structure

- `frontend/` - Contains the cinematic frontend interface (HTML, CSS, JS). Ready to be deployed to Vercel or similar static hosts.
- `backend/` - Contains the FastAPI application, NVIDIA integration logic, and endpoints. Ready to be deployed to Render, Vultr, or any Python runtime.

## Getting Started

### Backend
1. Navigate to the `backend/` directory.
2. Create a `.env` file and add your NVIDIA API key: `NVIDIA_API_KEY=your_key_here`
3. Install dependencies: `pip install -r requirements.txt`
4. Run the server: `python main.py` or `uvicorn main:app --reload`
5. The backend will start on `http://127.0.0.1:8000`.

### Frontend
1. Navigate to the `frontend/` directory.
2. Serve the static files using any HTTP server: `python -m http.server 3000`
3. Open `http://localhost:3000` in your browser.
4. (The frontend dynamically switches to the cloud backend URL in production, or uses `127.0.0.1` locally.)
