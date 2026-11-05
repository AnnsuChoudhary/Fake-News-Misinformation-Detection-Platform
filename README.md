# Fake News & Misinformation Detection Platform

This repository includes a FastAPI backend and a React/Vite frontend for fake-news detection.

## What is included
- FastAPI backend under [backend](backend) with endpoints for content, propagation, hybrid, and combined analysis.
- React/Vite frontend under [frontend](frontend), with local API calls proxied through Vite and `VITE_API_URL` available for deployment.
- Deployment configuration for Render and Netlify/Vercel.

## Model availability

The included `gnn_propagation_model.pth` supports propagation analysis. Content and hybrid analysis also require the trained `pheme_distilbert_model/` directory (or a compatible content checkpoint), which is not included in this repository. The API health endpoint reports which models are available, and the frontend disables modes that cannot run.

## Run locally

Install dependencies once from the repository root:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
cd frontend
npm install
```

Whenever you reopen the project, start the backend in one terminal from the repository root:
```bash
source .venv/bin/activate
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Start the frontend in a second terminal:
```bash
cd frontend
npm run dev -- --host 0.0.0.0
```

In VS Code, open port `5173` from the **Ports** tab. Keep both terminals running while using the app. Vite proxies local API requests to port `8000`.

During local development, the Vite server proxies API requests to `http://127.0.0.1:8000`. For a deployed frontend, set `VITE_API_URL` to the backend's public URL.

## Deployment
### Render (backend)
1. Create a new Web Service from this repository.
2. Set the build command to `pip install -r backend/requirements.txt`.
3. Use the start command from [backend/Procfile](backend/Procfile).
4. Deploy.

### Netlify or Vercel (frontend)
1. Point the deployment to the [frontend](frontend) folder.
2. Set the build command to `npm run build`.
3. Set the publish directory to `dist`.
4. Configure the `VITE_API_URL` environment variable to your deployed backend URL.

The backend reports missing or unloadable checkpoints in `/health` and returns a clear error response if an unavailable prediction mode is requested directly.
