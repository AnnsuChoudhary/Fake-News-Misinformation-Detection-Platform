# Fake News & Misinformation Detection Platform

This repository now includes a full-stack scaffold for serving the trained fake-news detection models through a FastAPI backend and a React/Vite frontend.

## What is included
- FastAPI backend under [backend](backend) with endpoints for content, propagation, hybrid, and combined analysis.
- React/Vite frontend under [frontend](frontend) with a polished single-page experience and environment-based API configuration.
- Deployment configuration for Render and Netlify/Vercel.

## Backend

### Local setup
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run the API
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Environment variables
Copy [backend/.env.example](backend/.env.example) if you need to override the default frontend origin.

## Frontend

### Local setup
```bash
cd frontend
npm install
cp .env.example .env
```

### Run the UI
```bash
cd frontend
npm run dev
```

The UI uses the VITE_API_URL variable from [frontend/.env](frontend/.env) and calls the backend at the configured URL.

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

## Notes
- The scaffold loads the provided model checkpoints rather than retraining them.
- If a model file is missing or cannot be loaded, the backend returns a clear error response instead of crashing.
