## Knit - Quick Start

Minimal setup to run the hierarchical extractor, backend API, and frontend UI.


### Demo Video

[![Demo Video](https://img.youtube.com/vi/wEEMZnU0OZc/0.jpg)](https://www.youtube.com/watch?v=wEEMZnU0OZc)

### 1) Start the Hierarchical Extractor (NLM Ingestor)

- Pull the Docker image (public GHCR):

```bash
docker pull ghcr.io/nlmatics/nlm-ingestor:latest
```

- Run the container (map container port 5001 to a host port of your choice, e.g. 5010):

```bash
docker run -p 5010:5001 ghcr.io/nlmatics/nlm-ingestor:latest
```

- Your llmsherpa_url to use with clients will be:

```text
http://localhost:5010/api/parseDocument?renderFormat=all
```

### 2) Backend API

From the `backend` directory, install dependencies and run the API with Uvicorn:

```bash
cd backend
uvicorn app.main:app --port 8000
```

- API will be available at `http://localhost:8000`

### 3) Frontend App

From the `frontend` directory, install deps and start the dev server:

```bash
cd frontend
npm install
npm run dev
```

- Frontend will be available on the port shown by the dev server (commonly `http://localhost:5173`).

### Notes
- Ensure the NLM Ingestor container is running before uploading documents.
- Set the frontend and backend to point to the correct `llmsherpa_url` (`http://localhost:5010/api/parseDocument?renderFormat=all`).
- By default, the backend listens on port 8000; adjust as needed.

# knit
Agentic AI Tutor for Adaptive Learning in EdTech

.env file
```
