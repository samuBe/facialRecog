# Tactical Face Scanner

A two-part facial-recognition demo:
- **Client** (`client/`): browser app for image upload/camera capture, local face detection + descriptor extraction (`face-api.js`), interactive face selection, and lookup/enrollment actions.
- **Server** (`server/`): FastAPI service that receives face embeddings and performs cosine-similarity search against a SQLite embedding store.

## 1) Run the server

```bash
cd "/Users/samuel/Documents/New project/server"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API endpoints:
- `GET /health`
- `GET /identities`
- `POST /enroll`
- `POST /search`

## 2) Run the browser client

Serve the client directory with any static server (camera access usually requires `http://localhost` or HTTPS):

```bash
cd "/Users/samuel/Documents/New project/client"
python3 -m http.server 5173
```

Then open `http://localhost:5173`.

## Usage flow

1. Upload an image or start camera and capture a frame.
2. Click **Extract Faces** to detect faces locally.
3. Select one detected face (thumbnail row or box click).
4. Click **Lookup Selected Face** to send the embedding to FastAPI.
5. Optional: set a label and click **Enroll Selected Face** to add/update an embedding in the DB.

## Notes

- Face detection + descriptor extraction runs locally in the browser with `face-api.js`.
- First model load downloads model files from CDN.
- The starter DB is seeded once from `server/data/seed_embeddings.json` on first launch.
- This is a prototype UI and not production-grade biometric security.
