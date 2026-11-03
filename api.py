from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from codec import decode, encode

app = FastAPI(title="IMAGE-LINE LSC Codec API", version="2.0")
WORK = Path("work")
WORK.mkdir(exist_ok=True)
JOBS: dict[str, Path] = {}
MAX_UPLOAD_BYTES = 100 * 1024 * 1024


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index():
    page = Path("index.html")
    if not page.exists():
        raise HTTPException(404, "index.html is missing")
    return page.read_text(encoding="utf-8")


@app.post("/encode")
async def encode_image(file: UploadFile = File(...), max_kb: int = Form(...)):
    if max_kb < 1:
        raise HTTPException(400, "max_kb must be at least 1")

    data = await file.read()
    if not data:
        raise HTTPException(400, "Uploaded image is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image exceeds the 100 MiB upload limit")

    token = uuid.uuid4().hex
    job_dir = WORK / token
    job_dir.mkdir(parents=True, exist_ok=False)
    suffix = Path(file.filename or "image.png").suffix or ".png"
    source = job_dir / f"source{suffix}"
    encoded = job_dir / "encoded.lsc"
    restored = job_dir / "restored.png"
    source.write_bytes(data)

    try:
        result = encode(str(source), str(encoded), max_kb * 1024)
        decode(str(encoded), str(restored))
    except (ValueError, OSError) as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        await file.close()

    JOBS[token] = job_dir
    return {
        "token": token,
        "filename": "encoded.lsc",
        "size_bytes": result["output_bytes"],
        "limit_bytes": result["limit_bytes"],
        "compression_ratio": result["compression_ratio"],
        "scale": result["scale"],
        "quality": result["quality"],
        "download_url": f"/files/{token}/encoded.lsc",
        "preview_url": f"/files/{token}/restored.png",
    }


@app.post("/decode")
async def decode_image(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(400, "Uploaded LSC file is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File exceeds the 100 MiB upload limit")

    token = uuid.uuid4().hex
    job_dir = WORK / token
    job_dir.mkdir(parents=True, exist_ok=False)
    source = job_dir / "uploaded.lsc"
    restored = job_dir / "restored.png"
    source.write_bytes(data)
    try:
        decode(str(source), str(restored))
    except (ValueError, OSError) as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        await file.close()

    JOBS[token] = job_dir
    return {
        "token": token,
        "filename": "restored.png",
        "download_url": f"/files/{token}/restored.png",
    }


@app.get("/files/{token}/{filename}")
def get_job_file(token: str, filename: str):
    if filename not in {"encoded.lsc", "restored.png"}:
        raise HTTPException(404, "File not found")
    job_dir = JOBS.get(token)
    if job_dir is None:
        raise HTTPException(404, "Job not found")
    path = job_dir / filename
    if not path.is_file():
        raise HTTPException(404, "File not found")
    return FileResponse(path, filename=filename)
