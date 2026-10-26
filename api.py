from pathlib import Path
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from codec import encode, decode

app = FastAPI(title="LSC Image Codec API", version="1.0")
WORK = Path("work")
WORK.mkdir(exist_ok=True)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/encode")
async def encode_image(file: UploadFile = File(...), max_kb: int = Form(...)):
    if max_kb < 1:
        raise HTTPException(400, "max_kb must be at least 1")
    suffix = Path(file.filename or "image.png").suffix or ".png"
    source = WORK / f"upload{suffix}"
    output = WORK / "encoded.lsc"
    source.write_bytes(await file.read())
    try:
        result = encode(str(source), str(output), max_kb * 1024)
    except Exception as exc:
        raise HTTPException(422, str(exc))
    return {
        "filename": "encoded.lsc",
        "size_bytes": result["output_bytes"],
        "limit_bytes": result["limit_bytes"],
        "compression_ratio": result["compression_ratio"],
        "download_url": "/download/encoded.lsc"
    }


@app.get("/download/{filename}")
def download(filename: str):
    if filename not in {"encoded.lsc", "restored.png"}:
        raise HTTPException(404, "File not found")
    path = WORK / filename
    if not path.exists():
        raise HTTPException(404, "File has not been created")
    return FileResponse(path, filename=filename)


@app.post("/decode")
async def decode_image(file: UploadFile = File(...)):
    source = WORK / "uploaded.lsc"
    output = WORK / "restored.png"
    source.write_bytes(await file.read())
    try:
        decode(str(source), str(output))
    except Exception as exc:
        raise HTTPException(422, str(exc))
    return {"filename": "restored.png", "download_url": "/download/restored.png"}
