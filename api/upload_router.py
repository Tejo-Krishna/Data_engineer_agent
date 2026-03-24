"""
File upload endpoint.

POST /api/upload  — accepts a multipart file, saves to uploads/, returns the saved path.
"""

import os
from pathlib import Path

import aiofiles
from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import JSONResponse

router = APIRouter()

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))


@router.post("/upload")
async def upload_file(file: UploadFile):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # Sanitise filename — strip path separators
    filename = Path(file.filename).name if file.filename else "upload.csv"
    save_path = UPLOAD_DIR / filename

    # If a file with the same name exists, add a counter suffix
    counter = 1
    stem = save_path.stem
    suffix = save_path.suffix
    while save_path.exists():
        save_path = UPLOAD_DIR / f"{stem}_{counter}{suffix}"
        counter += 1

    try:
        async with aiofiles.open(save_path, "wb") as f:
            content = await file.read()
            await f.write(content)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {exc}")

    return JSONResponse({
        "saved_path": str(save_path),
        "filename": save_path.name,
        "size_bytes": save_path.stat().st_size,
    })
