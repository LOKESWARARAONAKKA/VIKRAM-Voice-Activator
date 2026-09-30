from fastapi import FastAPI, UploadFile, File
from faster_whisper import WhisperModel
import os
import tempfile

from server.command_handler import process_command

app = FastAPI(title="VIKRAM ASR Server")

print("Loading Whisper model...")
model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"
)
print("Whisper model loaded successfully.")


@app.get("/")
def root():
    return {
        "status": "running",
        "service": "VIKRAM ASR Server"
    }


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename or ".wav")[1]

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as temp:
        temp.write(await file.read())
        temp_path = temp.name

    try:
        segments, info = model.transcribe(
            temp_path,
            beam_size=5,
            language="en"
        )

        text = " ".join(
            segment.text.strip()
            for segment in segments
        ).strip()

        result = process_command(text)

        return {
            "text": text,
            "language": info.language,
            "language_probability": info.language_probability,
            "command": result["command"],
            "action": result["action"]
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
