import os
import tempfile

import numpy as np
import soundfile as sf
import torch
import torchaudio
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

device = "cuda" if torch.cuda.is_available() else "cpu"
app = FastAPI(
    title="ASR API",
    description="API for ASR",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)   

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h").to(device)

@app.get("/ping")
async def ping():
    return {"message": "pong"}

@app.post("/asr")
async def transcribe(file: UploadFile = File(...)):
    if not file.filename.endswith('.mp3'):
        raise HTTPException(status_code=400, detail="Only MP3 files are supported")
    
    try:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        # Load audio file with soundfile
        audio_array, sample_rate = sf.read(temp_file_path)
        
        # Convert to mono if stereo
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            audio_array = torch.from_numpy(audio_array).float()
            audio_array = resampler(audio_array).numpy()
            sample_rate = 16000
        
        # Get duration
        duration = len(audio_array) / sample_rate
        
        # Normalize audio array
        audio_array = audio_array / np.max(np.abs(audio_array))
        
        # Process audio with Wav2Vec2
        input_values = processor(
            audio_array, 
            return_tensors="pt", 
            padding="longest",
            sampling_rate=sample_rate
        ).input_values.to(device)

        # Get model predictions
        with torch.no_grad():
            logits = model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)[0]

        # Clean up temporary file
        os.unlink(temp_file_path)

        return {
            "transcription": transcription,
            "duration": str(duration)
        }

    except Exception as e:
        # Clean up temporary file in case of error
        if 'temp_file_path' in locals():
            os.unlink(temp_file_path)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn

    # Get port from environment variable, default to 8000 if not set
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)