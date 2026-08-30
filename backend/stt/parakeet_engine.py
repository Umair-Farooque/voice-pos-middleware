import numpy as np
from typing import Optional
import tempfile
import os
import subprocess
import soundfile as sf


class ParakeetSTT:
    def __init__(self):
        try:
            from nano_parakeet import from_pretrained
            self.model = from_pretrained()
            self.use_nano_parakeet = True
        except ImportError:
            from transformers import pipeline
            self.model = pipeline(
                "automatic-speech-recognition",
                model="nvidia/parakeet-tdt-0.6b-v3",
                device="cpu"
            )
            self.use_nano_parakeet = False

    def transcribe(self, audio_data: bytes) -> str:
        webm_path = None
        wav_path = None
        
        try:
            webm_path = tempfile.NamedTemporaryFile(suffix=".webm", delete=False).name
            with open(webm_path, 'wb') as f:
                f.write(audio_data)
            
            wav_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
            
            try:
                subprocess.run([
                    'ffmpeg', '-i', webm_path, '-ar', '16000', '-ac', '1', '-loglevel', 'error', wav_path, '-y'
                ], check=True, capture_output=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                try:
                    data, samplerate = sf.read(webm_path)
                    if len(data.shape) > 1:
                        data = data.mean(axis=1)
                    sf.write(wav_path, data, samplerate)
                except Exception:
                    wav_path = webm_path
            
            audio_array, rate = sf.read(wav_path)
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)
            
            print(f"[STT] Audio shape: {audio_array.shape}, sample rate: {rate}")
            
            if self.use_nano_parakeet:
                result = self.model.transcribe(wav_path)
                return result.text.strip() if hasattr(result, 'text') else str(result).strip()
            else:
                result = self.model({"sampling_rate": rate, "raw": audio_array})
                print(f"[STT] Model result: {result}")
                return result["text"].strip()
        finally:
            if webm_path and os.path.exists(webm_path):
                os.unlink(webm_path)
            if wav_path and wav_path != webm_path and os.path.exists(wav_path):
                os.unlink(wav_path)
