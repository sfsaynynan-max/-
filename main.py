from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import re

app = FastAPI()

DEEPSEEK_API_KEY = ""  # من Railway Variables

class TranslateRequest(BaseModel):
    transcript: list  # Flutter يرسل الـ transcript جاهز
    target_language: str = "Arabic"

def refine_and_translate(transcript: list, target_language: str):
    full_text = ""
    for item in transcript:
        start = item['start']
        text = item['text']
        full_text += f"[{start:.2f}] {text}\n"

    prompt = f"""You are given a raw auto-generated YouTube transcript with timestamps.
Your job:
1. Fix grammar, punctuation, and broken sentences
2. Translate to {target_language}
3. Keep the EXACT timestamp format [seconds] at the start of each line
4. Return ONLY the corrected and translated transcript, nothing else

Transcript:
{full_text}"""

    response = requests.post(
        "https://api.deepseek.com/chat/completions",
        headers={
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3
        }
    )

    result = response.json()
    translated_text = result['choices'][0]['message']['content']

    subtitles = []
    lines = translated_text.strip().split('\n')
    for line in lines:
        match = re.match(r'\[(\d+\.?\d*)\]\s+(.+)', line)
        if match:
            subtitles.append({
                "start": float(match.group(1)),
                "text": match.group(2)
            })

    return subtitles

@app.get("/")
def root():
    return {"status": "Flext API running"}

@app.post("/translate")
def translate_video(req: TranslateRequest):
    if not req.transcript:
        raise HTTPException(status_code=400, detail="Transcript is empty")
    
    subtitles = refine_and_translate(req.transcript, req.target_language)
    return {"subtitles": subtitles}
