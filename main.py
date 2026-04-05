from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi
import requests
import re

app = FastAPI()

DEEPSEEK_API_KEY = ""  # سنضيفه في Railway

class TranslateRequest(BaseModel):
    video_url: str
    target_language: str = "Arabic"

def extract_video_id(url: str):
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11})',
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_transcript(video_id: str):
    try:
        ytt = YouTubeTranscriptApi()
        transcript = ytt.fetch(video_id, languages=['en', 'ar'])
        return [{'start': s.start, 'text': s.text} for s in transcript]
    except Exception:
        try:
            ytt = YouTubeTranscriptApi()
            transcript_list = ytt.list(video_id)
            transcript = transcript_list.find_generated_transcript(['en', 'ar']).fetch()
            return [{'start': s.start, 'text': s.text} for s in transcript]
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"No transcript found: {str(e)}")

def refine_and_translate(transcript: list, target_language: str):
    # نجمع النص كامل مع timestamps
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

    # نحول النص المترجم لقائمة مع timestamps
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
    video_id = extract_video_id(req.video_url)
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")

    transcript = get_transcript(video_id)
    subtitles = refine_and_translate(transcript, req.target_language)

    return {
        "video_id": video_id,
        "subtitles": subtitles
    }
