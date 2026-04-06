from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import re
import os
import subprocess
import json
import glob

app = FastAPI()

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

class TranslateRequest(BaseModel):
    video_url: str
    target_language: str = "Arabic"

def extract_video_id(url: str):
    patterns = [
        r'(?:v=)([0-9A-Za-z_-]{11})',
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})',
        r'shorts\/([0-9A-Za-z_-]{11})'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_transcript(video_id: str):
    url = f"https://www.youtube.com/watch?v={video_id}"
    output_path = f"/tmp/{video_id}"
    
    # تحديث yt-dlp لأحدث إصدار
    subprocess.run(['pip', 'install', '--upgrade', 'yt-dlp'], capture_output=True)
    
    # تنظيف أي ملفات قديمة
    for f in glob.glob(f"{output_path}*"):
        try:
            os.remove(f)
        except:
            pass

    # User-Agent لمتصفح حقيقي
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

    try:
        subprocess.run(
            [
                'yt-dlp',
                '--write-auto-sub',
                '--skip-download',
                '--sub-format', 'json3',
                '--sub-langs', 'en,ar,fr,es',
                '--user-agent', user_agent,
                '--extractor-retries', '5',
                '--no-warnings',
                '--no-check-certificates',
                '-o', output_path,
                url
            ],
            capture_output=True,
            text=True,
            timeout=90
        )

        files = glob.glob(f"{output_path}*.json3")
        if not files:
            # محاولة ثانية مع خيار مختلف
            subprocess.run(
                [
                    'yt-dlp',
                    '--write-sub',
                    '--skip-download',
                    '--sub-format', 'json3',
                    '--sub-langs', 'en,ar',
                    '--user-agent', user_agent,
                    '--extractor-retries', '5',
                    '-o', output_path,
                    url
                ],
                capture_output=True,
                text=True,
                timeout=90
            )
            files = glob.glob(f"{output_path}*.json3")

        if not files:
            raise Exception("No subtitle file generated even after retry")

        with open(files[0], 'r', encoding='utf-8') as f:
            data = json.load(f)

        transcript = []
        for event in data.get('events', []):
            segs = event.get('segs', [])
            text = ''.join(s.get('utf8', '') for s in segs).strip()
            if text and text != '\n':
                transcript.append({
                    'start': event['tStartMs'] / 1000.0,
                    'text': text
                })

        for f in files:
            try:
                os.remove(f)
            except:
                pass

        if not transcript:
            raise Exception("Transcript is empty")

        return transcript

    except Exception as e:
        for f in glob.glob(f"{output_path}*"):
            try:
                os.remove(f)
            except:
                pass
        raise HTTPException(status_code=404, detail=str(e))

def refine_and_translate(transcript: list, target_language: str):
    full_text = ""
    for item in transcript:
        full_text += f"[{item['start']:.2f}] {item['text']}\n"

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
        },
        timeout=90
    )

    result = response.json()
    translated_text = result['choices'][0]['message']['content']

    subtitles = []
    for line in translated_text.strip().split('\n'):
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

    return {"video_id": video_id, "subtitles": subtitles}
