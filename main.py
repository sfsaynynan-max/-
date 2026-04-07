import os
import httpx
import assemblyai as aai
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tempfile

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Keys من Environment Variables
ASSEMBLYAI_KEY = os.environ.get("ASSEMBLYAI_KEY")
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_KEY")

# ==============================
# تفريغ الصوت بـ AssemblyAI
# ==============================
@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form("auto")
):
    try:
        aai.settings.api_key = ASSEMBLYAI_KEY
        
        # حفظ الملف مؤقتاً
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        config = aai.TranscriptionConfig(
    language_detection=True,
    punctuate=True,
    speech_model=aai.SpeechModel.universal,
)
        
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(tmp_path, config=config)
        
        if transcript.status == aai.TranscriptStatus.error:
            raise HTTPException(status_code=500, detail=transcript.error)

        # تحويل النتيجة لصيغة SRT مع التوقيت
        words = transcript.words
        srt_entries = []
        
        chunk = []
        chunk_start = None
        chunk_end = None
        
        for i, word in enumerate(words):
            if chunk_start is None:
                chunk_start = word.start
            chunk.append(word.text)
            chunk_end = word.end
            
            if len(chunk) >= 8 or i == len(words) - 1:
                srt_entries.append({
                    "index": len(srt_entries) + 1,
                    "start": chunk_start,
                    "end": chunk_end,
                    "text": " ".join(chunk)
                })
                chunk = []
                chunk_start = None
        
        os.unlink(tmp_path)
        
        return {
            "success": True,
            "segments": srt_entries,
            "detected_language": transcript.language_code
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==============================
# ترجمة النص بـ DeepSeek
# ==============================
@app.post("/translate")
async def translate(
    segments: list = Form(...),
    target_language: str = Form(...),
):
    try:
        # نترجم كل segments دفعة واحدة لتوفير الوقت
        texts = [s["text"] for s in segments]
        combined = "\n---\n".join(texts)
        
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                "https://api.deepseek.com/chat/completions",
                headers={
                    "Authorization": f"Bearer {DEEPSEEK_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {
                            "role": "system",
                            "content": f"""You are a professional subtitle translator.
Translate the following subtitle segments to {target_language}.
Each segment is separated by ---
Return ONLY the translated segments separated by ---
Keep the same number of segments.
Keep translations natural and concise for subtitles."""
                        },
                        {
                            "role": "user",
                            "content": combined
                        }
                    ],
                    "temperature": 0.3
                }
            )
        
        result = response.json()
        translated_text = result["choices"][0]["message"]["content"]
        translated_parts = translated_text.split("\n---\n")
        
        # دمج الترجمة مع التوقيتات
        translated_segments = []
        for i, seg in enumerate(segments):
            translated_segments.append({
                "index": seg["index"],
                "start": seg["start"],
                "end": seg["end"],
                "original": seg["text"],
                "translated": translated_parts[i].strip() if i < len(translated_parts) else seg["text"]
            })
        
        return {
            "success": True,
            "segments": translated_segments
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==============================
# تفريغ + ترجمة في طلب واحد
# ==============================
@app.post("/process")
async def process(
    file: UploadFile = File(...),
    target_language: str = Form("English"),
    is_premium: str = Form("false")
):
    try:
        aai.settings.api_key = ASSEMBLYAI_KEY
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # تفريغ
        config = aai.TranscriptionConfig(
            language_detection=True,
            punctuate=True,
        )
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(tmp_path, config=config)
        
        if transcript.status == aai.TranscriptStatus.error:
            raise HTTPException(status_code=500, detail=transcript.error)

        words = transcript.words
        segments = []
        chunk = []
        chunk_start = None
        chunk_end = None
        
        for i, word in enumerate(words):
            if chunk_start is None:
                chunk_start = word.start
            chunk.append(word.text)
            chunk_end = word.end
            if len(chunk) >= 8 or i == len(words) - 1:
                segments.append({
                    "index": len(segments) + 1,
                    "start": chunk_start,
                    "end": chunk_end,
                    "text": " ".join(chunk)
                })
                chunk = []
                chunk_start = None

        os.unlink(tmp_path)

        # ترجمة
        texts = [s["text"] for s in segments]
        combined = "\n---\n".join(texts)
        
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                "https://api.deepseek.com/chat/completions",
                headers={
                    "Authorization": f"Bearer {DEEPSEEK_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {
                            "role": "system",
                            "content": f"""You are a professional subtitle translator.
Translate the following subtitle segments to {target_language}.
Each segment is separated by ---
Return ONLY the translated segments separated by ---
Keep the same number of segments."""
                        },
                        {
                            "role": "user",
                            "content": combined
                        }
                    ],
                    "temperature": 0.3
                }
            )
        
        result = response.json()
        translated_text = result["choices"][0]["message"]["content"]
        translated_parts = translated_text.split("\n---\n")
        
        final_segments = []
        for i, seg in enumerate(segments):
            final_segments.append({
                "index": seg["index"],
                "start": seg["start"],
                "end": seg["end"],
                "original": seg["text"],
                "translated": translated_parts[i].strip() if i < len(translated_parts) else seg["text"]
            })
        
        return {
            "success": True,
            "segments": final_segments,
            "detected_language": transcript.language_code
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok", "service": "flextman-api"}
