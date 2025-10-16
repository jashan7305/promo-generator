import ffmpeg
from faster_whisper import WhisperModel
from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

client = OpenAI(api_key=OPENROUTER_API_KEY)
transcription_model = WhisperModel("base", device="cpu")

def extract_audio(video_path, audio_path="temp.wav"):
    ffmpeg.input(video_path).output(audio_path, ac=1, ar=16000).run(overwrite_output=True)
    return audio_path

def transcribe_audio(audio_path):
    segments, info = transcription_model.transcribe(audio_path)

    results = []
    for segment in segments:
        results.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip()
        })

    return results

def get_imp_dialogues(segments, theme):
    text = "\n".join([f"[{s['start']:.2f}-{s['end']:.2f}] {s['text']}" for s in segments])

    prompt = f"""
    Here is a transcript of a video with timestamps:
    {text}
    
    The user wants a promo about the theme: "{theme}".
    Select the 5 most impactful dialogues that align with this theme.
    Respond as JSON: [{{"start": <float>, "end": <float>, "text": "<dialogue>"}}]
    """

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b:free",
        messages=[{"role": "user", "content": prompt}]
    )

    return json.loads(response.choices[0].message.content)

def expand_timestamps(dialogues, secs=5, video_duration=None):
    clips = []
    for d in dialogues:
        start = max(0, d["start"] - secs)
        end = min(video_duration, d["end"] + secs) if video_duration else d["end"] + secs
        clips.append((start, end))
    return clips

def stitch_clips(video_path, clips, output_path="promo.mp4"):
    streams = [ffmpeg.input(video_path, ss=s, to=e) for s, e in clips]
    joined = ffmpeg.concat(*streams)
    joined.output(output_path).run(overwrite_output=True)
    return output_path

def generate_promo(video_path, theme):
    audio_path = extract_audio(video_path)
    segments = transcribe_audio(audio_path)
    key_dialogues = get_imp_dialogues(segments, theme)
    
    probe = ffmpeg.probe(video_path)
    duration = float(probe['format']['duration'])
    
    clips = expand_timestamps(key_dialogues, secs=5, video_duration=duration)
    promo_path = stitch_clips(video_path, clips)
    
    return promo_path

def hello_world():
    return {"message": "hello world"}