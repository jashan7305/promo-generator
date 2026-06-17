import os
import shutil
import numpy as np
import ffmpeg
from faster_whisper import WhisperModel
from transformers import AutoTokenizer
from onnxruntime import InferenceSession
from sklearn.neighbors import NearestNeighbors

# Load models (Note: These will fail without environment dependencies)
try:
    transcription_model = WhisperModel("base", device="cpu")
    embedding_model = "Xenova/e5-small-v2"
    tokenizer = AutoTokenizer.from_pretrained(embedding_model)
    model_path = f"{embedding_model}/onnx/model.onnx"
    onnx_sess = InferenceSession(model_path, providers=["CPUExecutionProvider"])
except Exception:
    pass

def embed_texts(texts: list[str] | str) -> np.ndarray:
    """
    Computes embeddings for the provided texts.

    Args:
        texts: A single string or a list of strings.

    Returns:
        The normalized embeddings as a numpy array.
    """
    if isinstance(texts, str):
        texts = [texts]
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="np")
    outputs = onnx_sess.run(None, dict(inputs))
    embeddings = outputs[0]
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings

def extract_audio(video_path: str, temp_folder: str = "temp") -> str:
    """
    Extracts audio from a video file into a temporary WAV file.

    Args:
        video_path: Path to the input video.
        temp_folder: Directory to store the extracted audio.

    Returns:
        The path to the extracted audio file.
    """
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
    
    audio_path = os.path.join(temp_folder, "temp.wav")
    
    ffmpeg.input(video_path).output(audio_path, ac=1, ar=16000).run(overwrite_output=True)
    return audio_path

def transcribe_audio(audio_path: str) -> list[dict]:
    """
    Transcribes the audio file using a whisper model.

    Args:
        audio_path: Path to the audio file.

    Returns:
        A list of segments containing start time, end time, and transcribed text.
    """
    segments, info = transcription_model.transcribe(audio_path)

    results = []
    for segment in segments:
        results.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip()
        })
    return results

def get_imp_dialogues(dialogues: list[dict], theme: str, n_results: int = 3) -> list[dict]:
    """
    Finds the most important dialogues related to a theme using semantic similarity.

    Args:
        dialogues: A list of dialogue dictionaries.
        theme: The search theme.
        n_results: Number of top results to return.

    Returns:
        The most relevant dialogues.
    """
    if not dialogues:
        return []

    texts = [d["text"] for d in dialogues]
    dialogue_embs = embed_texts(texts)
    dialogue_embs = np.mean(dialogue_embs, axis=1)

    theme_emb = embed_texts(theme)
    theme_emb = np.mean(theme_emb, axis=1)

    knn = NearestNeighbors(n_neighbors=min(n_results, len(dialogues)), metric="cosine")
    knn.fit(dialogue_embs)
    distances, indices = knn.kneighbors(theme_emb)

    top_dialogues = [dialogues[i] for i in indices[0]]
    return top_dialogues

def expand_timestamps(dialogues: list[dict], secs: int = 3, video_duration: float | None = None) -> list[tuple[float, float]]:
    """
    Expands timestamps for clips.

    Args:
        dialogues: List of dialogue segments.
        secs: Padding in seconds.
        video_duration: Total duration of the video.

    Returns:
        List of (start, end) tuples.
    """
    clips = []
    for d in dialogues:
        start = max(0.0, d["start"] - secs)
        end = min(video_duration, d["end"] + secs) if video_duration else d["end"] + secs
        clips.append((start, end))
    return clips


def stitch_clips(video_path: str, clips: list[tuple[float, float]], audio_path: str = "temp/temp.wav", output_path: str = "promo.mp4") -> str:
    """
    Stitches video clips together.

    Args:
        video_path: Path to original video.
        clips: List of (start, end) tuples.
        audio_path: Path to audio file.
        output_path: Path for the final video output.

    Returns:
        The output path.
    """
    video_inputs = []
    audio_inputs = []

    for s, e in clips:
        video_inputs.append(ffmpeg.input(video_path, ss=s, to=e))
        audio_inputs.append(ffmpeg.input(audio_path, ss=s, to=e))

    streams = []
    for v, a in zip(video_inputs, audio_inputs):
        streams.extend([v.video, a.audio])

    joined = ffmpeg.concat(*streams, v=1, a=1)

    (
        joined
        .output(output_path)
        .overwrite_output()
        .run()
    )

    return output_path

def cleanup_and_move(temp_folder: str = "temp", promo_file: str = "promo.mp4", promo_folder: str = "promos") -> None:
    """
    Cleans up temporary files and moves the promo file to the output directory.

    Args:
        temp_folder: The temporary directory to clean.
        promo_file: The generated promo video file.
        promo_folder: The destination directory for the promo video.
    """
    if os.path.exists(temp_folder):
        for filename in os.listdir(temp_folder):
            file_path = os.path.join(temp_folder, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")
    
    if not os.path.exists(promo_folder):
        os.makedirs(promo_folder)
    
    if os.path.exists(promo_file):
        dest_path = os.path.join(promo_folder, promo_file)
        shutil.move(promo_file, dest_path)
        print(f"Moved {promo_file} to {dest_path}")
    else:
        print(f"{promo_file} not found.")

def generate_promo(video_path: str, theme: str) -> str:
    """
    Generates a promo video based on a theme.

    Args:
        video_path: Path to input video.
        theme: Theme for the promo.

    Returns:
        Path to the generated promo.
    """
    audio_path = extract_audio(video_path)
    segments = transcribe_audio(audio_path)
    key_dialogues = get_imp_dialogues(segments, theme)
    
    probe = ffmpeg.probe(video_path)
    duration = float(probe['format']['duration'])
    
    clips = expand_timestamps(key_dialogues, secs=3, video_duration=duration)
    promo_path = stitch_clips(video_path, clips)

    cleanup_and_move(temp_folder="temp", promo_file=promo_path, promo_folder="promos")

    return os.path.join("promos", "promo.mp4")

def hello_world() -> dict[str, str]:
    """
    Returns a simple greeting.

    Returns:
        A dictionary with a greeting message.
    """
    return {"message": "hello world"}
