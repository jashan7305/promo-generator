import os
# import json
import numpy as np
import shutil

import ffmpeg
# from faster_whisper import WhisperModel
# from transformers import AutoTokenizer
# from onnxruntime import InferenceSession
# from sklearn.neighbors import NearestNeighbors

# print(os.path.getmtime("promo.mp4"))

# transcription_model = WhisperModel("base", device="cpu")
# embedding_model = "Xenova/e5-small-v2"
# tokenizer = AutoTokenizer.from_pretrained(embedding_model)

# model_path = f"{embedding_model}/onnx/model.onnx"
# if not os.path.exists(model_path):
#     from huggingface_hub import snapshot_download
#     snapshot_download(embedding_model, local_dir=embedding_model)
# onnx_sess = InferenceSession(model_path, providers=["CPUExecutionProvider"])

def embed_texts(texts):
    """
    Embeds texts into vectors using the pre-loaded embedding model.

    Args:
        texts (str or list[str]): The text or list of texts to embed.

    Returns:
        np.ndarray: A numpy array containing the embeddings.
    """
    if isinstance(texts, str):
        texts = [texts]
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="np")
    outputs = onnx_sess.run(None, dict(inputs))
    embeddings = outputs[0]
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings

def extract_audio(video_path, temp_folder="temp"):
    """
    Extracts audio from the video file into a WAV format at 16kHz.

    Args:
        video_path (str): The path to the video file.
        temp_folder (str): The folder to save the temporary audio file.

    Returns:
        str: The path to the extracted audio file.
    """
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
    
    audio_path = os.path.join(temp_folder, "temp.wav")
    
    ffmpeg.input(video_path).output(audio_path, ac=1, ar=16000).run(overwrite_output=True)
    return audio_path

def transcribe_audio(audio_path):
    """
    Transcribes audio from a WAV file using the pre-loaded Whisper model.

    Args:
        audio_path (str): The path to the audio file.

    Returns:
        list[dict]: A list of transcript segments containing start, end, and text.
    """
    segments, info = transcription_model.transcribe(audio_path)

    results = []
    for segment in segments:
        results.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip()
        })
    # print(results)
    return results

def get_imp_dialogues(dialogues, theme, n_results=3):
    """
    Identifies important dialogues related to a given theme using semantic search.

    Args:
        dialogues (list[dict]): A list of dialogue segments.
        theme (str): The theme to search for.
        n_results (int): The number of top dialogues to return.

    Returns:
        list[dict]: A list of the most relevant dialogues.
    """
    if not dialogues:
        return []

    texts = [d["text"] for d in dialogues]
    dialogue_embs = embed_texts(texts)
    dialogue_embs = np.mean(dialogue_embs, axis=1)

    theme_emb = embed_texts(theme)
    theme_emb = np.mean(theme_emb, axis=1)
    # print(dialogue_embs)

    knn = NearestNeighbors(n_neighbors=min(n_results, len(dialogues)), metric="cosine")
    knn.fit(dialogue_embs)
    distances, indices = knn.kneighbors(theme_emb)

    top_dialogues = [dialogues[i] for i in indices[0]]
    # print(top_dialogues)
    return top_dialogues

def expand_timestamps(dialogues, secs=3, video_duration=None):
    """
    Expands the duration of dialogue clips.

    Args:
        dialogues (list[dict]): A list of dialogue segments.
        secs (int): The amount of time in seconds to expand by.
        video_duration (float, optional): Total video duration to cap expansion.

    Returns:
        list[tuple]: A list of tuples (start, end) for the clips.
    """
    clips = []
    for d in dialogues:
        start = max(0, d["start"] - secs)
        end = min(video_duration, d["end"] + secs) if video_duration else d["end"] + secs
        clips.append((start, end))
    return clips


def stitch_clips(video_path, clips, audio_path="temp/temp.wav", output_path="promo.mp4"):
    """
    Stitches together segments of video based on provided timestamps.

    Args:
        video_path (str): The path to the source video.
        clips (list[tuple]): A list of start/end tuples.
        audio_path (str): Path to the audio file.
        output_path (str): Path for the output video.

    Returns:
        str: The path to the stitched promo video.
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

def cleanup_and_move(temp_folder="temp", promo_file="promo.mp4", promo_folder="promos"):
    """
    Deletes all files in temp_folder and moves promo_file into promo_folder.
    Creates promo_folder if it doesn't exist.

    Args:
        temp_folder (str): Folder containing temporary files.
        promo_file (str): The promo video file to move.
        promo_folder (str): The destination directory.
    """
    # Delete all files in temp folder
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
    
    # Create promos folder if not exists
    if not os.path.exists(promo_folder):
        os.makedirs(promo_folder)
    
    # Move promo file
    if os.path.exists(promo_file):
        dest_path = os.path.join(promo_folder, promo_file)
        shutil.move(promo_file, dest_path)
        print(f"Moved {promo_file} to {dest_path}")
    else:
        print(f"{promo_file} not found.")

def generate_promo(video_path, theme):
    """
    Main workflow to generate a promo video based on a theme.

    Args:
        video_path (str): The input video path.
        theme (str): The theme for the promo.

    Returns:
        str: The final path to the generated promo video.
    """
    audio_path = extract_audio(video_path)
    segments = transcribe_audio(audio_path)
    key_dialogues = get_imp_dialogues(segments, theme)
    
    probe = ffmpeg.probe(video_path)
    duration = float(probe['format']['duration'])
    
    clips = expand_timestamps(key_dialogues, secs=3, video_duration=duration)
    promo_path = stitch_clips(video_path, clips)
    # print(os.path.getmtime("promo.mp4"))

    cleanup_and_move(temp_folder="temp", promo_file=promo_path, promo_folder="promos")

    return os.path.join("promos", "promo.mp4")

def hello_world():
    """
    Returns a greeting message.

    Returns:
        dict: A dictionary containing the greeting.
    """
    return {"message": "hello world"}
