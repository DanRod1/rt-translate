#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import os
import traceback
from html import unescape
from pathlib import Path
from typing import Any

import ctranslate2
import ffmpeg
from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download
from huggingface_hub.utils import disable_progress_bars
from moviepy import VideoFileClip
from pydub import AudioSegment
from pydub.utils import make_chunks
from pytubefix import YouTube
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    SeamlessM4Tv2ForTextToText,
)

# ---------------------------------------------------------------------------
# Caches modèles
# ---------------------------------------------------------------------------

WHISPER_MODEL_CACHE: dict[tuple[str, str, str], WhisperModel] = {}
TOKENIZER_CACHE: dict[str, Any] = {}
SEQ2SEQ_MODEL_CACHE: dict[str, Any] = {}
SEAMLESS_CACHE: dict[str, Any] = {}

ISO1_TO_ISO3 = {
    "af": "afr", "am": "amh", "ar": "arb", "as": "asm", "az": "aze", "ba": "bak",
    "be": "bel", "bg": "bul", "bn": "ben", "bo": "bod", "br": "bre", "bs": "bos",
    "ca": "cat", "cs": "ces", "cy": "cym", "da": "dan", "de": "deu", "el": "ell",
    "en": "eng", "es": "spa", "et": "est", "eu": "eus", "fa": "pes", "fi": "fin",
    "fo": "fao", "fr": "fra", "fy": "fry", "ga": "gle", "gd": "gla", "gl": "glg",
    "gu": "guj", "ha": "hau", "he": "heb", "hi": "hin", "hr": "hrv", "ht": "hat",
    "hu": "hun", "hy": "hye", "id": "ind", "is": "isl", "it": "ita", "ja": "jpn",
    "jw": "jav", "ka": "kat", "kk": "kaz", "km": "khm", "kn": "kan", "ko": "kor",
    "ku": "kmr", "ky": "kir", "la": "lat", "lb": "ltz", "ln": "lin", "lo": "lao",
    "lt": "lit", "lv": "lav", "mg": "mlg", "mi": "mri", "mk": "mkd", "ml": "mal",
    "mn": "mon", "mr": "mar", "ms": "msa", "mt": "mlt", "my": "mya", "ne": "npi",
    "nl": "nld", "nn": "nno", "no": "nor", "oc": "oci", "pa": "pan", "pl": "pol",
    "ps": "pus", "pt": "por", "ro": "ron", "ru": "rus", "sa": "san", "sd": "snd",
    "si": "sin", "sk": "slk", "sl": "slv", "sn": "sna", "so": "som", "sq": "sqi",
    "sr": "srp", "su": "sun", "sv": "swe", "sw": "swh", "ta": "tam", "te": "tel",
    "tg": "tgk", "th": "tha", "tl": "tgl", "tr": "tur", "tt": "tat", "uk": "ukr",
    "ur": "urd", "uz": "uzn", "vi": "vie", "xh": "xho", "yi": "yid", "yo": "yor",
    "zh": "cmn", "yue": "yue", "zh-CN": "cmn", "zh-TW": "cmn",
}

DEFAULT_CHUNK_MS = 5000


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def strip_file_scheme(value: str) -> str:
    if value.startswith("file://"):
        return value[7:]
    if value.startswith("file:"):
        return value[5:]
    return value


def build_subtitled_output_path(video_path: str) -> str:
    path = Path(video_path)
    return str(path.with_name(f"{path.stem}-sub{path.suffix}"))


def get_device_and_compute() -> tuple[str, str]:
    device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    supported = ctranslate2.get_supported_compute_types(device)

    compute = "float32"
    for pref in ("float16", "int8_float16", "bfloat16", "int8", "int8_float32", "float32"):
        if pref in supported:
            compute = pref
            break

    return device, compute


def get_whisper_model(model_name: str = "Systran/faster-whisper-large-v3") -> WhisperModel:
    device, compute = get_device_and_compute()
    cache_key = (model_name, device, compute)

    if cache_key not in WHISPER_MODEL_CACHE:
        WHISPER_MODEL_CACHE[cache_key] = WhisperModel(
            model_name,
            device=device,
            compute_type=compute,
        )
    return WHISPER_MODEL_CACHE[cache_key]


def get_seq2seq_model(model_name: str):
    if model_name not in TOKENIZER_CACHE:
        TOKENIZER_CACHE[model_name] = AutoTokenizer.from_pretrained(model_name)
    if model_name not in SEQ2SEQ_MODEL_CACHE:
        SEQ2SEQ_MODEL_CACHE[model_name] = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return TOKENIZER_CACHE[model_name], SEQ2SEQ_MODEL_CACHE[model_name]


def get_seamless_model(model_name: str = "facebook/seamless-m4t-v2-large"):
    if model_name not in TOKENIZER_CACHE:
        TOKENIZER_CACHE[model_name] = AutoTokenizer.from_pretrained(model_name)
    if model_name not in SEAMLESS_CACHE:
        SEAMLESS_CACHE[model_name] = SeamlessM4Tv2ForTextToText.from_pretrained(model_name)

    return TOKENIZER_CACHE[model_name], SEAMLESS_CACHE[model_name]


def translate_text_seq2seq(text: str, model_name: str) -> str:
    tokenizer, model = get_seq2seq_model(model_name)
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    generated_tokens = model.generate(**inputs, max_new_tokens=256)
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]


def translate_text_seamless(text: str, input_language: str, output_language: str) -> str:
    if input_language not in ISO1_TO_ISO3:
        raise ValueError(f"Langue source non supportée par SeamlessM4T-v2: {input_language}")
    if output_language not in ISO1_TO_ISO3:
        raise ValueError(f"Langue cible non supportée par SeamlessM4T-v2: {output_language}")

    tokenizer, model = get_seamless_model("facebook/seamless-m4t-v2-large")
    inputs = tokenizer(
        text,
        return_tensors="pt",
        src_lang=ISO1_TO_ISO3[input_language],
    )
    generated_tokens = model.generate(
        **inputs,
        tgt_lang=ISO1_TO_ISO3[output_language],
        max_new_tokens=256,
    )
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]


def detect_opus_model(input_language: str, output_language: str, fallback_model: str) -> str:
    candidate_repo = f"Helsinki-NLP/opus-mt-{input_language}-{output_language}"
    local_candidate = f"/home/drodriguez/dev/Helsinki-NLP/opus-mt-{input_language}-{output_language}"

    if os.path.exists(local_candidate):
        return local_candidate

    if fallback_model == "auto":
        return candidate_repo

    return fallback_model


def format_srt_timestamp(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    hours = total_ms // 3_600_000
    total_ms %= 3_600_000
    minutes = total_ms // 60_000
    total_ms %= 60_000
    secs = total_ms // 1000
    ms = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


# ---------------------------------------------------------------------------
# SRT
# ---------------------------------------------------------------------------

def generate_srt(
    srt_file: Any,
    entries: list[dict[str, Any]],
    input_language: str,
    output_language: str,
    multiple: bool = False,
) -> None:
    for idx, entry in enumerate(entries, start=1):
        start_ts = format_srt_timestamp(entry["start"])
        end_ts = format_srt_timestamp(entry["end"])

        srt_file.write(f"{idx}\n")
        srt_file.write(f"{start_ts} --> {end_ts}\n")

        if multiple:
            srt_file.write(f"{input_language} : {entry['source_text']}\n")
            srt_file.write(f"{output_language} : {entry['translated_text']}\n")
        else:
            if output_language == input_language:
                srt_file.write(f"{entry['source_text']}\n")
            else:
                srt_file.write(f"{entry['translated_text']}\n")

        srt_file.write("\n")


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args(parser: argparse.ArgumentParser | None = None):
    if parser is None:
        parser = argparse.ArgumentParser(
            description="Lance la traduction et la génération de sous-titres d'une vidéo locale ou YouTube"
        )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="increase output verbosity",
    )
    parser.add_argument(
        "-i",
        "--inputLanguage",
        action="store",
        default="fr",
        help="langue de la vidéo téléchargée",
    )
    parser.add_argument(
        "-o",
        "--outputLanguage",
        action="store",
        default="ru",
        help="langue cible de traduction",
    )
    parser.add_argument(
        "-u",
        "--url",
        action="store",
        default="https://youtu.be/aN0-DgKsBxY",
        help="url de la vidéo téléchargée",
    )
    parser.add_argument(
        "-a",
        "--audioDir",
        action="store",
        default="Audio",
        help="répertoire Audio cache",
    )
    parser.add_argument(
        "-P",
        "--videoPath",
        action="store",
        default="Video",
        help="répertoire Video cache ou fichier vidéo local",
    )
    parser.add_argument(
        "-M",
        "--initModele",
        action="store_true",
        help="Activate local model cache",
    )
    parser.add_argument(
        "-n",
        "--nameModel",
        action="store",
        default="auto",
        help="Model name ou 'auto' pour OPUS-MT détecté automatiquement",
    )

    args = parser.parse_args()

    if args.url.startswith("file:"):
        args.videoPath = strip_file_scheme(args.url)

    return args


# ---------------------------------------------------------------------------
# Téléchargement / extraction
# ---------------------------------------------------------------------------

def download_video_yt(url: str, audio_dir: str) -> dict[str, str]:
    cwd = os.getcwd()
    stream: dict[str, str] = {}

    ensure_dir(audio_dir)
    ensure_dir("Video")

    if url.startswith("file:"):
        local_video_path = strip_file_scheme(url)
        audio_path = os.path.join(audio_dir, "audio.mp3")

        with VideoFileClip(local_video_path) as video:
            if video.audio is None:
                raise RuntimeError("La vidéo locale ne contient pas de piste audio.")
            video.audio.write_audiofile(filename=audio_path)

        stream = {"audio": audio_path, "video": local_video_path}
    else:
        open(file=os.path.join(audio_dir, "buffer-rt-translate"), mode="wb+").close()

        try:
            yt = YouTube(url)
        except Exception:
            traceback.print_exc()
            raise

        if yt.age_restricted:
            yt.bypass_age_gate()

        # vidéo
        video_stream = yt.streams.get_highest_resolution()
        if video_stream is None:
            raise RuntimeError("Aucun flux vidéo trouvé sur la vidéo YouTube.")
        print(video_stream.default_filename)
        video_stream.download(output_path="Video")
        stream["video"] = f"{cwd}/Video/{video_stream.default_filename}"

        # audio
        preferred_audio = None
        for item in yt.streams.filter(only_audio=True):
            if getattr(item, "audio_codec", None) == "opus":
                preferred_audio = item
                break

        if preferred_audio is None:
            preferred_audio = yt.streams.get_audio_only()

        if preferred_audio is None:
            raise RuntimeError("Aucun flux audio trouvé sur la vidéo YouTube.")

        print(preferred_audio.default_filename)
        preferred_audio.download(output_path=audio_dir)
        stream["audio"] = f"{cwd}/{audio_dir}/{preferred_audio.default_filename}"

    return stream


def split_audio_file(stream: dict[str, str], chunk_length_ms: int = DEFAULT_CHUNK_MS) -> list[str]:
    myaudio = AudioSegment.from_file(stream["audio"])
    audio_dir = str(Path(stream["audio"]).parent)
    ensure_dir(audio_dir)

    chunks = make_chunks(myaudio, chunk_length_ms)

    split_wav: list[str] = []
    for i, chunk in enumerate(chunks):
        chunk_name = f"{audio_dir}/chunk{i:06d}.mp3"
        chunk.export(chunk_name, format="mp3")
        split_wav.append(chunk_name)

    return split_wav


# ---------------------------------------------------------------------------
# Modèles
# ---------------------------------------------------------------------------

def init_huge_model() -> None:
    disable_progress_bars()

    downloads = [
        ("Helsinki-NLP/opus-mt-fr-ru", "/home/drodriguez/dev/Helsinki-NLP/opus-mt-fr-ru/"),
        ("Helsinki-NLP/opus-mt-en-fr", "/home/drodriguez/dev/Helsinki-NLP/opus-mt-en-fr/"),
        ("Helsinki-NLP/opus-mt-fr-en", "/home/drodriguez/dev/Helsinki-NLP/opus-mt-fr-en/"),
        ("Helsinki-NLP/opus-mt-ru-hy", "/home/drodriguez/dev/Helsinki-NLP/opus-mt-ru-hy/"),
        ("Helsinki-NLP/opus-mt-fr-es", "/home/drodriguez/dev/Helsinki-NLP/opus-mt-fr-es/"),
        ("Helsinki-NLP/opus-mt-es-fr", "/home/drodriguez/dev/Helsinki-NLP/opus-mt-es-fr/"),
        ("Helsinki-NLP/opus-mt-ar-fr", "/home/drodriguez/dev/Helsinki-NLP/opus-mt-ar-fr/"),
        ("Systran/faster-whisper-large-v3", "/home/drodriguez/dev/whisper-large-v3"),
        ("SXenova/opus-mt-it-fr", "/home/drodriguez/dev/Xenova/opus-mt-it-fr"),
        ("facebook/seamless-m4t-v2-large", "/home/drodriguez/dev/facebook/seamless-m4t-v2-large"),
    ]

    for repo_id, local_dir in downloads:
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=local_dir,
            local_files_only=False,
            cache_dir=f"{local_dir}/.cache/",
            resume_download=True,
        )


# ---------------------------------------------------------------------------
# Transcription / traduction
# ---------------------------------------------------------------------------

def transcribe(
    chunk: str = "",
    input_language: str = "fr",
    output_language: str = "ru",
    model_name: str = "auto",
    verbose: bool = False,
    time_offset_seconds: float = 0.0,
) -> list[dict[str, Any]]:
    exclude = {
        "Merci d'avoir regardé cette vidéo !",
        "Sous-titres réalisés para la communauté d'Amara.org",
        "Subtitles ST' 501",
    }

    model = get_whisper_model("Systran/faster-whisper-large-v3")

    segments, info = model.transcribe(
        chunk,
        chunk_length=5,
        language=input_language,
        temperature=0.0,
        patience=0.2,
        beam_size=5,
        condition_on_previous_text=False,
        vad_filter=True,
    )

    results: list[dict[str, Any]] = []

    for s in segments:
        input_text = (s.text or "").strip()
        if not input_text:
            continue

        if input_text in exclude:
            translated_text = "oups what the fluck robot do not understand"
        elif input_language == output_language:
            translated_text = input_text
        else:
            effective_model_name = model_name
            if effective_model_name == "auto":
                effective_model_name = detect_opus_model(
                    input_language=input_language,
                    output_language=output_language,
                    fallback_model="auto",
                )

            if effective_model_name == "facebook/seamless-m4t-v2-large":
                translation = translate_text_seamless(
                    text=input_text,
                    input_language=input_language,
                    output_language=output_language,
                )
            else:
                translation = translate_text_seq2seq(
                    text=input_text,
                    model_name=effective_model_name,
                )

            translated_text = unescape(translation)

        absolute_start = float(s.start) + time_offset_seconds
        absolute_end = float(s.end) + time_offset_seconds

        if verbose:
            print("#############################################")
            print(f"# Code Language en entrée : {input_language}")
            print("#############################################")
            print(input_text)
            print("########################################################################")
            print(f"# Traduction : {input_language} => {output_language}")
            print("########################################################################")
            print(translated_text)
            print("########################################################################")
            print(f"# Timing absolu : {absolute_start:.3f} -> {absolute_end:.3f}")
            print("########################################################################")
            print()

        results.append(
            {
                "chunk": chunk,
                "start": absolute_start,
                "end": absolute_end,
                "source_text": input_text,
                "translated_text": translated_text,
            }
        )

    return results


# ---------------------------------------------------------------------------
# Sous-titrage vidéo
# ---------------------------------------------------------------------------

def burn_subtitles(
    options,
    stream: dict[str, str],
    srt_path: str,
) -> str:
    width = 800
    height = 600
    style = "OutlineColour=&H100000000,BorderStyle=3,Outline=1,Shadow=0,Fontsize=13"

    if options.url.startswith("file:"):
        source_video = options.videoPath
        source_audio = os.path.join(options.audioDir, "audio.mp3")
    else:
        source_video = stream["video"]
        source_audio = stream["audio"]

    output = build_subtitled_output_path(source_video)

    audio_stream = ffmpeg.input(source_audio).audio
    video_stream = (
        ffmpeg
        .input(source_video)
        .video
        .filter("scale", width, height)
        .filter("pad", "iw", "ih+250", 0, 0, color="black")
        .filter("subtitles", filename=srt_path, force_style=style)
    )

    (
        ffmpeg
        .output(video_stream, audio_stream, output, vcodec="libx264")
        .run(overwrite_output=True)
    )

    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    options = parse_args()

    ensure_dir(options.audioDir)

    if options.initModele:
        init_huge_model()

    stream = download_video_yt(url=options.url, audio_dir=options.audioDir)
    chunks = split_audio_file(stream, chunk_length_ms=DEFAULT_CHUNK_MS)

    all_entries: list[dict[str, Any]] = []

    try:
        for chunk_index, chunk in enumerate(chunks):
            size_bytes = os.path.getsize(chunk)
            print(chunk)

            if size_bytes < 100:
                continue

            chunk_offset_seconds = (chunk_index * DEFAULT_CHUNK_MS) / 1000.0

            entries = transcribe(
                chunk=chunk,
                input_language=options.inputLanguage,
                output_language=options.outputLanguage,
                model_name=options.nameModel,
                verbose=options.verbose,
                time_offset_seconds=chunk_offset_seconds,
            )

            all_entries.extend(entries)

        # sécurité : tri global par temps
        all_entries.sort(key=lambda x: (x["start"], x["end"]))

        srt_path = f"{options.audioDir}/audio.srt"
        with open(srt_path, "w", encoding="utf-8") as file:
            generate_srt(
                srt_file=file,
                entries=all_entries,
                input_language=options.inputLanguage,
                output_language=options.outputLanguage,
                multiple=options.verbose,
            )

        output_video = burn_subtitles(
            options=options,
            stream=stream,
            srt_path=srt_path,
        )
        print(f"SRT généré : {srt_path}")
        print(f"Vidéo sous-titrée générée : {output_video}")
    finally:
        for f in glob.glob(f"{options.audioDir}/chunk*.mp3"):
            try:
                os.remove(f)
            except OSError:
                pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())