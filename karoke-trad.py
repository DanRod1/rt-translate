from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf


SILENCE_AMPLITUDE_THRESHOLD = 0.02

PHONEMIZER_AVAILABLE = True
try:
    from phonemizer import phonemize
except Exception:
    PHONEMIZER_AVAILABLE = False
    phonemize = None


TRANSFORMERS_AVAILABLE = True
try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
except Exception:
    TRANSFORMERS_AVAILABLE = False
    AutoModelForSeq2SeqLM = None
    AutoTokenizer = None


ONOMATOPOEIA_PATTERNS = [
    r"^(o+h+|oh+)$",
    r"^(a+h+|ah+)$",
    r"^(h+a+)+$",
    r"^(ou+i+|oui+|ouais+|ouai+)$",
    r"^(ye+a+h+|yeah+|ye+h+)$",
    r"^(la+[-\s]?)+la*$",
    r"^(na+[-\s]?)+na*$",
    r"^(mm+|hmm+|hum+|mmm+)$",
    r"^(uh+|um+|erm+|eh+)$",
    r"^(hey+|yo+)$",
    r"^(wo+u*h+|woo+h+)$",
    r"^(ba+|pa+|da+|ta+)$",
]


@dataclass
class MulanFragment:
    id: int
    text: str
    original_text: str
    translated_text: str
    translation_applied: bool
    tokens: List[str]
    syllable_count: int
    phonemes: str
    phoneme_count: int
    weight: float
    source_line: int
    is_onomatopoeia: bool
    explicit_duration: Optional[float]
    sync_offset: float


@dataclass
class VocalRegion:
    id: int
    start: float
    end: float
    duration: float
    start_boundary_reason: str
    end_boundary_reason: str


@dataclass
class BlankBoundary:
    id: int
    start: float
    end: float
    duration: float
    boundary_time: float


@dataclass
class TimedSubtitle:
    id: int
    start: float
    end: float
    text: str
    fragment_id: int
    source_line: int
    original_text: str
    translated_text: str
    translation_applied: bool
    phonemes: str
    syllable_count: int
    phoneme_count: int
    is_onomatopoeia: bool
    explicit_duration: Optional[float]
    sync_offset: float


@dataclass
class LocalTextTranslator:
    model_path: str
    input_language: str
    output_language: str
    max_length: int
    tokenizer: Any
    model: Any

    def translate(self, text: str) -> str:
        if not text.strip():
            return text

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
        )

        generated = self.model.generate(
            **encoded,
            max_length=self.max_length,
        )

        translated = self.tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
        )[0]

        return normalize_text(str(translated))


def log(message: str) -> None:
    print(message, flush=True)


def warn(message: str) -> None:
    print(f"[WARN] {message}", flush=True)


def fail(message: str, exit_code: int = 1) -> None:
    print(f"[ERROR] {message}", flush=True)
    sys.exit(exit_code)


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def normalize_text(text: str) -> str:
    text = text.replace("…", "...")
    text = text.replace("’", "'")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,;:!?])", r"\1", text)
    return text.strip()


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.size == 0:
        return audio.astype(np.float32)
    if getattr(audio, "ndim", 1) == 2:
        return np.mean(audio, axis=1).astype(np.float32)
    return audio.astype(np.float32)


def is_parenthesized_text(text: str) -> bool:
    text = text.strip()
    return text.startswith("(") and text.endswith(")")


def normalize_onomatopoeia_candidate(text: str) -> str:
    text = text.strip()
    if is_parenthesized_text(text):
        text = text[1:-1]

    text = text.lower()
    text = text.replace("’", "'")
    text = re.sub(r"[^\wÀ-ÿ\s'-]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_duration_metadata_value(raw_value: str) -> Dict[str, Any]:
    value = raw_value.strip().replace(" ", "").replace(",", ".")

    if not re.fullmatch(r"[+-]?[0-9]+(?:\.[0-9]+)?", value):
        return {}

    numeric_value = float(value)

    if value.startswith("+") or value.startswith("-"):
        return {"sync_offset": numeric_value}

    return {"explicit_duration": numeric_value}


def split_parenthesized_inline_metadata(text: str) -> Tuple[str, Dict[str, Any]]:
    raw = text.strip()
    metadata: Dict[str, Any] = {}

    if not is_parenthesized_text(raw):
        return raw, metadata

    inner = raw[1:-1].strip()

    if "--" not in inner:
        return "", metadata

    useful_text, comment = inner.split("--", 1)
    useful_text = useful_text.strip()
    comment = comment.strip()

    duration_match = re.search(
        r"\bdure\s*:\s*([+-]?\s*[0-9]+(?:[.,][0-9]+)?)",
        comment,
        flags=re.IGNORECASE,
    )

    if duration_match:
        metadata.update(parse_duration_metadata_value(duration_match.group(1)))

    return useful_text, metadata


def strip_mulan_markup(line: str) -> str:
    line = re.sub(r"<[^>]+>", " ", line)
    line = re.sub(r"\[[^\]]+\]", " ", line)
    line = re.sub(r"\{[^}]+\}", " ", line)
    return normalize_text(line)


def parse_mulan_payload_from_line(line: str) -> Optional[str]:
    raw = line.strip()
    if not raw:
        return None
    if raw.startswith("#") or raw.startswith("//"):
        return None

    if "=" in raw:
        key, value = raw.split("=", 1)
        key = key.strip().lower()
        if key in {
            "text",
            "lyric",
            "lyrics",
            "line",
            "sentence",
            "phrase",
            "fragment",
            "content",
        }:
            return value.strip()

    if "\t" in raw:
        parts = [p.strip() for p in raw.split("\t") if p.strip()]
        if parts:
            return parts[-1]

    if "|" in raw:
        parts = [p.strip() for p in raw.split("|") if p.strip()]
        if parts:
            return parts[-1]

    return raw


def tokenize_text(text: str) -> List[str]:
    return re.findall(r"\b[\wÀ-ÿ'-]+\b", text, flags=re.UNICODE)


def estimate_syllables_in_token(token: str) -> int:
    groups = re.findall(
        r"[aeiouyàâäéèêëîïôöùûüÿœæ]+",
        token.lower(),
        flags=re.UNICODE,
    )
    return max(1, len(groups))


def estimate_syllables(tokens: List[str]) -> int:
    if not tokens:
        return 1
    return sum(estimate_syllables_in_token(token) for token in tokens)


def fallback_phonemize_text(text: str) -> str:
    text = normalize_text(text).lower()
    text = re.sub(r"[^\w\sÀ-ÿ'-]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join(list(text.replace(" ", "|")))


def normalize_text_for_phonetic_weight(text: str) -> str:
    normalized = normalize_text(text).lower()
    tokens = tokenize_text(normalized)
    normalized_tokens: List[str] = []

    for token in tokens:
        t = token.strip().lower()

        # Pluriels nominaux/adjectivaux généralement muets.
        if len(t) > 3 and t.endswith("s") and not t.endswith(("as", "os", "us", "is")):
            t = t[:-1]

        # Féminin/pluriel adjectival souvent muet : grandes -> grand.
        if len(t) > 4 and t.endswith("es"):
            t = t[:-2]

        # Terminaisons verbales fréquentes non proportionnelles au chant : parlent -> parle.
        if len(t) > 5 and t.endswith("ent"):
            t = t[:-3]

        # Imparfait / conditionnel pluriel : parlaient -> parlai.
        if len(t) > 7 and t.endswith("aient"):
            t = t[:-5] + "ai"

        # Futur pluriel : parleront -> parler.
        if len(t) > 7 and t.endswith("eront"):
            t = t[:-5] + "er"

        # Conditionnel pluriel : parleraient -> parlerai.
        if len(t) > 8 and t.endswith("eraient"):
            t = t[:-7] + "erai"

        if t:
            normalized_tokens.append(t)

    return " ".join(normalized_tokens)


def count_repeated_vocalic_stretch(text: str) -> float:
    normalized = normalize_text(text).lower()
    stretches = re.findall(
        r"[aeiouyàâäéèêëîïôöùûüÿœæ]{2,}",
        normalized,
        flags=re.UNICODE,
    )
    return float(sum(max(0, len(stretch) - 1) for stretch in stretches))


def count_repeated_consonant_stretch(text: str) -> float:
    normalized = normalize_text(text).lower()
    stretches = re.findall(
        r"[bcdfghjklmnpqrstvwxzç]{2,}",
        normalized,
        flags=re.UNICODE,
    )
    return float(sum(max(0, len(stretch) - 1) * 0.35 for stretch in stretches))


def phonemize_text(
    text: str,
    language: str,
    backend: str,
    is_onomatopoeia: bool,
) -> str:
    if is_onomatopoeia:
        return normalize_onomatopoeia_candidate(text)

    if not PHONEMIZER_AVAILABLE:
        return fallback_phonemize_text(text)

    try:
        result = phonemize(
            text,
            language=language,
            backend=backend,
            strip=True,
            preserve_punctuation=False,
            with_stress=False,
            njobs=1,
        )
        return normalize_text(str(result))
    except Exception as exc:
        warn(f"Phonemizer indisponible pour une ligne, fallback simple: {exc}")
        return fallback_phonemize_text(text)


def count_phoneme_units(phonemes: str) -> int:
    tokens = re.findall(r"\S+", phonemes)
    if tokens:
        return max(1, len(tokens))
    return max(1, len(phonemes.replace(" ", "")))


def estimate_phonetic_duration_weight(
    text: str,
    tokens: List[str],
    syllable_count: int,
    phoneme_count: int,
    is_onomatopoeia: bool,
) -> float:
    # Ici, text est déjà censé être phonetic_text.
    normalized = normalize_text_for_phonetic_weight(text)
    phonetic_tokens = tokenize_text(normalized)

    token_count = max(1, len(phonetic_tokens))
    base_syllables = max(1.0, float(estimate_syllables(phonetic_tokens)))
    base_phonemes = max(1.0, float(count_phoneme_units(fallback_phonemize_text(normalized))))

    vocalic_stretch = count_repeated_vocalic_stretch(normalized)
    consonant_stretch = count_repeated_consonant_stretch(normalized)

    punctuation_hold = 0.0
    punctuation_hold += text.count("...") * 1.2
    punctuation_hold += text.count("!") * 0.35
    punctuation_hold += text.count("?") * 0.25

    short_word_penalty = 0.0
    if token_count <= 3 and base_syllables <= 4:
        short_word_penalty = -0.25

    phonetic_weight = (
        base_syllables * 1.00
        + base_phonemes * 0.18
        + vocalic_stretch * 0.75
        + consonant_stretch * 0.35
        + punctuation_hold
        + short_word_penalty
    )

    if is_onomatopoeia:
        phonetic_weight *= 1.35

    return max(1.0, phonetic_weight)


def compute_fragment_weight(
    text: str,
    tokens: List[str],
    syllable_count: int,
    phoneme_count: int,
    timing_mode: str,
    is_onomatopoeia: bool,
) -> float:
    if timing_mode == "syllables":
        weight = float(max(1, syllable_count))
    elif timing_mode == "phonemes":
        weight = estimate_phonetic_duration_weight(
            text=text,
            tokens=tokens,
            syllable_count=syllable_count,
            phoneme_count=phoneme_count,
            is_onomatopoeia=is_onomatopoeia,
        )
    elif timing_mode == "hybrid":
        phonetic_weight = estimate_phonetic_duration_weight(
            text=text,
            tokens=tokens,
            syllable_count=syllable_count,
            phoneme_count=phoneme_count,
            is_onomatopoeia=is_onomatopoeia,
        )
        syllable_weight = float(max(1, syllable_count))
        weight = phonetic_weight * 0.75 + syllable_weight * 0.25
    else:
        raise ValueError(f"timing_mode non supporté: {timing_mode}")

    return max(1.0, weight)


def should_translate_text(
    text: str,
    input_language: str,
    output_language: str,
    translation_model: str,
) -> bool:
    if not text.strip():
        return False

    if not output_language.strip():
        return False

    if not translation_model.strip():
        return False

    input_lang = input_language.strip().lower()
    output_lang = output_language.strip().lower()

    if input_lang and output_lang and input_lang == output_lang:
        return False

    return True


def create_local_text_translator(
    input_language: str,
    output_language: str,
    translation_model: str,
    translation_max_length: int,
    local_files_only: bool,
) -> Optional[LocalTextTranslator]:
    if not output_language.strip():
        return None

    if not translation_model.strip():
        return None

    if not TRANSFORMERS_AVAILABLE:
        warn("transformers indisponible, traduction locale désactivée.")
        return None

    if AutoTokenizer is None or AutoModelForSeq2SeqLM is None:
        warn("transformers importé partiellement, traduction locale désactivée.")
        return None

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            translation_model,
            local_files_only=local_files_only,
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            translation_model,
            local_files_only=local_files_only,
        )
        model.eval()

        return LocalTextTranslator(
            model_path=translation_model,
            input_language=input_language,
            output_language=output_language,
            max_length=translation_max_length,
            tokenizer=tokenizer,
            model=model,
        )
    except Exception as exc:
        warn(f"Impossible de charger le modèle de traduction locale: {exc}")
        return None


def translate_text_value(
    text: str,
    translator: Optional[LocalTextTranslator],
    input_language: str,
    output_language: str,
    translation_model: str,
) -> Tuple[str, bool]:
    if not should_translate_text(
        text=text,
        input_language=input_language,
        output_language=output_language,
        translation_model=translation_model,
    ):
        return text, False

    if translator is None:
        warn("Traduction demandée mais traducteur local indisponible, texte source conservé.")
        return text, False

    try:
        translated = translator.translate(text)
        if not translated:
            return text, False
        return translated, True
    except Exception as exc:
        warn(f"Traduction locale échouée pour une ligne, texte source conservé: {exc}")
        return text, False


def detect_onomatopoeia_from_payload(payload: str) -> bool:
    if not is_parenthesized_text(payload):
        return False

    payload_inner = payload.strip()[1:-1].strip()

    if "--" in payload_inner:
        payload_inner_before_comment, _payload_comment = payload_inner.split("--", 1)
        onomatopoeia_candidate = payload_inner_before_comment.strip()
    else:
        onomatopoeia_candidate = payload_inner

    if not onomatopoeia_candidate:
        return False

    normalized_candidate = normalize_onomatopoeia_candidate(onomatopoeia_candidate)
    candidate_tokens = normalized_candidate.split()

    if not candidate_tokens:
        return False

    for candidate_token in candidate_tokens:
        candidate_token = candidate_token.strip("-'")
        if not candidate_token:
            return False

        candidate_matched = any(
            re.fullmatch(pattern, candidate_token, flags=re.IGNORECASE)
            for pattern in ONOMATOPOEIA_PATTERNS
        )

        if not candidate_matched:
            return False

    return True


def parse_mulan_text(
    path: str,
    input_language: str,
    output_language: str,
    translation_model: str,
    translator: Optional[LocalTextTranslator],
    phoneme_language: str,
    phoneme_backend: str,
    timing_mode: str,
) -> List[MulanFragment]:
    raw = read_text_file(path)
    fragments: List[MulanFragment] = []

    for source_line, raw_line in enumerate(raw.splitlines(), start=1):
        payload = parse_mulan_payload_from_line(raw_line)
        if payload is None:
            continue

        payload_text, metadata = split_parenthesized_inline_metadata(payload)
        is_onomatopoeia = detect_onomatopoeia_from_payload(payload)

        # Texte source propre, jamais normalisé phonétiquement pour l'affichage.
        original_text = strip_mulan_markup(payload_text)

        if not original_text and not metadata:
            continue

        if original_text:
            translated_text, translation_applied = translate_text_value(
                text=original_text,
                translator=translator,
                input_language=input_language,
                output_language=output_language,
                translation_model=translation_model,
            )
            display_text = translated_text
        else:
            translated_text = ""
            translation_applied = False
            display_text = ""

        timing_text = original_text

        if not timing_text:
            fragments.append(
                MulanFragment(
                    id=len(fragments),
                    text="",
                    original_text="",
                    translated_text="",
                    translation_applied=False,
                    tokens=[],
                    syllable_count=1,
                    phonemes="",
                    phoneme_count=1,
                    weight=1.0,
                    source_line=source_line,
                    is_onomatopoeia=False,
                    explicit_duration=metadata.get("explicit_duration"),
                    sync_offset=float(metadata.get("sync_offset", 0.0)),
                )
            )
            continue

        # Texte phonétique : utilisé uniquement pour le calcul de durée.
        phonetic_text = normalize_text_for_phonetic_weight(timing_text)
        tokens = tokenize_text(phonetic_text)

        if not tokens:
            continue

        syllable_count = estimate_syllables(tokens)

        phonemes = phonemize_text(
            text=phonetic_text,
            language=phoneme_language,
            backend=phoneme_backend,
            is_onomatopoeia=is_onomatopoeia,
        )

        phoneme_count = count_phoneme_units(phonemes)

        weight = compute_fragment_weight(
            text=phonetic_text,
            tokens=tokens,
            syllable_count=syllable_count,
            phoneme_count=phoneme_count,
            timing_mode=timing_mode,
            is_onomatopoeia=is_onomatopoeia,
        )

        fragments.append(
            MulanFragment(
                id=len(fragments),
                text=display_text,
                original_text=original_text,
                translated_text=translated_text,
                translation_applied=translation_applied,
                tokens=tokens,
                syllable_count=syllable_count,
                phonemes=phonemes,
                phoneme_count=phoneme_count,
                weight=weight,
                source_line=source_line,
                is_onomatopoeia=is_onomatopoeia,
                explicit_duration=metadata.get("explicit_duration"),
                sync_offset=float(metadata.get("sync_offset", 0.0)),
            )
        )

    return fragments


def convert_media_to_wav(
    input_path: str,
    output_wav_path: str,
    sample_rate: int = 44100,
    channels: int = 2,
) -> str:
    ensure_dir(os.path.dirname(output_wav_path))

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        output_wav_path,
    ]

    log(f"[INFO] Conversion média -> wav: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            "Echec ffmpeg lors de la conversion en wav.\n"
            f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )

    if not os.path.isfile(output_wav_path):
        raise FileNotFoundError(f"WAV de sortie introuvable: {output_wav_path}")

    return output_wav_path


def extract_vocal_and_instrumental_with_demucs(
    input_media_path: str,
    output_dir: str,
    demucs_model: str = "htdemucs",
) -> Tuple[str, str]:
    ensure_dir(output_dir)

    prepared_wav = os.path.join(output_dir, "00_input_prepared.wav")
    convert_media_to_wav(
        input_path=input_media_path,
        output_wav_path=prepared_wav,
        sample_rate=44100,
        channels=2,
    )

    demucs_out = os.path.join(output_dir, "demucs_stems")

    cmd = [
        sys.executable,
        "-m",
        "demucs.separate",
        "-n",
        demucs_model,
        "--two-stems=vocals",
        "-o",
        demucs_out,
        prepared_wav,
    ]

    log(f"[INFO] Demucs: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            "Echec Demucs.\n"
            f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )

    stem_dir = os.path.join(demucs_out, demucs_model, Path(prepared_wav).stem)

    vocal_path = os.path.join(stem_dir, "vocals.wav")
    instrumental_path = os.path.join(stem_dir, "no_vocals.wav")

    if not os.path.isfile(vocal_path):
        raise FileNotFoundError(f"Stem vocal Demucs introuvable: {vocal_path}")

    if not os.path.isfile(instrumental_path):
        raise FileNotFoundError(f"Stem instrumental Demucs introuvable: {instrumental_path}")

    return vocal_path, instrumental_path


def make_vocal_region(
    region_id: int,
    start: float,
    end: float,
    start_boundary_reason: str,
    end_boundary_reason: str,
) -> VocalRegion:
    return VocalRegion(
        id=region_id,
        start=start,
        end=end,
        duration=max(0.0, end - start),
        start_boundary_reason=start_boundary_reason,
        end_boundary_reason=end_boundary_reason,
    )


def make_vocal_activity_mask_from_amplitude(audio: np.ndarray) -> np.ndarray:
    audio = to_mono(audio).astype(np.float32)
    return np.abs(audio) > SILENCE_AMPLITUDE_THRESHOLD


def smooth_vocal_activity_mask(
    vocal_activity_mask: np.ndarray,
    sr: int,
    min_voice_duration: float,
) -> np.ndarray:
    if vocal_activity_mask.size == 0:
        return vocal_activity_mask

    min_voice_samples = max(1, int(min_voice_duration * sr))
    mask = vocal_activity_mask.copy()

    idx = 0
    n = len(mask)

    while idx < n:
        if not mask[idx]:
            idx += 1
            continue

        start = idx
        while idx < n and mask[idx]:
            idx += 1
        end = idx

        if end - start < min_voice_samples:
            mask[start:end] = False

    return mask


def detect_blank_boundaries(
    vocal_activity_mask: np.ndarray,
    sr: int,
    min_blank_duration: float,
    blank_boundary_position: str,
) -> Tuple[List[BlankBoundary], Dict[str, Any]]:
    if blank_boundary_position not in {"start", "middle", "end"}:
        raise ValueError(f"blank_boundary_position non supporté: {blank_boundary_position}")

    if vocal_activity_mask.size == 0:
        return [], {
            "blank_detection_mode": "duration_only",
            "min_blank_duration": min_blank_duration,
            "blank_count": 0,
        }

    min_blank_samples = max(1, int(min_blank_duration * sr))
    blanks: List[BlankBoundary] = []

    idx = 0
    n = len(vocal_activity_mask)

    while idx < n:
        if vocal_activity_mask[idx]:
            idx += 1
            continue

        blank_start = idx

        while idx < n and not vocal_activity_mask[idx]:
            idx += 1

        blank_end = idx
        blank_len = blank_end - blank_start

        if blank_len >= min_blank_samples:
            start_time = blank_start / sr
            end_time = blank_end / sr
            duration = end_time - start_time

            if blank_boundary_position == "start":
                boundary_time = start_time
            elif blank_boundary_position == "end":
                boundary_time = end_time
            else:
                boundary_time = (start_time + end_time) / 2.0

            blanks.append(
                BlankBoundary(
                    id=len(blanks),
                    start=start_time,
                    end=end_time,
                    duration=duration,
                    boundary_time=boundary_time,
                )
            )

    return blanks, {
        "blank_detection_mode": "duration_only",
        "silence_amplitude_threshold": SILENCE_AMPLITUDE_THRESHOLD,
        "min_blank_duration": min_blank_duration,
        "min_blank_samples": min_blank_samples,
        "blank_boundary_position": blank_boundary_position,
        "blank_count": len(blanks),
        "mask_semantics": "True=voice/activity, False=blank",
        "boundary_semantics": "boundary_time selected from start/middle/end of each blank",
    }


def build_regions_from_blank_boundaries(
    blanks: List[BlankBoundary],
    duration_seconds: float,
    pad_start: float,
    pad_end: float,
) -> List[VocalRegion]:
    boundaries = [0.0] + [blank.boundary_time for blank in blanks] + [duration_seconds]
    boundaries = sorted(set(boundaries))

    regions: List[VocalRegion] = []

    for idx in range(len(boundaries) - 1):
        raw_start = boundaries[idx]
        raw_end = boundaries[idx + 1]

        if raw_end <= raw_start:
            continue

        start = raw_start
        end = raw_end

        if idx > 0:
            start = max(0.0, start - pad_start)

        if idx < len(boundaries) - 2:
            end = min(duration_seconds, end + pad_end)

        if regions:
            start = max(start, regions[-1].end)

        if end <= start:
            continue

        regions.append(
            make_vocal_region(
                region_id=len(regions),
                start=start,
                end=end,
                start_boundary_reason="start" if idx == 0 else "blank_duration",
                end_boundary_reason="end" if idx == len(boundaries) - 2 else "blank_duration",
            )
        )

    for i, region in enumerate(regions):
        region.id = i

    return regions


def detect_vocal_regions_from_blanks(
    vocal_path: str,
    min_blank_duration: float,
    min_voice_duration: float,
    blank_boundary_position: str,
    pad_start: float,
    pad_end: float,
) -> Tuple[List[VocalRegion], List[BlankBoundary], Dict[str, Any]]:
    audio, sr = sf.read(vocal_path, always_2d=False)
    audio = to_mono(audio).astype(np.float32)

    if audio.size == 0:
        raise RuntimeError("Stem vocal vide, impossible de calculer une timeline.")

    duration_seconds = len(audio) / sr

    vocal_activity_mask = make_vocal_activity_mask_from_amplitude(audio)

    vocal_activity_mask = smooth_vocal_activity_mask(
        vocal_activity_mask=vocal_activity_mask,
        sr=sr,
        min_voice_duration=min_voice_duration,
    )

    blank_boundaries, detection_debug = detect_blank_boundaries(
        vocal_activity_mask=vocal_activity_mask,
        sr=sr,
        min_blank_duration=min_blank_duration,
        blank_boundary_position=blank_boundary_position,
    )

    if not blank_boundaries:
        warn(
            "Aucun blanc exploitable détecté dans la stem vocal Demucs. "
            "Fallback: une seule région globale."
        )
        regions = [
            make_vocal_region(
                region_id=0,
                start=0.0,
                end=duration_seconds,
                start_boundary_reason="start",
                end_boundary_reason="end_no_blank_detected",
            )
        ]
    else:
        regions = build_regions_from_blank_boundaries(
            blanks=blank_boundaries,
            duration_seconds=duration_seconds,
            pad_start=pad_start,
            pad_end=pad_end,
        )

    if not regions:
        raise RuntimeError("Aucune région temporelle valide n'a été construite.")

    debug = {
        "sample_rate": sr,
        "duration": duration_seconds,
        "timeline_mode": "blank_duration_only",
        "silence_amplitude_threshold": SILENCE_AMPLITUDE_THRESHOLD,
        "min_voice_duration": min_voice_duration,
        "blank_detection": detection_debug,
        "region_count": len(regions),
    }

    return regions, blank_boundaries, debug


def make_subtitle(
    idx: int,
    fragment: MulanFragment,
    start: float,
    end: float,
) -> TimedSubtitle:
    # Le sync_offset n'est PAS appliqué ici.
    # Il est appliqué directement sur current_time dans assign_mulan_lines_to_vocal_regions,
    # afin d'être cumulatif et valable pour tous les fragments suivants.
    safe_start = max(0.0, start)
    safe_end = max(safe_start + 0.01, end)

    return TimedSubtitle(
        id=idx,
        start=safe_start,
        end=safe_end,
        text=fragment.text,
        original_text=fragment.original_text,
        translated_text=fragment.translated_text,
        translation_applied=fragment.translation_applied,
        phonemes=fragment.phonemes,
        syllable_count=fragment.syllable_count,
        phoneme_count=fragment.phoneme_count,
        is_onomatopoeia=fragment.is_onomatopoeia,
        explicit_duration=fragment.explicit_duration,
        sync_offset=fragment.sync_offset,
        fragment_id=fragment.id,
        source_line=fragment.source_line,
    )


def get_fixed_fragment_duration(
    fragment: MulanFragment,
    onomatopoeia_fixed_duration: float,
    fixed_duration_compression_factor: float,
) -> Optional[float]:
    if fragment.explicit_duration is not None:
        return max(0.01, fragment.explicit_duration) * fixed_duration_compression_factor

    if fragment.is_onomatopoeia:
        return max(0.01, onomatopoeia_fixed_duration) * fixed_duration_compression_factor

    return None


def find_region_index_for_time(
    regions: List[VocalRegion],
    current_index: int,
    time_value: float,
) -> int:
    # Autorise les corrections négatives : region_index peut reculer.
    if not regions:
        return 0

    time_value = max(0.0, time_value)

    if 0 <= current_index < len(regions):
        current = regions[current_index]
        if current.start <= time_value < current.end:
            return current_index

    for idx, region in enumerate(regions):
        if region.start <= time_value < region.end:
            return idx

    if time_value < regions[0].start:
        return 0

    return len(regions) - 1


def align_time_to_region(
    regions: List[VocalRegion],
    region_index: int,
    current_time: float,
) -> Tuple[int, float]:
    if not regions:
        return 0, max(0.0, current_time)

    current_time = max(0.0, current_time)
    region_index = find_region_index_for_time(
        regions=regions,
        current_index=region_index,
        time_value=current_time,
    )

    region = regions[region_index]

    if current_time < region.start:
        current_time = region.start

    # Si la correction tombe après la dernière région, on garde la dernière borne.
    if current_time > region.end and region_index == len(regions) - 1:
        current_time = region.end

    return region_index, current_time


def assign_mulan_lines_to_vocal_regions(
    fragments: List[MulanFragment],
    vocal_regions: List[VocalRegion],
    onomatopoeia_fixed_duration: float,
) -> Tuple[List[TimedSubtitle], Dict[str, Any]]:
    if not fragments:
        raise RuntimeError("Aucun fragment Mulan exploitable : impossible de générer un SRT.")

    if not vocal_regions:
        raise RuntimeError("Aucune région vocale exploitable : impossible de générer un SRT.")

    if onomatopoeia_fixed_duration <= 0.0:
        raise ValueError("onomatopoeia_fixed_duration doit être strictement positif.")

    valid_regions = [
        region
        for region in vocal_regions
        if region.end > region.start and region.duration > 0.0
    ]

    if not valid_regions:
        raise RuntimeError("Aucune région vocale avec durée positive : impossible de générer un SRT.")

    total_region_duration = sum(region.end - region.start for region in valid_regions)

    if total_region_duration <= 0.0:
        raise RuntimeError("Durée totale des régions vocales nulle.")

    explicit_duration_fragments = [
        fragment for fragment in fragments
        if fragment.original_text and fragment.explicit_duration is not None
    ]

    fixed_onomatopoeia_fragments = [
        fragment
        for fragment in fragments
        if fragment.original_text and fragment.explicit_duration is None and fragment.is_onomatopoeia
    ]

    normal_fragments = [
        fragment
        for fragment in fragments
        if fragment.original_text and fragment.explicit_duration is None and not fragment.is_onomatopoeia
    ]

    reserved_explicit_duration = sum(
        max(0.01, fragment.explicit_duration or 0.0)
        for fragment in explicit_duration_fragments
    )

    reserved_onomatopoeia_duration = (
        len(fixed_onomatopoeia_fragments) * onomatopoeia_fixed_duration
    )

    reserved_fixed_duration = reserved_explicit_duration + reserved_onomatopoeia_duration

    if reserved_fixed_duration >= total_region_duration:
        warn(
            "La durée réservée aux fragments fixes dépasse ou égale la durée totale disponible. "
            "Fallback: compression proportionnelle des durées fixes."
        )
        fixed_duration_compression_factor = total_region_duration / max(0.01, reserved_fixed_duration)
        available_weighted_duration = 0.01
    else:
        fixed_duration_compression_factor = 1.0
        available_weighted_duration = total_region_duration - reserved_fixed_duration

    effective_onomatopoeia_fixed_duration = (
        onomatopoeia_fixed_duration * fixed_duration_compression_factor
    )

    total_normal_fragment_weight = sum(
        max(1.0, fragment.weight)
        for fragment in normal_fragments
    )

    subtitles: List[TimedSubtitle] = []
    region_index = 0
    current_time = valid_regions[0].start
    unmapped_fragments = 0
    timeline_control_fragments = 0
    cumulative_sync_offset = 0.0
    negative_sync_corrections = 0
    positive_sync_corrections = 0
    region_rewinds = 0
    sync_correction_events: List[Dict[str, Any]] = []

    for fragment in fragments:
        # Correction cumulative, positive OU négative.
        # Elle s'applique avant le fragment courant et donc à toute la suite.
        if fragment.sync_offset != 0.0:
            previous_time = current_time
            previous_region_index = region_index

            current_time = max(0.0, current_time + fragment.sync_offset)
            cumulative_sync_offset += fragment.sync_offset

            region_index, current_time = align_time_to_region(
                regions=valid_regions,
                region_index=region_index,
                current_time=current_time,
            )

            if fragment.sync_offset < 0:
                negative_sync_corrections += 1
            else:
                positive_sync_corrections += 1

            if region_index < previous_region_index:
                region_rewinds += 1

            sync_correction_events.append(
                {
                    "fragment_id": fragment.id,
                    "source_line": fragment.source_line,
                    "sync_offset": fragment.sync_offset,
                    "previous_time": previous_time,
                    "new_time": current_time,
                    "previous_region_index": previous_region_index,
                    "new_region_index": region_index,
                    "cumulative_sync_offset": cumulative_sync_offset,
                    "applies_to": "current_fragment_and_all_following_fragments",
                }
            )

        if not fragment.original_text:
            timeline_control_fragments += 1
            continue

        fixed_duration = get_fixed_fragment_duration(
            fragment=fragment,
            onomatopoeia_fixed_duration=onomatopoeia_fixed_duration,
            fixed_duration_compression_factor=fixed_duration_compression_factor,
        )

        if fixed_duration is not None:
            target_duration = fixed_duration
        else:
            fragment_weight = max(1.0, fragment.weight)

            if total_normal_fragment_weight <= 0.0:
                target_duration = available_weighted_duration / max(1, len(normal_fragments))
            else:
                target_duration = available_weighted_duration * (
                    fragment_weight / total_normal_fragment_weight
                )

        region_index, current_time = align_time_to_region(
            regions=valid_regions,
            region_index=region_index,
            current_time=current_time,
        )

        if region_index >= len(valid_regions):
            unmapped_fragments += 1
            last_end = valid_regions[-1].end
            subtitles.append(make_subtitle(len(subtitles), fragment, last_end, last_end))
            continue

        start = max(current_time, valid_regions[region_index].start)
        end = start
        remaining_duration = target_duration

        while remaining_duration > 0.0 and region_index < len(valid_regions):
            region = valid_regions[region_index]

            if current_time < region.start:
                current_time = region.start

            available_duration = region.end - current_time

            if available_duration <= 0.0:
                region_index += 1
                if region_index < len(valid_regions):
                    current_time = valid_regions[region_index].start
                continue

            consumed_duration = min(available_duration, remaining_duration)
            end = current_time + consumed_duration
            current_time = end
            remaining_duration -= consumed_duration

            if current_time >= region.end:
                region_index += 1
                if region_index < len(valid_regions):
                    current_time = valid_regions[region_index].start

        if end <= start:
            end = min(start + 0.01, valid_regions[-1].end)

        subtitles.append(make_subtitle(len(subtitles), fragment, start, end))

    subtitles = sorted(
        subtitles,
        key=lambda subtitle: (
            subtitle.fragment_id,
            subtitle.source_line,
        ),
    )

    return subtitles, {
        "mapping_rule": "dure_signed_sync_offset_is_cumulative_timeline_correction_unsigned_is_explicit_duration",
        "dure_metadata_rule": {
            "unsigned": "dure: 15 => fenêtre explicite de 15 secondes",
            "signed_positive": "dure: +1 => avance la timeline courante et tous les fragments suivants",
            "signed_negative": "dure: -1 => recule la timeline courante et tous les fragments suivants",
            "scope": "uniquement dans une parenthèse complète",
            "timeline_control": "(-- dure: +1) ou (-- dure: -1) => contrôle de timeline sans sous-titre",
            "text_fragment_control": "(texte -- dure: +1/-1) => correction appliquée avant ce fragment et conservée pour la suite",
            "subtitle_order_rule": "export_order_by_fragment_index_without_time_rewrite",
        },
        "fragments_count": len(fragments),
        "timeline_control_fragments": timeline_control_fragments,
        "normal_fragments_count": len(normal_fragments),
        "explicit_duration_fragments_count": len(explicit_duration_fragments),
        "sync_offset_fragments_count": sum(1 for fragment in fragments if fragment.sync_offset != 0.0),
        "positive_sync_corrections": positive_sync_corrections,
        "negative_sync_corrections": negative_sync_corrections,
        "region_rewinds": region_rewinds,
        "cumulative_sync_offset": cumulative_sync_offset,
        "sync_correction_events": sync_correction_events,
        "fixed_onomatopoeia_fragments_count": len(fixed_onomatopoeia_fragments),
        "vocal_regions_count": len(vocal_regions),
        "valid_vocal_regions_count": len(valid_regions),
        "unmapped_fragments": unmapped_fragments,
        "unused_vocal_regions": max(0, len(valid_regions) - region_index - 1),
        "total_region_duration": total_region_duration,
        "total_normal_fragment_weight": total_normal_fragment_weight,
        "onomatopoeia_fixed_duration": onomatopoeia_fixed_duration,
        "effective_onomatopoeia_fixed_duration": effective_onomatopoeia_fixed_duration,
        "reserved_explicit_duration": reserved_explicit_duration,
        "reserved_onomatopoeia_duration": reserved_onomatopoeia_duration,
        "reserved_fixed_duration": reserved_fixed_duration,
        "fixed_duration_compression_factor": fixed_duration_compression_factor,
        "available_weighted_duration": available_weighted_duration,
    }


def format_srt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    ms = int(round(seconds * 1000))
    h = ms // 3_600_000
    ms %= 3_600_000
    m = ms // 60_000
    ms %= 60_000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def export_srt(path: str, subtitles: List[TimedSubtitle]) -> None:
    with open(path, "w", encoding="utf-8") as file:
        for i, sub in enumerate(subtitles, start=1):
            file.write(f"{i}\n")
            file.write(f"{format_srt_time(sub.start)} --> {format_srt_time(sub.end)}\n")
            file.write(sub.text.strip() + "\n\n")


def export_lrc(path: str, subtitles: List[TimedSubtitle]) -> None:
    with open(path, "w", encoding="utf-8") as file:
        for sub in subtitles:
            minutes = int(sub.start // 60)
            seconds = sub.start - minutes * 60
            file.write(f"[{minutes:02d}:{seconds:05.2f}] {sub.text.strip()}\n")


def export_enhanced_lrc(path: str, subtitles: List[TimedSubtitle]) -> None:
    with open(path, "w", encoding="utf-8") as file:
        for sub in subtitles:
            minutes = int(sub.start // 60)
            seconds = sub.start - minutes * 60
            file.write(f"[{minutes:02d}:{seconds:05.2f}]{sub.text.strip()}\n")


def export_txt(path: str, subtitles: List[TimedSubtitle]) -> None:
    with open(path, "w", encoding="utf-8") as file:
        for sub in subtitles:
            file.write(sub.text.strip() + "\n")


def export_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def is_video_file(path: str) -> bool:
    return Path(path).suffix.lower() in {
        ".mp4",
        ".mkv",
        ".avi",
        ".mov",
        ".webm",
        ".flv",
        ".m4v",
    }


def burn_srt_to_video(
    input_video_path: str,
    input_srt_path: str,
    output_video_path: str,
) -> str:
    ensure_dir(os.path.dirname(output_video_path))

    if not os.path.isfile(input_video_path):
        raise FileNotFoundError(f"Vidéo introuvable: {input_video_path}")

    if not os.path.isfile(input_srt_path):
        raise FileNotFoundError(f"SRT introuvable: {input_srt_path}")

    srt_for_filter = (
        input_srt_path
        .replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", r"\'")
    )

    subtitle_style = (
        "FontName=Arial,"
        "FontSize=20,"
        "PrimaryColour=&H00FFFFFF,"
        "BackColour=&H00000000,"
        "OutlineColour=&H00000000,"
        "BorderStyle=3,"
        "Outline=1,"
        "Shadow=0,"
        "MarginV=25,"
        "Alignment=2"
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_video_path,
        "-vf",
        f"subtitles='{srt_for_filter}':force_style='{subtitle_style}'",
        "-c:a",
        "copy",
        output_video_path,
    ]

    log(f"[INFO] Burn subtitles to video: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            "Echec ffmpeg lors de l'incrustation du SRT dans la vidéo.\n"
            f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )

    if not os.path.isfile(output_video_path):
        raise FileNotFoundError(f"Vidéo sous-titrée non générée: {output_video_path}")

    return output_video_path


def process_mulan_demucs_timeline_pipeline(
    mulan_text_path: str,
    input_media_path: str,
    output_dir: str,
    input_language: str,
    output_language: str,
    translation_model: str,
    translation_max_length: int,
    translation_local_files_only: bool,
    phoneme_language: str,
    phoneme_backend: str,
    timing_mode: str,
    onomatopoeia_fixed_duration: float,
    demucs_model: str,
    min_blank_duration: float,
    min_voice_duration: float,
    blank_boundary_position: str,
    pad_start: float,
    pad_end: float,
    burn_video: bool,
) -> Dict[str, Any]:
    ensure_dir(output_dir)

    mulan_text_path = str(Path(mulan_text_path).resolve())
    input_media_path = str(Path(input_media_path).resolve())

    if not os.path.isfile(mulan_text_path):
        raise FileNotFoundError(f"Fichier Mulan text introuvable: {mulan_text_path}")

    if not os.path.isfile(input_media_path):
        raise FileNotFoundError(f"Fichier média introuvable: {input_media_path}")

    translator = create_local_text_translator(
        input_language=input_language,
        output_language=output_language,
        translation_model=translation_model,
        translation_max_length=translation_max_length,
        local_files_only=translation_local_files_only,
    )

    fragments = parse_mulan_text(
        path=mulan_text_path,
        input_language=input_language,
        output_language=output_language,
        translation_model=translation_model,
        translator=translator,
        phoneme_language=phoneme_language,
        phoneme_backend=phoneme_backend,
        timing_mode=timing_mode,
    )

    vocal_path, instrumental_path = extract_vocal_and_instrumental_with_demucs(
        input_media_path=input_media_path,
        output_dir=output_dir,
        demucs_model=demucs_model,
    )

    vocal_regions, blank_boundaries, detection_debug = detect_vocal_regions_from_blanks(
        vocal_path=vocal_path,
        min_blank_duration=min_blank_duration,
        min_voice_duration=min_voice_duration,
        blank_boundary_position=blank_boundary_position,
        pad_start=pad_start,
        pad_end=pad_end,
    )

    subtitles, mapping_debug = assign_mulan_lines_to_vocal_regions(
        fragments=fragments,
        vocal_regions=vocal_regions,
        onomatopoeia_fixed_duration=onomatopoeia_fixed_duration,
    )

    txt_path = os.path.join(output_dir, "mulan_clean_lines.txt")
    json_path = os.path.join(output_dir, "mulan_demucs_timeline.json")
    srt_path = os.path.join(output_dir, "mulan_demucs_timeline.srt")
    lrc_path = os.path.join(output_dir, "mulan_demucs_timeline.lrc")
    enhanced_lrc_path = os.path.join(output_dir, "mulan_demucs_timeline.enhanced.lrc")

    export_txt(txt_path, subtitles)
    export_srt(srt_path, subtitles)
    export_lrc(lrc_path, subtitles)
    export_enhanced_lrc(enhanced_lrc_path, subtitles)

    burned_video_path = None
    if burn_video:
        if not is_video_file(input_media_path):
            warn(f"Le média ne semble pas être une vidéo supportée: {input_media_path}")
        else:
            burned_video_path = os.path.join(output_dir, "video_subtitled_from_mulan_demucs.mp4")
            burn_srt_to_video(
                input_video_path=input_media_path,
                input_srt_path=srt_path,
                output_video_path=burned_video_path,
            )

    meta = {
        "mulan_text_path": mulan_text_path,
        "input_media_path": input_media_path,
        "vocal_path": vocal_path,
        "instrumental_path": instrumental_path,
        "demucs_model": demucs_model,
        "input_language": input_language,
        "output_language": output_language,
        "translation_model": translation_model,
        "translation_max_length": translation_max_length,
        "translation_local_files_only": translation_local_files_only,
        "translation_enabled": bool(output_language.strip() and translation_model.strip()),
        "translator_loaded": translator is not None,
        "transformers_available": TRANSFORMERS_AVAILABLE,
        "phoneme_language": phoneme_language,
        "phoneme_backend": phoneme_backend,
        "phonemizer_available": PHONEMIZER_AVAILABLE,
        "timing_mode": timing_mode,
        "translation_timing_policy": "timing_on_original_text_display_translated_text",
        "phonetic_timing_policy": "orthography_preserved_for_display_phonetic_normalization_used_only_for_weight",
        "onomatopoeia_fixed_duration": onomatopoeia_fixed_duration,
        "effective_onomatopoeia_fixed_duration": mapping_debug.get(
            "effective_onomatopoeia_fixed_duration"
        ),
        "timeline_mode": "demucs_vocal_blank_duration_only_projection",
        "silence_amplitude_threshold": SILENCE_AMPLITUDE_THRESHOLD,
        "min_blank_duration": min_blank_duration,
        "min_voice_duration": min_voice_duration,
        "blank_boundary_position": blank_boundary_position,
        "mapping_rule": mapping_debug.get("mapping_rule"),
        "dure_metadata_rule": mapping_debug.get("dure_metadata_rule"),
        "fragment_count": len(fragments),
        "timeline_control_count": sum(1 for fragment in fragments if not fragment.original_text),
        "onomatopoeia_count": sum(1 for fragment in fragments if fragment.is_onomatopoeia),
        "translated_fragment_count": sum(1 for fragment in fragments if fragment.translation_applied),
        "explicit_duration_count": sum(1 for fragment in fragments if fragment.explicit_duration is not None),
        "sync_offset_count": sum(1 for fragment in fragments if fragment.sync_offset != 0.0),
        "blank_boundary_count": len(blank_boundaries),
        "vocal_region_count": len(vocal_regions),
        "subtitle_count": len(subtitles),
        "srt_path": srt_path,
        "lrc_path": lrc_path,
        "enhanced_lrc_path": enhanced_lrc_path,
        "burned_video_path": burned_video_path,
        "asr_enabled": False,
        "whisperx_enabled": False,
        "demucs_enabled": True,
        "detection": detection_debug,
        "mapping": mapping_debug,
        "onomatopoeia_patterns": ONOMATOPOEIA_PATTERNS,
    }

    payload = {
        "meta": meta,
        "fragments": [asdict(fragment) for fragment in fragments],
        "blank_boundaries": [asdict(blank) for blank in blank_boundaries],
        "vocal_regions": [asdict(region) for region in vocal_regions],
        "subtitles": [asdict(subtitle) for subtitle in subtitles],
    }

    export_json(json_path, payload)

    return {
        "meta": meta,
        "fragments": fragments,
        "blank_boundaries": blank_boundaries,
        "vocal_regions": vocal_regions,
        "subtitles": subtitles,
        "txt_path": txt_path,
        "json_path": json_path,
        "srt_path": srt_path,
        "lrc_path": lrc_path,
        "enhanced_lrc_path": enhanced_lrc_path,
        "burned_video_path": burned_video_path,
        "vocal_path": vocal_path,
        "instrumental_path": instrumental_path,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Pipeline Mulan text + média -> traduction locale optionnelle "
            "-> Demucs vocal/instrumental -> régions temporelles "
            "-> projection pondérée Mulan -> SRT/LRC/JSON -> burn vidéo."
        )
    )

    parser.add_argument("-m", "--mulan-text", required=True, help="Fichier Mulan text contenant les paroles")
    parser.add_argument("-a", "--input-media", required=True, help="Audio/vidéo source")
    parser.add_argument("-o", "--output-dir", default="./out", help="Dossier de sortie")

    parser.add_argument("--input-language", default="auto", help="Langue source texte, ex: auto, fr, en")
    parser.add_argument("--output-language", default="", help="Langue cible. Vide = traduction désactivée.")
    parser.add_argument(
        "--translation-model",
        default="/home/drodriguez/dev/Helsinki-NLP/opus-mt-fr-es",
        help="Chemin local ou nom de modèle transformers Seq2Seq. Vide = traduction désactivée.",
    )
    parser.add_argument(
        "--translation-max-length",
        type=int,
        default=256,
        help="Longueur maximale de génération du modèle de traduction.",
    )
    parser.add_argument(
        "--translation-allow-download",
        action="store_false",
        dest="translation_local_files_only",
        default=True,
        help="Autorise transformers à télécharger le modèle si absent localement.",
    )

    parser.add_argument("--phoneme-language", default="fr-fr", help="Langue phonemizer")
    parser.add_argument("--phoneme-backend", default="espeak", help="Backend phonemizer")
    parser.add_argument(
        "--timing-mode",
        default="hybrid",
        choices=["syllables", "phonemes", "hybrid"],
        help="Mode de pondération timing des fragments Mulan",
    )
    parser.add_argument(
        "--onomatopoeia-fixed-duration",
        type=float,
        default=0.65,
        help="Durée fixe en secondes pour chaque onomatopée sans durée explicite.",
    )
    parser.add_argument("--demucs-model", default="htdemucs")

    parser.add_argument("--min-blank-duration", type=float, default=0.25)
    parser.add_argument("--min-voice-duration", type=float, default=0.5)
    parser.add_argument(
        "--blank-boundary-position",
        default="middle",
        choices=["start", "middle", "end"],
    )

    parser.add_argument("--pad-start", type=float, default=0.05)
    parser.add_argument("--pad-end", type=float, default=0.10)

    parser.add_argument(
        "--no-burn-video",
        action="store_false",
        dest="burn_video",
        default=True,
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        result = process_mulan_demucs_timeline_pipeline(
            mulan_text_path=args.mulan_text,
            input_media_path=args.input_media,
            output_dir=args.output_dir,
            input_language=args.input_language,
            output_language=args.output_language,
            translation_model=args.translation_model,
            translation_max_length=args.translation_max_length,
            translation_local_files_only=args.translation_local_files_only,
            phoneme_language=args.phoneme_language,
            phoneme_backend=args.phoneme_backend,
            timing_mode=args.timing_mode,
            onomatopoeia_fixed_duration=args.onomatopoeia_fixed_duration,
            demucs_model=args.demucs_model,
            min_blank_duration=args.min_blank_duration,
            min_voice_duration=args.min_voice_duration,
            blank_boundary_position=args.blank_boundary_position,
            pad_start=args.pad_start,
            pad_end=args.pad_end,
            burn_video=args.burn_video,
        )

        log("")
        log("=== RESULTAT ===")
        log(f"VOCAL       : {result['vocal_path']}")
        log(f"INSTRUMENTAL: {result['instrumental_path']}")
        log(f"TXT         : {result['txt_path']}")
        log(f"JSON        : {result['json_path']}")
        log(f"SRT         : {result['srt_path']}")
        log(f"LRC         : {result['lrc_path']}")
        log(f"ELRC        : {result['enhanced_lrc_path']}")

        if result.get("burned_video_path"):
            log(f"VIDEO       : {result['burned_video_path']}")

        log("")
        log(f"Fragments Mulan       : {len(result['fragments'])}")
        log(f"Contrôles timeline    : {sum(1 for fragment in result['fragments'] if not fragment.original_text)}")
        log(f"Onomatopées           : {sum(1 for fragment in result['fragments'] if fragment.is_onomatopoeia)}")
        log(f"Fragments traduits    : {sum(1 for fragment in result['fragments'] if fragment.translation_applied)}")
        log(f"Durées explicites     : {sum(1 for fragment in result['fragments'] if fragment.explicit_duration is not None)}")
        log(f"Offsets synchro       : {sum(1 for fragment in result['fragments'] if fragment.sync_offset != 0.0)}")
        log(f"Blancs détectés       : {len(result['blank_boundaries'])}")
        log(f"Régions vocales       : {len(result['vocal_regions'])}")
        log("")

        for sub in result["subtitles"]:
            markers = []
            if sub.is_onomatopoeia:
                markers.append("ONOM")
            if sub.translation_applied:
                markers.append("TRAD")
            if sub.explicit_duration is not None:
                markers.append(f"DURE={sub.explicit_duration:.2f}s")
            if sub.sync_offset != 0.0:
                markers.append(f"SYNC={sub.sync_offset:+.2f}s")

            marker = f" [{' '.join(markers)}]" if markers else ""
            log(f"[{sub.start:8.2f} -> {sub.end:8.2f}]{marker} {sub.text}")

    except Exception as exc:
        fail(f"Echec pipeline Mulan/Demucs timeline: {exc}\n{traceback.format_exc()}")


if __name__ == "__main__":
    main()
