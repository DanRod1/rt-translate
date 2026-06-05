from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml
from youtube_transcript_api import YouTubeTranscriptApi

try:
    import torch
except Exception:
    torch = None

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
except Exception:
    AutoModelForSeq2SeqLM = None
    AutoTokenizer = None


@dataclass
class CaptionSegment:
    start: float
    duration: float
    text: str
    source_text: str = ""
    distance_ratio: float = 1.0
    mulan_index: int = -1
    inserted: bool = False


class LocalTranslator:
    def __init__(
        self,
        model_path: str,
        source_language: str,
        target_language: str,
        max_length: int = 256,
        local_files_only: bool = True,
        device: str = "cpu",
        num_beams: int = 1,
    ):
        if AutoTokenizer is None or AutoModelForSeq2SeqLM is None:
            raise RuntimeError("transformers n'est pas disponible.")

        self.source_language = source_language.strip().lower()
        self.target_language = target_language.strip().lower()
        self.device = device
        self.max_length = max_length
        self.num_beams = num_beams

        if not self.source_language:
            raise ValueError("translation.source_language est obligatoire.")

        if not self.target_language:
            raise ValueError("translation.target_language est obligatoire.")

        resolved_model_path = Path(model_path).expanduser().resolve()

        if local_files_only and not resolved_model_path.exists():
            raise FileNotFoundError(f"Modèle local introuvable: {resolved_model_path}")

        load_path = str(resolved_model_path) if resolved_model_path.exists() else model_path

        self.tokenizer = AutoTokenizer.from_pretrained(
            load_path,
            local_files_only=local_files_only,
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            load_path,
            local_files_only=local_files_only,
        )

        if device != "cpu":
            self.model.to(device)

        self.model.eval()

    def translate(self, text: str) -> str:
        if not text.strip():
            return text

        encoded = self.tokenizer(text, return_tensors="pt", truncation=True)

        if self.device != "cpu":
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

        if torch is not None:
            with torch.inference_mode():
                generated = self.model.generate(
                    **encoded,
                    max_length=self.max_length,
                    num_beams=self.num_beams,
                )
        else:
            generated = self.model.generate(
                **encoded,
                max_length=self.max_length,
                num_beams=self.num_beams,
            )

        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]


def log(message: str) -> None:
    print(message, flush=True)


def normalize_text(text: str) -> str:
    text = str(text).replace("…", "...").replace("’", "'")
    return re.sub(r"\s+", " ", text).strip()


def normalize_for_distance(text: str) -> str:
    text = normalize_text(text).lower()
    return re.sub(r"[^\wÀ-ÿ]+", "", text, flags=re.UNICODE)


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    if len(a) < len(b):
        a, b = b, a

    previous = list(range(len(b) + 1))

    for i, ca in enumerate(a, start=1):
        current = [i]

        for j, cb in enumerate(b, start=1):
            current.append(
                min(
                    current[j - 1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (0 if ca == cb else 1),
                )
            )

        previous = current

    return previous[-1]


def normalized_char_distance_ratio(a: str, b: str) -> float:
    na = normalize_for_distance(a)
    nb = normalize_for_distance(b)
    return levenshtein_distance(na, nb) / max(len(na), len(nb), 1)


def extract_video_id(url: str) -> str:
    patterns = [
        r"(?:v=)([A-Za-z0-9_-]{11})",
        r"(?:youtu\.be/)([A-Za-z0-9_-]{11})",
        r"(?:shorts/)([A-Za-z0-9_-]{11})",
        r"(?:embed/)([A-Za-z0-9_-]{11})",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url.strip()):
        return url.strip()

    raise ValueError(f"Impossible d'extraire l'id YouTube depuis: {url}")


def format_ass_time(seconds: float) -> str:
    cs = int(round(max(0.0, float(seconds)) * 100))
    h = cs // 360000
    cs %= 360000
    m = cs // 6000
    cs %= 6000
    s = cs // 100
    cs %= 100
    return f"{h}:{m:02}:{s:02}.{cs:02}"


def parse_mulan_payload_from_line(line: str) -> Optional[str]:
    raw = line.strip()

    if not raw or raw.startswith("#") or raw.startswith("//"):
        return None

    if "=" in raw:
        key, value = raw.split("=", 1)

        if key.strip().lower() in {
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


def strip_mulan_markup(line: str) -> str:
    line = re.sub(r"<[^>]+>", " ", line)
    line = re.sub(r"\[[^\]]+\]", " ", line)
    line = re.sub(r"\{[^}]+\}", " ", line)
    return normalize_text(line)


def split_parenthesized_inline_metadata(text: str) -> tuple[str, dict[str, Any]]:
    raw = text.strip()
    metadata: dict[str, Any] = {}

    if not (raw.startswith("(") and raw.endswith(")")):
        return raw, metadata

    inner = raw[1:-1].strip()

    if "--" not in inner:
        return raw, metadata

    useful_text, comment = inner.split("--", 1)

    duration_match = re.search(
        r"\bdure\s*:\s*([+-]?\s*[0-9]+(?:[.,][0-9]+)?)",
        comment,
        flags=re.IGNORECASE,
    )

    if duration_match:
        value = duration_match.group(1).replace(" ", "").replace(",", ".")
        number = float(value)

        if value.startswith("+") or value.startswith("-"):
            metadata["sync_offset"] = number
        else:
            metadata["explicit_duration"] = number

    return useful_text.strip(), metadata


def load_mulan_lines(path: str) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []

    with open(path, "r", encoding="utf-8") as f:
        for source_line, raw_line in enumerate(f, start=1):
            payload = parse_mulan_payload_from_line(raw_line)

            if payload is None:
                continue

            text, metadata = split_parenthesized_inline_metadata(payload)
            clean = strip_mulan_markup(text)

            if not clean and not metadata:
                continue

            lines.append(
                {
                    "source_line": source_line,
                    "text": clean,
                    "sync_offset": float(metadata.get("sync_offset", 0.0)),
                    "explicit_duration": metadata.get("explicit_duration"),
                }
            )

    return lines


def fetch_youtube_transcript(
    video_id: str,
    languages: list[str],
) -> list[CaptionSegment]:
    api = YouTubeTranscriptApi()
    transcript = api.fetch(video_id, languages=languages)

    return [
        CaptionSegment(
            start=float(snippet.start),
            duration=float(snippet.duration),
            text=normalize_text(snippet.text),
        )
        for snippet in transcript
    ]


def merge_missing_lines_limited(
    missing: list[dict[str, Any]],
    matched_text: str,
    max_merged_chars: int = 90,
    max_missing_lines: int = 2,
) -> str:
    selected: list[str] = []
    matched_text = normalize_text(matched_text)

    if max_missing_lines <= 0 or max_merged_chars <= 0:
        return matched_text

    for item in reversed(missing):
        text = normalize_text(item.get("text", ""))

        if not text:
            continue

        candidate_lines = [text] + list(reversed(selected)) + [matched_text]
        candidate = " ".join(candidate_lines)

        if len(candidate) > max_merged_chars:
            break

        if len(selected) >= max_missing_lines:
            break

        selected.append(text)

    selected = list(reversed(selected))

    if not selected:
        return matched_text

    return normalize_text(" ".join(selected + [matched_text]))


def align_mulan_to_asr_by_char_distance(
    asr_segments: list[CaptionSegment],
    mulan_items: list[dict[str, Any]],
    max_distance_ratio: float = 0.10,
    search_window: int = 12,
    insert_missing_between_matches: bool = True,
    max_merged_chars: int = 90,
    max_missing_lines: int = 2,
) -> list[CaptionSegment]:
    lyric_items = [item for item in mulan_items if item.get("text")]
    aligned: list[CaptionSegment] = []
    mulan_cursor = 0

    for seg in asr_segments:
        best_idx = -1
        best_ratio = 1.0
        search_end = min(len(lyric_items), mulan_cursor + max(1, search_window))

        for idx in range(mulan_cursor, search_end):
            ratio = normalized_char_distance_ratio(seg.text, lyric_items[idx]["text"])

            if ratio < best_ratio:
                best_ratio = ratio
                best_idx = idx

            if ratio == 0.0:
                break

        if best_idx >= 0 and best_ratio <= max_distance_ratio:
            missing = lyric_items[mulan_cursor:best_idx]
            matched = dict(lyric_items[best_idx])

            if missing and insert_missing_between_matches:
                matched["text"] = merge_missing_lines_limited(
                    missing,
                    matched["text"],
                    max_merged_chars=max_merged_chars,
                    max_missing_lines=max_missing_lines,
                )

            aligned.append(
                CaptionSegment(
                    start=seg.start,
                    duration=seg.duration,
                    text=matched["text"],
                    source_text=seg.text,
                    distance_ratio=best_ratio,
                    mulan_index=best_idx,
                    inserted=False,
                )
            )

            mulan_cursor = best_idx + 1

        else:
            aligned.append(
                CaptionSegment(
                    start=seg.start,
                    duration=seg.duration,
                    text=seg.text,
                    source_text=seg.text,
                    distance_ratio=best_ratio,
                    mulan_index=-1,
                    inserted=False,
                )
            )

    aligned.sort(key=lambda s: (s.start, s.start + s.duration))
    return aligned


def apply_mulan_timeline_adjustments(
    segments: list[CaptionSegment],
    mulan_items: list[dict[str, Any]],
) -> list[CaptionSegment]:
    lyric_items = [item for item in mulan_items if item.get("text")]

    adjusted: list[CaptionSegment] = []
    cumulative_offset = 0.0
    last_mulan_index = -1

    for seg in segments:
        if seg.mulan_index >= 0:
            for idx in range(last_mulan_index + 1, seg.mulan_index + 1):
                cumulative_offset += float(
                    lyric_items[idx].get("sync_offset", 0.0) or 0.0
                )

            last_mulan_index = max(last_mulan_index, seg.mulan_index)

        duration = seg.duration

        if seg.mulan_index >= 0:
            explicit_duration = lyric_items[seg.mulan_index].get("explicit_duration")

            if explicit_duration is not None:
                duration = max(0.01, float(explicit_duration))

        adjusted.append(
            CaptionSegment(
                start=max(0.0, seg.start + cumulative_offset),
                duration=duration,
                text=seg.text,
                source_text=seg.source_text,
                distance_ratio=seg.distance_ratio,
                mulan_index=seg.mulan_index,
                inserted=seg.inserted,
            )
        )

    adjusted.sort(key=lambda s: (s.start, s.start + s.duration))
    return adjusted


def looks_like_language(text: str, language: str) -> bool:
    t = f" {normalize_text(text).lower()} "

    markers = {
        "fr": [
            " je ", " tu ", " il ", " elle ", " nous ", " vous ",
            " les ", " des ", " une ", " un ", " que ", " qui ",
            " pas ", " dans ", " avec ", " pour ",
        ],
        "en": [
            " i ", " you ", " he ", " she ", " we ", " they ",
            " the ", " and ", " not ", " my ", " your ", " is ",
            " are ", " in ", " with ", " for ",
        ],
        "es": [
            " yo ", " tú ", " tu ", " él ", " ella ", " nosotros ",
            " los ", " las ", " una ", " un ", " que ", " no ",
            " mi ", " en ", " con ", " para ", " y ",
        ],
    }

    language = language.lower().split("-")[0]
    return sum(1 for marker in markers.get(language, []) if marker in t) >= 1


def should_translate_text(
    text: str,
    source_language: str,
    target_language: str,
    skip_if_target_detected: bool = True,
) -> bool:
    if not text.strip():
        return False

    source_language = source_language.lower().split("-")[0]
    target_language = target_language.lower().split("-")[0]

    source_detected = looks_like_language(text, source_language)
    target_detected = looks_like_language(text, target_language)

    if skip_if_target_detected and target_detected and not source_detected:
        return False

    return True


def maybe_translate_segments(
    segments: list[CaptionSegment],
    translator: Optional[LocalTranslator],
    log_progress: bool,
    skip_if_target_detected: bool = True,
) -> list[CaptionSegment]:
    if translator is None:
        return segments

    out: list[CaptionSegment] = []

    for idx, seg in enumerate(segments, start=1):
        if not should_translate_text(
            seg.text,
            translator.source_language,
            translator.target_language,
            skip_if_target_detected=skip_if_target_detected,
        ):
            if log_progress:
                log(f"[INFO] Segment {idx}/{len(segments)} non traduit")
            out.append(seg)
            continue

        if log_progress:
            log(f"[INFO] Traduction segment {idx}/{len(segments)}")

        out.append(
            CaptionSegment(
                start=seg.start,
                duration=seg.duration,
                text=normalize_text(translator.translate(seg.text)),
                source_text=seg.source_text,
                distance_ratio=seg.distance_ratio,
                mulan_index=seg.mulan_index,
                inserted=seg.inserted,
            )
        )

    return out


def wrap_text_max_chars(text: str, max_chars_per_line: int) -> str:
    text = normalize_text(text)

    if max_chars_per_line <= 0:
        return text

    words = text.split()

    if not words:
        return text

    lines: list[str] = []
    current: list[str] = []
    current_len = 0

    for word in words:
        if not current:
            current = [word]
            current_len = len(word)
            continue

        candidate_len = current_len + 1 + len(word)

        if candidate_len <= max_chars_per_line:
            current.append(word)
            current_len = candidate_len
        else:
            lines.append(" ".join(current))
            current = [word]
            current_len = len(word)

    if current:
        lines.append(" ".join(current))

    return "\n".join(lines)


def apply_max_chars_per_line(
    segments: list[CaptionSegment],
    max_chars_per_line: int,
) -> list[CaptionSegment]:
    if max_chars_per_line <= 0:
        return segments

    return [
        CaptionSegment(
            start=seg.start,
            duration=seg.duration,
            text=wrap_text_max_chars(seg.text, max_chars_per_line),
            source_text=seg.source_text,
            distance_ratio=seg.distance_ratio,
            mulan_index=seg.mulan_index,
            inserted=seg.inserted,
        )
        for seg in segments
    ]


def split_text_by_timelines(
    text: str,
    timelines: list[CaptionSegment],
    max_chars_per_line: int = 42,
) -> list[CaptionSegment]:
    text = normalize_text(text)
    words = text.split()

    timelines = sorted(timelines, key=lambda s: (s.start, s.start + s.duration))

    if not words or not timelines:
        return []

    total_duration = sum(max(0.01, seg.duration) for seg in timelines)
    chunks: list[list[str]] = [[] for _ in timelines]
    word_index = 0

    for i, seg in enumerate(timelines):
        if i == len(timelines) - 1:
            chunks[i] = words[word_index:]
            break

        ratio = max(0.01, seg.duration) / total_duration
        target_count = max(1, round(len(words) * ratio))

        remaining_words = len(words) - word_index
        remaining_slots = len(timelines) - i

        target_count = min(target_count, max(1, remaining_words - remaining_slots + 1))

        chunks[i] = words[word_index:word_index + target_count]
        word_index += target_count

    out: list[CaptionSegment] = []

    for seg, chunk_words in zip(timelines, chunks):
        chunk_text = normalize_text(" ".join(chunk_words))

        if not chunk_text:
            continue

        out.append(
            CaptionSegment(
                start=seg.start,
                duration=seg.duration,
                text=wrap_text_max_chars(chunk_text, max_chars_per_line),
                source_text=seg.source_text,
                distance_ratio=seg.distance_ratio,
                mulan_index=seg.mulan_index,
                inserted=seg.inserted,
            )
        )

    return out


def replace_bad_fragment_group_with_merged_text(
    segments: list[CaptionSegment],
    max_group_gap: float = 1.25,
    min_group_segments: int = 3,
    merged_duration_factor: float = 1.8,
    merged_text_ratio: float = 0.55,
    max_chars_per_line: int = 42,
) -> list[CaptionSegment]:
    ordered = sorted(segments, key=lambda s: (s.start, s.start + s.duration))
    result: list[CaptionSegment] = []
    i = 0

    while i < len(ordered):
        current = ordered[i]
        group_start = current.start
        group_end = current.start + current.duration

        group: list[CaptionSegment] = [current]
        j = i + 1

        while j < len(ordered):
            nxt = ordered[j]
            nxt_start = nxt.start
            nxt_end = nxt.start + nxt.duration

            if nxt_start - group_end > max_group_gap:
                break

            group.append(nxt)
            group_end = max(group_end, nxt_end)
            j += 1

        if len(group) < min_group_segments:
            result.extend(group)
            i = j
            continue

        group_duration = max(0.01, group_end - group_start)
        candidate_merged: Optional[CaptionSegment] = None

        for seg in group:
            other_text_len = sum(len(x.text) for x in group if x is not seg)
            long_enough = len(seg.text) > other_text_len * merged_text_ratio
            duration_enough = seg.duration >= group_duration / merged_duration_factor

            if long_enough and duration_enough:
                candidate_merged = seg
                break

        if candidate_merged is None:
            result.extend(group)
            i = j
            continue

        timelines = [seg for seg in group if seg is not candidate_merged]

        if not timelines:
            result.append(candidate_merged)
            i = j
            continue

        result.extend(
            split_text_by_timelines(
                candidate_merged.text,
                timelines,
                max_chars_per_line=max_chars_per_line,
            )
        )

        i = j

    result.sort(key=lambda s: (s.start, s.start + s.duration))
    return result


def assign_ass_tracks(
    segments: list[CaptionSegment],
    max_tracks: int = 3,
) -> list[tuple[CaptionSegment, int]]:
    ordered = sorted(segments, key=lambda s: (s.start, s.start + s.duration))

    active: list[tuple[CaptionSegment, int]] = []
    result: list[tuple[CaptionSegment, int]] = []

    for seg in ordered:
        start = seg.start

        active = [
            (active_seg, track)
            for active_seg, track in active
            if active_seg.start + active_seg.duration > start
        ]

        active = [
            (active_seg, track + 1)
            for active_seg, track in active
            if track + 1 < max_tracks
        ]

        result = [
            (
                old_seg,
                old_track + 1
                if old_seg.start + old_seg.duration > start
                and old_track + 1 < max_tracks
                else old_track,
            )
            for old_seg, old_track in result
        ]

        active.append((seg, 0))
        result.append((seg, 0))

    return result


def ass_escape(text: str) -> str:
    text = str(text)
    text = text.replace("\\", r"\\")
    text = text.replace("{", r"\{").replace("}", r"\}")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\n", r"\N")
    return text.strip()


def ass_filter_path(path: str) -> str:
    return (
        path
        .replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", r"\'")
    )


def write_ass(
    path: str,
    segments: list[CaptionSegment],
    play_res_x: int = 1920,
    play_res_y: int = 1080,
    font_name: str = "Arial",
    font_size: int = 54,
    margin_v: int = 45,
    line_gap: int = 72,
    max_tracks: int = 3,
) -> None:
    ordered_with_tracks = assign_ass_tracks(segments, max_tracks=max_tracks)

    with open(path, "w", encoding="utf-8") as f:
        f.write("[Script Info]\n")
        f.write("ScriptType: v4.00+\n")
        f.write("WrapStyle: 2\n")
        f.write("ScaledBorderAndShadow: yes\n")
        f.write(f"PlayResX: {play_res_x}\n")
        f.write(f"PlayResY: {play_res_y}\n\n")

        f.write("[V4+ Styles]\n")
        f.write(
            "Format: "
            "Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,"
            "OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,"
            "ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,"
            "Alignment,MarginL,MarginR,MarginV,Encoding\n"
        )
        f.write(
            "Style: "
            f"Default,{font_name},{font_size},"
            "&H00FFFFFF,&H000000FF,&H00000000,&H64000000,"
            "0,0,0,0,"
            "100,100,0,0,"
            "1,2,0,"
            f"2,60,60,{margin_v},1\n\n"
        )

        f.write("[Events]\n")
        f.write(
            "Format: "
            "Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"
        )

        for idx, (seg, track) in enumerate(ordered_with_tracks):
            if not seg.text.strip() or seg.duration <= 0:
                continue

            start = format_ass_time(seg.start)
            end = format_ass_time(seg.start + seg.duration)

            margin_v_for_track = margin_v + track * line_gap
            text = ass_escape(seg.text)

            f.write(
                f"Dialogue: 0,{start},{end},Default,,"
                f"0,0,{margin_v_for_track},,{text}\n"
            )


def write_txt(path: str, segments: list[CaptionSegment]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for seg in segments:
            f.write(seg.text.strip() + "\n")


def tsv_escape(value: str) -> str:
    return str(value).replace("\t", " ").replace("\n", " ").replace("\r", " ")


def write_debug_tsv(path: str, segments: list[CaptionSegment]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "start\tend\tdistance_ratio\tinserted\tmulan_index\tasr_text\tfinal_text\n"
        )

        for seg in segments:
            f.write(
                f"{seg.start:.3f}\t"
                f"{seg.start + seg.duration:.3f}\t"
                f"{seg.distance_ratio:.3f}\t"
                f"{int(seg.inserted)}\t"
                f"{seg.mulan_index}\t"
                f"{tsv_escape(seg.source_text)}\t"
                f"{tsv_escape(seg.text)}\n"
            )


def build_translator(config: dict[str, Any]) -> Optional[LocalTranslator]:
    translation = config.get("translation", {}) or {}

    if not translation.get("enabled", False):
        log("[INFO] Traduction désactivée.")
        return None

    source_language = str(translation.get("source_language", "")).strip()
    target_language = str(translation.get("target_language", "")).strip()

    if not source_language:
        raise ValueError(
            "translation.source_language est obligatoire quand translation.enabled=true."
        )

    if not target_language:
        raise ValueError(
            "translation.target_language est obligatoire quand translation.enabled=true."
        )

    if source_language == target_language:
        log("[INFO] source_language == target_language, traduction désactivée.")
        return None

    return LocalTranslator(
        model_path=str(translation["model_path"]),
        source_language=source_language,
        target_language=target_language,
        max_length=int(translation.get("max_length", 256)),
        local_files_only=bool(translation.get("local_files_only", True)),
        device=str(translation.get("device", "cpu")),
        num_beams=int(translation.get("num_beams", 1)),
    )


def build_ytdlp_cmd(config: dict[str, Any]) -> list[str]:
    ytdlp = config.get("youtube_download", {}) or {}
    cmd = [sys.executable, "-m", "yt_dlp", "--no-playlist"]

    if ytdlp.get("cookies_from_browser"):
        cmd += ["--cookies-from-browser", str(ytdlp["cookies_from_browser"])]

    if ytdlp.get("cookies"):
        cmd += ["--cookies", str(Path(ytdlp["cookies"]).expanduser())]

    if ytdlp.get("remote_components"):
        cmd += ["--remote-components", str(ytdlp["remote_components"])]

    if ytdlp.get("no_check_certificate", False):
        cmd += ["--no-check-certificate"]

    return cmd


def run_cmd(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    log(f"[INFO] CMD: {' '.join(cmd)}")
    return subprocess.run(cmd, capture_output=False, text=True)


def download_youtube_video(
    youtube_url: str,
    output_dir: str,
    video_id: str,
    config: dict[str, Any],
) -> Optional[str]:
    ytdlp = config.get("youtube_download", {}) or {}

    if not ytdlp.get("enabled", False):
        log("[INFO] Téléchargement vidéo désactivé.")
        return None

    work_dir = Path(output_dir) / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    output_template = str(work_dir / f"{video_id}.%(ext)s")
    fmt = ytdlp.get("format", "bv*[height<=720]+ba/b[height<=720]/best")

    cmd = build_ytdlp_cmd(config) + [
        "-f",
        fmt,
        "--merge-output-format",
        "mp4",
        "-o",
        output_template,
        youtube_url,
    ]

    result = run_cmd(cmd)

    if result.returncode != 0:
        raise RuntimeError("Échec téléchargement vidéo YouTube.")

    video_candidates = [
        p
        for p in sorted(work_dir.glob(f"{video_id}.*"))
        if p.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov", ".m4v"}
    ]

    if not video_candidates:
        raise FileNotFoundError(f"Aucune vidéo téléchargée trouvée dans {work_dir}")

    return str(video_candidates[0])


def burn_ass_to_video(
    input_video_path: str,
    input_ass_path: str,
    output_video_path: str,
) -> str:
    Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)

    if not os.path.isfile(input_video_path):
        raise FileNotFoundError(f"Vidéo introuvable: {input_video_path}")

    if not os.path.isfile(input_ass_path):
        raise FileNotFoundError(f"ASS introuvable: {input_ass_path}")

    ass_for_filter = ass_filter_path(input_ass_path)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_video_path,
        "-vf",
        f"ass='{ass_for_filter}'",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "copy",
        output_video_path,
    ]

    result = run_cmd(cmd)

    if result.returncode != 0:
        raise RuntimeError("Échec burn ASS dans la vidéo.")

    return output_video_path


def maybe_download_and_burn(
    youtube_url: str,
    video_id: str,
    ass_path: str,
    output_dir: str,
    config: dict[str, Any],
) -> Optional[str]:
    burn_cfg = config.get("burn", {}) or {}

    if not burn_cfg.get("enabled", False):
        log("[INFO] Burn vidéo désactivé.")
        return None

    input_video = burn_cfg.get("input_video")

    if input_video:
        video_path = str(Path(input_video).expanduser())
    else:
        video_path = download_youtube_video(
            youtube_url,
            output_dir,
            video_id,
            config,
        )

    if not video_path:
        return None

    output_video = burn_cfg.get("output_video") or str(
        Path(output_dir) / f"{video_id}.burned.mp4"
    )

    return burn_ass_to_video(video_path, ass_path, output_video)


def main() -> None:
    config_path = os.environ.get("CONFIG", "config.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    youtube_url = config["youtube_url"]
    mulan_file = config["mulan_file"]

    languages = config.get("languages", ["fr", "en"])
    output_dir = config.get("output_dir", "./out")

    alignment_cfg = config.get("alignment", {}) or {}
    ass_cfg = config.get("ass", {}) or {}

    os.makedirs(output_dir, exist_ok=True)

    video_id = extract_video_id(youtube_url)

    log(f"[INFO] video_id={video_id}")
    log(f"[INFO] fetch YouTube ASR languages={languages}")

    asr_segments = fetch_youtube_transcript(video_id, languages)

    log(f"[INFO] ASR segments={len(asr_segments)}")

    mulan_items = load_mulan_lines(mulan_file)

    log(f"[INFO] Mulan items={len(mulan_items)}")

    aligned = align_mulan_to_asr_by_char_distance(
        asr_segments,
        mulan_items,
        max_distance_ratio=float(alignment_cfg.get("max_distance_ratio", 0.10)),
        search_window=int(alignment_cfg.get("search_window", 12)),
        insert_missing_between_matches=bool(
            alignment_cfg.get("insert_missing_between_matches", True)
        ),
        max_merged_chars=int(alignment_cfg.get("max_merged_chars", 90)),
        max_missing_lines=int(alignment_cfg.get("max_missing_lines", 2)),
    )

    adjusted = apply_mulan_timeline_adjustments(aligned, mulan_items)

    translator = build_translator(config)

    translation_cfg = config.get("translation", {}) or {}

    final_segments = maybe_translate_segments(
        adjusted,
        translator,
        log_progress=bool(translation_cfg.get("log_progress", True)),
        skip_if_target_detected=bool(
            translation_cfg.get("skip_if_target_detected", True)
        ),
    )

    final_segments.sort(key=lambda s: (s.start, s.start + s.duration))

    final_segments = replace_bad_fragment_group_with_merged_text(
        final_segments,
        max_group_gap=float(alignment_cfg.get("merged_fix_max_group_gap", 1.25)),
        min_group_segments=int(alignment_cfg.get("merged_fix_min_group_segments", 3)),
        merged_duration_factor=float(
            alignment_cfg.get("merged_fix_duration_factor", 1.8)
        ),
        merged_text_ratio=float(alignment_cfg.get("merged_fix_text_ratio", 0.55)),
        max_chars_per_line=int(ass_cfg.get("max_chars_per_line", 42)),
    )

    final_segments = apply_max_chars_per_line(
        final_segments,
        int(ass_cfg.get("max_chars_per_line", 42)),
    )

    suffix = "translated" if translator is not None else "mulan"

    ass_path = os.path.join(output_dir, f"{video_id}.{suffix}.ass")
    txt_path = os.path.join(output_dir, f"{video_id}.{suffix}.txt")
    debug_path = os.path.join(output_dir, f"{video_id}.debug.tsv")

    write_ass(
        ass_path,
        final_segments,
        play_res_x=int(ass_cfg.get("play_res_x", 1920)),
        play_res_y=int(ass_cfg.get("play_res_y", 1080)),
        font_name=str(ass_cfg.get("font_name", "Arial")),
        font_size=int(ass_cfg.get("font_size", 54)),
        margin_v=int(ass_cfg.get("margin_v", 45)),
        line_gap=int(ass_cfg.get("line_gap", 72)),
        max_tracks=int(ass_cfg.get("max_tracks", 3)),
    )

    write_txt(txt_path, final_segments)
    write_debug_tsv(debug_path, final_segments)

    log(f"[OK] ASS   : {ass_path}")
    log(f"[OK] TXT   : {txt_path}")
    log(f"[OK] DEBUG : {debug_path}")

    burned_video = maybe_download_and_burn(
        youtube_url,
        video_id,
        ass_path,
        output_dir,
        config,
    )

    if burned_video:
        log(f"[OK] VIDEO : {burned_video}")


if __name__ == "__main__":
    main()