#!/usr/bin/env python3
import argparse
import re
import shutil
import textwrap
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import ffmpeg


@dataclass
class Segment:
    start: float
    end: float
    text: str
    kind: str  # "speech"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate SRT subtitles from lyrics/text using a stems-driven timeline. "
            "The SRT timing is determined by comparing isolated vocals and instrumental stems. "
            "No aeneas alignment is used."
        )
    )
    parser.add_argument("-V", "--video-input", required=True, help="Input video path")
    parser.add_argument("-T", "--text-input", required=True, help="Input text path")
    parser.add_argument("-S", "--subtitle-output", default="output.srt", help="Output SRT path")
    parser.add_argument("-X", "--video-output", default="output_subtitled.mp4", help="Output subtitled video path")

    parser.add_argument("--audio-wav", default="audio_16k_mono.wav", help="Temporary extracted WAV path")
    parser.add_argument("--vocals-wav", default="vocals_16k_mono.wav", help="Temporary isolated vocals WAV path")
    parser.add_argument("--instrumental-wav", default="instrumental_16k_mono.wav", help="Temporary isolated instrumental WAV path")
    parser.add_argument("--spleeter-output-dir", default="spleeter_output", help="Temporary Spleeter output directory")
    parser.add_argument("--clean-text", default="cleaned_text.txt", help="Temporary cleaned text path")

    parser.add_argument(
        "--translate-to",
        default=None,
        help="Target language for subtitle translation, e.g. en, fr, es, de, it",
    )

    parser.add_argument(
        "--vocal-dominance-threshold",
        type=float,
        default=1.00,
        help="Minimum vocals/instrumental RMS ratio to consider a frame as vocal-dominant",
    )
    parser.add_argument(
        "--analysis-window",
        type=float,
        default=0.1,
        help="Analysis window in seconds for vocals/instrumental comparison",
    )
    parser.add_argument(
        "--min-vocal-region",
        type=float,
        default=3.0,
        help="Minimum vocal-dominant region duration in seconds",
    )
    parser.add_argument(
        "--merge-gap",
        type=float,
        default=0.18,
        help="Merge neighboring vocal regions separated by gaps shorter than this",
    )
    parser.add_argument(
        "--min-text-chars",
        type=int,
        default=28,
        help="Minimum chars before keeping a chunk isolated; smaller chunks are merged with neighbors",
    )

    parser.add_argument(
        "--chars-per-second",
        type=float,
        default=14.0,
        help="Target characters per second used to compute dynamic subtitle width",
    )
    parser.add_argument(
        "--max-lines-per-subtitle",
        type=int,
        default=1,
        help="Maximum number of lines per subtitle block",
    )

    parser.add_argument(
        "--subtitle-offset",
        type=float,
        default=0.0,
        help="Global subtitle offset in seconds. Positive delays subtitles, negative advances them.",
    )

    parser.add_argument(
        "--style",
        default="FontName=Arial,FontSize=18,BorderStyle=3,Outline=1,Shadow=0,MarginV=20",
        help="ASS force_style string for burned subtitles",
    )
    parser.add_argument("--keep-temp", action="store_true", help="Keep intermediate files")
    parser.add_argument("--dump-vocal-regions", default=None, help="Optional path to dump detected vocal regions")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    return parser.parse_args()


def log(msg: str, verbose: bool = True) -> None:
    if verbose:
        print(msg)


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\[[^\]]*\]", "", text)
    text = text.replace("(", "").replace(")", "")

    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]

    text = "\n".join(lines)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text_for_subtitles(text: str) -> List[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    chunks = re.split(r"\n", text)
    return [c.strip() for c in chunks if c.strip()]


def translate_fragments(
    fragments: List[str],
    target_lang: str,
    verbose: bool = False,
) -> List[str]:
    from deep_translator import GoogleTranslator

    translator = GoogleTranslator(source="auto", target=target_lang)
    translated: List[str] = []

    for text in fragments:
        try:
            out = translator.translate(text)
            if not out:
                out = text
            if verbose and out != text:
                print(f"[TR] {text} -> {out}")
        except Exception as e:
            if verbose:
                print(f"[TR][WARN] translation failed for: {text!r} ({e})")
            out = text

        translated.append(out)

    return translated


def write_text_file(text: str, path: Path) -> None:
    path.write_text(text, encoding="utf-8")


def extract_audio_to_wav(video_input: Path, wav_output: Path) -> None:
    try:
        (
            ffmpeg
            .input(str(video_input))
            .output(
                str(wav_output),
                vn=None,
                ac=1,
                ar=16000,
                acodec="pcm_s16le",
            )
            .overwrite_output()
            .run(quiet=True)
        )
    except ffmpeg.Error as e:
        stderr = e.stderr.decode("utf-8", errors="replace") if e.stderr else "<empty>"
        raise RuntimeError(f"FFmpeg audio extraction failed:\n{stderr}") from e


def convert_wav_to_mono_16k(input_wav: Path, output_wav: Path) -> None:
    try:
        (
            ffmpeg
            .input(str(input_wav))
            .output(
                str(output_wav),
                ac=1,
                ar=16000,
                acodec="pcm_s16le",
            )
            .overwrite_output()
            .run(quiet=True)
        )
    except ffmpeg.Error as e:
        stderr = e.stderr.decode("utf-8", errors="replace") if e.stderr else "<empty>"
        raise RuntimeError(f"FFmpeg mono conversion failed for {input_wav}:\n{stderr}") from e


def isolate_stems_with_spleeter(audio_wav: Path, vocals_wav: Path, instrumental_wav: Path, output_dir: Path) -> None:
    from spleeter.separator import Separator

    output_dir.mkdir(parents=True, exist_ok=True)

    separator = Separator("spleeter:2stems", multiprocess=False)
    separator.separate_to_file(str(audio_wav), str(output_dir))

    stem_dir = output_dir / audio_wav.stem
    generated_vocals = stem_dir / "vocals.wav"
    generated_instr = stem_dir / "accompaniment.wav"

    if not generated_vocals.exists():
        raise RuntimeError(f"Spleeter did not generate expected vocals file: {generated_vocals}")
    if not generated_instr.exists():
        raise RuntimeError(f"Spleeter did not generate expected accompaniment file: {generated_instr}")

    convert_wav_to_mono_16k(generated_vocals, vocals_wav)
    convert_wav_to_mono_16k(generated_instr, instrumental_wav)


def load_audio_mono_16k(path: Path):
    import numpy as np

    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()

        if sampwidth != 2:
            raise RuntimeError(f"Unsupported sample width for {path}: {sampwidth}")

        raw = wf.readframes(n_frames)
        audio = np.frombuffer(raw, dtype=np.int16).astype("float32") / 32768.0

        if n_channels <= 0:
            raise RuntimeError(f"Invalid number of channels for {path}: {n_channels}")

        if n_channels > 1:
            if len(audio) % n_channels != 0:
                raise RuntimeError(
                    f"Invalid PCM frame layout for {path}: "
                    f"samples={len(audio)}, channels={n_channels}"
                )
            audio = audio.reshape(-1, n_channels).mean(axis=1)

    return audio, framerate


def get_wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        n_frames = wf.getnframes()
        framerate = wf.getframerate()
        if framerate <= 0:
            raise RuntimeError(f"Invalid sample rate in {path}: {framerate}")
        return n_frames / float(framerate)


def rms_energy(signal, start_idx: int, end_idx: int) -> float:
    import numpy as np

    if end_idx <= start_idx:
        return 0.0

    chunk = signal[start_idx:end_idx]
    if len(chunk) == 0:
        return 0.0

    return float((np.mean(np.square(chunk)) + 1e-12) ** 0.5)


def merge_intervals(intervals: List[Tuple[float, float]], gap_tolerance: float = 0.0) -> List[Tuple[float, float]]:
    if not intervals:
        return []

    intervals = sorted(intervals, key=lambda x: x[0])
    merged: List[Tuple[float, float]] = [intervals[0]]

    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + gap_tolerance:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    return merged


def build_vocal_regions_from_stems(
    vocals_signal,
    instrumental_signal,
    sr: int,
    total_duration: float,
    analysis_window: float,
    vocal_dominance_threshold: float,
    min_vocal_region: float,
    merge_gap: float,
) -> List[Tuple[float, float]]:
    raw_regions: List[Tuple[float, float]] = []

    step = max(1, int(round(analysis_window * sr)))
    total_samples = min(len(vocals_signal), len(instrumental_signal))
    if total_samples <= 0:
        return []

    current_start: Optional[float] = None
    pos = 0

    while pos < total_samples:
        end = min(total_samples, pos + step)

        v = rms_energy(vocals_signal, pos, end)
        i = rms_energy(instrumental_signal, pos, end)
        ratio = v / (i + 1e-9)

        t0 = pos / sr

        if ratio >= vocal_dominance_threshold:
            if current_start is None:
                current_start = t0
        else:
            if current_start is not None:
                raw_regions.append((current_start, t0))
                current_start = None

        pos = end

    if current_start is not None:
        raw_regions.append((current_start, total_samples / sr))

    merged = merge_intervals(raw_regions, gap_tolerance=merge_gap)

    filtered: List[Tuple[float, float]] = []
    for start, end in merged:
        if end - start >= min_vocal_region:
            filtered.append((max(0.0, start), min(total_duration, end)))

    return filtered


def split_text_proportionally(text: str, durations: List[float]) -> List[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return [""] * len(durations)

    if not durations:
        return []

    if len(durations) == 1:
        return [cleaned]

    words = cleaned.split()
    if not words:
        return [""] * len(durations)

    total_duration = sum(max(0.0, d) for d in durations)
    if total_duration <= 0:
        chunk_size = max(1, round(len(words) / len(durations)))
        chunks: List[str] = []
        pos = 0
        for idx in range(len(durations)):
            if idx == len(durations) - 1:
                chunk_words = words[pos:]
            else:
                chunk_words = words[pos:pos + chunk_size]
            chunks.append(" ".join(chunk_words).strip())
            pos += chunk_size
        return chunks

    targets: List[int] = []
    remaining_words = len(words)
    remaining_duration = total_duration

    for idx, d in enumerate(durations):
        if idx == len(durations) - 1:
            count = remaining_words
        else:
            ratio = max(0.0, d) / remaining_duration if remaining_duration > 0 else 0.0
            count = max(1, int(round(remaining_words * ratio)))
            max_allowed = remaining_words - (len(durations) - idx - 1)
            count = min(count, max_allowed)

        targets.append(count)
        remaining_words -= count
        remaining_duration -= max(0.0, d)

    chunks: List[str] = []
    pos = 0
    for count in targets:
        chunk_words = words[pos:pos + count]
        chunks.append(" ".join(chunk_words).strip())
        pos += count

    return chunks


def merge_subregions_by_text_size(
    sub_regions: List[Tuple[float, float]],
    text_chunks: List[str],
    min_text_chars: int,
) -> Tuple[List[Tuple[float, float]], List[str]]:
    if not sub_regions or not text_chunks or len(sub_regions) != len(text_chunks):
        return sub_regions, text_chunks

    merged_regions: List[Tuple[float, float]] = []
    merged_texts: List[str] = []

    cur_start, cur_end = sub_regions[0]
    cur_text = text_chunks[0].strip()

    for (r0, r1), txt in zip(sub_regions[1:], text_chunks[1:]):
        txt = txt.strip()

        if len(cur_text) < min_text_chars:
            cur_end = r1
            cur_text = f"{cur_text} {txt}".strip()
        else:
            merged_regions.append((cur_start, cur_end))
            merged_texts.append(cur_text)
            cur_start, cur_end = r0, r1
            cur_text = txt

    merged_regions.append((cur_start, cur_end))
    merged_texts.append(cur_text)

    if len(merged_texts) >= 2 and len(merged_texts[-1]) < min_text_chars:
        prev_start, _ = merged_regions[-2]
        _, last_end = merged_regions[-1]
        merged_regions[-2] = (prev_start, last_end)
        merged_texts[-2] = f"{merged_texts[-2]} {merged_texts[-1]}".strip()
        merged_regions.pop()
        merged_texts.pop()

    return merged_regions, merged_texts


def merge_speech_segments(segments: List[Segment], tolerance: float = 0.05) -> List[Segment]:
    if not segments:
        return []

    segments = sorted(segments, key=lambda s: s.start)
    merged = [segments[0]]

    for seg in segments[1:]:
        prev = merged[-1]
        if seg.start <= prev.end + tolerance:
            merged_text = f"{prev.text.replace(chr(10), ' ')} {seg.text.replace(chr(10), ' ')}".strip()
            merged[-1] = Segment(
                start=prev.start,
                end=max(prev.end, seg.end),
                text=merged_text,
                kind="speech",
            )
        else:
            merged.append(seg)

    return merged


def wrap_text_dynamic(
    text: str,
    start: float,
    end: float,
    chars_per_second: float,
    max_lines: int,
) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return ""

    duration = max(0.1, end - start)
    target_width = max(1, int(duration * chars_per_second))
    width = max(8, min(target_width, len(cleaned)))

    lines = textwrap.wrap(
        cleaned,
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )

    if not lines:
        return cleaned

    if len(lines) <= max_lines:
        return "\n".join(lines)

    words = cleaned.split()
    if not words:
        return cleaned

    target_total = len(cleaned)
    target_per_line = max(1, target_total // max_lines)

    buckets: List[List[str]] = [[]]
    current_len = 0

    for word in words:
        projected = current_len + (1 if current_len else 0) + len(word)
        used_words = sum(len(b) for b in buckets)
        remaining_words = len(words) - used_words - 1
        remaining_lines = max_lines - len(buckets)

        if (
            buckets[-1]
            and projected > target_per_line
            and len(buckets) < max_lines
            and remaining_words >= remaining_lines
        ):
            buckets.append([word])
            current_len = len(word)
        else:
            buckets[-1].append(word)
            current_len = projected

    merged_lines = [" ".join(bucket).strip() for bucket in buckets if bucket]

    if len(merged_lines) > max_lines:
        head = merged_lines[: max_lines - 1]
        tail = " ".join(merged_lines[max_lines - 1:]).strip()
        merged_lines = head + [tail]

    return "\n".join(merged_lines)


def allocate_region_counts_by_text_weight(
    fragments: List[str],
    n_regions: int,
) -> List[int]:
    if not fragments or n_regions <= 0:
        return []

    weights = [max(1, len(re.sub(r"\s+", " ", frag).strip())) for frag in fragments]
    total_weight = sum(weights)

    assigned_counts: List[int] = []
    remaining_regions = n_regions
    remaining_weight = total_weight

    for i, weight in enumerate(weights):
        if i == len(weights) - 1:
            count = remaining_regions
        else:
            ratio = weight / remaining_weight if remaining_weight > 0 else 0.0
            count = max(1, int(round(remaining_regions * ratio)))
            max_allowed = remaining_regions - (len(fragments) - i - 1)
            count = min(count, max_allowed)

        assigned_counts.append(count)
        remaining_regions -= count
        remaining_weight -= weight

    return assigned_counts


def normalize_fragments_to_regions(
    fragments: List[str],
    vocal_regions: List[Tuple[float, float]],
) -> List[str]:
    if not fragments:
        return []
    if len(fragments) <= len(vocal_regions):
        return fragments[:]

    keep = len(vocal_regions)
    if keep <= 0:
        return []

    head = fragments[:keep - 1]
    tail = " ".join(fragments[keep - 1:]).strip()
    return head + [tail]


def build_segments_from_vocal_timeline(
    fragments: List[str],
    vocal_regions: List[Tuple[float, float]],
    total_duration: float,
    chars_per_second: float,
    max_lines_per_subtitle: int,
    min_text_chars: int,
) -> List[Segment]:
    speech_segments: List[Segment] = []

    if not fragments or not vocal_regions:
        return []

    fragments = normalize_fragments_to_regions(fragments, vocal_regions)
    assigned_counts = allocate_region_counts_by_text_weight(fragments, len(vocal_regions))

    pos = 0
    for fragment, count in zip(fragments, assigned_counts):
        assigned_regions = vocal_regions[pos:pos + count]
        pos += count

        if not assigned_regions:
            continue

        durations = [max(0.0, r1 - r0) for r0, r1 in assigned_regions]
        text_chunks = split_text_proportionally(fragment, durations)

        assigned_regions, text_chunks = merge_subregions_by_text_size(
            sub_regions=assigned_regions,
            text_chunks=text_chunks,
            min_text_chars=min_text_chars,
        )

        for (start, end), chunk_text in zip(assigned_regions, text_chunks):
            start = max(0.0, start)
            end = min(total_duration, end)

            if end <= start:
                continue

            chunk_text = re.sub(r"\s+", " ", chunk_text).strip()
            if not chunk_text:
                continue

            formatted_text = wrap_text_dynamic(
                text=chunk_text,
                start=start,
                end=end,
                chars_per_second=chars_per_second,
                max_lines=max_lines_per_subtitle,
            )

            if formatted_text:
                speech_segments.append(
                    Segment(
                        start=start,
                        end=end,
                        text=formatted_text,
                        kind="speech",
                    )
                )

    return merge_speech_segments(speech_segments, tolerance=0.05)


def apply_offset(segments: List[Segment], offset: float, total_duration: float) -> List[Segment]:
    shifted: List[Segment] = []
    for seg in segments:
        start = max(0.0, seg.start + offset)
        end = min(total_duration, seg.end + offset)
        if end > start:
            shifted.append(Segment(start=start, end=end, text=seg.text, kind=seg.kind))
    return shifted


def format_srt_timestamp(seconds: float) -> str:
    total_ms = int(round(max(0.0, seconds) * 1000))
    hours = total_ms // 3_600_000
    minutes = (total_ms % 3_600_000) // 60_000
    secs = (total_ms % 60_000) // 1000
    millis = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_srt(segments: List[Segment], srt_path: Path) -> None:
    with srt_path.open("w", encoding="utf-8") as f:
        for idx, seg in enumerate(segments, start=1):
            f.write(f"{idx}\n")
            log(f'idx => {idx}')
            f.write(f"{format_srt_timestamp(seg.start)} --> {format_srt_timestamp(seg.end)}\n")
            log(f"{format_srt_timestamp(seg.start)} --> {format_srt_timestamp(seg.end)}\n")
            f.write(seg.text.strip() + "\n\n")
            log(seg.text.strip() + "\n\n")


def dump_regions(regions: List[Tuple[float, float]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for start, end in regions:
            f.write(f"{start:.3f}\t{end:.3f}\n")


def burn_subtitles(video_input: Path, subtitle_file: Path, video_output: Path, style: str) -> None:
    try:
        input_stream = ffmpeg.input(str(video_input))
        video_stream = input_stream.video.filter("subtitles", str(subtitle_file), force_style=style)
        audio_stream = input_stream.audio

        (
            ffmpeg
            .output(
                video_stream,
                audio_stream,
                str(video_output),
                vcodec="libx264",
                acodec="copy",
            )
            .overwrite_output()
            .run(quiet=True)
        )
    except ffmpeg.Error as e:
        stderr = e.stderr.decode("utf-8", errors="replace") if e.stderr else "<empty>"
        raise RuntimeError(f"FFmpeg subtitle burn failed:\n{stderr}") from e


def remove_files(paths: Iterable[Path]) -> None:
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass


def remove_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        try:
            shutil.rmtree(path, ignore_errors=True)
        except Exception:
            pass


def main() -> int:
    args = parse_args()

    video_input = Path(args.video_input)
    text_input = Path(args.text_input)
    subtitle_output = Path(args.subtitle_output)
    video_output = Path(args.video_output)

    audio_wav = Path(args.audio_wav)
    vocals_wav = Path(args.vocals_wav)
    instrumental_wav = Path(args.instrumental_wav)
    spleeter_output_dir = Path(args.spleeter_output_dir)
    clean_text_path = Path(args.clean_text)

    if not video_input.exists():
        raise FileNotFoundError(f"Video not found: {video_input}")
    if not text_input.exists():
        raise FileNotFoundError(f"Text not found: {text_input}")

    raw_text = text_input.read_text(encoding="utf-8")
    cleaned = clean_text(raw_text)
    if not cleaned:
        raise RuntimeError("Input text is empty after cleaning.")

    fragments = split_text_for_subtitles(cleaned)
    if not fragments:
        raise RuntimeError("No text fragments could be built from the input text.")

    if args.translate_to:
        log(f"[TR] Translating fragments to {args.translate_to}...", args.verbose)
        fragments = translate_fragments(
            fragments=fragments,
            target_lang=args.translate_to,
            verbose=args.verbose,
        )

    log("[1/7] Writing cleaned text...", args.verbose)
    write_text_file(cleaned, clean_text_path)

    log("[2/7] Extracting audio to 16 kHz mono WAV...", args.verbose)
    extract_audio_to_wav(video_input, audio_wav)

    log("[3/7] Isolating stems with Spleeter...", args.verbose)
    isolate_stems_with_spleeter(audio_wav, vocals_wav, instrumental_wav, spleeter_output_dir)

    log("[4/7] Loading vocals/instrumental stems...", args.verbose)
    vocals_signal, sr_v = load_audio_mono_16k(vocals_wav)
    instrumental_signal, sr_i = load_audio_mono_16k(instrumental_wav)
    if sr_v != sr_i:
        raise RuntimeError(f"Sample rate mismatch: vocals={sr_v}, instrumental={sr_i}")

    log("[5/7] Building vocal timeline from vocals vs instrumental comparison...", args.verbose)
    total_duration = get_wav_duration(instrumental_wav)
    vocal_regions = build_vocal_regions_from_stems(
        vocals_signal=vocals_signal,
        instrumental_signal=instrumental_signal,
        sr=sr_v,
        total_duration=total_duration,
        analysis_window=args.analysis_window,
        vocal_dominance_threshold=args.vocal_dominance_threshold,
        min_vocal_region=args.min_vocal_region,
        merge_gap=args.merge_gap,
    )

    if args.verbose:
        print(f"[DEBUG] fragments count: {len(fragments)}")
        print(f"[DEBUG] vocal_regions count: {len(vocal_regions)}")

    if args.dump_vocal_regions:
        dump_regions(vocal_regions, Path(args.dump_vocal_regions))

    if not vocal_regions:
        raise RuntimeError("No vocal regions detected from stem comparison.")

    log("[6/7] Projecting text onto stem-driven vocal timeline...", args.verbose)
    final_segments = build_segments_from_vocal_timeline(
        fragments=fragments,
        vocal_regions=vocal_regions,
        total_duration=total_duration,
        chars_per_second=args.chars_per_second,
        max_lines_per_subtitle=args.max_lines_per_subtitle,
        min_text_chars=args.min_text_chars,
    )

    final_segments = apply_offset(final_segments, args.subtitle_offset, total_duration)

    if not final_segments:
        raise RuntimeError("No SRT segments were generated from the vocal timeline.")

    log("[7/7] Writing SRT and burning subtitles...", args.verbose)
    write_srt(final_segments, subtitle_output)

    if not subtitle_output.exists():
        raise RuntimeError(f"SRT was not created: {subtitle_output}")

    burn_subtitles(video_input, subtitle_output, video_output, args.style)

    print(f"SRT created: {subtitle_output.resolve()}")
    print(f"Video created: {video_output.resolve()}")

    if not args.keep_temp:
        remove_files([
            audio_wav,
            vocals_wav,
            instrumental_wav,
            clean_text_path,
        ])
        remove_dirs([spleeter_output_dir])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())