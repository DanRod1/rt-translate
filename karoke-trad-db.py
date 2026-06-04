#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
import re
import subprocess
import sys
import tempfile
import traceback
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from rapidfuzz import fuzz
except Exception:
    fuzz = None

TRANSFORMERS_AVAILABLE = True
try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
except Exception:
    TRANSFORMERS_AVAILABLE = False
    AutoModelForSeq2SeqLM = None
    AutoTokenizer = None

YOUTUBE_TRANSCRIPT_API_AVAILABLE = True
try:
    from youtube_transcript_api import YouTubeTranscriptApi
except Exception:
    YOUTUBE_TRANSCRIPT_API_AVAILABLE = False
    YouTubeTranscriptApi = None


@dataclass
class CaptionSegment:
    id: int
    start: float
    end: float
    text: str
    source: str
    language: str


@dataclass
class MulanFragment:
    id: int
    text: str
    original_text: str
    translated_text: str
    translation_applied: bool
    tokens: List[str]
    weight: float
    source_line: int
    explicit_duration: Optional[float]
    sync_offset: float


@dataclass
class TimedSubtitle:
    id: int
    start: float
    end: float
    text: str
    fragment_id: int
    caption_id: Optional[int]
    source_line: int
    original_text: str
    translated_text: str
    translation_applied: bool
    explicit_duration: Optional[float]
    sync_offset: float
    match_score: float
    caption_text: str


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
        encoded = self.tokenizer(text, return_tensors="pt", truncation=True)
        generated = self.model.generate(**encoded, max_length=self.max_length)
        translated = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
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
    text = html.unescape(str(text))
    text = text.replace("…", "...")
    text = text.replace("’", "'")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,;:!?])", r"\1", text)
    return text.strip()


def normalize_for_match(text: str) -> str:
    text = normalize_text(text).lower()
    text = re.sub(r"[^\wÀ-ÿ' -]+", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_text(text: str) -> List[str]:
    return re.findall(r"\b[\wÀ-ÿ'-]+\b", text, flags=re.UNICODE)


def estimate_syllables_in_token(token: str) -> int:
    groups = re.findall(r"[aeiouyàâäéèêëîïôöùûüÿœæ]+", token.lower(), flags=re.UNICODE)
    return max(1, len(groups))


def estimate_weight(text: str) -> float:
    tokens = tokenize_text(normalize_for_match(text))
    if not tokens:
        return 1.0
    syllables = sum(estimate_syllables_in_token(t) for t in tokens)
    chars = sum(len(t) for t in tokens)
    stretch = sum(max(0, len(m.group(0)) - 1) for m in re.finditer(r"([aeiouyàâäéèêëîïôöùûüÿœæ])\1+", text.lower()))
    return max(1.0, float(syllables) + chars * 0.08 + stretch * 0.75)


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def is_parenthesized_text(text: str) -> bool:
    text = text.strip()
    return text.startswith("(") and text.endswith(")")


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
        return raw, metadata
    useful_text, comment = inner.split("--", 1)
    useful_text = useful_text.strip()
    comment = comment.strip()
    duration_match = re.search(r"\bdure\s*:\s*([+-]?\s*[0-9]+(?:[.,][0-9]+)?)", comment, flags=re.IGNORECASE)
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
    if not raw or raw.startswith("#") or raw.startswith("//"):
        return None
    if "=" in raw:
        key, value = raw.split("=", 1)
        if key.strip().lower() in {"text", "lyric", "lyrics", "line", "sentence", "phrase", "fragment", "content"}:
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


def should_translate_text(text: str, input_language: str, output_language: str, translation_model: str) -> bool:
    if not text.strip() or not output_language.strip() or not translation_model.strip():
        return False
    if input_language.strip().lower() and input_language.strip().lower() == output_language.strip().lower():
        return False
    return True


def create_local_text_translator(input_language: str, output_language: str, translation_model: str, translation_max_length: int, local_files_only: bool) -> Optional[LocalTextTranslator]:
    if not output_language.strip() or not translation_model.strip():
        return None
    if not TRANSFORMERS_AVAILABLE or AutoTokenizer is None or AutoModelForSeq2SeqLM is None:
        warn("transformers indisponible, traduction locale désactivée.")
        return None
    try:
        tokenizer = AutoTokenizer.from_pretrained(translation_model, local_files_only=local_files_only)
        model = AutoModelForSeq2SeqLM.from_pretrained(translation_model, local_files_only=local_files_only)
        model.eval()
        return LocalTextTranslator(translation_model, input_language, output_language, translation_max_length, tokenizer, model)
    except Exception as exc:
        warn(f"Impossible de charger le modèle de traduction locale: {exc}")
        return None


def translate_text_value(text: str, translator: Optional[LocalTextTranslator], input_language: str, output_language: str, translation_model: str) -> Tuple[str, bool]:
    if not should_translate_text(text, input_language, output_language, translation_model):
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


def parse_mulan_text(path: str, input_language: str, output_language: str, translation_model: str, translator: Optional[LocalTextTranslator]) -> List[MulanFragment]:
    raw = read_text_file(path)
    fragments: List[MulanFragment] = []
    for source_line, raw_line in enumerate(raw.splitlines(), start=1):
        payload = parse_mulan_payload_from_line(raw_line)
        if payload is None:
            continue
        payload_text, metadata = split_parenthesized_inline_metadata(payload)
        original_text = strip_mulan_markup(payload_text)
        if not original_text and not metadata:
            continue
        if original_text:
            translated_text, translation_applied = translate_text_value(original_text, translator, input_language, output_language, translation_model)
            display_text = translated_text
            tokens = tokenize_text(original_text)
            weight = estimate_weight(original_text)
        else:
            translated_text = ""
            translation_applied = False
            display_text = ""
            tokens = []
            weight = 1.0
        fragments.append(MulanFragment(
            id=len(fragments),
            text=display_text,
            original_text=original_text,
            translated_text=translated_text,
            translation_applied=translation_applied,
            tokens=tokens,
            weight=weight,
            source_line=source_line,
            explicit_duration=metadata.get("explicit_duration"),
            sync_offset=float(metadata.get("sync_offset", 0.0)),
        ))
    return fragments


def extract_youtube_video_id(url: str) -> str:
    raw = url.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", raw):
        return raw
    parsed = urllib.parse.urlparse(raw)
    host = parsed.netloc.lower()
    if "youtu.be" in host:
        vid = parsed.path.strip("/").split("/")[0]
        if vid:
            return vid
    qs = urllib.parse.parse_qs(parsed.query)
    if "v" in qs and qs["v"]:
        return qs["v"][0]
    m = re.search(r"/(?:shorts|embed|live)/([A-Za-z0-9_-]{11})", parsed.path)
    if m:
        return m.group(1)
    raise ValueError(f"Impossible d'extraire l'id YouTube depuis: {url}")


def run_cmd(cmd: Sequence[str], check: bool = False) -> subprocess.CompletedProcess[str]:
    log("[INFO] CMD: " + " ".join(cmd))
    result = subprocess.run(list(cmd), capture_output=True, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"Commande échouée: {' '.join(cmd)}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
    return result


def build_ytdlp_base_cmd(cookies_from_browser: str = "chrome", cookies: str = "", remote_components: str = "ejs:github", no_check_certificate: bool = False) -> List[str]:
    cmd = [sys.executable, "-m", "yt_dlp", "--no-playlist"]
    if cookies_from_browser:
        cmd += ["--cookies-from-browser", cookies_from_browser]
    if cookies:
        cmd += ["--cookies", cookies]
    if remote_components:
        cmd += ["--remote-components", remote_components]
    if no_check_certificate:
        cmd += ["--no-check-certificate"]
    return cmd


def export_cookies_txt(work_dir: str, cookies_from_browser: str, cookies: str, remote_components: str, no_check_certificate: bool) -> Optional[str]:
    if cookies and os.path.isfile(cookies):
        return cookies
    if not cookies_from_browser:
        return None
    ensure_dir(work_dir)
    cookie_path = os.path.join(work_dir, "yt_cookies.txt")
    cmd = build_ytdlp_base_cmd(cookies_from_browser, "", remote_components, no_check_certificate) + [
        "--cookies", cookie_path,
        "--skip-download",
        "--simulate",
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    ]
    result = run_cmd(cmd, check=False)
    if not os.path.isfile(cookie_path):
        warn(f"Export cookies yt-dlp non disponible. STDERR: {result.stderr.strip()[:500]}")
        return None
    return cookie_path


YOUTUBE_COOKIE_NAMES = {
    "SID",
    "__Secure-1PSID",
    "__Secure-3PSID",
    "HSID",
    "SSID",
    "APISID",
    "SAPISID",
    "__Secure-1PAPISID",
    "__Secure-3PAPISID",
    "LOGIN_INFO",
    "VISITOR_INFO1_LIVE",
    "VISITOR_PRIVACY_METADATA",
    "PREF",
    "YSC",
    "CONSENT",
    "SOCS",
}


def build_filtered_youtube_cookie_header(cookie_path: Optional[str], max_cookie_header_bytes: int = 12000) -> str:
    """Build a small Cookie header from a Netscape cookie file.

    Exported Chrome profiles can contain thousands of cookies. Sending all of them
    to YouTube often triggers HTTP 413. Keep only the cookies useful for YouTube
    auth/session/caption access.
    """
    if not cookie_path or not os.path.isfile(cookie_path):
        return ""

    selected: List[Tuple[int, str, str]] = []
    with open(cookie_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if not line.strip():
                continue
            if line.startswith("#HttpOnly_"):
                line = line.replace("#HttpOnly_", "", 1)
            elif line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) < 7:
                continue

            domain = parts[0].lower()
            name = parts[5]
            value = parts[6]

            if not value:
                continue
            if "youtube.com" not in domain and "google.com" not in domain:
                continue
            if name not in YOUTUBE_COOKIE_NAMES:
                continue

            # Auth/session cookies first, volatile visitor cookies later.
            priority = 10
            if "SID" in name or "SAPISID" in name or name in {"LOGIN_INFO", "CONSENT", "SOCS"}:
                priority = 0
            elif name in {"VISITOR_INFO1_LIVE", "VISITOR_PRIVACY_METADATA", "PREF", "YSC"}:
                priority = 5
            selected.append((priority, name, value))

    # De-duplicate by cookie name, keeping the first highest-priority occurrence.
    selected.sort(key=lambda item: item[0])
    seen = set()
    pairs: List[str] = []
    total = 0
    for _priority, name, value in selected:
        if name in seen:
            continue
        pair = f"{name}={value}"
        pair_len = len(pair.encode("utf-8")) + (2 if pairs else 0)
        if total + pair_len > max_cookie_header_bytes:
            warn(f"Cookie header limité à {max_cookie_header_bytes} octets pour éviter HTTP 413.")
            break
        pairs.append(pair)
        seen.add(name)
        total += pair_len

    if pairs:
        log(f"[INFO] Cookie header filtré: {len(pairs)} cookies, {total} octets")
    return "; ".join(pairs)


def http_get_text(url: str, cookie_path: Optional[str] = None, timeout: int = 30) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0 Safari/537.36",
        "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7,es;q=0.6",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/json;q=0.8,*/*;q=0.7",
    }
    cookie_header = build_filtered_youtube_cookie_header(cookie_path)
    if cookie_header:
        headers["Cookie"] = cookie_header
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
        charset = resp.headers.get_content_charset() or "utf-8"
        return data.decode(charset, errors="replace")


def extract_balanced_json_after_marker(page: str, marker: str) -> Optional[Dict[str, Any]]:
    idx = page.find(marker)
    if idx < 0:
        return None
    eq = page.find("=", idx)
    if eq < 0:
        return None
    start = page.find("{", eq)
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(page)):
        ch = page[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    raw = page[start:i + 1]
                    try:
                        return json.loads(raw)
                    except Exception:
                        return None
    return None


def find_caption_tracks(obj: Any) -> List[Dict[str, Any]]:
    tracks: List[Dict[str, Any]] = []
    def walk(x: Any) -> None:
        if isinstance(x, dict):
            if "captionTracks" in x and isinstance(x["captionTracks"], list):
                for item in x["captionTracks"]:
                    if isinstance(item, dict) and item.get("baseUrl"):
                        tracks.append(item)
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(obj)
    return tracks


def track_language_code(track: Dict[str, Any]) -> str:
    return str(track.get("languageCode") or track.get("vssId") or "").lstrip(".")


def track_name(track: Dict[str, Any]) -> str:
    name = track.get("name")
    if isinstance(name, dict):
        if "simpleText" in name:
            return str(name["simpleText"])
        runs = name.get("runs")
        if isinstance(runs, list):
            return "".join(str(r.get("text", "")) for r in runs if isinstance(r, dict))
    return str(name or "")


def score_track(track: Dict[str, Any], preferred_langs: List[str]) -> Tuple[int, int, str]:
    lang = track_language_code(track)
    kind = str(track.get("kind", ""))
    is_asr = 1 if kind == "asr" else 0
    lang_rank = 999
    for i, pref in enumerate(preferred_langs):
        p = pref.replace(".*", "")
        if pref == "all" or lang == p or lang.startswith(p + "-"):
            lang_rank = i
            break
    return (lang_rank, -is_asr, lang)


def caption_url_with_format(base_url: str, fmt: str = "json3") -> str:
    parsed = urllib.parse.urlparse(base_url)
    qs = urllib.parse.parse_qs(parsed.query)
    qs["fmt"] = [fmt]
    new_query = urllib.parse.urlencode(qs, doseq=True)
    return urllib.parse.urlunparse(parsed._replace(query=new_query))


def parse_youtube_json3(raw: str, language: str, source: str) -> List[CaptionSegment]:
    data = json.loads(raw)
    events = data.get("events", [])
    segments: List[CaptionSegment] = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        if "segs" not in ev:
            continue
        start = float(ev.get("tStartMs", 0)) / 1000.0
        dur = float(ev.get("dDurationMs", 0)) / 1000.0
        text = "".join(str(seg.get("utf8", "")) for seg in ev.get("segs", []) if isinstance(seg, dict))
        text = normalize_text(text.replace("\n", " "))
        if not text:
            continue
        end = start + max(0.05, dur)
        segments.append(CaptionSegment(len(segments), start, end, text, source, language))
    for i in range(len(segments) - 1):
        if segments[i].end <= segments[i].start or segments[i].end > segments[i + 1].start:
            segments[i].end = max(segments[i].start + 0.05, segments[i + 1].start)
    return segments


def parse_srt_time(value: str) -> float:
    value = value.strip().replace(",", ".")
    m = re.match(r"(?:(\d+):)?(\d{2}):(\d{2})\.(\d{1,3})", value)
    if not m:
        return 0.0
    h = int(m.group(1) or 0)
    minute = int(m.group(2))
    sec = int(m.group(3))
    ms = int(m.group(4).ljust(3, "0")[:3])
    return h * 3600 + minute * 60 + sec + ms / 1000.0


def parse_srt_or_vtt(path: str, language: str, source: str) -> List[CaptionSegment]:
    raw = read_text_file(path)
    raw = raw.replace("\ufeff", "")
    blocks = re.split(r"\n\s*\n", raw.strip(), flags=re.MULTILINE)
    segments: List[CaptionSegment] = []
    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        time_idx = None
        for idx, ln in enumerate(lines):
            if "-->" in ln:
                time_idx = idx
                break
        if time_idx is None:
            continue
        left, right = lines[time_idx].split("-->", 1)
        start = parse_srt_time(left)
        right = right.split()[0]
        end = parse_srt_time(right)
        text = normalize_text(" ".join(lines[time_idx + 1:]))
        text = re.sub(r"<[^>]+>", " ", text)
        text = normalize_text(text)
        if text and end > start:
            segments.append(CaptionSegment(len(segments), start, end, text, source, language))
    return segments


def get_captiontracks_direct(youtube_url: str, work_dir: str, preferred_langs: List[str], cookies_from_browser: str, cookies: str, remote_components: str, no_check_certificate: bool) -> Tuple[List[CaptionSegment], Dict[str, Any]]:
    ensure_dir(work_dir)
    cookie_path = export_cookies_txt(work_dir, cookies_from_browser, cookies, remote_components, no_check_certificate)
    page = http_get_text(youtube_url, cookie_path=cookie_path)
    page_path = os.path.join(work_dir, "youtube_watch_page.html")
    with open(page_path, "w", encoding="utf-8") as f:
        f.write(page)

    player = extract_balanced_json_after_marker(page, "ytInitialPlayerResponse")
    tracks: List[Dict[str, Any]] = []
    if player:
        tracks = find_caption_tracks(player)
    if not tracks:
        # Fallback regex: parfois la JSON est échappée dans la page.
        unescaped = page.encode("utf-8", "ignore").decode("unicode_escape", errors="ignore")
        player = extract_balanced_json_after_marker(unescaped, "ytInitialPlayerResponse")
        if player:
            tracks = find_caption_tracks(player)

    tracks_debug = []
    for t in tracks:
        tracks_debug.append({
            "languageCode": track_language_code(t),
            "kind": t.get("kind", ""),
            "name": track_name(t),
            "isTranslatable": t.get("isTranslatable"),
            "vssId": t.get("vssId"),
        })
    with open(os.path.join(work_dir, "direct_caption_tracks.json"), "w", encoding="utf-8") as f:
        json.dump(tracks_debug, f, ensure_ascii=False, indent=2)

    if not tracks:
        return [], {"source": "direct_captiontracks", "page_path": page_path, "cookie_path": cookie_path, "tracks": []}

    tracks = sorted(tracks, key=lambda t: score_track(t, preferred_langs))
    errors: List[str] = []
    for track in tracks:
        lang = track_language_code(track)
        if score_track(track, preferred_langs)[0] >= 999 and "all" not in preferred_langs:
            continue
        for fmt in ["json3", "srv3", "vtt"]:
            url = caption_url_with_format(str(track["baseUrl"]), fmt=fmt)
            try:
                raw = http_get_text(url, cookie_path=cookie_path)
                out_path = os.path.join(work_dir, f"direct_caption_{lang or 'unknown'}.{fmt}")
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(raw)
                if fmt == "json3":
                    segments = parse_youtube_json3(raw, lang, "direct_captiontracks_json3")
                elif fmt == "vtt":
                    segments = parse_srt_or_vtt(out_path, lang, "direct_captiontracks_vtt")
                else:
                    continue
                if segments:
                    return segments, {"source": "direct_captiontracks", "track": {"language": lang, "kind": track.get("kind"), "name": track_name(track)}, "track_count": len(tracks), "raw_path": out_path, "cookie_path": cookie_path}
            except Exception as exc:
                errors.append(f"{lang}/{fmt}: {exc}")
    return [], {"source": "direct_captiontracks", "track_count": len(tracks), "errors": errors, "cookie_path": cookie_path}


def get_captiontracks_via_ytdlp_dumpjson(youtube_url: str, work_dir: str, preferred_langs: List[str], cookies_from_browser: str, cookies: str, remote_components: str, no_check_certificate: bool) -> Tuple[List[CaptionSegment], Dict[str, Any]]:
    ensure_dir(work_dir)
    cmd = build_ytdlp_base_cmd(cookies_from_browser, cookies, remote_components, no_check_certificate) + ["--dump-json", youtube_url]
    result = run_cmd(cmd, check=False)
    with open(os.path.join(work_dir, "yt_dlp_dump_json_stdout.txt"), "w", encoding="utf-8") as f:
        f.write(result.stdout)
    with open(os.path.join(work_dir, "yt_dlp_dump_json_stderr.txt"), "w", encoding="utf-8") as f:
        f.write(result.stderr)
    if result.returncode != 0 or not result.stdout.strip():
        return [], {"source": "ytdlp_dumpjson_caption_url", "returncode": result.returncode, "stderr": result.stderr[-2000:]}
    try:
        info = json.loads(result.stdout.splitlines()[-1])
    except Exception as exc:
        return [], {"source": "ytdlp_dumpjson_caption_url", "error": str(exc)}
    pools = []
    for key in ["automatic_captions", "subtitles"]:
        value = info.get(key) or {}
        if isinstance(value, dict):
            for lang, items in value.items():
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict) and item.get("url"):
                            pools.append((key, lang, item))
    def pool_score(item: Tuple[str, str, Dict[str, Any]]) -> Tuple[int, int, str]:
        key, lang, data = item
        lang_rank = 999
        for i, pref in enumerate(preferred_langs):
            p = pref.replace(".*", "")
            if pref == "all" or lang == p or lang.startswith(p + "-"):
                lang_rank = i
                break
        ext_rank = 0 if data.get("ext") in {"json3", "srv3", "vtt", "srt"} else 5
        return (lang_rank, ext_rank, lang)
    pools.sort(key=pool_score)
    errors = []
    for key, lang, item in pools:
        if pool_score((key, lang, item))[0] >= 999 and "all" not in preferred_langs:
            continue
        url = item["url"]
        ext = item.get("ext") or "json3"
        try:
            raw = http_get_text(url)
            out_path = os.path.join(work_dir, f"ytdlp_caption_url_{lang}.{ext}")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(raw)
            if ext == "json3":
                segs = parse_youtube_json3(raw, lang, f"ytdlp_{key}_json3")
            elif ext in {"vtt", "srt"}:
                segs = parse_srt_or_vtt(out_path, lang, f"ytdlp_{key}_{ext}")
            else:
                continue
            if segs:
                return segs, {"source": "ytdlp_dumpjson_caption_url", "pool": key, "language": lang, "raw_path": out_path}
        except Exception as exc:
            errors.append(f"{lang}/{ext}: {exc}")
    return [], {"source": "ytdlp_dumpjson_caption_url", "pools": len(pools), "errors": errors}


def get_captions_youtube_transcript_api(video_id: str, preferred_langs: List[str]) -> Tuple[List[CaptionSegment], Dict[str, Any]]:
    if not YOUTUBE_TRANSCRIPT_API_AVAILABLE or YouTubeTranscriptApi is None:
        return [], {"source": "youtube_transcript_api", "available": False}
    langs = []
    for pref in preferred_langs:
        if pref == "all":
            continue
        base = pref.replace(".*", "")
        if base and base not in langs:
            langs.append(base)
    if not langs:
        langs = ["es", "fr", "en"]
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=langs)
        segments = []
        for item in transcript:
            start = float(item.get("start", 0.0))
            dur = float(item.get("duration", 0.0))
            text = normalize_text(item.get("text", ""))
            if text:
                segments.append(CaptionSegment(len(segments), start, start + max(0.05, dur), text, "youtube_transcript_api", langs[0]))
        return segments, {"source": "youtube_transcript_api", "languages": langs}
    except Exception as exc:
        return [], {"source": "youtube_transcript_api", "error": str(exc)}


def get_captions_ytdlp_write_subs(youtube_url: str, output_dir: str, subtitle_lang_expr: str, cookies_from_browser: str, cookies: str, remote_components: str, no_check_certificate: bool) -> Tuple[List[CaptionSegment], Dict[str, Any]]:
    work_dir = os.path.join(output_dir, "work")
    ensure_dir(work_dir)
    output_template = os.path.join(work_dir, "youtube_source.%(ext)s")
    for mode in ["--write-subs", "--write-auto-subs"]:
        cmd = build_ytdlp_base_cmd(cookies_from_browser, cookies, remote_components, no_check_certificate) + [
            "--skip-download", "--sub-langs", subtitle_lang_expr, "--convert-subs", "srt", "-o", output_template, mode, youtube_url,
        ]
        result = run_cmd(cmd, check=False)
        with open(os.path.join(work_dir, f"yt_dlp_{mode.strip('-')}_stdout.txt"), "w", encoding="utf-8") as f:
            f.write(result.stdout)
        with open(os.path.join(work_dir, f"yt_dlp_{mode.strip('-')}_stderr.txt"), "w", encoding="utf-8") as f:
            f.write(result.stderr)
        candidates = sorted(Path(work_dir).glob("youtube_source*.srt")) + sorted(Path(work_dir).glob("youtube_source*.vtt"))
        for path in candidates:
            segs = parse_srt_or_vtt(str(path), "", f"yt_dlp_{mode.strip('-')}")
            if segs:
                return segs, {"source": f"yt_dlp_{mode.strip('-')}", "path": str(path)}
    list_cmd = build_ytdlp_base_cmd(cookies_from_browser, cookies, remote_components, no_check_certificate) + ["--list-subs", youtube_url]
    result = run_cmd(list_cmd, check=False)
    with open(os.path.join(work_dir, "yt_dlp_list_subs_stdout.txt"), "w", encoding="utf-8") as f:
        f.write(result.stdout)
    with open(os.path.join(work_dir, "yt_dlp_list_subs_stderr.txt"), "w", encoding="utf-8") as f:
        f.write(result.stderr)
    return [], {"source": "yt_dlp_write_subs", "list_subs_stdout": result.stdout[-4000:], "list_subs_stderr": result.stderr[-4000:]}


def build_preferred_langs(primary: str, fallback: str) -> List[str]:
    out = []
    for raw in [primary, fallback]:
        for part in str(raw or "").split(","):
            p = part.strip()
            if p and p not in out:
                out.append(p)
    if "all" not in out:
        out.append("all")
    return out


def download_youtube_asr_captions(youtube_url: str, output_dir: str, subtitle_language: str, fallback_subtitle_languages: str, cookies_from_browser: str, cookies: str, remote_components: str, no_check_certificate: bool) -> Dict[str, Any]:
    work_dir = os.path.join(output_dir, "work")
    ensure_dir(work_dir)
    video_id = extract_youtube_video_id(youtube_url)
    preferred_langs = build_preferred_langs(subtitle_language, fallback_subtitle_languages)
    debug: List[Dict[str, Any]] = []

    segs, meta = get_captiontracks_direct(youtube_url, work_dir, preferred_langs, cookies_from_browser, cookies, remote_components, no_check_certificate)
    debug.append(meta)
    if segs:
        return {"video_id": video_id, "segments": segs, "source": meta.get("source"), "meta": meta, "debug": debug}

    segs, meta = get_captiontracks_via_ytdlp_dumpjson(youtube_url, work_dir, preferred_langs, cookies_from_browser, cookies, remote_components, no_check_certificate)
    debug.append(meta)
    if segs:
        return {"video_id": video_id, "segments": segs, "source": meta.get("source"), "meta": meta, "debug": debug}

    segs, meta = get_captions_youtube_transcript_api(video_id, preferred_langs)
    debug.append(meta)
    if segs:
        return {"video_id": video_id, "segments": segs, "source": meta.get("source"), "meta": meta, "debug": debug}

    subtitle_expr = ",".join([p for p in preferred_langs if p != "all"] + ["all"])
    segs, meta = get_captions_ytdlp_write_subs(youtube_url, output_dir, subtitle_expr, cookies_from_browser, cookies, remote_components, no_check_certificate)
    debug.append(meta)
    if segs:
        return {"video_id": video_id, "segments": segs, "source": meta.get("source"), "meta": meta, "debug": debug}

    with open(os.path.join(work_dir, "caption_retrieval_debug.json"), "w", encoding="utf-8") as f:
        json.dump(debug, f, ensure_ascii=False, indent=2)
    raise RuntimeError("Aucune transcription ASR YouTube récupérée. Tentatives: direct captionTracks/baseUrl, yt-dlp dump-json caption URLs, youtube-transcript-api, yt-dlp write-auto-subs. Voir work/caption_retrieval_debug.json.")


def similarity(a: str, b: str) -> float:
    aa = normalize_for_match(a)
    bb = normalize_for_match(b)
    if not aa or not bb:
        return 0.0
    if fuzz is not None:
        return float(fuzz.token_set_ratio(aa, bb))
    sa = set(tokenize_text(aa))
    sb = set(tokenize_text(bb))
    if not sa or not sb:
        return 0.0
    return 100.0 * len(sa & sb) / max(1, len(sa | sb))


def align_mulan_to_captions(fragments: List[MulanFragment], captions: List[CaptionSegment], max_caption_group: int = 3, min_match_score: float = 20.0) -> Tuple[List[TimedSubtitle], Dict[str, Any]]:
    if not captions:
        raise RuntimeError("Aucun segment caption disponible.")
    subtitles: List[TimedSubtitle] = []
    cap_idx = 0
    cumulative_offset = 0.0
    sync_events: List[Dict[str, Any]] = []
    unmatched = 0

    text_fragments = [f for f in fragments if f.original_text]
    for fragment in fragments:
        if fragment.sync_offset != 0.0:
            before = cumulative_offset
            cumulative_offset += fragment.sync_offset
            sync_events.append({"fragment_id": fragment.id, "source_line": fragment.source_line, "sync_offset": fragment.sync_offset, "before": before, "after": cumulative_offset})
        if not fragment.original_text:
            continue

        best_score = -1.0
        best_start_idx = cap_idx
        best_end_idx = cap_idx
        search_end = min(len(captions), cap_idx + max(12, max_caption_group * 3))
        for start_idx in range(cap_idx, search_end):
            joined = ""
            for end_idx in range(start_idx, min(len(captions), start_idx + max_caption_group)):
                joined = normalize_text((joined + " " + captions[end_idx].text).strip())
                score = similarity(fragment.original_text, joined)
                # Bonus séquentiel: évite des sauts trop agressifs.
                score -= max(0, start_idx - cap_idx) * 1.5
                if score > best_score:
                    best_score = score
                    best_start_idx = start_idx
                    best_end_idx = end_idx

        if best_score < min_match_score and cap_idx < len(captions):
            best_start_idx = cap_idx
            best_end_idx = cap_idx
            best_score = similarity(fragment.original_text, captions[cap_idx].text)
            unmatched += 1

        if best_start_idx >= len(captions):
            last = captions[-1]
            start = last.end + cumulative_offset
            end = start + max(0.5, fragment.weight * 0.35)
            caption_text = ""
            caption_id = None
        else:
            selected = captions[best_start_idx:best_end_idx + 1]
            start = selected[0].start + cumulative_offset
            end = selected[-1].end + cumulative_offset
            caption_text = normalize_text(" ".join(c.text for c in selected))
            caption_id = selected[0].id
            cap_idx = max(cap_idx, best_end_idx + 1)

        if fragment.explicit_duration is not None:
            end = start + max(0.05, fragment.explicit_duration)

        if subtitles and start < subtitles[-1].end:
            start = subtitles[-1].end
            if end <= start:
                end = start + 0.05
        end = max(start + 0.05, end)

        subtitles.append(TimedSubtitle(
            id=len(subtitles),
            start=max(0.0, start),
            end=max(0.05, end),
            text=fragment.text,
            fragment_id=fragment.id,
            caption_id=caption_id,
            source_line=fragment.source_line,
            original_text=fragment.original_text,
            translated_text=fragment.translated_text,
            translation_applied=fragment.translation_applied,
            explicit_duration=fragment.explicit_duration,
            sync_offset=fragment.sync_offset,
            match_score=max(0.0, best_score),
            caption_text=caption_text,
        ))

    return subtitles, {
        "alignment_mode": "youtube_asr_timing_direct_captiontracks_mulan_text_reference",
        "caption_count": len(captions),
        "fragment_count": len(fragments),
        "text_fragment_count": len(text_fragments),
        "subtitle_count": len(subtitles),
        "unmatched_low_score_count": unmatched,
        "min_match_score": min_match_score,
        "max_caption_group": max_caption_group,
        "cumulative_offset": cumulative_offset,
        "sync_events": sync_events,
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
    with open(path, "w", encoding="utf-8") as f:
        for i, sub in enumerate(subtitles, start=1):
            f.write(f"{i}\n")
            f.write(f"{format_srt_time(sub.start)} --> {format_srt_time(sub.end)}\n")
            f.write(sub.text.strip() + "\n\n")


def export_lrc(path: str, subtitles: List[TimedSubtitle], enhanced: bool = False) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for sub in subtitles:
            minutes = int(sub.start // 60)
            seconds = sub.start - minutes * 60
            sep = "" if enhanced else " "
            f.write(f"[{minutes:02d}:{seconds:05.2f}]{sep}{sub.text.strip()}\n")


def export_txt(path: str, subtitles: List[TimedSubtitle]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for sub in subtitles:
            f.write(sub.text.strip() + "\n")


def export_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def burn_srt_to_video(input_video_path: str, input_srt_path: str, output_video_path: str) -> str:
    ensure_dir(os.path.dirname(output_video_path))
    if not os.path.isfile(input_video_path):
        raise FileNotFoundError(f"Vidéo locale introuvable: {input_video_path}")
    srt_for_filter = input_srt_path.replace("\\", "\\\\").replace(":", "\\:").replace("'", r"\'")
    subtitle_style = "FontName=Arial,FontSize=20,PrimaryColour=&H00FFFFFF,BackColour=&H00000000,OutlineColour=&H00000000,BorderStyle=3,Outline=1,Shadow=0,MarginV=25,Alignment=2"
    cmd = ["ffmpeg", "-y", "-i", input_video_path, "-vf", f"subtitles='{srt_for_filter}':force_style='{subtitle_style}'", "-c:a", "copy", output_video_path]
    result = run_cmd(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Echec ffmpeg burn.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
    return output_video_path


def process_youtube_mulan_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    ensure_dir(args.output_dir)
    work_dir = os.path.join(args.output_dir, "work")
    ensure_dir(work_dir)

    translator = create_local_text_translator(args.input_language, args.output_language, args.translation_model, args.translation_max_length, args.translation_local_files_only)
    fragments = parse_mulan_text(args.mulan_text, args.input_language, args.output_language, args.translation_model, translator)
    yt_payload = download_youtube_asr_captions(args.youtube_url, args.output_dir, args.subtitle_language, args.fallback_subtitle_languages, args.cookies_from_browser, args.cookies, args.remote_components, args.no_check_certificate)
    captions: List[CaptionSegment] = yt_payload["segments"]

    subtitles, alignment_debug = align_mulan_to_captions(fragments, captions, args.max_caption_group, args.min_match_score)

    srt_path = os.path.join(args.output_dir, "youtube_mulan_corrected.srt")
    lrc_path = os.path.join(args.output_dir, "youtube_mulan_corrected.lrc")
    enhanced_lrc_path = os.path.join(args.output_dir, "youtube_mulan_corrected.enhanced.lrc")
    txt_path = os.path.join(args.output_dir, "youtube_mulan_corrected.txt")
    json_path = os.path.join(args.output_dir, "youtube_mulan_corrected.json")

    export_srt(srt_path, subtitles)
    export_lrc(lrc_path, subtitles, enhanced=False)
    export_lrc(enhanced_lrc_path, subtitles, enhanced=True)
    export_txt(txt_path, subtitles)

    burned_video_path = None
    if args.burn_video:
        if not args.input_video:
            warn("--burn-video demandé mais --input-video absent. Burn ignoré.")
        else:
            burned_video_path = os.path.join(args.output_dir, "video_subtitled.mp4")
            burn_srt_to_video(args.input_video, srt_path, burned_video_path)

    meta = {
        "youtube_url": args.youtube_url,
        "youtube_video_id": yt_payload.get("video_id"),
        "caption_source": yt_payload.get("source"),
        "mulan_text": str(Path(args.mulan_text).resolve()),
        "input_video": str(Path(args.input_video).resolve()) if args.input_video else "",
        "translation_enabled": bool(args.output_language.strip() and args.translation_model.strip()),
        "translator_loaded": translator is not None,
        "caption_count": len(captions),
        "fragment_count": len(fragments),
        "subtitle_count": len(subtitles),
        "srt_path": srt_path,
        "lrc_path": lrc_path,
        "enhanced_lrc_path": enhanced_lrc_path,
        "txt_path": txt_path,
        "json_path": json_path,
        "burned_video_path": burned_video_path,
        "caption_retrieval": yt_payload.get("meta"),
        "caption_retrieval_debug": yt_payload.get("debug"),
        "alignment": alignment_debug,
    }
    payload = {"meta": meta, "captions": [asdict(c) for c in captions], "fragments": [asdict(f) for f in fragments], "subtitles": [asdict(s) for s in subtitles]}
    export_json(json_path, payload)

    return {"meta": meta, "subtitles": subtitles, "captions": captions, "fragments": fragments, "srt_path": srt_path, "lrc_path": lrc_path, "enhanced_lrc_path": enhanced_lrc_path, "txt_path": txt_path, "json_path": json_path, "burned_video_path": burned_video_path}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YouTube ASR captions direct captionTracks -> correction texte Mulan -> traduction optionnelle -> SRT/LRC/JSON -> burn vidéo locale optionnel.")
    parser.add_argument("--youtube-url", required=True, help="URL YouTube source. Sert à récupérer les captions/transcriptions ASR, pas forcément la vidéo.")
    parser.add_argument("-m", "--mulan-text", required=True, help="Fichier Mulan text de référence/correction.")
    parser.add_argument("-o", "--output-dir", default="./youtube_mulan_out", help="Dossier de sortie.")
    parser.add_argument("--input-video", default="", help="Vidéo locale optionnelle pour le burn.")
    parser.add_argument("--burn-video", action="store_true", help="Incruste le SRT sur --input-video dans output-dir/video_subtitled.mp4.")

    parser.add_argument("--subtitle-language", default="es", help="Langue prioritaire captions, ex: es, fr, en.")
    parser.add_argument("--fallback-subtitle-languages", default="es.*,fr,fr.*,en,en.*", help="Langues fallback séparées par virgules.")
    parser.add_argument("--cookies-from-browser", default="chrome", help="Navigateur pour cookies yt-dlp, ex: chrome. Vide = désactivé.")
    parser.add_argument("--cookies", default="", help="Fichier cookies.txt alternatif.")
    parser.add_argument("--remote-components", default="ejs:github", help="Option yt-dlp --remote-components, utile pour EJS.")
    parser.add_argument("--no-check-certificate", action="store_true")

    parser.add_argument("--input-language", default="auto", help="Langue source Mulan.")
    parser.add_argument("--output-language", default="", help="Langue cible. Vide = traduction désactivée.")
    parser.add_argument("--translation-model", default="", help="Chemin local ou nom modèle transformers Seq2Seq. Vide = traduction désactivée.")
    parser.add_argument("--translation-max-length", type=int, default=256)
    parser.add_argument("--translation-allow-download", action="store_false", dest="translation_local_files_only", default=True)

    parser.add_argument("--max-caption-group", type=int, default=3, help="Nombre maximal de segments ASR regroupés pour matcher une ligne Mulan.")
    parser.add_argument("--min-match-score", type=float, default=20.0, help="Score minimal avant fallback séquentiel.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        result = process_youtube_mulan_pipeline(args)
        log("\n=== RESULTAT ===")
        log(f"Caption source : {result['meta']['caption_source']}")
        log(f"Captions       : {len(result['captions'])}")
        log(f"Fragments      : {len(result['fragments'])}")
        log(f"Subtitles      : {len(result['subtitles'])}")
        log(f"SRT            : {result['srt_path']}")
        log(f"LRC            : {result['lrc_path']}")
        log(f"ELRC           : {result['enhanced_lrc_path']}")
        log(f"TXT            : {result['txt_path']}")
        log(f"JSON           : {result['json_path']}")
        if result.get("burned_video_path"):
            log(f"VIDEO          : {result['burned_video_path']}")
        log("")
        for sub in result["subtitles"][:30]:
            log(f"[{sub.start:8.2f} -> {sub.end:8.2f}] score={sub.match_score:5.1f} {sub.text}")
        if len(result["subtitles"]) > 30:
            log(f"... {len(result['subtitles']) - 30} sous-titres supplémentaires")
    except Exception as exc:
        fail(f"Echec pipeline YouTube/Mulan: {exc}\n{traceback.format_exc()}")


if __name__ == "__main__":
    main()
