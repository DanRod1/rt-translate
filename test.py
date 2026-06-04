import re
from youtube_transcript_api import YouTubeTranscriptApi


def extract_video_id(url: str) -> str:
    m = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url)
    if not m:
        raise ValueError("Impossible d'extraire l'id YouTube")
    return m.group(1)


def format_srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))

    hours = ms // 3600000
    ms %= 3600000

    minutes = ms // 60000
    ms %= 60000

    secs = ms // 1000
    ms %= 1000

    return f"{hours:02}:{minutes:02}:{secs:02},{ms:03}"


def transcript_to_srt(transcript, output_file: str):
    with open(output_file, "w", encoding="utf-8") as f:
        for idx, snippet in enumerate(transcript, start=1):
            start = snippet.start
            end = snippet.start + snippet.duration

            f.write(f"{idx}\n")
            f.write(
                f"{format_srt_time(start)} --> "
                f"{format_srt_time(end)}\n"
            )
            f.write(f"{snippet.text}\n\n")


url = "https://youtu.be/Uq2K6O8X2nM"

video_id = extract_video_id(url)

ytt_api = YouTubeTranscriptApi()

transcript = ytt_api.fetch(
    video_id,
    languages=["es", "fr", "en"]
)

transcript_to_srt(
    transcript,
    f"{video_id}.srt"
)

print(f"SRT créé : {video_id}.srt")
