#!/usr/bin/env python3
import openai
import os
import pyaudio
import wave

openai.api_key = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

import openai

with open("/home/drodriguez/dev/rt-translate-oa/Audio/easy_english.mp3", "rb") as audio_file:
    transcript = openai.Audio.transcribe(
        file = audio_file,
        model = "whisper-1",
        response_format="text",
        language="en"
    )
print(transcript)

with open("/home/drodriguez/dev/rt-translate-oa/Audio/easy_english.mp3", "rb") as audio_file:
    transcript2 = openai.Audio.transcribe(
        file = audio_file,
        model = "whisper-1",
        response_format="srt",
        language="en"
    )
print(transcript2)

with open("/home/drodriguez/dev/rt-translate-oa/Audio/easy_spanish.mp3", "rb") as audio_file:
    transcript_es = openai.Audio.transcribe(
        file = audio_file,
        model = "whisper-1",
        response_format="text",
        language="es"
    )
print(transcript_es)

with open("/home/drodriguez/dev/rt-translate-oa/Audio/easy_spanish.mp3", "rb") as audio_file:
    translate = openai.Audio.translate(
        file = audio_file,
        model = "whisper-1",
        response_format="text",
        language="en"
    )
print(translate)
