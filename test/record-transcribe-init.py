#!/usr/bin/env python3
import openai
import os
import pyaudio
import whisper

openai.api_key = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# send audio file to transalate
def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()



with open("Audio/easy_english.mp3", "rb") as audio_file:
    transcript = openai.Audio.transcribe(
        file = audio_file,
        model = "whisper-1",
        response_format="text",
        language="en"
    )
print(transcript)

with open("Audio/easy_english.mp3", "rb") as audio_file:
    transcript2 = openai.Audio.transcribe(
        file = audio_file,
        model = "whisper-1",
        response_format="text",
        language="en"
    )
print(transcript2)

with open("Audio/easy_spanish.mp3", "rb") as audio_file:
    translate = openai.Audio.translate(
        file = audio_file,
        model = "whisper-1",
        response_format="text",
        language="en"
    )
print(translate)
