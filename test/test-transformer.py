#!/usr/bin/env python3

import huggingface_hub
from transformers import pipeline
import openai
import os
import pyaudio
import whisper

# CLEF OPENAI  pour accèder au service de transcription
openai.api_key = os.environ["OPENAI_KEY"]

# Donwload du model OPUSMT
snapshot_download(repo_id="Helsinki-NLP/opus-mt-en-ru", 
                repo_type='model',
                local_dir='/home/drodriguez/dev/opus-mt-en-ru/',
                local_files_only=False,
                cache_dir='/home/drodriguez/dev/opus-mt-en-ru/.cache/')

snapshot_download(repo_id="Helsinki-NLP/opus-mt-ru-hy", 
                repo_type='model',
                local_dir='/home/drodriguez/dev/opus-mt-ru-hy/',
                local_files_only=False,
                cache_dir='/home/drodriguez/dev/opus-mt-ru-hy/.cache/')


with open("Audio/easy_english.mp3", "rb") as audio_file:
    easyEnglish = openai.Audio.transcribe(
        file = audio_file,
        model = "whisper-1",
        response_format="text",
        language="en"
    )
print(easyEnglish)


with open("Audio/easy_spanish.mp3", "rb") as audio_file:
    esaySpanish = openai.Audio.translate(
        file = audio_file,
        model = "whisper-1",
        response_format="text",
        language="en"
    )
print(esaySpanish)

pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ru")
EnRu = pipe(easyEnglish)
EsRu = pipe(esaySpanish)

EnRuTxt = EnRu[0]['translation_text']
EsRuTxt = EsRu[0]['translation_text']

pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-ru-hy")
EnRuAr = pipe(EnRuTxt)
EsRuAr = pipe(EsRuTxt)

EnRuArTxt = EnRuAr[0]['translation_text']
EsRuaArTxt = EsRuAr[0]['translation_text']

print(EnRuTxt)
print(EsRuTxt)
print(EnRuAr)
print(EsRuAr)
