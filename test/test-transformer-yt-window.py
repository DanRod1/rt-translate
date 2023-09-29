#!/usr/bin/env python3

from pytube import YouTube
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from pydub.utils import make_chunks
from huggingface_hub import snapshot_download
from huggingface_hub.utils import are_progress_bars_disabled, disable_progress_bars, enable_progress_bars
from transformers import pipeline
from html import unescape
from datetime import datetime
import openai
import os
import pyaudio
import whisper
import proglog

def convert_video_to_audio_moviepy(video_file : str, output_ext : str = 'wav'):
    """Converts video to audio using MoviePy library
    that uses `ffmpeg` under the hood"""
    filename, ext = os.path.splitext(video_file)
    clip = VideoFileClip(video_file)
    clip.audio.write_audiofile(f"{filename}.{output_ext}",logger=None)
    return filename+'.'+output_ext

def progress_callback(stream, chunk, bytes_remaining):
        total_size = stream.filesize
        progress_percent = ((total_size - bytes_remaining) / total_size) * 100
        time_spent = datetime.now() - start
        minutes_left = round(time_spent.total_seconds() / progress_percent * (100 - progress_percent) /60, 1)
        print(str(int(progress_percent)) + '% | ' + str(minutes_left) + 'mins ... ', end='')    

def download_video_yt(url: str, path: str, video_file: str):
    """Download the video url on youtube"""
    start = datetime.now()
    yt = YouTube(url)
    stream = yt.streams.get_highest_resolution()
    stream.download(output_path=path,filename=video_file)
    return path+video_file

def split_audio_file(audio_file: str) :    
    myaudio = AudioSegment.from_file(audio_file , "wav") 
    chunk_length_ms = 10000 # pydub calculates in millisec
    chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec

    #Export all of the individual chunks as wav files
    splitWav = []

    for i, chunk in enumerate(chunks):
        chunk_name = f"/tmp/chunk{i}.wav"
        chunk.export(chunk_name, format="wav")
        splitWav.append(chunk_name)
    return splitWav

def initHugeModel():
# Donwload du model OPUSMT
    disable_progress_bars()
    snapshot_download(repo_id="Helsinki-NLP/opus-mt-en-ru", 
                    repo_type='model',
                    local_dir='/home/drodriguez/dev/opus-mt-en-ru/',
                    local_files_only=False,
                    cache_dir='/home/drodriguez/dev/opus-mt-fr-ru/.cache/')

    snapshot_download(repo_id="Helsinki-NLP/opus-mt-ru-hy", 
                    repo_type='model',
                    local_dir='/home/drodriguez/dev/opus-mt-ru-hy/',
                    local_files_only=False,
                    cache_dir='/home/drodriguez/dev/opus-mt-ru-hy/.cache/')

def chunkVideos(chunks: list ):
    exclude = [ 'Merci d\'avoir regardé cette vidéo !\n',
                'Sous-titres réalisés para la communauté d\'Amara.org\n'
              ]
    for waves in chunks :
        with open(waves, "rb") as audio_file:
            SousTitre = openai.Audio.transcribe(
                file = audio_file,
                model = "whisper-1",
                response_format="text",
                language="en"
            )

        pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ru")
        FrRu = pipe(SousTitre.lower())
        FrRuTxt = unescape(FrRu[0]['translation_text'])

        #pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-ru-hy")
        #FrRuAr = pipe(FrRuTxt)
        #FrRuArTxt = unescape(FrRuAr[0]['translation_text'])
        if SousTitre not in exclude :
            print("#############################")
            print(f"# Phrases en Anglais utilisé {waves}")
            print("#############################")
            print(SousTitre)
            print("#############################")
            print("# Traduction selon le model opus mt en => ru")
            print("#############################")
            print(FrRuTxt)
            #print("#############################")
            #print("# Traduction selon le model opus mt ru => Armenien")
            #print("#############################")
            #print(FrRuArTxt)
            print("#############################")

# CLEF OPENAI  pour accèder au service de transcription
openai.api_key = os.environ["OPENAI_KEY"]
initHugeModel()

video = download_video_yt(url="https://youtu.be/4PSijc4Chtc",path='/tmp/',video_file='video.mp4')
audio = convert_video_to_audio_moviepy(video_file=video)
chunks = split_audio_file(audio)

chunkVideos(chunks=chunks)
