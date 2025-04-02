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
import traceback
import io

def convert_video_to_audio_moviepy(video_file : str, output_ext : str = 'wav'):
    """Converts video to audio using MoviePy library
    that uses `ffmpeg` under the hood"""
    filename, ext = os.path.splitext(video_file)
    clip = VideoFileClip(video_file)
    clip.audio.write_audiofile(f"{filename}.{output_ext}",logger=None)
    return filename+'.'+output_ext

def progress_callback(stream = None, file_handler = None, start = None ):
    if stream is None :
        return None
    
    total_size = stream.filesize
    progress_percent = ((total_size - 10) / total_size) * 100
    time_spent = datetime.now() - start
    minutes_left = round(time_spent.total_seconds() / progress_percent * (100 - progress_percent) /60, 1)
    print(str(int(progress_percent)) + '% | ' + str(minutes_left) + 'mins ... ', end='')  

    buffer = stream.stream_to_buffer(buffer=file_handler)

    print("downloaded")
    exclude = [ 'Merci d\'avoir regardé cette vidéo !\n',
                'Sous-titres réalisés para la communauté d\'Amara.org\n'
              ]
    
    with open(buffer, "rb") as audio_file:
        SousTitre = openai.Audio.transcribe(
            file = audio_file,
            model = "whisper-1",
            response_format="text",
            language="fr"
        )

    pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-ru")
    FrRu = pipe(SousTitre.lower())
    FrRuTxt = unescape(FrRu[0]['translation_text'])

    pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-ru-hy")
    FrRuAr = pipe(FrRuTxt)
    FrRuArTxt = unescape(FrRuAr[0]['translation_text'])
    if SousTitre not in exclude :
        print("#############################")
        print(f"# Phrases en francais detectée")
        print("#############################")
        print(SousTitre)
        print("#############################")
        print("# Traduction selon le model opus mt en => ru")
        print("#############################")
        print(FrRuTxt)
        print("#############################")
        print("# Traduction selon le model opus mt ru => Armenien")
        print("#############################")
        print(FrRuArTxt)
        print("#############################")


def download_video_yt(url: str, path: str, video_file: str):
    """Download the video url on youtube"""
    file_ = open(file='/tmp/buffer-rt-translate', mode='wb+')
    file_handle = io.open(file=file_.fileno(), mode='wb+', closefd=False )
    chunk = bytes(2048)
    stream = None
    start = datetime.now()
    try:
        yt = YouTube(url, on_progress_callback=progress_callback(stream=stream ,file_handler=file_handle,start=start))
    except:
        traceback.print_exc()
    
    stream = yt.streams.get_highest_resolution()
        
    stream.on_progress(chunk=chunk, file_handler=file_handle, bytes_remaining=int(10))
    stream.download(output_path=path,filename=video_file)
    file_.close()
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
                    local_dir='/home/drodriguez/dev/opus-mt-fr-ru/',
                    local_files_only=False,
                    cache_dir='/home/drodriguez/dev/opus-mt-fr-ru/.cache/')

    snapshot_download(repo_id="Helsinki-NLP/opus-mt-ru-hy", 
                    repo_type='model',
                    local_dir='/home/drodriguez/dev/opus-mt-ru-hy/',
                    local_files_only=False,
                    cache_dir='/home/drodriguez/dev/opus-mt-ru-hy/.cache/')

# CLEF OPENAI  pour accèder au service de transcription
openai.api_key = os.environ["OPENAI_KEY"]
initHugeModel()

video = download_video_yt(url="https://youtu.be/aN0-DgKsBxY",path='/tmp/',video_file='video.mp4')
audio = convert_video_to_audio_moviepy(video_file=video)
chunks = split_audio_file(audio)
