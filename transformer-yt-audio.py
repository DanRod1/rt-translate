#!/usr/bin/env python3

from pytube import YouTube
from pydub import AudioSegment
from pydub.utils import make_chunks
from huggingface_hub import snapshot_download
from huggingface_hub.utils import are_progress_bars_disabled, disable_progress_bars, enable_progress_bars
from transformers import pipeline
from html import unescape
import openai
import os
import pyaudio
import whisper
import proglog
import traceback
import argparse
import re
from datetime import timedelta
from moviepy.editor import *

def generateSrt(srtFile: any , data :dict = {}, outputLanguage :str = '', init: list = ['00','00','00,000']):
    for keys in data :
        pattern = '\d+'
        result = re.search(pattern=pattern,string=keys) 
        index = int(result.group())
        seconds = (index + 1) * 10
        end = str(timedelta(seconds=seconds)).split(':')
        print(f'{index}')
        srtFile.write(f'{index}\n')
        print(f'{init}')
        srtFile.write(f'{":".join(init)} --> ')
        count = 0
        returnValue = []
        for unit in end :
            if len(str(unit)) == 1 :
                tmp = '0{unit}'.format(unit=unit)
                init[count] = tmp
                srtFile.write(f'{tmp}:')
            elif count == 2 :
                tmp = '{unit},000'.format(unit=unit)
                init[count] = tmp
                srtFile.write(f'{tmp}')
            else:
                init[count]  = str(unit)
                srtFile.write(f' {str(unit)}:')
            count += 1
        returnValue = init
        srtFile.write(f' \n')
        srtFile.write(f'{data[keys][outputLanguage]}\n')
        print(f'{returnValue}')     
    return returnValue

def parseARgs(parser = None ):
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")
    parser.add_argument("-i", "--inputLanguage", action="store", default = 'fr',
                    help="langue de la video téléchargé")
    parser.add_argument("-o", "--outputLanguage", action="store", default = 'ru',
                    help="langue de la video téléchargé")
    parser.add_argument("-u", "--url", action="store", default = 'https://youtu.be/aN0-DgKsBxY',
                    help="url de la video téléchargé")
    parser.add_argument("-a", "--audioDir", action="store", default = 'Audio',
                    help="repertoire Audio Cache")
    args = parser.parse_args()
    
    if args.verbose:
        print(f"usage is transformer-yt-audio.py -v | --verbose \n")
    else:
        return args

def transcribe(chunk :str = '', inputLanguage :str = 'fr', outputLanguage :str = 'ru'):
    exclude = [ 'Merci d\'avoir regardé cette vidéo !\n',
                'Sous-titres réalisés para la communauté d\'Amara.org\n'
              ]
    
    with open(chunk, "rb") as audio_file:
        SousTitre = openai.Audio.transcribe(
            file = audio_file,
            model = "whisper-1",
            response_format="text",
            language=inputLanguage
        )

    pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-"+inputLanguage+"-"+outputLanguage)
    intputText = pipe(SousTitre.lower())
    translateSubtile = unescape(intputText[0]['translation_text'])

    if SousTitre not in exclude :
        print("#############################################")
        print(f"# Code Language en entrée : {inputLanguage} ")
        print("#############################################")
        print(SousTitre)
        print("########################################################################")
        print(f"# Traduction selon le model opus mt {inputLanguage} => {outputLanguage}")
        print("########################################################################")
        print(translateSubtile)
        print(f"\n")
        subtitles = {chunk:{inputLanguage:SousTitre,outputLanguage:translateSubtile}}
    else :
        print("#############################################")
        print(f"# Code Language en entrée : {inputLanguage} ")
        print("#############################################")
        print(SousTitre)
        print("########################################################################")
        print(f"# Traduction selon le model opus mt {inputLanguage} => {outputLanguage}")
        print("########################################################################")
        print('oups what the fluck robot do not understand')
        print(f"\n")
        subtitles = {chunk:{inputLanguage:SousTitre,outputLanguage:'oups what the fluck robot do not understand'}}
    return subtitles


def download_video_yt(url: str, audioDir: str ):
    if re.match('^file:/',url):
        video = VideoFileClip(url) # 2.
        audio = video.audio # 3.
        audio.write_audiofile(audioDir+'/audio.mp3') # 4.
    else:
        """Download the video url on youtube"""
        file_ = open(file=audioDir+'/buffer-rt-translate', mode='wb+')
        try:
            yt = YouTube(url, use_oauth=False, allow_oauth_cache=True)
        except:
            traceback.print_exc()
        if yt.age_restricted:
            yt.bypass_age_gate()
        
        for key, values in yt.streams.itag_index.items() :
            if values.is_progressive is False and values.audio_codec == 'opus' :
                yt.streams.get_by_itag(key).download(output_path=audioDir,filename='audio.mp3')
    return audioDir+'/audio.mp3'

def split_audio_file(audio_file: str, audioDir:str) :    
    #myaudio = AudioSegment.from_file(audio_file , codec="opus") 
    myaudio = AudioSegment.from_mp3(audio_file) 
    chunk_length_ms = 10000 # pydub calculates in millisec
    chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec

    #Export all of the individual chunks as wav files
    splitWav = []

    for i, chunk in enumerate(chunks):
        chunk_name = f"{audioDir}/chunk{i}.mp3"
        chunk.export(chunk_name, format="mp3")
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

    snapshot_download(repo_id="Helsinki-NLP/opus-mt-fr-es", 
                    repo_type='model',
                    local_dir='/home/drodriguez/dev/opus-mt-fr-es/',
                    local_files_only=False,
                    cache_dir='/home/drodriguez/dev/opus-mt-fr-es/.cache/')

# CLEF OPENAI  pour accèder au service de transcription
openai.api_key = os.environ["OPENAI_KEY"]

options = parseARgs(argparse.ArgumentParser())
if __name__ == '__main__':
    initHugeModel()
    audio_file = download_video_yt(url = options.url,audioDir=options.audioDir)
    chunks = split_audio_file(audio_file, audioDir = options.audioDir)
    turn = 0
    file = open(f"{options.audioDir}/audio.srt", "w+")
    for chunk in chunks : 
        srt = transcribe(chunk = chunk, inputLanguage = options.inputLanguage, outputLanguage = options.outputLanguage)
        if turn == 0 :
            timescale = generateSrt(srtFile = file, data = srt, outputLanguage = options.outputLanguage)
        else :
            timescale = generateSrt(srtFile = file, data = srt, outputLanguage = options.outputLanguage,init = timescale)            
        turn += 1
    file.close()
