from pytubefix import YouTube
from pydub import AudioSegment
from pydub.utils import make_chunks
from huggingface_hub import snapshot_download
from huggingface_hub.utils import are_progress_bars_disabled, disable_progress_bars, enable_progress_bars
from urllib.request import urlretrieve
from transformers import pipeline
from html import unescape
from openai import OpenAI
import os, glob

client = OpenAI(api_key=os.environ["OPENAI_KEY"])
import pyaudio
from faster_whisper import WhisperModel
import proglog
import traceback
import argparse
import re
from datetime import timedelta
from moviepy import VideoFileClip
import ffmpeg
import ctranslate2

def generateSrt(srtFile: any , data :dict = {}, outputLanguage :str = '', init: list = ['00','00','00,000'], verbose :bool = False):
    returnValue = list()
    for keys in data :
        pattern = '\d+'
        result = re.search(pattern=pattern,string=keys) 
        index = int(result.group())
        seconds = (index + 1) * 10
        end = str(timedelta(seconds=seconds)).split(':')
        if verbose : print(f'{index}')
        srtFile.write(f'{index}\n')
        if verbose : print(f'{init}')
        srtFile.write(f'{":".join(init)} --> ')
        count = 0
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
        print(f'in progress {data}')    
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
    parser.add_argument("-P", "--videoPath", action="store", default = 'Video',
                    help="repertoire Video Cache")
    parser.add_argument("-M", "--initModele", action="store_true",
                    help="Activate local Model cache")
    parser.add_argument("-n", "--nameModel", action="store", default="Helsinki-NLP/opus-mt-fr-ru",
                    help="Model name")

    args = parser.parse_args()

    if re.match('^file:/',args.url):
        videoPath = re.sub('^file:','',args.url)
        args.videoPath = videoPath
    else:
        args.videoPath = args.url


    return args

def transcribe(chunk :str = '', inputLanguage :str = 'fr', outputLanguage :str = 'ru', verbose : bool = False):
    exclude = [ 'Merci d\'avoir regardé cette vidéo !\n',
                'Sous-titres réalisés para la communauté d\'Amara.org\n'
              ]
    intputText = ''
    translateSubtile = ''
    subtitles = dict()
    # client = OpenAI(api_key="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")
    device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    supported = ctranslate2.get_supported_compute_types(device)
    for pref in ("float16","int8_float16","bfloat16","int8","int8_float32","float32"):
        if pref in supported:
            compute = pref; break
    model = WhisperModel("Systran/faster-whisper-large-v3", device=device, compute_type=compute)        
    # --- transcribe
    segments, info = model.transcribe(
        chunk,
        chunk_length=10,
        language=inputLanguage,
        temperature=0.0,
        patience=0.2,
        beam_size=5,
        condition_on_previous_text=False,
        vad_filter=True
    )
    segment = list(segments)
    subtitles = {chunk:{inputLanguage:'...',outputLanguage:'...'}}
    for s in segment:
        model_name = "Helsinki-NLP/opus-mt-"+inputLanguage+"-"+outputLanguage
        # Step 2: Text-to-text (Marian)
        translator = pipeline("translation", model=model_name)
        # Translate
        translation = translator(s.text)[0]["translation_text"]

        intputText = s.text 
        translateSubtile = unescape(translation)

        if intputText not in exclude :
            if verbose :
                print("#############################################")
                print(f"# Code Language en entrée : {inputLanguage} ")
                print("#############################################")
                print(intputText)
                print("########################################################################")
                print(f"# Traduction selon le model opus mt {inputLanguage} => {outputLanguage}")
                print("########################################################################")
                print(translateSubtile)
                print(f"\n")
            subtitles = {chunk:{inputLanguage:intputText,outputLanguage:translateSubtile}}
        else :
            if verbose :
                print("#############################################")
                print(f"# Code Language en entrée : {inputLanguage} ")
                print("#############################################")
                print(intputText)
                print("########################################################################")
                print(f"# Traduction selon le model opus mt {inputLanguage} => {outputLanguage}")
                print("########################################################################")
                print('oups what the fluck robot do not understand')
                print(f"\n")
            subtitles = {chunk:{inputLanguage:intputText,outputLanguage:'oups what the fluck robot do not understand'}}
    return subtitles


def download_video_yt(url: str, audioDir: str ):
    cwd = os.getcwd()
    stream = {}

    if re.match('^file:/',url):
        url = re.sub('^(file:/)(.*)','\\2',url)

        video = VideoFileClip(url)
        video.audio.write_audiofile(filename=f'{audioDir}/audio.mp3')
        stream = { 'audio': f'{audioDir}/audio.mp3', 'video': url }
    else:
        """Download the video url on youtube"""
        open(file=audioDir+'/buffer-rt-translate', mode='wb+')
        try:
            yt = YouTube(url)
        except:
            traceback.print_exc()
        if yt.age_restricted:
            yt.bypass_age_gate()

        for index in yt.streams.all():
            if index.audio_codec == 'mp4a.40.2' and index.type == 'video':
                print(index.default_filename)
                yt.streams.get_highest_resolution().download(output_path='Video')
                stream['Video'] = f'{cwd}/Video/{index.default_filename}'
                break
        for index in yt.streams.all():
            if index.audio_codec == 'opus' and index.type == 'audio':
                print(index.default_filename)
                yt.streams.get_audio_only().download(output_path=f'{audioDir}')
                stream['audio'] = f'{cwd}/{audioDir}/{index.default_filename}'
                break
    return stream

def split_audio_file(stream: dict ) :    
    myaudio = AudioSegment.from_file(stream['audio']) 
    audioDir = re.sub('(.*)/(.*\.\w+)','\\1', stream['audio'] )
    #myaudio = AudioSegment.from_mp3(audio_file)  
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
                    local_dir='/home/drodriguez/dev/Helsinki-NLP/opus-mt-fr-ru/',
                    local_files_only=False,
                    cache_dir='/home/drodriguez/dev/Helsinki-NLP/opus-mt-fr-ru/.cache/')
    snapshot_download(repo_id="Helsinki-NLP/opus-mt-en-fr", 
                    repo_type='model',
                    local_dir='/home/drodriguez/dev/Helsinki-NLP/opus-mt-en-fr/',
                    local_files_only=False,
                    cache_dir='/home/drodriguez/dev/Helsinki-NLP/opus-mt-en-fr/.cache/')
    snapshot_download(repo_id="Helsinki-NLP/opus-mt-ru-hy", 
                    repo_type='model',
                    local_dir='/home/drodriguez/dev/Helsinki-NLP/opus-mt-ru-hy/',
                    local_files_only=False,
                    cache_dir='/home/drodriguez/dev/Helsinki-NLP/opus-mt-ru-hy/.cache/')
    snapshot_download(repo_id="Helsinki-NLP/opus-mt-fr-es", 
                    repo_type='model',
                    local_dir='/home/drodriguez/dev/Helsinki-NLP/opus-mt-fr-es/',
                    local_files_only=False,
                    cache_dir='/home/drodriguez/dev/Helsinki-NLP/opus-mt-fr-es/.cache/')
    snapshot_download(repo_id="Helsinki-NLP/opus-mt-es-fr", 
                    repo_type='model',
                    local_dir='/home/drodriguez/dev/Helsinki-NLP/opus-mt-es-fr/',
                    local_files_only=False,
                    cache_dir='/home/drodriguez/dev/Helsinki-NLP/opus-mt-es-fr/.cache/')
    snapshot_download(repo_id="Helsinki-NLP/opus-mt-ar-fr", 
                    repo_type='model',
                    local_dir='/home/drodriguez/dev/opus-mt-ar-fr/',
                    local_files_only=False,
                    cache_dir='/home/drodriguez/dev/opus-mt-ar-fr/.cache/')
    snapshot_download(repo_id="Systran/faster-whisper-large-v3", 
            repo_type='model',
            local_dir='/home/drodriguez/dev/whisper-large-v3',
            local_files_only=False,
            cache_dir='/home/drodriguez/dev/whisper-large-v3/.cache/')


# CLEF OPENAI  pour accèder au service de transcription

options = parseARgs(argparse.ArgumentParser(description='Lance la traduction et la génération de sous-tritre d\'une vidéo local ou youtube'))
if __name__ == '__main__':
    if options.initModele :
        initHugeModel()

    stream = download_video_yt(url = options.url,audioDir=options.audioDir)
    chunks = split_audio_file(stream)
    turn = 0
    file = open(f"{options.audioDir}/audio.srt", "w+")
    for chunk in chunks : 
        size_bytes = os.path.getsize(chunk)
        size_kb = size_bytes / 1024
        print(f'{chunk}')
        if size_bytes < 100: next

        srt = transcribe(chunk = chunk, inputLanguage = options.inputLanguage, outputLanguage = options.outputLanguage, verbose = options.verbose)
        if turn == 0 :
            timescale = generateSrt(srtFile = file, data = srt, outputLanguage = options.inputLanguage)
        else :
            timescale = generateSrt(srtFile = file, data = srt, outputLanguage = options.inputLanguage,init = timescale)            
        turn += 1
    file.close()

    width=800
    height=600
    # Manage 2 kind Local or youtube url ( probably url globaly if metadata)    
    scale_expr_w = f"iw*min({width}/iw\\,{height}/ih)"
    scale_expr_h = f"ih*min({width}/iw\\,{height}/ih)"

    # pad: centre sur la toile cible
    pad_x = f"({width}-iw)/2"
    pad_y = f"({height}-ih)/2"
    if re.match('^file:/',options.url):
        style = "OutlineColour=&H100000000,BorderStyle=3,Outline=1,Shadow=0,Fontsize=13"
        output = re.sub('\.','-sub.',options.videoPath)
        audio_stream = ffmpeg.input(options.audioDir+'/audio.mp3').audio 
        video_stream = ffmpeg.input(options.videoPath).video 
    else:
        style = "OutlineColour=&H100000000,BorderStyle=3,Outline=1,Shadow=0,Fontsize=13"
        output = re.sub('\.','-sub.',stream['Video'])
        audio_stream = ffmpeg.input(stream['audio']).audio 
        video_stream = ffmpeg.input(stream['Video']).video

    first = (
            ffmpeg
            .input(options.videoPath)
            .filter('scale', width, height )
            .filter(
                "pad",
                "iw",             # largeur = largeur d’origine → pas de pad à gauche/droite
                f"ih+250",  # on ajoute du pad en bas
                0,                # x=0 → aligné à gauche
                0,                # y=0 → contenu collé en haut
                color="black"     # couleur du pad
            )
            .filter('subtitles',filename=file.name,force_style=style)
            .output(video_stream, audio_stream,output,vcodec='libx264')
            .run(overwrite_output=True)
    )

    for f in glob.glob(f'{options.audioDir}/chunk*.mp3'):
        os.remove(f)

