# rt-translate
Simple script which extract the audio part of a youtube video à translate with hudgeFace model Opus  

usage is :  
rt-translate -v | --verbose          => simple args with no value to verbosity  
             -i | --inputLanguage    => simple string defining the language of the video by default fr means French  
             -o | --outputLanguage   => simple string defining the translate language for the video by default ru means russian  
             -a | --audioDir         => simple string defining th cache and working directory by default ./Audio  


# 🎬 Stems-Driven Subtitle Generator transform.py

Génération automatique de sous-titres (SRT) à partir d’un texte (lyrics) en utilisant une **timeline dérivée des stems audio (vocals vs instrumental)**.

⚠️ Aucun forced alignment (`aeneas`) n’est utilisé.  
La timeline est **entièrement pilotée par la détection vocale**.

---

# 🧠 Principe

Le pipeline fonctionne ainsi :

1. Extraction audio depuis la vidéo
2. Séparation audio en :
   - voix (`vocals`)
   - instrumental (`instrumental`)
3. Détection des zones vocales
4. Projection du texte sur ces zones
5. Génération du SRT
6. Incrustation dans la vidéo

---

# 🔁 Pipeline

```text
vidéo
  → extraction audio
  → séparation stems (Spleeter)
  → vocals.wav / instrumental.wav
  → analyse RMS (vocals vs instrumental)
  → vocal_regions
  → projection du texte
  → segments SRT
  → export SRT
  → burn dans la vidéo