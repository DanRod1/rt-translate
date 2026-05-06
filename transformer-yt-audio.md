VIDEO
  ↓
FFmpeg extract audio
  ↓
16k mono WAV
  ↓
Spleeter
  ↓
vocals.wav + accompaniment.wav
  ↓
RMS analysis
  ↓
vocal regions
  ↓
text fragmentation
  ↓
text projection onto regions
  ↓
subtitle segments
  ↓
SRT
  ↓
FFmpeg subtitle burn

# Documentation d’Utilisation — Générateur SRT Basé sur Stems Audio

# Présentation

Ce script génère automatiquement des sous-titres `.srt` à partir :

- d’une vidéo
- d’un fichier texte / paroles

Le système :

1. extrait l’audio
2. sépare voix et instrumental avec Spleeter
3. détecte les régions vocales
4. projette le texte sur la timeline détectée
5. génère un fichier SRT
6. incruste les sous-titres dans la vidéo

---

# Principe de Fonctionnement

Le script ne réalise PAS :

- d’alignement phonétique
- de forced alignment
- de reconnaissance vocale ASR

Le système fonctionne uniquement via :

```text
comparaison énergétique vocals.wav vs accompaniment.wav
```

Le texte est ensuite réparti statistiquement sur les régions vocales détectées.

---

# Dépendances

# Python

Recommandé :

```bash
python >= 3.10
```

---

# FFmpeg

FFmpeg doit être installé et accessible dans le PATH.

Vérification :

```bash
ffmpeg -version
```

---

# Dépendances Python

Installation :

```bash
pip install ffmpeg-python spleeter deep-translator numpy
```

---

# Dépendances système possibles

Selon environnement :

```bash
apt install ffmpeg libsndfile1
```

---

# Structure Générale

```text
video.mp4
lyrics.txt
↓
script.py
↓
output.srt
output_subtitled.mp4
```

---

# Utilisation Minimale

```bash
python script.py \
  -V video.mp4 \
  -T lyrics.txt
```

---

# Résultats générés

Par défaut :

```text
output.srt
output_subtitled.mp4
```

---

# Exemple Complet

```bash
python script.py \
  -V song.mp4 \
  -T lyrics.txt \
  -S subtitles.srt \
  -X subtitled.mp4 \
  --translate-to fr \
  --vocal-dominance-threshold 0.8 \
  --analysis-window 0.08 \
  --min-vocal-region 1.5 \
  --merge-gap 0.12 \
  --chars-per-second 14 \
  --max-lines-per-subtitle 2 \
  --subtitle-offset -0.2 \
  --verbose
```

---

# Arguments CLI

# Entrées / sorties

## `-V`, `--video-input`

Vidéo source.

```bash
-V input.mp4
```

Obligatoire.

---

## `-T`, `--text-input`

Fichier texte source.

```bash
-T lyrics.txt
```

Obligatoire.

---

## `-S`, `--subtitle-output`

Chemin du SRT généré.

Défaut :

```text
output.srt
```

---

## `-X`, `--video-output`

Vidéo finale avec sous-titres burnés.

Défaut :

```text
output_subtitled.mp4
```

---

# Fichiers temporaires

## `--audio-wav`

Audio extrait temporaire.

Défaut :

```text
audio_16k_mono.wav
```

---

## `--vocals-wav`

Stem vocal généré.

Défaut :

```text
vocals_16k_mono.wav
```

---

## `--instrumental-wav`

Stem instrumental généré.

Défaut :

```text
instrumental_16k_mono.wav
```

---

## `--spleeter-output-dir`

Répertoire temporaire Spleeter.

Défaut :

```text
spleeter_output
```

---

## `--clean-text`

Texte nettoyé intermédiaire.

Défaut :

```text
cleaned_text.txt
```

---

# Traduction

## `--translate-to`

Traduit les fragments avant génération.

Exemple :

```bash
--translate-to fr
```

Langues typiques :

```text
fr
en
es
de
it
ja
ko
```

---

# Détection vocale

# `--vocal-dominance-threshold`

Seuil :

```text
vocals_rms / instrumental_rms
```

Défaut :

```text
1.0
```

---

## Valeurs recommandées

### Détection permissive

```bash
--vocal-dominance-threshold 0.5
```

### Équilibré

```bash
--vocal-dominance-threshold 0.8
```

### Très strict

```bash
--vocal-dominance-threshold 1.2
```

---

# `--analysis-window`

Taille fenêtre RMS en secondes.

Défaut :

```text
0.1
```

---

## Valeurs typiques

### Détection fine

```bash
--analysis-window 0.05
```

### Stable

```bash
--analysis-window 0.12
```

---

# `--min-vocal-region`

Durée minimale d’une région vocale.

Défaut :

```text
3.0
```

---

## Attention

Valeur trop élevée :

- supprime phrases courtes
- supprime interjections
- supprime respirations

---

## Recommandations

### Chant dense

```bash
--min-vocal-region 0.8
```

### Discours lent

```bash
--min-vocal-region 2.0
```

---

# `--merge-gap`

Fusionne régions proches.

Défaut :

```text
0.18
```

---

## Exemple

Si deux régions sont séparées par :

```text
< 0.18 secondes
```

elles sont fusionnées.

---

# Gestion du texte

# `--min-text-chars`

Taille minimale avant fusion de sous-régions.

Défaut :

```text
28
```

---

## Effets

Valeur faible :

- plus de sous-titres
- plus fragmenté

Valeur élevée :

- plus de fusion
- sous-titres plus longs

---

# `--chars-per-second`

Contrôle largeur dynamique des sous-titres.

Défaut :

```text
14
```

---

## Valeurs recommandées

### Lecture lente

```bash
--chars-per-second 10
```

### Standard

```bash
--chars-per-second 14
```

### Rapide

```bash
--chars-per-second 18
```

---

# `--max-lines-per-subtitle`

Nombre maximal de lignes.

Défaut :

```text
1
```

---

## Exemple

```bash
--max-lines-per-subtitle 2
```

---

# Décalage temporel

# `--subtitle-offset`

Décale tous les sous-titres.

---

## Retarder

```bash
--subtitle-offset 0.5
```

---

## Avancer

```bash
--subtitle-offset -0.25
```

---

# Style des sous-titres

# `--style`

Style ASS injecté dans FFmpeg.

Défaut :

```text
FontName=Arial,FontSize=18,BorderStyle=3,Outline=1,Shadow=0,MarginV=20
```

---

# Exemple personnalisé

```bash
--style "FontName=Arial,FontSize=24,PrimaryColour=&H00FFFFFF,Outline=2"
```

---

# Debug / Diagnostic

# `--verbose`

Active logs détaillés.

```bash
--verbose
```

---

# `--dump-vocal-regions`

Sauvegarde les régions détectées.

Exemple :

```bash
--dump-vocal-regions regions.txt
```

Format :

```text
0.000    2.513
3.002    5.871
...
```

---

# `--keep-temp`

Conserve fichiers temporaires.

```bash
--keep-temp
```

Utile pour :

- debug
- inspection stems
- analyse timeline

---

# Pipeline Interne

# Étape 1 — Extraction audio

```text
video.mp4
↓
audio_16k_mono.wav
```

---

# Étape 2 — Séparation stems

```text
audio.wav
↓
vocals.wav
accompaniment.wav
```

---

# Étape 3 — Détection vocale

Calcul :

```text
vocals_rms / instrumental_rms
```

---

# Étape 4 — Construction régions

```text
(start, end)
```

---

# Étape 5 — Projection texte

Le texte est réparti :

```text
proportionnellement à la durée des régions
```

---

# Étape 6 — Génération SRT

```text
1
00:00:00,000 --> 00:00:02,000
Hello world
```

---

# Étape 7 — Burn vidéo

FFmpeg :

```text
subtitles=output.srt
```

---

# Limitations Importantes

# Ce système n’est PAS un forced aligner

Il ne connaît pas :

- phonèmes
- syllabes
- mots réellement chantés

---

# Le système effectue :

```text
projection statistique du texte
```

---

# Conséquences

Le timing peut être :

- approximatif
- dérivant
- incorrect sur certains morceaux

---

# Cas difficiles

# Genres problématiques

- metal
- EDM
- orchestral
- hip-hop compressé
- mixages très denses

---

# Cas difficiles

- voix faibles
- choeurs
- réverbérations fortes
- voix doublées
- voix très compressées

---

# Paramètres recommandés

# Musique dense

```bash
--vocal-dominance-threshold 0.5 \
--min-vocal-region 0.8 \
--analysis-window 0.06
```

---

# Podcast / voix claire

```bash
--vocal-dominance-threshold 1.0 \
--min-vocal-region 2.0
```

---

# Sous-titres plus lisibles

```bash
--max-lines-per-subtitle 2 \
--chars-per-second 12
```

---

# Workflow recommandé debug

```bash
python script.py \
  -V input.mp4 \
  -T lyrics.txt \
  --keep-temp \
  --dump-vocal-regions regions.txt \
  --verbose
```

Puis vérifier :

- vocals.wav
- accompaniment.wav
- regions.txt
- output.srt

---

# Problèmes Courants

# Aucun segment détecté

Erreur :

```text
No vocal regions detected from stem comparison.
```

---

## Causes possibles

- seuil trop élevé
- voix faibles
- séparation Spleeter mauvaise

---

## Solution

Réduire :

```bash
--vocal-dominance-threshold 0.5
```

---

# Sous-titres trop fusionnés

Réduire :

```bash
--min-text-chars
```

Exemple :

```bash
--min-text-chars 12
```

---

# Sous-titres trop fragmentés

Augmenter :

```bash
--min-text-chars 40
```

---

# Mauvais timing

Tester :

```bash
--analysis-window 0.05
```

ou :

```bash
--analysis-window 0.12
```

---

# Décalage global

Corriger avec :

```bash
--subtitle-offset
```

---

# Conclusion

Ce script est :

- un générateur de sous-titres heuristique
- basé sur énergie audio
- sans alignement phonétique réel

---

# Nature réelle du système

Le pipeline réalise :

```text
détection approximative de régions vocales
+
projection statistique du texte
```

et non :

```text
alignement texte/audio exact
```
