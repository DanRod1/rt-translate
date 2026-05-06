# mulan_demucs_timeline.py

## Objectif

Générer une timeline de sous-titres depuis un fichier Mulan text et un média audio/vidéo sans ASR.

Le système repose sur :

- séparation vocale Demucs
- détection des blancs vocaux
- projection pondérée des fragments texte
- corrections manuelles de synchronisation dans le Mulan text

---

# Pipeline

```text
Mulan text
    +
média audio/vidéo
    ↓
traduction locale optionnelle
    ↓
conversion WAV
    ↓
Demucs vocals / no_vocals
    ↓
détection des blancs vocaux
    ↓
construction des régions temporelles
    ↓
projection pondérée des fragments Mulan
    ↓
exports SRT / LRC / JSON / TXT
    ↓
burn vidéo optionnel
```

---

# Philosophie du projet

Le projet cherche à contourner l’absence d’ASR.

Le Mulan text reste la source canonique de :

- l’ordre des sous-titres
- la structure logique
- la correction de synchronisation

Le système ne cherche pas à reconstruire automatiquement les paroles.

---

# Principes de synchronisation

## Source canonique

L’ordre du fichier Mulan est toujours prioritaire.

Le système ne réordonne jamais les sous-titres selon les timestamps.

---

# Format Mulan supporté

## Texte simple

```text
Je marche seul
```

---

## Onomatopées

```text
(oh oh)
(yeah yeah)
(hmmmmm)
```

---

## Métadonnées inline

```text
(oh oh -- dure: 5)
```

---

# Règles `dure:`

## Durée explicite

```text
(-- dure: 15)
```

ou :

```text
(texte -- dure: 15)
```

→ force une fenêtre temporelle explicite.

---

## Correction positive

```text
(-- dure: +1)
```

→ avance la timeline courante de 1 seconde.

Tous les fragments suivants héritent de cette correction.

---

## Correction négative

```text
(-- dure: -1)
```

→ recule la timeline courante de 1 seconde.

Tous les fragments suivants héritent également de cette correction.

---

# Exemple de correction cumulative

```text
ligne 1
(-- dure: +1)
ligne 2
ligne 3
```

Résultat :

```text
ligne 2 et ligne 3 sont décalées de +1 seconde.
```

---

# Exemple rétroactif

```text
ligne 1
(-- dure: -1)
ligne 2
ligne 3
```

Résultat :

```text
ligne 2 et ligne 3 commencent 1 seconde plus tôt.
```

Le système peut revenir dans une région vocale précédente.

---

# Séparation des textes

Le système distingue explicitement :

```text
display_text
original_text
phonetic_text
```

---

## display_text

Texte exporté dans :

- SRT
- LRC
- TXT

Orthographe conservée.

---

## original_text

Texte Mulan nettoyé.

Utilisé comme base logique.

---

## phonetic_text

Texte normalisé uniquement pour :

- poids temporel
- estimation phonétique
- projection temporelle

Jamais exporté.

---

# Normalisation phonétique

Le système retire certaines règles grammaticales peu audibles :

## Pluriels

```text
histoires -> histoire
grandes -> grand
```

---

## Conjugaisons

```text
parlent -> parle
aiment -> aime
parleraient -> parlerai
```

---

# Objectif de la normalisation

Réduire les désynchronisations liées à :

- marques grammaticales muettes
- longueurs textuelles trompeuses
- débit vocal réel différent du texte écrit

---

# Pondération phonétique

Le poids temporel repose sur :

- syllabes
- unités phonétiques
- allongements vocaliques
- répétitions consonantiques
- ponctuation expressive
- onomatopées

---

# Exemples

## Allongement vocalique

```text
Ooooooh
```

→ poids augmenté.

---

## Ponctuation expressive

```text
Je suis là...
```

→ poids augmenté.

---

## Onomatopée

```text
(hmmmm)
```

→ pondération spéciale.

---

# Traduction locale

## Principe

La traduction est optionnelle et locale via `transformers`.

Le timing reste calculé sur le texte source.

---

# Exemple

```bash
--input-language fr
--output-language es
--translation-model /models/opus-mt-fr-es
```

---

# Important

La traduction :

- ne modifie pas le timing
- ne modifie pas les poids
- ne modifie pas les régions vocales

Elle agit uniquement sur le texte affiché.

---

# Détection des régions vocales

Le système utilise :

```text
Demucs vocals.wav
```

Puis :

- conversion mono
- détection amplitude
- détection des blancs
- construction des régions

---

# Construction des régions

Les régions sont construites entre les blancs détectés.

Les frontières peuvent être :

- début du blanc
- milieu du blanc
- fin du blanc

selon :

```bash
--blank-boundary-position
```

---

# Détection des blancs

Le système utilise :

```python
SILENCE_AMPLITUDE_THRESHOLD
```

et :

```bash
--min-blank-duration
```

---

# Compression des durées fixes

Si les durées fixes dépassent le temps disponible :

```text
dure explicite
+
onomatopées fixes
```

alors le système applique :

```text
compression proportionnelle
```

---

# Exports générés

## TXT

```text
mulan_clean_lines.txt
```

---

## JSON

```text
mulan_demucs_timeline.json
```

---

## SRT

```text
mulan_demucs_timeline.srt
```

---

## LRC

```text
mulan_demucs_timeline.lrc
```

---

## Enhanced LRC

```text
mulan_demucs_timeline.enhanced.lrc
```

---

# Burn vidéo

Optionnel via ffmpeg.

Le système incruste automatiquement le SRT dans la vidéo.

---

# Modes de timing

## syllables

Pondération syllabique simple.

---

## phonemes

Pondération phonétique avancée.

---

## hybrid

Mode recommandé.

Combine :

- poids phonétique
- poids syllabique

---

# Dépendances principales

- ffmpeg
- demucs
- numpy
- soundfile
- phonemizer
- transformers (optionnel)

---

# Aucun ASR

Le pipeline n’utilise :

- ni Whisper
- ni WhisperX
- ni alignement forcé ASR

Toute la synchronisation repose sur :

```text
texte Mulan
+
régions vocales Demucs
+
corrections manuelles éventuelles
```

---

# Design important

Le système privilégie :

- traitement offline
- reproductibilité
- contrôle manuel
- absence de dépendance API

Aucune étape ne nécessite un fournisseur LLM.

# Mode opérateur

## Commande minimale

```bash
python mulan_demucs_timeline.py \
  --mulan-text paroles.mulan.txt \
  --input-media chanson.mp4 \
  --output-dir ./out
```

---

## Commande recommandée

```bash
python mulan_demucs_timeline.py \
  --mulan-text paroles.mulan.txt \
  --input-media chanson.mp4 \
  --output-dir ./out \
  --input-language fr \
  --output-language es \
  --translation-model /home/drodriguez/dev/Helsinki-NLP/opus-mt-fr-es \
  --phoneme-language fr-fr \
  --phoneme-backend espeak \
  --timing-mode hybrid \
  --demucs-model htdemucs \
  --min-blank-duration 0.25 \
  --min-voice-duration 0.5 \
  --blank-boundary-position middle \
  --pad-start 0.05 \
  --pad-end 0.10
```

---

## Sans burn vidéo

```bash
python mulan_demucs_timeline.py \
  --mulan-text paroles.mulan.txt \
  --input-media chanson.mp4 \
  --output-dir ./out \
  --no-burn-video
```

---

# Détail des arguments

## `--mulan-text` / `-m`

Fichier texte Mulan contenant les paroles.

```bash
--mulan-text paroles.mulan.txt
```

Obligatoire.

---

## `--input-media` / `-a`

Fichier audio ou vidéo source.

```bash
--input-media chanson.mp4
```

Formats typiques :

```text
mp3, wav, flac, mp4, mkv, mov
```

Obligatoire.

---

## `--output-dir` / `-o`

Dossier de sortie.

```bash
--output-dir ./out
```

Par défaut :

```text
./out
```

---

## `--input-language`

Langue source du texte.

```bash
--input-language fr
```

Exemples :

```text
fr
en
es
auto
```

Par défaut :

```text
auto
```

---

## `--output-language`

Langue cible de traduction.

```bash
--output-language es
```

Vide = traduction désactivée.

```bash
--output-language ""
```

---

## `--translation-model`

Chemin local ou nom du modèle de traduction `transformers`.

```bash
--translation-model /home/drodriguez/dev/Helsinki-NLP/opus-mt-fr-es
```

La traduction est active uniquement si :

```text
--output-language non vide
+
--translation-model non vide
```

---

## `--translation-max-length`

Longueur maximale de génération du modèle de traduction.

```bash
--translation-max-length 256
```

Par défaut :

```text
256
```

---

## `--translation-allow-download`

Autorise `transformers` à télécharger un modèle absent localement.

```bash
--translation-allow-download
```

Par défaut, le mode local est privilégié.

---

## `--phoneme-language`

Langue utilisée par `phonemizer`.

```bash
--phoneme-language fr-fr
```

Exemples :

```text
fr-fr
en-us
es
```

---

## `--phoneme-backend`

Backend phonétique.

```bash
--phoneme-backend espeak
```

Valeur courante :

```text
espeak
```

---

## `--timing-mode`

Mode de calcul du poids temporel.

```bash
--timing-mode hybrid
```

Valeurs possibles :

```text
syllables
phonemes
hybrid
```

### `syllables`

Utilise principalement le nombre de syllabes.

### `phonemes`

Utilise la pondération phonétique avancée.

### `hybrid`

Mode recommandé.

Combine :

```text
75% phonétique
25% syllabique
```

---

## `--onomatopoeia-fixed-duration`

Durée fixe des onomatopées sans durée explicite.

```bash
--onomatopoeia-fixed-duration 0.65
```

Par défaut :

```text
0.65 seconde
```

Exemple :

```text
(oh oh)
```

sera projeté avec une durée fixe si aucune durée `dure:` n’est donnée.

---

## `--demucs-model`

Modèle Demucs utilisé.

```bash
--demucs-model htdemucs
```

Par défaut :

```text
htdemucs
```

---

## `--min-blank-duration`

Durée minimale d’un blanc pour créer une frontière temporelle.

```bash
--min-blank-duration 0.25
```

Plus la valeur est basse, plus le système détecte de coupures.

Plus la valeur est haute, plus les régions sont longues.

---

## `--min-voice-duration`

Durée minimale d’une activité vocale conservée.

```bash
--min-voice-duration 0.5
```

Attention : une valeur trop haute peut supprimer des vocalisations courtes.

Exemples concernés :

```text
oh
ah
hey
cris courts
attaques syllabiques
```

---

## `--blank-boundary-position`

Position utilisée dans le blanc pour couper la timeline.

```bash
--blank-boundary-position middle
```

Valeurs possibles :

```text
start
middle
end
```

### `start`

Coupe au début du blanc.

### `middle`

Coupe au milieu du blanc.

Mode équilibré.

### `end`

Coupe à la fin du blanc.

---

## `--pad-start`

Marge ajoutée au début des régions vocales.

```bash
--pad-start 0.05
```

Permet d’éviter de couper une attaque vocale.

---

## `--pad-end`

Marge ajoutée à la fin des régions vocales.

```bash
--pad-end 0.10
```

Permet d’éviter de couper une fin de phrase chantée.

---

## `--no-burn-video`

Désactive l’incrustation des sous-titres dans la vidéo.

```bash
--no-burn-video
```

Sans cet argument, le burn vidéo est actif par défaut.

---

# Lecture opérateur des sorties

Après exécution, le script affiche :

```text
VOCAL       : chemin vocals.wav
INSTRUMENTAL: chemin no_vocals.wav
TXT         : chemin texte nettoyé
JSON        : chemin debug complet
SRT         : chemin SRT
LRC         : chemin LRC
ELRC        : chemin enhanced LRC
VIDEO       : chemin vidéo sous-titrée
```

---

# Fichier JSON de diagnostic

Le fichier :

```text
mulan_demucs_timeline.json
```

contient :

```text
meta
fragments
blank_boundaries
vocal_regions
subtitles
```

---

## `meta`

Résumé global du traitement.

Contient notamment :

```text
modèle Demucs
langues
mode timing
nombre de fragments
nombre de corrections sync
nombre de blancs détectés
nombre de régions vocales
chemins de sortie
```

---

## `fragments`

Liste des fragments Mulan analysés.

Chaque fragment contient :

```text
original_text
translated_text
tokens
syllable_count
phonemes
phoneme_count
weight
is_onomatopoeia
explicit_duration
sync_offset
```

---

## `blank_boundaries`

Liste des blancs vocaux détectés.

Chaque blanc contient :

```text
start
end
duration
boundary_time
```

---

## `vocal_regions`

Régions temporelles construites entre les blancs.

Chaque région contient :

```text
start
end
duration
start_boundary_reason
end_boundary_reason
```

---

## `subtitles`

Sous-titres finaux exportés.

Chaque sous-titre contient :

```text
start
end
text
original_text
translated_text
fragment_id
source_line
sync_offset
```

---

# Mode correction opérateur

## Décaler toute la suite vers l’avant

```text
(-- dure: +1)
```

Effet :

```text
tous les fragments suivants commencent 1 seconde plus tard
```

---

## Décaler toute la suite vers l’arrière

```text
(-- dure: -1)
```

Effet :

```text
tous les fragments suivants commencent 1 seconde plus tôt
```

---

## Forcer la durée d’un fragment

```text
(oh oh -- dure: 3)
```

Effet :

```text
le fragment dure 3 secondes
```

---

# Workflow recommandé

## 1. Premier passage

Lancer avec les paramètres standards.

```bash
python mulan_demucs_timeline.py \
  -m paroles.mulan.txt \
  -a chanson.mp4 \
  -o ./out
```

---

## 2. Vérifier le SRT

Ouvrir :

```text
./out/mulan_demucs_timeline.srt
```

ou la vidéo générée.

---

## 3. Corriger dans le Mulan text

Ajouter uniquement les corrections nécessaires :

```text
(-- dure: +0.5)
(-- dure: -0.25)
(oh oh -- dure: 2)
```

---

## 4. Relancer

```bash
python mulan_demucs_timeline.py \
  -m paroles.mulan.txt \
  -a chanson.mp4 \
  -o ./out
```

---

# Règle opérateur importante

Ne pas corriger chaque ligne manuellement.

La correction `dure:+/-X` est cumulative.

Elle doit être utilisée comme un recalage de timeline à partir d’un point précis.

---

# Exemple complet Mulan

```text
Je marche seul
Dans la nuit qui tombe
(-- dure: +0.4)
Et je t'appelle encore
(oh oh -- dure: 2)
(-- dure: -0.3)
Mais personne ne répond
```

Effet :

```text
les lignes après +0.4 sont avancées dans la timeline
l’onomatopée dure explicitement 2 secondes
les lignes après -0.3 sont recalées plus tôt
```
