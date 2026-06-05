# Documentation — Pipeline Mulan / YouTube ASR vers ASS

## 1. Objectif

Ce script génère un fichier de sous-titres **ASS** à partir de deux sources :

1. une transcription YouTube ASR récupérée via `youtube_transcript_api` ;
2. un fichier texte Mulan contenant le texte de référence.

Le but est de produire des sous-titres plus propres que l'ASR brut, avec :

- alignement du texte Mulan sur les timestamps YouTube ;
- traduction locale optionnelle ;
- sortie au format `.ass` ;
- gestion des chevauchements par position verticale ;
- limitation du nombre de caractères par ligne ;
- correction des gros blocs fusionnés générés par certains alignements ;
- burn vidéo optionnel via FFmpeg.

---

## 2. Pourquoi ASS au lieu de SRT

Le format SRT est limité :

- pas de vraie notion de position verticale par événement ;
- pas de styles avancés ;
- pas de contrôle fiable des sous-titres simultanés ;
- comportement variable selon les lecteurs.

Le format ASS permet :

- `MarginV` pour positionner verticalement les sous-titres ;
- `Style` pour définir police, taille, contour, alignement ;
- `Layer` pour définir la priorité de rendu ;
- événements simultanés ;
- sauts de ligne explicites via `\N`.

Dans ce projet, ASS est utilisé pour éviter de reconstruire artificiellement des blocs multi-lignes comme en SRT.

---

## 3. Pipeline global

```text
config.yaml
   ↓
lecture URL YouTube + fichier Mulan
   ↓
récupération transcript YouTube ASR
   ↓
chargement lignes Mulan
   ↓
alignement Mulan ↔ ASR par distance de caractères
   ↓
application des offsets/durées Mulan
   ↓
traduction optionnelle
   ↓
correction des blocs fusionnés
   ↓
limitation des caractères par ligne
   ↓
assignation des pistes verticales ASS
   ↓
écriture .ass / .txt / debug.tsv
   ↓
burn vidéo optionnel
```

---

## 4. Structure des segments

Le script manipule des objets `CaptionSegment`.

```python
@dataclass
class CaptionSegment:
    start: float
    duration: float
    text: str
    source_text: str = ""
    distance_ratio: float = 1.0
    mulan_index: int = -1
    inserted: bool = False
```

### Champs

| Champ | Rôle |
|---|---|
| `start` | Début du segment en secondes |
| `duration` | Durée du segment en secondes |
| `text` | Texte final du sous-titre |
| `source_text` | Texte ASR source utilisé pour debug |
| `distance_ratio` | Distance normalisée entre ASR et Mulan |
| `mulan_index` | Index de la ligne Mulan alignée |
| `inserted` | Indique si le segment a été inséré artificiellement |

---

## 5. Chargement du fichier Mulan

Le fichier Mulan est lu ligne par ligne.

Les formats acceptés sont :

```text
Texte simple
text=Texte
lyric=Texte
line=Texte
fragment=Texte
colonne1|colonne2|Texte
colonne1<TAB>colonne2<TAB>Texte
```

Les lignes ignorées :

```text
# commentaire
// commentaire
ligne vide
```

Le nettoyage retire :

- balises `<...>` ;
- blocs `[...]` ;
- blocs `{...}` ;
- espaces multiples.

---

## 6. Métadonnées inline Mulan

Le script supporte des métadonnées entre parenthèses :

```text
(Texte -- dure: 2.5)
(Texte -- dure: +0.4)
(Texte -- dure: -0.2)
```

### Interprétation

| Forme | Effet |
|---|---|
| `dure: 2.5` | force une durée explicite de 2.5 secondes |
| `dure: +0.4` | ajoute un offset cumulatif de +0.4 seconde |
| `dure: -0.2` | ajoute un offset cumulatif de -0.2 seconde |

Les offsets sont cumulés selon l'ordre des lignes Mulan.

---

## 7. Alignement Mulan ↔ ASR

L'alignement se fait par distance de Levenshtein normalisée.

Le texte est normalisé :

- passage en minuscules ;
- retrait de ponctuation ;
- conservation des caractères accentués ;
- suppression des espaces.

La distance est calculée ainsi :

```python
distance = levenshtein(normalized_asr, normalized_mulan) / max_length
```

Un match est accepté si :

```yaml
distance <= alignment.max_distance_ratio
```

Par défaut :

```yaml
max_distance_ratio: 0.10
```

---

## 8. Fenêtre de recherche

Pour chaque segment ASR, le script cherche une correspondance Mulan dans une fenêtre limitée.

```yaml
alignment:
  search_window: 12
```

Cela veut dire :

```text
chercher parmi les 12 prochaines lignes Mulan à partir du curseur courant
```

Cette stratégie évite de réaligner trop loin dans le texte, mais peut échouer si le texte Mulan diverge fortement de l'ASR.

---

## 9. Gestion des lignes Mulan manquantes

Quand le script trouve un match Mulan plus loin que le curseur courant, il y a des lignes Mulan intermédiaires non alignées.

Ancienne logique problématique :

```text
créer des segments artificiels avant le match
```

Cela provoquait des chevauchements.

Nouvelle logique :

```text
fusionner seulement quelques lignes manquantes dans le segment matché
```

La fusion est limitée par :

```yaml
alignment:
  max_merged_chars: 90
  max_missing_lines: 2
```

### Effet

Si Mulan contient :

```text
ligne A
ligne B
ligne C
```

et que `ligne C` est matchée avec l'ASR, alors le script peut produire :

```text
ligne B ligne C
```

ou :

```text
ligne A ligne B ligne C
```

mais seulement si les limites `max_merged_chars` et `max_missing_lines` le permettent.

---

## 10. Pourquoi limiter les merges

Une fusion illimitée peut produire de très gros blocs :

```text
phrase 1 phrase 2 phrase 3 phrase 4 phrase 5 phrase 6 ...
```

Ce type de bloc est difficile à lire et perturbe ensuite :

- le wrapping ;
- la traduction ;
- la distribution sur les timelines ;
- la lisibilité dans ASS.

Les paramètres importants sont :

```yaml
max_merged_chars: 90
max_missing_lines: 2
```

Recommandations :

| Cas | Valeur conseillée |
|---|---|
| Sous-titres courts | `max_merged_chars: 60` |
| Chansons / paroles | `max_merged_chars: 90` |
| Texte parlé dense | `max_merged_chars: 110` |
| Éviter gros blocs | `max_missing_lines: 1` |
| Autoriser rattrapage léger | `max_missing_lines: 2` |

---

## 11. Correction des blocs fusionnés globaux

Certains alignements produisent un gros segment contenant une traduction correcte de plusieurs petits segments précédents.

Exemple problématique :

```ass
Dialogue: 0,0:00:19.32,0:00:25.76,...,La ciudad del mes de mayo tiene labios
Dialogue: 1,0:00:22.76,0:00:25.76,...,- ¿Qué?
Dialogue: 2,0:00:26.52,0:00:32.16,...,La hermosa mariquita te invita a ti
...
Dialogue: 9,0:01:03.56,0:01:11.84,...,La Bella del mes de mayo tiene ...
```

Dans ce cas, le segment 9 contient parfois le bon texte global, mais avec une mauvaise timeline.

La correction appliquée est :

```text
garder le texte du gros bloc
redistribuer ce texte sur les timelines des petits fragments
supprimer le gros bloc comme événement unique
```

Paramètres :

```yaml
alignment:
  merged_fix_max_group_gap: 1.25
  merged_fix_min_group_segments: 3
  merged_fix_duration_factor: 1.8
  merged_fix_text_ratio: 0.55
```

### Rôle des paramètres

| Paramètre | Rôle |
|---|---|
| `merged_fix_max_group_gap` | gap maximal entre fragments pour les considérer dans le même groupe |
| `merged_fix_min_group_segments` | nombre minimal de segments dans un groupe |
| `merged_fix_duration_factor` | détecte si un segment couvre une grande partie du groupe |
| `merged_fix_text_ratio` | détecte si un segment contient beaucoup plus de texte que les autres |

---

## 12. Redistribution du texte fusionné

Quand un gros bloc est détecté, son texte est découpé proportionnellement aux durées des petits segments.

Exemple :

```text
texte fusionné :
"La Bella del mes de mayo tiene labios hinchados ..."

timelines :
19.32 → 25.76
22.76 → 25.76
26.52 → 32.16
```

Résultat :

```text
19.32 → 25.76 : La Bella del mes de mayo...
22.76 → 25.76 : ...
26.52 → 32.16 : ...
```

La découpe se fait par mots, pas par caractères, afin d'éviter de couper un mot au milieu.

---

## 13. Traduction locale

La traduction est optionnelle.

Configuration :

```yaml
translation:
  enabled: false
  model_path: "/home/drodriguez/models/opus-mt-fr-es"
  source_language: "fr"
  target_language: "es"
  max_length: 256
  local_files_only: true
  device: "cpu"
  num_beams: 1
  log_progress: true
  skip_if_target_detected: true
```

Le modèle est chargé via Hugging Face Transformers :

```python
AutoTokenizer.from_pretrained(...)
AutoModelForSeq2SeqLM.from_pretrained(...)
```

### Mode local

Si `local_files_only: true`, le chemin local doit exister.

Sinon le script lève :

```text
FileNotFoundError: Modèle local introuvable
```

### CPU

Pour une machine sans GPU NVIDIA :

```yaml
device: "cpu"
```

---

## 14. Détection de langue simplifiée

Le script utilise une détection très simple par mots marqueurs.

Exemples pour le français :

```text
je, tu, il, elle, les, des, que, qui, pas, dans, avec, pour
```

Cette détection sert uniquement à éviter de retraduire un segment déjà dans la langue cible.

Paramètre :

```yaml
skip_if_target_detected: true
```

Si la détection provoque trop de faux positifs, utiliser :

```yaml
skip_if_target_detected: false
```

---

## 15. Limitation du nombre de caractères par ligne

Avant génération du fichier ASS, le texte est reformaté selon :

```yaml
ass:
  max_chars_per_line: 42
```

Exemple :

```text
This is a very long subtitle that should not stay on a single line
```

devient :

```text
This is a very long subtitle that
should not stay on a single line
```

Dans ASS, les retours ligne sont écrits sous forme :

```text
\N
```

---

## 16. Gestion verticale des chevauchements

ASS ne décale pas automatiquement les sous-titres qui se chevauchent.

Le script assigne donc une piste verticale à chaque segment.

```text
track 0 = ligne basse
track 1 = ligne au-dessus
track 2 = encore au-dessus
```

Le positionnement est fait via `MarginV`.

```python
margin_v_for_track = margin_v + track * line_gap
```

Configuration :

```yaml
ass:
  margin_v: 45
  line_gap: 72
  max_tracks: 3
```

### Effet

| Track | Position |
|---|---|
| `0` | bas |
| `1` | au-dessus |
| `2` | encore au-dessus |

---

## 17. Comportement demandé : la première ligne monte

Quand un nouveau sous-titre chevauche un ancien :

```text
ancien actif
nouveau arrive
```

Le comportement voulu est :

```text
ancien monte
nouveau prend la ligne basse
```

Visuellement :

```text
ancien
nouveau
```

Le script décale donc les anciens segments actifs vers le haut quand un nouveau segment arrive.

---

## 18. Notion de Layer ASS

Dans ASS, `Layer` ne signifie pas piste verticale.

Exemple :

```ass
Dialogue: 0,0:00:10.00,0:00:15.00,Default,,0,0,45,,Texte A
Dialogue: 1,0:00:10.00,0:00:15.00,Default,,0,0,45,,Texte B
```

Ici :

```text
Layer 1 est dessiné au-dessus de Layer 0
```

Mais les deux textes restent à la même position si `MarginV` est identique.

### Important

```text
Layer ≠ position verticale
Layer = priorité de rendu
```

Dans ce projet, la position verticale est gérée par :

```text
MarginV
```

pas par :

```text
Layer
```

Le code final écrit donc :

```ass
Dialogue: 0,...
```

pour tous les événements, et utilise `MarginV` pour les pistes.

---

## 19. Format ASS généré

Exemple de header :

```ass
[Script Info]
ScriptType: v4.00+
WrapStyle: 2
ScaledBorderAndShadow: yes
PlayResX: 1920
PlayResY: 1080
```

Style :

```ass
[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Default,Arial,54,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,2,0,2,60,60,45,1
```

Événement :

```ass
Dialogue: 0,0:00:19.32,0:00:25.76,Default,,0,0,117,,Texte
```

Champs :

```text
Dialogue: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
```

---

## 20. Échappement ASS

Le texte est échappé avant écriture.

Transformations :

| Entrée | Sortie |
|---|---|
| `\` | `\\` |
| `{` | `\{` |
| `}` | `\}` |
| retour ligne | `\N` |

Cela évite que des caractères spéciaux soient interprétés comme des tags ASS.

---

## 21. Burn vidéo avec FFmpeg

Si `burn.enabled: true`, le script produit une vidéo avec sous-titres incrustés.

Commande utilisée :

```bash
ffmpeg -y \
  -i input.mp4 \
  -vf "ass='file.ass'" \
  -c:v libx264 \
  -preset veryfast \
  -crf 23 \
  -c:a copy \
  output.mp4
```

Configuration :

```yaml
burn:
  enabled: false
  input_video: ""
  output_video: ""
```

Si `input_video` est vide, le script tente de télécharger la vidéo YouTube via `yt-dlp`.

---

## 22. Téléchargement YouTube optionnel

Configuration :

```yaml
youtube_download:
  enabled: false
  format: "bv*[height<=720]+ba/b[height<=720]/best"
  cookies_from_browser: ""
  cookies: ""
  remote_components: ""
  no_check_certificate: false
```

Le téléchargement utilise :

```bash
python -m yt_dlp --no-playlist
```

Options supportées :

| Option | Rôle |
|---|---|
| `cookies_from_browser` | récupère les cookies depuis un navigateur |
| `cookies` | utilise un fichier cookies |
| `remote_components` | option yt-dlp |
| `no_check_certificate` | désactive la vérification TLS |
| `format` | sélection du format vidéo/audio |

---

## 23. Fichiers de sortie

Le script génère :

```text
out/<video_id>.mulan.ass
out/<video_id>.mulan.txt
out/<video_id>.debug.tsv
```

ou, si traduction activée :

```text
out/<video_id>.translated.ass
out/<video_id>.translated.txt
out/<video_id>.debug.tsv
```

### `.ass`

Fichier principal de sous-titres.

### `.txt`

Texte final brut, une entrée par segment.

### `.debug.tsv`

Fichier de debug avec :

```text
start
end
distance_ratio
inserted
mulan_index
asr_text
final_text
```

---

## 24. Configuration complète

```yaml
youtube_url: "https://www.youtube.com/watch?v=XXXXXXXXXXX"
mulan_file: "./mulan.txt"

languages:
  - fr
  - en

output_dir: "./out"

alignment:
  max_distance_ratio: 0.10
  search_window: 12
  insert_missing_between_matches: true
  max_merged_chars: 90
  max_missing_lines: 2

  merged_fix_max_group_gap: 1.25
  merged_fix_min_group_segments: 3
  merged_fix_duration_factor: 1.8
  merged_fix_text_ratio: 0.55

translation:
  enabled: false
  model_path: "/home/drodriguez/models/opus-mt-fr-es"
  source_language: "fr"
  target_language: "es"
  max_length: 256
  local_files_only: true
  device: "cpu"
  num_beams: 1
  log_progress: true
  skip_if_target_detected: true

ass:
  play_res_x: 1920
  play_res_y: 1080
  font_name: Arial
  font_size: 54
  margin_v: 45
  line_gap: 72
  max_tracks: 3
  max_chars_per_line: 42

youtube_download:
  enabled: false
  format: "bv*[height<=720]+ba/b[height<=720]/best"
  cookies_from_browser: ""
  cookies: ""
  remote_components: ""
  no_check_certificate: false

burn:
  enabled: false
  input_video: ""
  output_video: ""
```

---

## 25. Paramètres recommandés

### Cas général

```yaml
alignment:
  max_distance_ratio: 0.10
  search_window: 12
  max_merged_chars: 90
  max_missing_lines: 2

ass:
  max_chars_per_line: 42
  max_tracks: 3
```

### Si trop de mauvais matchs

```yaml
alignment:
  max_distance_ratio: 0.06
  search_window: 8
```

### Si trop peu de matchs

```yaml
alignment:
  max_distance_ratio: 0.15
  search_window: 18
```

### Si les blocs sont trop longs

```yaml
alignment:
  max_merged_chars: 60
  max_missing_lines: 1

ass:
  max_chars_per_line: 36
```

### Si les sous-titres montent trop haut

```yaml
ass:
  max_tracks: 2
  line_gap: 60
```

### Si les sous-titres se chevauchent encore visuellement

```yaml
ass:
  line_gap: 90
  max_tracks: 4
```

---

## 26. Dépannage

### Les sous-titres sont trop longs

Réduire :

```yaml
ass:
  max_chars_per_line: 36
```

et/ou :

```yaml
alignment:
  max_merged_chars: 60
  max_missing_lines: 1
```

---

### Le texte correct apparaît comme gros bloc final

Activer ou ajuster :

```yaml
alignment:
  merged_fix_max_group_gap: 1.25
  merged_fix_min_group_segments: 3
  merged_fix_duration_factor: 1.8
  merged_fix_text_ratio: 0.55
```

Si le bloc n'est pas détecté, essayer :

```yaml
merged_fix_text_ratio: 0.40
merged_fix_duration_factor: 2.5
```

---

### Les sous-titres montent mais l'ordre semble inversé

Rappel :

```text
track 0 = bas
track 1 = au-dessus
track 2 = encore au-dessus
```

Le comportement voulu est :

```text
ancien monte
nouveau reste en bas
```

Si tu veux l'inverse, il faut modifier `assign_ass_tracks`.

---

### Le champ Layer augmente dans le fichier ASS

Ce n'est pas souhaité dans la version finale.

Le code doit écrire :

```python
f"Dialogue: 0,{start},{end},Default,,"
```

et non :

```python
f"Dialogue: {idx},{start},{end},Default,,"
```

La piste verticale doit être portée par :

```text
MarginV
```

pas par :

```text
Layer
```

---

### FFmpeg échoue avec le filtre ASS

Vérifier :

```bash
ffmpeg -filters | grep ass
```

Le build FFmpeg doit inclure libass.

Tester aussi le chemin :

```bash
ls -l out/video.ass
```

Si le chemin contient des caractères spéciaux, le script applique déjà un échappement via `ass_filter_path`.

---

### Le modèle de traduction est introuvable

Avec :

```yaml
local_files_only: true
```

le chemin doit exister :

```bash
ls -l /home/drodriguez/models/opus-mt-fr-es
```

Sinon désactiver le mode local :

```yaml
local_files_only: false
```

ou télécharger le modèle localement.

---

## 27. Commande d'exécution

Par défaut :

```bash
python script.py
```

Avec un fichier de config explicite :

```bash
CONFIG=config.yaml python script.py
```

---

## 28. Dépendances Python

```bash
pip install pyyaml youtube-transcript-api transformers torch yt-dlp
```

Si traduction désactivée, `transformers` et `torch` ne sont nécessaires que si le module est importé dans l'environnement.

Pour le burn vidéo :

```bash
sudo apt install ffmpeg
```

---

## 29. Limites connues

1. L'alignement repose sur une distance de caractères, pas sur un vrai alignement phonétique.
2. La détection de langue est volontairement simplifiée.
3. La redistribution du gros bloc se fait proportionnellement aux durées, pas au sens linguistique.
4. Les longues phrases peuvent être coupées au mauvais endroit si le texte source est très compact.
5. `max_tracks` limite le nombre de pistes visibles ; au-delà, des segments peuvent partager une piste.
6. Le format ASS ne résout pas automatiquement les chevauchements : le script les gère via `MarginV`.

---

## 30. Résumé de la logique finale

```text
Mulan fournit le texte de référence.
YouTube ASR fournit la timeline.
Le script aligne Mulan sur ASR.
Les fragments manquants sont fusionnés avec limite.
Les gros blocs corrects sont redistribués sur les petites timelines.
La traduction est optionnelle.
Le texte final est wrapé.
ASS positionne les sous-titres par MarginV.
Layer reste à 0.
FFmpeg peut incruster le fichier ASS dans la vidéo.
```