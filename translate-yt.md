# Documentation `config.yaml`

Ce fichier configure le pipeline :

YouTube ASR → alignement Mulan → ajustement timeline → traduction optionnelle → génération SRT/TXT/debug → burn vidéo optionnel.

## Exemple complet

```yaml
youtube_url: "https://www.youtube.com/watch?v=XXXXXXXXXXX"
mulan_file: "./mulan.txt"
languages: ["fr", "en"]
output_dir: "./out"

alignment:
  max_distance_ratio: 0.15
  search_window: 20
  insert_missing_between_matches: true

subtitle_stack:
  enabled: true
  max_lines: 3
  min_duration: 0.15

translation:
  enabled: true
  source_language: fr
  target_language: en
  model_path: /home/drodriguez/dev/opus-mt-fr-en
  local_files_only: true
  device: cpu
  num_beams: 1
  max_length: 256
  log_progress: true
  skip_if_target_detected: true

youtube_download:
  enabled: false
  cookies_from_browser: chrome
  remote_components: ejs:github
  format: "bv*[height<=720]+ba/b[height<=720]/best"
  no_check_certificate: false

burn:
  enabled: false
  input_video: ""
  output_video: ""
```

---

# Paramètres racine

## `youtube_url`

URL YouTube ou identifiant vidéo YouTube.

Exemples :

```yaml
youtube_url: "https://www.youtube.com/watch?v=XXXXXXXXXXX"
```

ou :

```yaml
youtube_url: "XXXXXXXXXXX"
```

Obligatoire.

---

## `mulan_file`

Chemin vers le fichier texte Mulan servant de référence textuelle.

```yaml
mulan_file: "./mulan.txt"
```

Obligatoire.

Le fichier Mulan sert uniquement de référence textuelle et de support de corrections. Il ne remplace pas directement la timeline YouTube.

---

## `languages`

Liste des langues à demander à `youtube_transcript_api`.

```yaml
languages: ["fr", "en"]
```

Le script essaie de récupérer une transcription YouTube dans cet ordre.

Exemples :

```yaml
languages: ["fr"]
```

```yaml
languages: ["es", "fr", "en"]
```

---

## `output_dir`

Répertoire de sortie.

```yaml
output_dir: "./out"
```

Le script génère notamment :

```text
out/<video_id>.mulan.srt
out/<video_id>.mulan.txt
out/<video_id>.debug.tsv
```

Si la traduction est activée :

```text
out/<video_id>.translated.srt
out/<video_id>.translated.txt
out/<video_id>.debug.tsv
```

---

# Bloc `alignment`

Configure l’alignement entre la transcription ASR YouTube et le fichier Mulan.

```yaml
alignment:
  max_distance_ratio: 0.15
  search_window: 20
  insert_missing_between_matches: true
```

---

## `alignment.max_distance_ratio`

Seuil maximum de différence accepté entre une ligne ASR et une ligne Mulan.

```yaml
max_distance_ratio: 0.15
```

La valeur est basée sur une distance de Levenshtein normalisée :

```text
0.00 = textes identiques
0.05 = très proche
0.10 = strict
0.15 = recommandé
0.20 = tolérant
0.30 = risqué
```

Recommandation :

```yaml
max_distance_ratio: 0.15
```

Pour ASR YouTube + paroles Mulan, `0.10` peut être trop strict.

---

## `alignment.search_window`

Nombre maximum de lignes Mulan regardées à partir de la position courante.

```yaml
search_window: 20
```

Ce paramètre évite que le script aille matcher un refrain identique beaucoup plus loin dans le fichier.

Valeurs recommandées :

```text
8 à 12  = strict
15 à 20 = recommandé
30+     = plus tolérant mais risque de mauvais match
```

---

## `alignment.insert_missing_between_matches`

Autorise l’insertion de lignes Mulan manquantes entre deux lignes alignées.

```yaml
insert_missing_between_matches: true
```

Si `true`, lorsque le script trouve une ligne Mulan plus loin que la position attendue, il insère les lignes intermédiaires juste avant le segment trouvé.

Si `false`, les lignes non matchées ne sont pas ajoutées.

Pour éviter les accumulations parasites :

```yaml
insert_missing_between_matches: false
```

Pour un karaoké plus complet :

```yaml
insert_missing_between_matches: true
```

Important : le script ne doit pas ajouter automatiquement toutes les lignes restantes en fin de SRT.

---

# Bloc `subtitle_stack`

Configure l’affichage multi-lignes de bas en haut.

```yaml
subtitle_stack:
  enabled: true
  max_lines: 3
  min_duration: 0.15
```

---

## `subtitle_stack.enabled`

Active ou désactive l’empilement temporel des sous-titres.

```yaml
enabled: true
```

Si activé, plusieurs lignes actives peuvent être affichées dans un même bloc SRT.

---

## `subtitle_stack.max_lines`

Nombre maximum de lignes visibles simultanément.

```yaml
max_lines: 3
```

Exemple visuel :

```text
ligne récente
ligne précédente
ligne ancienne
```

En SRT, la dernière ligne du bloc est affichée en bas. Le script inverse donc l’ordre pour construire visuellement la timeline de bas en haut.

---

## `subtitle_stack.min_duration`

Durée minimale d’un bloc SRT généré par découpage temporel.

```yaml
min_duration: 0.15
```

Cela évite de générer des blocs trop courts.

Valeurs recommandées :

```text
0.10 = très réactif
0.15 = recommandé
0.25 = plus stable
```

---

# Bloc `translation`

Configure la traduction locale via un modèle Hugging Face compatible `transformers`.

```yaml
translation:
  enabled: true
  source_language: fr
  target_language: en
  model_path: /home/drodriguez/dev/opus-mt-fr-en
  local_files_only: true
  device: cpu
  num_beams: 1
  max_length: 256
  log_progress: true
  skip_if_target_detected: true
```

---

## `translation.enabled`

Active ou désactive la traduction.

```yaml
enabled: true
```

Si `false`, le SRT final utilise le texte Mulan aligné.

---

## `translation.source_language`

Langue source attendue.

```yaml
source_language: fr
```

Exemples :

```yaml
source_language: fr
source_language: es
source_language: en
```

---

## `translation.target_language`

Langue cible.

```yaml
target_language: en
```

Si `source_language` et `target_language` sont identiques, la traduction est désactivée automatiquement.

---

## `translation.model_path`

Chemin local ou nom Hugging Face du modèle.

```yaml
model_path: /home/drodriguez/dev/opus-mt-fr-en
```

Exemples locaux :

```yaml
model_path: /home/drodriguez/dev/opus-mt-fr-en
model_path: /home/drodriguez/dev/opus-mt-fr-es
```

---

## `translation.local_files_only`

Force l’utilisation de fichiers locaux uniquement.

```yaml
local_files_only: true
```

Recommandé si le modèle a déjà été téléchargé localement.

---

## `translation.device`

Périphérique d’exécution.

```yaml
device: cpu
```

Valeurs typiques :

```yaml
device: cpu
device: cuda
```

Sur machine sans NVIDIA, utiliser :

```yaml
device: cpu
```

---

## `translation.num_beams`

Nombre de beams pour la génération.

```yaml
num_beams: 1
```

Recommandation CPU :

```yaml
num_beams: 1
```

Un nombre plus élevé peut améliorer certains résultats mais ralentit fortement.

---

## `translation.max_length`

Longueur maximale générée par le modèle.

```yaml
max_length: 256
```

Pour des paroles courtes, `128` ou `256` suffit généralement.

---

## `translation.log_progress`

Affiche la progression segment par segment.

```yaml
log_progress: true
```

---

## `translation.skip_if_target_detected`

Évite de retraduire une ligne qui semble déjà être dans la langue cible.

```yaml
skip_if_target_detected: true
```

Exemple : si `source_language: fr` et `target_language: en`, une ligne déjà détectée comme anglaise est conservée telle quelle.

Utile quand les paroles mélangent français et anglais.

---

# Bloc `youtube_download`

Configure le téléchargement vidéo par `yt-dlp`.

```yaml
youtube_download:
  enabled: false
  cookies_from_browser: chrome
  remote_components: ejs:github
  format: "bv*[height<=720]+ba/b[height<=720]/best"
  no_check_certificate: false
```

---

## `youtube_download.enabled`

Active le téléchargement de la vidéo YouTube.

```yaml
enabled: true
```

Nécessaire uniquement si `burn.enabled: true` et si aucun `burn.input_video` local n’est fourni.

---

## `youtube_download.cookies_from_browser`

Utilise les cookies du navigateur.

```yaml
cookies_from_browser: chrome
```

Utile pour les vidéos nécessitant session, âge, restrictions ou challenge YouTube.

---

## `youtube_download.cookies`

Chemin vers un fichier cookies exporté.

```yaml
cookies: "./cookies.txt"
```

Alternative à `cookies_from_browser`.

---

## `youtube_download.remote_components`

Charge les composants JavaScript distants nécessaires à certains challenges YouTube.

```yaml
remote_components: ejs:github
```

Utile avec `yt-dlp` lorsque YouTube impose un challenge `n`.

---

## `youtube_download.format`

Format demandé à `yt-dlp`.

```yaml
format: "bv*[height<=720]+ba/b[height<=720]/best"
```

Ce format limite la vidéo à 720p maximum.

Pour plus léger :

```yaml
format: "bv*[height<=480]+ba/b[height<=480]/best"
```

---

## `youtube_download.no_check_certificate`

Désactive la vérification TLS.

```yaml
no_check_certificate: false
```

À laisser à `false` sauf problème réseau spécifique.

---

# Bloc `burn`

Configure l’incrustation du SRT dans la vidéo avec FFmpeg.

```yaml
burn:
  enabled: false
  input_video: ""
  output_video: ""
```

---

## `burn.enabled`

Active le burn vidéo.

```yaml
enabled: true
```

Si activé, le script génère une vidéo avec sous-titres incrustés.

---

## `burn.input_video`

Chemin vers une vidéo locale.

```yaml
input_video: "/home/drodriguez/Vidéos/input.mp4"
```

Si ce champ est renseigné, le script n’a pas besoin de télécharger la vidéo YouTube.

---

## `burn.output_video`

Chemin de sortie de la vidéo burnée.

```yaml
output_video: "./out/video_burned.mp4"
```

Si vide, le script utilise :

```text
out/<video_id>.burned.mp4
```

---

# Exemple minimal sans traduction

```yaml
youtube_url: "https://www.youtube.com/watch?v=XXXXXXXXXXX"
mulan_file: "./mulan.txt"
languages: ["fr"]
output_dir: "./out"

alignment:
  max_distance_ratio: 0.15
  search_window: 20
  insert_missing_between_matches: true

subtitle_stack:
  enabled: false

translation:
  enabled: false

youtube_download:
  enabled: false

burn:
  enabled: false
```

---

# Exemple avec traduction FR → EN

```yaml
youtube_url: "https://www.youtube.com/watch?v=XXXXXXXXXXX"
mulan_file: "./mulan.txt"
languages: ["fr", "en"]
output_dir: "./out"

alignment:
  max_distance_ratio: 0.15
  search_window: 20
  insert_missing_between_matches: true

subtitle_stack:
  enabled: true
  max_lines: 3
  min_duration: 0.15

translation:
  enabled: true
  source_language: fr
  target_language: en
  model_path: /home/drodriguez/dev/opus-mt-fr-en
  local_files_only: true
  device: cpu
  num_beams: 1
  max_length: 256
  log_progress: true
  skip_if_target_detected: true

youtube_download:
  enabled: false

burn:
  enabled: false
```

---

# Exemple avec burn vidéo locale

```yaml
youtube_url: "https://www.youtube.com/watch?v=XXXXXXXXXXX"
mulan_file: "./mulan.txt"
languages: ["fr"]
output_dir: "./out"

alignment:
  max_distance_ratio: 0.15
  search_window: 20
  insert_missing_between_matches: true

subtitle_stack:
  enabled: true
  max_lines: 3
  min_duration: 0.15

translation:
  enabled: false

youtube_download:
  enabled: false

burn:
  enabled: true
  input_video: "/home/drodriguez/Vidéos/input.mp4"
  output_video: "./out/input_burned.mp4"
```

---

# Exemple avec téléchargement YouTube + burn

```yaml
youtube_url: "https://www.youtube.com/watch?v=XXXXXXXXXXX"
mulan_file: "./mulan.txt"
languages: ["fr"]
output_dir: "./out"

alignment:
  max_distance_ratio: 0.15
  search_window: 20
  insert_missing_between_matches: true

subtitle_stack:
  enabled: true
  max_lines: 3
  min_duration: 0.15

translation:
  enabled: false

youtube_download:
  enabled: true
  cookies_from_browser: chrome
  remote_components: ejs:github
  format: "bv*[height<=720]+ba/b[height<=720]/best"
  no_check_certificate: false

burn:
  enabled: true
  input_video: ""
  output_video: "./out/youtube_burned.mp4"
```

---

# Recommandation par défaut

Pour ton usage actuel, configuration conseillée :

```yaml
alignment:
  max_distance_ratio: 0.15
  search_window: 20
  insert_missing_between_matches: true

subtitle_stack:
  enabled: true
  max_lines: 3
  min_duration: 0.15

translation:
  enabled: true
  source_language: fr
  target_language: en
  model_path: /home/drodriguez/dev/opus-mt-fr-en
  local_files_only: true
  device: cpu
  num_beams: 1
  max_length: 256
  log_progress: true
  skip_if_target_detected: true
```

