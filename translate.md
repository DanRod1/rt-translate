# Analyse Technique Complète du Pipeline SRT Basé sur Spleeter

## Résumé

Le script implémente un pipeline de génération de sous-titres basé sur :

1. extraction audio depuis une vidéo
2. séparation des stems avec Spleeter
3. comparaison voix/instrumental
4. génération d’une timeline vocale
5. projection proportionnelle du texte
6. génération SRT
7. burn FFmpeg

Le pipeline est fonctionnel conceptuellement mais présente plusieurs défauts structurels importants dans la logique d’alignement texte/audio.

Le problème principal est que le système mélange :

- régions vocales détectées
- unités textuelles
- découpage logique des paroles
- segmentation temporelle

sans modèle explicite de correspondance entre ces éléments.

---

# Architecture Générale

## Pipeline global

```text
VIDEO
  ↓
Extraction WAV mono 16k
  ↓
Spleeter 2 stems
  ↓
vocals.wav + accompaniment.wav
  ↓
Analyse RMS comparative
  ↓
Détection régions vocales
  ↓
Projection fragments texte
  ↓
Découpage proportionnel
  ↓
Segments SRT
  ↓
Burn FFmpeg


