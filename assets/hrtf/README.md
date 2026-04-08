# HRTF Asset Directory

Place `.sofa` files here for personalised HRTF profiles.

## Supported Sources

- [CIPIC HRTF Database](https://www.ece.ucdavis.edu/cipic/spatial-sound/hrtf-data/)
- [LISTEN HRTF Database](http://recherche.ircam.fr/equipes/salles/listen/)
- [3D Tune-In Toolkit](https://www.3d-tune-in.eu/)
- [SOFA Conventions](https://www.sofaconventions.org/)

## Format

Files must use the AES69-2022 SOFA format (SimpleFreeFieldHRIR convention).
Claudio loads these via `claudio.sofa_loader.load_sofa()`.

If no `.sofa` files are present, the system falls back to procedural
Woodworth-Schlosberg synthesis (see `hrtf_data.py`).
