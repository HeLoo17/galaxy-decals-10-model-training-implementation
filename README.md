# AI Galaxy Explorer — Streamlit prototype

## Run locally

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Open the local Streamlit URL shown in the terminal.

## What is implemented

- Aladin Lite v3 embedded as a Streamlit custom component.
- Pan and zoom astronomy viewer.
- Draw-rectangle interaction.
- Converts the rectangle's four corners from view pixels to ICRS RA/Dec.
- Retrieves an approximate Legacy Survey DR10 JPEG cutout around the selected region.
- Upload-image testing tab.
- Clearly marked placeholders for the binary and Galaxy10 morphology models.

## Files to edit for your notebook models

In `app.py`, complete:

1. `build_binary_model()` — create the binary galaxy/non-galaxy architecture.
2. `build_morphology_model()` — create your exact ResNet18 architecture with 10 outputs.
3. `preprocess_for_models()` — reproduce the notebook preprocessing exactly.
4. `predict_binary()` — return `(is_galaxy, confidence)`.
5. `predict_morphology()` — return `(label, confidence, probabilities)`.

Then update `load_models()` to load the state dictionaries.

Expected weight paths:

- `models/galaxy_non_galaxy.pth`
- `models/galaxy10_resnet18.pth`

The model architecture, class index mapping, channel order, image size and normalization must match training.

## Important scientific limitation

The current cutout retrieval uses a rendered JPEG from the Legacy Survey viewer. It is suitable for prototyping the viewer-to-cutout interaction, but it is not guaranteed to reproduce the original Galaxy10 DECals g/r/z input arrays.

Before presenting external-image predictions as valid, retrieve the corresponding calibrated g, r and z band data (or an equivalent compatible data product), reproduce the notebook's preprocessing, and evaluate on a held-out external sample.

A binary detector also needs representative negative examples such as stars, empty sky, image artifacts and other non-galaxy sources.

## Aladin Lite selection implementation

The frontend uses Aladin Lite's `select("rect", callback)` API and `pix2world()` to convert the four rectangle corners to sky coordinates.

- Aladin Lite docs: https://cds-astro.github.io/aladin-lite/Aladin.html
- Streamlit custom components v1: https://docs.streamlit.io/develop/concepts/custom-components/components-v1/intro
- Legacy Survey viewer: https://www.legacysurvey.org/viewer

## Deployment notes

For Streamlit Community Cloud, push this folder to GitHub and set `app.py` as the entry point.

The model weight files may exceed GitHub's normal file-size limits. If so, store them using Git LFS or an appropriate model artifact hosting service, and download them securely during app startup. Keep the repository's `models/` folder free of large weights until you choose the hosting method.
