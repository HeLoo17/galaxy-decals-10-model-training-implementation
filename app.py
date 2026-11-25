"""
AI Galaxy Explorer — Streamlit prototype

Run:
    pip install streamlit requests pillow numpy torch torchvision
    streamlit run app.py

Model integration:
    The Galaxy10 morphology pipeline is fully wired. Copy the notebook's
    export outputs into models/:

        models/galaxy10_resnet18.pth   (required — focal-loss weights)
        models/export_metadata.json    (recommended — training config)

    Any analysed image is resized automatically to the training input
    recorded in the metadata (fallback 128x128, LANCZOS + /255 +
    ImageNet normalization). For sky snaps the app additionally re-fetches
    the selected position at the Galaxy10 training pixel scale (0.262"/px),
    auto-centres on the brightest source and crops the ~67" training field
    before inference, so the network sees the same effective resolution it
    was trained on. The binary galaxy/non-galaxy detector is an optional
    placeholder.

IMPORTANT:
Aladin's sky view is for navigation/selection. The Legacy Survey JPEG
cutout is a display composite, NOT guaranteed to be numerically equivalent
to the three g/r/z channels used by Galaxy10 DECals. Do not claim the
notebook's validation accuracy applies to these external images until
you retrieve compatible band data and validate cross-survey performance.
"""

from __future__ import annotations

import io
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

try:
    import torch
    import torch.nn.functional as F
    import torchvision
except ImportError:
    torch = None
    F = None
    torchvision = None


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

APP_DIR = Path(__file__).parent
FRONTEND_DIR = APP_DIR / "frontend"
MODEL_DIR = APP_DIR / "models"

BINARY_MODEL_PATH = MODEL_DIR / "galaxy_non_galaxy.pth"
MORPHOLOGY_MODEL_PATH = MODEL_DIR / "galaxy10_resnet18.pth"
METADATA_PATH = MODEL_DIR / "export_metadata.json"

# Fallback input configuration when export_metadata.json is absent;
# they reproduce the notebook defaults (CNN_SIZE, ImageNet stats).
FALLBACK_IMG_SIZE = 128
FALLBACK_MEAN = [0.485, 0.456, 0.406]
FALLBACK_STD = [0.229, 0.224, 0.225]

# Pillow moved LANCZOS to Image.Resampling in Pillow >= 9.1.
try:
    LANCZOS_RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:  # pragma: no cover - older Pillow
    LANCZOS_RESAMPLE = Image.LANCZOS

GALAXY10_LABELS = [
    "Disturbed",
    "Merging",
    "Round Smooth",
    "In-between Round Smooth",
    "Cigar Shaped Smooth",
    "Barred Spiral",
    "Unbarred Tight Spiral",
    "Unbarred Loose Spiral",
    "Edge-on without Bulge",
    "Edge-on with Bulge",
]

LEGACY_CUTOUT_URL = "https://www.legacysurvey.org/viewer/cutout.jpg"

# Legacy Survey cutout geometry (documented limits of the DR10 service).
#   - max 512 px per side ("The maximum size for cutouts (in number of
#     pixels) is currently 512", legacysurvey.org DR10 description)
#   - native pixel scale 0.262 arcsec/px, the same scale the Galaxy10
#     DECaLS training crops use
#   - the app zooms out at most to 2.0 arcsec/px (512 px then covers ~17')
CUTOUT_MAX_PIXELS = 512
CUTOUT_MIN_PIXSCALE = 0.262
CUTOUT_MAX_PIXSCALE = 2.0
# Margin added around the drawn selection so objects touching the selection
# edge are not clipped in the snapped image.
CUTOUT_PADDING = 0.10

# Model-view acquisition: reproduce the Galaxy10 DECaLS training crop
# exactly. Training data: 256x256 px at the native 0.262"/px (~67" field)
# centred on the galaxy, LANCZOS-resized to the model input. For prediction
# the app re-fetches the selected position at that same pixel scale and
# crops the same ~67" field, independent of how wide the drawn selection
# was, so the effective pixel scale always matches training.
MODEL_VIEW_PIXELS = 512       # fetched frame at the native scale (134" field)
MODEL_VIEW_CROP_PX = 256      # Galaxy10 DECaLS original crop size
MODEL_VIEW_MAX_SHIFT_PX = 96  # re-centring clamp (~25" of sky)
MODEL_VIEW_SEARCH_FRAC = 0.5  # centroid search window = central 50% of frame

# Predefined navigation targets for the Aladin viewer. Keys are shown in the
# sidebar dropdown, values are the identifiers passed to Aladin Lite and
# resolved by Sesame (Messier/NGC names and coordinates both work).
SKY_TARGETS = {
    "Andromeda Galaxy (M31)": "M 31",
    "Triangulum Galaxy (M33)": "M 33",
    "Whirlpool Galaxy (M51)": "M 51",
    "Pinwheel Galaxy (M101)": "M 101",
    "Sombrero Galaxy (M104)": "M 104",
    "Orion Nebula (M42)": "M 42",
    "NGC 1300": "NGC 1300",
    "Milky Way Centre (Sgr A*)": "Sgr A*",
}

DEFAULT_SKY_TARGET = "NGC 1300"

st.set_page_config(
    page_title="AI Galaxy Explorer",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------
# Aladin Lite Streamlit component
# ---------------------------------------------------------------------

@st.cache_resource
def get_aladin_component():
    return components.declare_component(
        "aladin_rectangle_selector",
        path=str(FRONTEND_DIR),
    )


def render_aladin():
    """Render Aladin Lite and receive selected rectangle sky coordinates."""
    aladin_component = get_aladin_component()
    return aladin_component(
        height=650,
        target=st.session_state.get("sky_target", DEFAULT_SKY_TARGET),
        survey=st.session_state.get("sky_survey", "P/DSS2/color"),
        fov=float(st.session_state.get("sky_fov", 0.5)),
        # Incremented by the "Load sky" button so the viewer re-centres even
        # when the target identifier itself has not changed.
        target_nonce=int(st.session_state.get("target_nonce", 0)),
        key="aladin_sky_component",
        default=None,
    )


# ---------------------------------------------------------------------
# Galaxy10 model wiring (exported by the notebook, cell 5.4)
# ---------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_model_metadata() -> Optional[dict]:
    """
    Read models/export_metadata.json written by notebook cell 5.4.

    It records the training-time image size, class order and
    normalization statistics, so the app can resize ANY picture to the
    exact dimension the model was trained on without manual
    configuration. Returns None (the app then falls back to 128 px and
    ImageNet statistics) when the file is missing or malformed.
    """
    if not METADATA_PATH.exists():
        return None
    try:
        return json.loads(METADATA_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def model_input_config() -> tuple[int, list, list, list]:
    """
    (img_size, class_names, mean, std) for preprocessing.

    Values come from export_metadata.json; the fallbacks reproduce the
    notebook defaults (CNN_SIZE = 128, ImageNet mean/std).
    """
    meta = load_model_metadata() or {}
    raw_size = meta.get("img_size", FALLBACK_IMG_SIZE)
    try:
        img_size = int(raw_size)
    except (TypeError, ValueError):
        img_size = FALLBACK_IMG_SIZE
    if img_size <= 0:
        img_size = FALLBACK_IMG_SIZE
    classes_raw = meta.get("classes")
    classes = (
        [str(c) for c in classes_raw] if classes_raw else list(GALAXY10_LABELS)
    )
    norm = meta.get("normalization") or {}
    mean = [float(v) for v in norm.get("mean", FALLBACK_MEAN)]
    std = [float(v) for v in norm.get("std", FALLBACK_STD)]
    return img_size, classes, mean, std


def build_morphology_model(num_classes: int = 10):
    """
    Rebuild the notebook architecture exactly: torchvision ResNet-18
    (ImageNet backbone during training) with the classifier head
    replaced by Linear(512, num_classes) — identical to the notebook's
    build_model().
    """
    if torch is None or torchvision is None:
        raise RuntimeError("PyTorch and torchvision are required.")
    model = torchvision.models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


@st.cache_resource
def load_models():
    """
    Load the notebook-exported Galaxy10 morphology model.

    Expects models/galaxy10_resnet18.pth (focal-loss best-epoch
    state_dict, CPU tensors). The binary galaxy/non-galaxy detector is
    still optional and returned as None until you train and wire it;
    the app then classifies images directly with the Galaxy10 model.
    """
    if torch is None or torchvision is None:
        return None, None
    if not MORPHOLOGY_MODEL_PATH.exists():
        return None, None

    _img_size, classes, _mean, _std = model_input_config()
    model = build_morphology_model(num_classes=len(classes))
    state = torch.load(
        MORPHOLOGY_MODEL_PATH, map_location="cpu", weights_only=True
    )
    model.load_state_dict(state)
    model.eval()
    return None, model


def preprocess_for_models(image: Image.Image):
    """
    AUTOMATIC RESIZE to the model's training dimension + exact notebook
    preprocessing.

    Any input image (snapped cutout, uploaded photo, any aspect ratio)
    is LANCZOS-resized to img_size x img_size — taken from
    export_metadata.json (img_size), falling back to 128 — then scaled
    to [0, 1] and normalized with the training statistics, reproducing
    the notebook's GalaxyDataset(train=False) pipeline:

        uint8 RGB -> float / 255.0 -> (x - mean) / std

    Returns a float tensor shaped [1, 3, img_size, img_size].
    """
    img_size, _classes, mean, std = model_input_config()
    resized = image.convert("RGB").resize(
        (img_size, img_size), LANCZOS_RESAMPLE
    )
    x = torch.from_numpy(
        np.asarray(resized, dtype=np.float32) / 255.0
    ).permute(2, 0, 1)
    mean_t = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    return ((x - mean_t) / std_t).unsqueeze(0)


def predict_binary(binary_model, image: Image.Image):
    """
    TODO: still a placeholder — wire your binary detector here.

    Must return (is_galaxy: bool, confidence: float) and ensure the
    positive class index really corresponds to 'galaxy'. Until a binary
    model is loaded, the app classifies every analysed image directly
    with the Galaxy10 model.
    """
    raise NotImplementedError("Implement binary galaxy/non-galaxy inference.")


def predict_morphology(morphology_model, image: Image.Image):
    """
    10-class morphology inference: preprocess -> forward -> softmax.

    Returns (predicted_label, confidence, probabilities). The label
    order comes from export_metadata.json, identical to the notebook's
    CLASS_NAMES training order.
    """
    if torch is None:
        raise RuntimeError("PyTorch is not available.")
    _img_size, classes, _mean, _std = model_input_config()
    with torch.no_grad():
        logits = morphology_model(preprocess_for_models(image))
        probs = torch.softmax(logits, dim=1)[0]
    confidence, idx = probs.max(0)
    return classes[int(idx)], float(confidence), [float(p) for p in probs]


# ---------------------------------------------------------------------
# Astronomy image retrieval
# ---------------------------------------------------------------------

def _fmt_angle(deg: float) -> str:
    """Format a small angle as arcseconds or arcminutes for display."""
    arcsec = deg * 3600.0
    if arcsec < 120.0:
        return f"{arcsec:.1f}\""
    return f"{arcsec / 60.0:.1f}'"


def _angular_extents(corners: list[list[float]]) -> tuple[float, float]:
    """
    Approximate the selection's angular width and height in degrees.

    Returns the RA/Dec extents of the bounding box around the four drawn
    corners, so the snapped cutout always covers the full selection even
    when the viewer is rotated or the corner list starts at a different
    vertex. RA spans are corrected for the convergence of meridians at the
    mean declination and wrap correctly across the 0/360 degree meridian.
    """
    if len(corners) < 4:
        raise ValueError("The selected rectangle did not return four corners.")

    ras = [float(p[0]) for p in corners]
    decs = [float(p[1]) for p in corners]

    # RA span measured around the circular mean, so selections crossing
    # RA=0 (e.g. 359.9 -> 0.1) are not inflated to a full circle.
    radians = np.radians(ras)
    center_ra = math.degrees(
        math.atan2(np.sin(radians).mean(), np.cos(radians).mean())
    )
    deltas = [((ra - center_ra + 180.0) % 360.0 - 180.0) for ra in ras]
    ra_span = max(deltas) - min(deltas)

    mean_dec = math.radians(float(np.mean(decs)))
    width = max(ra_span * math.cos(mean_dec), 1.0 / 3600.0)
    height = max(max(decs) - min(decs), 1.0 / 3600.0)
    return width, height


def selection_center(corners: list[list[float]]) -> tuple[float, float]:
    """Approximate RA/Dec center, handling RA wrap-around."""
    ras = [float(p[0]) for p in corners]
    decs = [float(p[1]) for p in corners]

    # Circular mean for RA avoids issues around 0/360 degrees.
    radians = np.radians(ras)
    mean_ra = math.degrees(
        math.atan2(np.sin(radians).mean(), np.cos(radians).mean())
    ) % 360.0
    return mean_ra, float(np.mean(decs))


def compute_cutout_geometry(width_deg: float, height_deg: float) -> dict:
    """
    Decide pixel scale and output size for a cutout request.

    The pixel scale is the coarsest one that still fits the (padded)
    selection inside the service's 512 px-per-side limit, so large
    selections are snapped at lower resolution instead of being cropped.
    Tiny selections stay at the DR10 native scale (0.262 arcsec/pixel,
    the same scale as the Galaxy10 DECaLS training crops).

    Returns the pixel scale, requested width/height in pixels, the sky
    coverage actually delivered, and a `cropped` flag for selections
    larger than the service can cover (512 px x 2.0 arcsec/px = ~17
    arcminutes).
    """
    span_w = width_deg * 3600.0 * (1.0 + CUTOUT_PADDING)  # arcsec
    span_h = height_deg * 3600.0 * (1.0 + CUTOUT_PADDING)

    pixscale = max(span_w, span_h) / CUTOUT_MAX_PIXELS
    pixscale = max(CUTOUT_MIN_PIXSCALE, min(CUTOUT_MAX_PIXSCALE, pixscale))

    width_px = int(min(CUTOUT_MAX_PIXELS, max(1, round(span_w / pixscale))))
    height_px = int(min(CUTOUT_MAX_PIXELS, max(1, round(span_h / pixscale))))

    covered_w = width_px * pixscale
    covered_h = height_px * pixscale
    # Half-pixel tolerance before flagging that the request was cropped.
    cropped = (
        covered_w < span_w - 0.5 * pixscale
        or covered_h < span_h - 0.5 * pixscale
    )

    return {
        "pixscale": pixscale,
        "width_px": width_px,
        "height_px": height_px,
        "covered_arcsec": (covered_w, covered_h),
        "cropped": cropped,
    }


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_legacy_survey_cutout(
    ra: float, dec: float, width_deg: float, height_deg: float
):
    """
    Fetch a colour JPEG cutout matching the drawn selection.

    The cutout is requested with the selection's own width and height
    (non-square selections stay non-square) plus CUTOUT_PADDING margin,
    so galaxies that touch the selection edges remain fully inside the
    snapped image.

    This is a visual prototype input. It is not a substitute for the
    calibrated individual g/r/z band arrays required for scientifically
    faithful Galaxy10 inference.
    """
    geometry = compute_cutout_geometry(width_deg, height_deg)
    params = {
        "ra": ra,
        "dec": dec,
        "layer": "ls-dr10",
        "pixscale": geometry["pixscale"],
        "width": geometry["width_px"],
        "height": geometry["height_px"],
    }
    response = requests.get(LEGACY_CUTOUT_URL, params=params, timeout=30)
    response.raise_for_status()
    image = Image.open(io.BytesIO(response.content)).convert("RGB")
    return {"image": image, **geometry}


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_model_view_cutout(ra: float, dec: float):
    """
    Fetch the training-scale frame used to build the model input.

    Requests a 512x512 px cutout at the DR10 native 0.262"/px — the same
    pixel scale as the Galaxy10 DECaLS training images. The margin around
    the central 256 px training crop gives the auto-centring room to find
    the object's centroid even when the selection centre is a little off.
    """
    params = {
        "ra": ra,
        "dec": dec,
        "layer": "ls-dr10",
        "pixscale": CUTOUT_MIN_PIXSCALE,
        "width": MODEL_VIEW_PIXELS,
        "height": MODEL_VIEW_PIXELS,
    }
    response = requests.get(LEGACY_CUTOUT_URL, params=params, timeout=30)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")


def find_bright_centroid(
    image,
    search_frac: float = MODEL_VIEW_SEARCH_FRAC,
    max_shift_px: float = MODEL_VIEW_MAX_SHIFT_PX,
):
    """
    Luminance-weighted centroid of the bright source near the frame centre.

    Pure pixel-space operation on the fetched cutout (no sky-coordinate
    maths): inside the central search window, pixels above a robust
    'background + noise' threshold are weighted by their excess luminance.
    The shift from the frame centre is clamped, so a bright neighbour
    (a star or another galaxy) cannot drag the crop away from the target.
    Returns (cx, cy) in pixels.
    """
    arr = np.asarray(image.convert("L"), dtype=np.float32)
    h, w = arr.shape
    centre_x, centre_y = w / 2.0, h / 2.0
    half = int(min(w, h) * search_frac / 2.0)
    if half < 8:
        return centre_x, centre_y

    x0, y0 = int(centre_x) - half, int(centre_y) - half
    window = arr[y0:y0 + 2 * half, x0:x0 + 2 * half]

    background = float(np.median(window))
    noise = 1.4826 * float(np.median(np.abs(window - background)))
    weights = np.clip(window - background - max(4.0 * noise, 15.0), 0.0, None)
    total = float(weights.sum())
    if total <= 0.0:
        return centre_x, centre_y

    ys, xs = np.indices(window.shape)
    sx = float((xs * weights).sum() / total)
    sy = float((ys * weights).sum() / total)
    dx = float(np.clip(sx - half, -max_shift_px, max_shift_px))
    dy = float(np.clip(sy - half, -max_shift_px, max_shift_px))
    return centre_x + dx, centre_y + dy


def build_model_view(frame_image, auto_center: bool = True) -> dict:
    """
    Crop the Galaxy10-scale view that the model should receive.

    Takes the training-scale frame (fetched at 0.262"/px), optionally
    re-centres it on the brightest source, and crops the central 256x256 px
    (~67" x 67") — the exact field of view of the Galaxy10 DECaLS training
    crops. `preprocess_for_models` then resizes this crop to the model
    input with the notebook's LANCZOS pipeline, so the acquisition geometry
    and preprocessing match training end to end.
    """
    w, h = frame_image.size
    centre_x, centre_y = w / 2.0, h / 2.0
    cx, cy = centre_x, centre_y
    centred_on = "selection centre"
    if auto_center:
        cx, cy = find_bright_centroid(frame_image)
        centred_on = "brightest source"

    crop_px = min(MODEL_VIEW_CROP_PX, w, h)
    left = int(round(min(max(cx - crop_px / 2.0, 0.0), w - crop_px)))
    top = int(round(min(max(cy - crop_px / 2.0, 0.0), h - crop_px)))
    crop = frame_image.crop((left, top, left + crop_px, top + crop_px))
    return {
        "image": crop,
        "centred_on": centred_on,
        "shift_px": (int(round(cx - centre_x)), int(round(cy - centre_y))),
        "crop_px": crop_px,
    }


# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------

st.title("🌌 AI Galaxy Explorer")
st.caption(
    "Explore astronomical sky surveys, select an object and analyse it "
    "with your Galaxy10-trained models."
)

with st.sidebar:
    st.header("Viewer settings")
    target_mode = st.radio(
        "Target selection mode",
        options=["Preset targets", "Custom target"],
        index=0,
        key="target_mode",
        help="Preset targets offer well-known objects in a dropdown. "
        "Custom target accepts any object name or RA/Dec in degrees.",
    )

    if target_mode == "Preset targets":
        selected_target = st.selectbox(
            "Select sky target",
            options=list(SKY_TARGETS.keys()),
            index=list(SKY_TARGETS.keys()).index(DEFAULT_SKY_TARGET),
            key="preset_sky_target",
        )
        resolved_target = SKY_TARGETS[selected_target]
        st.caption(f"Aladin target: `{resolved_target}`")
    else:
        custom_target = st.text_input(
            "Custom target",
            value=st.session_state.get("custom_sky_target", ""),
            key="custom_sky_target",
            help="Object name (e.g. M87, NGC 1300) or RA/Dec coordinates "
            "in degrees, e.g. 83.63 +22.01.",
        )
        resolved_target = custom_target.strip() or DEFAULT_SKY_TARGET

    # Consumed by the Aladin component; navigation happens in the browser
    # via gotoObject() without recreating the viewer.
    st.session_state["sky_target"] = resolved_target

    if st.button("Load sky / re-centre viewer"):
        st.session_state["target_nonce"] = (
            st.session_state.get("target_nonce", 0) + 1
        )
    st.selectbox(
        "Initial Aladin survey",
        options=[
            "P/DSS2/color",
            "P/DSS2/red",
            "P/DSS2/blue",
            "P/2MASS/color",
            "P/SDSS9/color",
            "P/allWISE/color",
        ],
        index=0,
        key="sky_survey",
        help="DSS2, 2MASS and allWISE cover the whole sky. SDSS9 only "
        "covers part of the northern sky — a black viewer at southern "
        "targets means no data there, not a bug. If a survey fails to "
        "load, the viewer falls back to DSS2 colour automatically.",
    )
    st.slider(
        "Initial field of view (degrees)",
        min_value=0.02,
        max_value=5.0,
        value=float(st.session_state.get("sky_fov", 0.5)),
        step=0.02,
        key="sky_fov",
    )

    st.divider()
    st.subheader("Model status")
    binary_model, morphology_model = load_models()
    if binary_model is not None:
        st.success("Galaxy / non-galaxy model loaded")
    else:
        st.warning("Galaxy detector: optional, not connected")

    if morphology_model is not None:
        st.success("Galaxy10 morphology model loaded")
        img_size, classes, _mean, _std = model_input_config()
        st.caption(
            f"Every picture is resized automatically to {img_size}×{img_size} px"
            f" (the model's training size, LANCZOS) before prediction — "
            f"{len(classes)} morphology classes. Sky snaps are additionally "
            "re-fetched at the training pixel scale (0.262\"/px) and "
            "auto-centred before inference."
        )
    else:
        st.warning(
            "Galaxy10 ResNet18: not loaded — copy models/galaxy10_resnet18.pth "
            "(and export_metadata.json) from the notebook's outputs/models/."
        )

    st.caption("Model weights are expected in the local `models/` folder.")


tab_explore, tab_upload, tab_about = st.tabs(
    ["🛰️ Sky Explorer", "📤 Upload an image", "ℹ️ About / setup"]
)

with tab_explore:
    st.subheader("Explore the sky")
    st.markdown(
        "Choose a sky target in the sidebar (preset dropdown or custom "
        "name/coordinates) and press **Load sky** to centre the viewer. "
        "Use Aladin Lite to pan and zoom. Click **Draw rectangle**, then drag "
        "a box around one object — any size or shape works, and the snapped "
        "cutout reproduces your box (plus a 10% margin) instead of forcing "
        "it into a square. The selected sky coordinates will be returned "
        "to this application."
    )

    selection = render_aladin()

    if selection and isinstance(selection, dict) and selection.get("corners"):
        corners = selection["corners"]
        try:
            ra, dec = selection_center(corners)
            width_deg, height_deg = _angular_extents(corners)

            st.session_state["selected_sky_region"] = {
                "corners": corners,
                "ra": ra,
                "dec": dec,
                "width_deg": width_deg,
                "height_deg": height_deg,
            }
        except Exception as exc:
            st.error(f"Could not interpret the selected region: {exc}")

    region = st.session_state.get("selected_sky_region")

    if region:
        st.success("A rectangular sky region has been selected.")
        col1, col2, col3 = st.columns(3)
        col1.metric("Right Ascension", f"{region['ra']:.6f}°")
        col2.metric("Declination", f"{region['dec']:.6f}°")
        col3.metric(
            "Selection size",
            f"{_fmt_angle(region['width_deg'])} × {_fmt_angle(region['height_deg'])}",
        )

        with st.expander("Selection coordinates"):
            st.json(region["corners"])

        if st.button("Retrieve selected image cutout", type="primary"):
            try:
                with st.spinner("Retrieving Legacy Survey cutout..."):
                    snap = fetch_legacy_survey_cutout(
                        region["ra"],
                        region["dec"],
                        region["width_deg"],
                        region["height_deg"],
                    )
                st.session_state["selected_cutout"] = snap["image"]
                st.session_state["selected_cutout_info"] = {
                    key: value for key, value in snap.items() if key != "image"
                }
                st.session_state["selected_cutout_source"] = "Legacy Survey DR10"
            except Exception as exc:
                st.error(f"Could not retrieve the image cutout: {exc}")

            # Training-scale frame for the model input (a second, cached
            # request at the Galaxy10 pixel scale — see 'Model input').
            try:
                with st.spinner("Fetching training-scale model view..."):
                    st.session_state["model_view_frame"] = fetch_model_view_cutout(
                        region["ra"], region["dec"]
                    )
                st.session_state["model_view_key"] = (region["ra"], region["dec"])
                st.session_state.pop("model_view_error", None)
            except Exception as exc:
                st.session_state["model_view_frame"] = None
                st.session_state["model_view_error"] = str(exc)

    cutout = st.session_state.get("selected_cutout")
    snap_info = st.session_state.get("selected_cutout_info")
    if cutout is not None:
        st.divider()
        st.subheader("Selected image")
        st.image(cutout, caption="Legacy Survey DR10 colour cutout", width=320)
        if snap_info:
            covered_w, covered_h = snap_info["covered_arcsec"]
            st.caption(
                f"Snapped {snap_info['width_px']}×{snap_info['height_px']} px "
                f"at {snap_info['pixscale']:.3f}\" per pixel — covers "
                f"{_fmt_angle(covered_w / 3600.0)} × "
                f"{_fmt_angle(covered_h / 3600.0)} of sky "
                f"(selection + {round(CUTOUT_PADDING * 100)}% margin)."
            )
        if snap_info and snap_info["cropped"]:
            st.warning(
                "This selection is larger than the biggest cutout the Legacy "
                "Survey service can return (512 px per side; at 2.0\" per "
                "pixel that is ≈17′ of sky). The image shows only the central "
                "part of the selection — draw a tighter rectangle to capture "
                "the whole object."
            )
        st.caption(
            "For prediction the app does not squeeze this (possibly wide) "
            "selection into the network. Below, it re-fetches the same spot "
            "at the Galaxy10 training scale (0.262\" per pixel, ≈67″ field), "
            "auto-centres it on the object and feeds that to the model — "
            "see 'Model input'."
        )
        st.warning(
            "Prototype note: this JPEG is a rendered colour composite. "
            "The Galaxy10 model was trained on g/r/z image channels; verify "
            "that the input data and preprocessing match before trusting "
            "predictions."
        )

        st.divider()
        st.subheader("Model input (auto-prepared)")
        img_size, _classes, _mean, _std = model_input_config()
        model_view_frame = st.session_state.get("model_view_frame")
        if (
            model_view_frame is None
            and st.session_state.get("model_view_key") is None
            and region
        ):
            # Sessions that already hold a snap from an older app version:
            # fetch the training-scale frame lazily, exactly once.
            try:
                model_view_frame = fetch_model_view_cutout(
                    region["ra"], region["dec"]
                )
                st.session_state["model_view_frame"] = model_view_frame
                st.session_state["model_view_key"] = (region["ra"], region["dec"])
            except Exception as exc:
                st.session_state["model_view_error"] = str(exc)

        model_view = None
        if model_view_frame is not None:
            auto_center = st.checkbox(
                "Auto-centre the model view on the brightest source",
                value=True,
                key="auto_center_model_view",
                help="Locates the luminance centroid near the selection "
                "centre and crops the training field around it, so the "
                "target sits centred like in the Galaxy10 training images.",
            )
            model_view = build_model_view(model_view_frame, auto_center)
            model_fov = model_view["crop_px"] * CUTOUT_MIN_PIXSCALE

            col_input_view, col_input_choice = st.columns([1, 2])
            with col_input_view:
                preview = model_view["image"].resize(
                    (img_size, img_size), LANCZOS_RESAMPLE
                )
                st.image(
                    preview,
                    caption=(
                        f"Exact model input ({img_size}×{img_size} px) — "
                        f"{model_fov:.0f}″ field at "
                        f"{CUTOUT_MIN_PIXSCALE}″/px"
                    ),
                    width=256,
                )
            with col_input_choice:
                st.markdown(
                    "The model was trained on Galaxy10 DECaLS crops of "
                    f"{MODEL_VIEW_CROP_PX} px at {CUTOUT_MIN_PIXSCALE}\"/px "
                    f"(≈67″ field), LANCZOS-resized to {img_size}×{img_size}. "
                    "This training-scale view reproduces that input exactly, "
                    "no matter how wide the drawn selection was."
                )
                st.caption(
                    f"Centred on the {model_view['centred_on']} "
                    f"(shift {model_view['shift_px'][0]:+d}, "
                    f"{model_view['shift_px'][1]:+d} px from the frame centre)."
                )
                prediction_source = st.radio(
                    "Image used for prediction",
                    [
                        "Training-scale view (recommended)",
                        "Full selection cutout",
                    ],
                    key="prediction_source",
                )
                if prediction_source.startswith("Training-scale"):
                    exact = model_view["crop_px"] == MODEL_VIEW_CROP_PX
                    st.caption(
                        f"Effective resolution {model_fov / img_size:.2f}\""
                        f" per model pixel — "
                        + ("identical" if exact else "close")
                        + " to the training crops."
                    )
                elif snap_info:
                    covered_w, covered_h = snap_info["covered_arcsec"]
                    eff = max(covered_w, covered_h) / img_size
                    train_eff = (
                        MODEL_VIEW_CROP_PX * CUTOUT_MIN_PIXSCALE
                    ) / img_size
                    factor = eff / train_eff
                    st.caption(
                        f"Effective resolution {eff:.2f}\" per model pixel — "
                        + (
                            f"{factor:.1f}× coarser"
                            if factor >= 1.0
                            else f"{1.0 / factor:.1f}× finer"
                        )
                        + " than training, and the non-square cutout is "
                        "stretched to a square (distorted aspect ratio)."
                    )
        else:
            st.warning(
                "Training-scale model view unavailable"
                + (
                    f" ({st.session_state['model_view_error']})."
                    if st.session_state.get("model_view_error")
                    else "."
                )
                + " Prediction will use the selection cutout."
            )

        if st.button("Analyse selected object", type="primary", key="analyse_sky"):
            binary_model, morphology_model = load_models()

            if morphology_model is None:
                st.error(
                    "Galaxy10 model not found. Copy the notebook export "
                    "(models/galaxy10_resnet18.pth + export_metadata.json) "
                    "into the app's models/ folder."
                )
            else:
                if model_view is not None and str(
                    st.session_state.get("prediction_source", "")
                ).startswith("Training-scale"):
                    prediction_image = model_view["image"]
                    input_desc = "training-scale model view"
                else:
                    prediction_image = cutout
                    input_desc = "full selection cutout"

                try:
                    with st.spinner("Resizing input and classifying..."):
                        if binary_model is not None:
                            is_galaxy, binary_conf = predict_binary(
                                binary_model, prediction_image
                            )
                        else:
                            # Optional detector not connected: classify directly.
                            is_galaxy, binary_conf = True, None

                        if not is_galaxy:
                            st.session_state["last_result"] = {
                                "kind": "non_galaxy",
                                "confidence": binary_conf,
                            }
                        else:
                            label, confidence, probabilities = predict_morphology(
                                morphology_model, prediction_image
                            )
                            galaxy_result = {
                                "kind": "galaxy",
                                "label": label,
                                "confidence": confidence,
                                "probabilities": probabilities,
                                "input": input_desc,
                            }
                            if binary_conf is not None:
                                galaxy_result["binary_confidence"] = binary_conf
                            st.session_state["last_result"] = galaxy_result
                except NotImplementedError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.exception(exc)

    result = st.session_state.get("last_result")
    if result:
        st.divider()
        st.subheader("Analysis result")
        if result["kind"] == "non_galaxy":
            st.error(
                "The selected region was classified as non-galaxy. "
                f"Detector score: {result['confidence']:.1%}"
            )
        else:
            st.success(f"Predicted morphology: **{result['label']}**")
            st.metric("Morphology model score", f"{result['confidence']:.1%}")
            if result.get("input"):
                st.caption(f"Analysed input: {result['input']}.")
            if "binary_confidence" in result:
                st.metric(
                    "Galaxy detector score",
                    f"{result['binary_confidence']:.1%}",
                )
            _img_size, classes, _mean, _std = model_input_config()
            probs = result["probabilities"]
            if probs:
                st.bar_chart(
                    {cls: float(p) for cls, p in zip(classes, probs)}
                )


with tab_upload:
    st.subheader("Test the models with an uploaded image")
    st.write(
        "Use this tab to test the model pipeline with a local image. Any "
        "picture size or aspect ratio works — the image is automatically "
        "resized to the model's training input before prediction."
    )
    uploaded = st.file_uploader(
        "Upload an astronomical image",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
    )

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded image", width=350)

        if st.button("Analyse uploaded image"):
            binary_model, morphology_model = load_models()
            if morphology_model is None:
                st.info(
                    "Galaxy10 model not found. Copy the notebook export "
                    "(models/galaxy10_resnet18.pth + export_metadata.json) "
                    "into the app's models/ folder."
                )
            else:
                try:
                    with st.spinner("Resizing input and classifying..."):
                        if binary_model is not None:
                            is_galaxy, binary_conf = predict_binary(
                                binary_model, image
                            )
                        else:
                            is_galaxy, binary_conf = True, None

                    if not is_galaxy:
                        st.error(f"Non-galaxy detected (score {binary_conf:.1%}).")
                    else:
                        label, confidence, probabilities = predict_morphology(
                            morphology_model, image
                        )
                        st.success(f"Galaxy type: {label}")
                        st.metric("Morphology score", f"{confidence:.1%}")
                        _img_size, classes, _mean, _std = model_input_config()
                        st.bar_chart(
                            {
                                cls: float(p)
                                for cls, p in zip(classes, probabilities)
                            }
                        )
                except NotImplementedError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.exception(exc)


with tab_about:
    st.subheader("Prototype notes")
    st.markdown(
        """
        **Current flow**

        1. Aladin Lite displays a sky-survey viewer centred on the target
           chosen in the sidebar (preset dropdown or custom name/coordinates);
           navigation uses gotoObject() without recreating the viewer.
        2. User draws a rectangle around an object.
        3. Aladin returns the rectangle's sky-coordinate corners.
        4. The app requests a matching Legacy Survey DR10 cutout sized to
           the selection (non-square supported, 10% padding, 512 px cap).
        5. For prediction it re-fetches the same position at the training
           pixel scale (512 px @ 0.262"/px), auto-centres on the brightest
           source and crops the 256 px ≈67" Galaxy10 training field — the
           exact acquisition geometry of the training data.
        6. A binary model should reject non-galaxy regions.
        7. The Galaxy10 ResNet18 model classifies galaxy morphology.
        8. Any analysed image is resized automatically to the model's
           training input (export_metadata.json -> img_size, fallback
           128x128, LANCZOS) and normalized exactly like training.

        **Model files expected**

        - `models/galaxy10_resnet18.pth` — required (notebook export)
        - `models/export_metadata.json` — recommended (auto resize config)
        - `models/galaxy_non_galaxy.pth` — optional binary detector

        **Snap size limits**

        - The rectangle you draw is free-size; there is no fixed selection
          box. The snapped cutout follows the rectangle's own shape.
        - DR10 cutouts are capped at 512 px per side (service limit) and the
          app zooms out at most to 2.0 arcsec/pixel, so one snap covers at
          most ≈17′ of sky.
        - Large preset targets (M31 ≈3°, M33 ≈70′, M101 ≈29′, M42 ≈1°) cannot
          fit into a single cutout — snap their bright core instead.
        - Galaxy10 DECaLS training crops span ≈67″ (256 px at 0.262″/px).
          You no longer need to snap a similar size: the app re-fetches a
          training-scale, auto-centred ≈67″ view for prediction.

        **Black or empty sky after switching surveys**

        - SDSS9 covers only part of the northern sky — it has no data
          around southern targets such as NGC 1300, M42 or Sgr A*. DSS2,
          2MASS and allWISE are all-sky.
        - If the requested HiPS cannot be fetched at all (CDS server
          outage, network problem), the viewer falls back to DSS2 colour
          and shows a status message instead of staying black (aladin-lite
          issue #358: its own auto-fallback only exists since v3.9).

        **Wired up (no longer placeholders)**

        - `build_morphology_model()` — torchvision ResNet-18,
          fc -> Linear(512, 10), identical to the notebook's build_model()
        - `load_models()` — loads `galaxy10_resnet18.pth` with
          `weights_only=True`
        - `preprocess_for_models()` — automatic LANCZOS resize to the
          training size + /255 + ImageNet normalization
        - `predict_morphology()` — softmax over the 10 notebook classes
        - `fetch_model_view_cutout()` — training-scale frame (0.262"/px)
        - `find_bright_centroid()` — clamped luminance centroid
        - `build_model_view()` — 256 px ≈67" Galaxy10-scale crop

        **Still to complete (optional)**

        - `build_binary_model()` and `predict_binary()` — the binary
          galaxy/non-galaxy gate; until it is connected, every analysed
          image is classified directly by the Galaxy10 model.

        The architecture and preprocessing match the notebook's cell 5.4
        export metadata exactly.
        """
    )
    st.info(
        "The binary classifier should be trained with representative "
        "non-galaxy examples (stars, empty sky, artifacts, nebulae, etc.). "
        "A softmax confidence threshold on the 10-class model alone is not "
        "a dependable non-galaxy detector."
    )
    st.markdown(
        "**Data compatibility:** Aladin's displayed survey and the Legacy "
        "Survey cutout service are separate from the model's training data "
        "pipeline. For proper Galaxy10-style inference, retrieve compatible "
        "g/r/z data and apply the notebook's original channel ordering, "
        "normalization and resizing."
    )

st.caption(
    "Educational prototype. Predictions are model outputs, not authoritative "
    "astronomical classifications."
)