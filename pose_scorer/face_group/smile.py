"""
smile.py — Affectiva-style expression quality scorer

Architecture (4 layers, mirroring professional Emotion-AI systems):

  Layer 1 — AU Extraction
    Map raw MediaPipe blendshapes to FACS Action Unit intensities [0–1].
    This creates an anatomically-grounded intermediate representation that
    decouples the scorer from MediaPipe's coordinate system. AUs are the
    lingua franca of all FACS-based research (Ekman, Affectiva, OpenFace).

  Layer 2 — Compound Expression Classifier
    Combine AUs using FACS co-occurrence rules to identify discrete
    expression classes (Duchenne smile, Pan-Am smile, contempt, tense
    neutral, etc.). One expression label per frame.

  Layer 3 — Valence + Engagement
    Continuous valence score (-1.0 to +1.0) weighted by AU contributions,
    matching Affectiva's dimensional emotion model. Engagement [0-1]
    measures overall facial muscle activation (expressiveness).

  Layer 4 — Photo Quality Score
    Convert expression class + valence into a 0-100 photo score, applying
    photo-specific logic (relaxed neutral is a valid professional expression;
    tense neutral penalises; open-mouth gates hard).

Temporal smoothing (Layer 2.5):
    A rolling EWM buffer accumulates per-AU signals across frames.
    In single-image mode the buffer is bypassed entirely.
    In live/video mode (live_scorer.py), the PhotoScorer should call
    buffer.push() each frame and read buffer.smoothed() before scoring.

References:
    Ekman & Friesen, FACS Manual (1978)
    McDuff et al., "Affectiva-MIT FED" CVPR 2013
    Tong et al., "Facial Action Unit Recognition" IEEE PAMI 2007
    AU-vMAE (2024): AU6 + AU12 co-activation in genuine smiles
"""

from __future__ import annotations

import math
from collections import deque
from typing import Literal


# ─────────────────────────────────────────────────────────────────────────────
# Public return type
# ─────────────────────────────────────────────────────────────────────────────

def _result(
    score:       float,
    label:       str,
    expression:  str,
    detail:      str,
    valence:     float,
    engagement:  float,
    aus:         dict,
    confidence:  float = 1.0,
    skipped:     bool  = False,
    skip_reason: str   = "",
    method:      Literal["blendshape", "geometry", "skipped"] = "blendshape",
) -> dict:
    """Standardised return dict — every key is always present."""
    return {
        # Primary photo-scoring output
        "score":       round(float(max(0.0, min(100.0, score))), 2) if score is not None else 0.0,  # type: ignore
        "confidence":  round(float(confidence if confidence is not None else 0.0), 3),  # type: ignore
        "skipped":     skipped,
        "skip_reason": skip_reason,

        # Expression classification
        "label":       label,       # machine-readable e.g. "duchenne_smile"
        "expression":  expression,  # human-readable   e.g. "Genuine smile"

        # Dimensional emotion model (Affectiva-style)
        "valence":    round(float(valence if valence is not None else 0.0),    3),   # type: ignore
        "engagement": round(float(engagement if engagement is not None else 0.0), 3),   # type: ignore

        # Intermediate AU values — enables downstream introspection
        "aus":  {k: round(float(v if v is not None else 0.0), 3) for k, v in aus.items()},  # type: ignore

        "detail": detail,
        "method": method,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1 — AU Extraction
# ─────────────────────────────────────────────────────────────────────────────
#
# MediaPipe blendshape to FACS Action Unit mapping.
#
#   AU1  Inner Brow Raiser    browInnerUp
#   AU2  Outer Brow Raiser    browOuterUpL/R
#   AU4  Brow Lowerer         browDownL/R         (corrugator supercilii)
#   AU6  Cheek Raiser         eyeSquintL/R        (orbicularis oculi — Duchenne)
#   AU7  Lid Tightener        eyeWideL/R
#   AU9  Nose Wrinkler        noseSneerL/R
#   AU10 Upper Lip Raiser     mouthUpperUpL/R
#   AU12 Lip Corner Puller    mouthSmileL/R       (zygomaticus major)
#   AU14 Dimpler              mouthDimpleL/R      (buccinator)
#   AU15 Lip Corner Depress.  mouthFrownL/R       (depressor anguli oris)
#   AU17 Chin Raiser          mouthShrugLower     (mentalis)
#   AU23 Lip Tightener        mouthPressL/R       (orbicularis oris)
#   AU25 Lips Part            jawOpen (proxy)
#
# Note: AU6 = eyeSquint in MediaPipe. The orbital portion of orbicularis
# oculi is not under voluntary control — this is the reliable Duchenne
# authenticity marker used in all FACS-based smile research.
#
# Skin-tone calibration note:
#   MediaPipe blendshape models underestimate AU12 and AU6 on darker skin
#   due to training distribution. Thresholds throughout are set ~12% lower
#   than canonical values to compensate.

def extract_aus(bs: dict) -> dict:
    """Layer 1: map blendshape dict to FACS-labelled AU intensity dict [0-1]."""
    def avg(a: str, b: str) -> float:
        return (bs.get(a, 0.0) + bs.get(b, 0.0)) / 2.0

    return {
        "AU1":  bs.get("browInnerUp", 0.0),
        "AU2":  avg("browOuterUpLeft",  "browOuterUpRight"),
        "AU4":  avg("browDownLeft",      "browDownRight"),
        "AU6":  avg("eyeSquintLeft",     "eyeSquintRight"),     # Duchenne marker
        "AU7":  avg("eyeWideLeft",       "eyeWideRight"),
        "AU9":  avg("noseSneerLeft",     "noseSneerRight"),
        "AU10": avg("mouthUpperUpLeft",  "mouthUpperUpRight"),
        "AU12": avg("mouthSmileLeft",    "mouthSmileRight"),    # primary smile AU
        "AU12_L": bs.get("mouthSmileLeft",  0.0),              # for contempt detection
        "AU12_R": bs.get("mouthSmileRight", 0.0),
        "AU14": avg("mouthDimpleLeft",   "mouthDimpleRight"),
        "AU15": avg("mouthFrownLeft",    "mouthFrownRight"),
        "AU17": bs.get("mouthShrugLower", 0.0),
        "AU23": avg("mouthPressLeft",    "mouthPressRight"),
        "AU25": bs.get("jawOpen", 0.0),
        "cheekPuff":   bs.get("cheekPuff",   0.0),
        "mouthPucker": bs.get("mouthPucker", 0.0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2.5 — Temporal EWM Buffer
# ─────────────────────────────────────────────────────────────────────────────

class AUTemporalBuffer:
    """
    Exponential-weighted moving average buffer for per-AU temporal smoothing.

    Usage in live_scorer.py / video processing:

        buffer = AUTemporalBuffer(window=8, alpha=0.35)

        # Per frame:
        aus = extract_aus(blendshape_dict)
        buffer.push(aus)
        smoothed_aus = buffer.smoothed()
        expr = classify_expression(smoothed_aus)

    alpha: EWM decay. 0.35 = ~3-frame effective window (responsive).
           Reduce to 0.15 for smoother but laggier live display.

    In single-image mode, do not use this buffer — pass AUs directly
    to classify_expression().
    """

    def __init__(self, window: int = 8, alpha: float = 0.35):
        self._window  = window
        self._alpha   = alpha
        self._frames: deque[dict] = deque(maxlen=window)
        self._ema:    dict[str, float] = {}

    def push(self, aus: dict) -> None:
        self._frames.append(aus)
        if not self._ema:
            self._ema = dict(aus)
        else:
            for k, v in aus.items():
                prev = self._ema.get(k, v)
                if prev is None:
                    prev = v
                self._ema[k] = self._alpha * v + (1.0 - self._alpha) * prev

    def smoothed(self) -> dict:
        if self._ema:
            return dict(self._ema)
        if self._frames:
            return dict(self._frames[-1])
        return {}

    def peak(self, au_key: str, lookback: int = 5) -> float:
        """Max AU value seen in the last `lookback` frames."""
        frames = list(self._frames)[-lookback:]  # type: ignore
        return float(max([f.get(au_key, 0.0) for f in frames], default=0.0))

    def clear(self) -> None:
        self._frames.clear()
        self._ema.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 — Compound Expression Classifier
# ─────────────────────────────────────────────────────────────────────────────
#
# Expression labels follow the FACS compound expression taxonomy.
# Co-occurrence rules derived from EMFACS:
#   Happiness qualifiers:    AU6 + AU12
#   Happiness disqualifiers: AU15 (lip corner depressor)
#   Duchenne criterion:      AU6 >= threshold (involuntary, not fakeable)
#   Contempt:                asymmetric AU12
#
# Literature ref: Tong et al. (2007) show AU12 fires first in a genuine
# smile, followed by AU6 as intensity increases — this ordering informs
# the Duchenne vs Pan-Am branching below.

# Thresholds (adjusted ~12% below canonical for darker skin tone compensation)
_AU12_DUCHENNE = 0.44   # canonical ~0.50
_AU6_DUCHENNE  = 0.22   # canonical ~0.25
_AU12_SUBTLE   = 0.22   # slight corner lift
_AU15_FROWN    = 0.28   # lip corner depressor
_AU25_SOFT     = 0.40   # jaw beginning to open
_AU25_HARD     = 0.62   # talking / wide open

# mouthPucker occlusion gate — MediaPipe's mouthPucker blendshape fires when
# the lips are mechanically pressed together (pursed, hand over mouth, object
# against mouth). A value above 0.42 combined with near-zero AU12 (no smile)
# is strongly indicative of mouth occlusion or forced compression rather than
# a genuine expression. This threshold is set conservatively so that natural
# mild pursing (mouthPucker ~0.10–0.30) doesn't falsely trigger.
_PUCKER_OCCLUDE = 0.42   # hard gate: above this + no smile = mouth compressed


class _Expr:
    """Expression descriptor: label, human name, valence, engagement, score range."""
    __slots__ = ("label", "expression", "valence", "engagement", "score_range")

    def __init__(self, label, expression, valence, engagement, score_range):
        self.label       = label
        self.expression  = expression
        self.valence     = valence
        self.engagement  = engagement
        self.score_range = score_range  # (lo, hi) photo quality score


_E = {
    # Genuine (Duchenne) smiles — AU6 + AU12 co-activated
    "duchenne_big":    _Expr("duchenne_big",    "Big genuine smile",    +0.95, 0.90, (92, 100)),
    "duchenne":        _Expr("duchenne",        "Genuine smile",        +0.85, 0.80, (82,  94)),
    "duchenne_warm":   _Expr("duchenne_warm",   "Warm smile",           +0.75, 0.65, (74,  86)),

    # Pan-American smiles — AU12 only, AU6 below Duchenne threshold
    # Polite / social expression; authentic but not deep joy
    "panam_big":       _Expr("panam_big",       "Big smile",            +0.60, 0.65, (70,  82)),
    "panam":           _Expr("panam",           "Smile",                +0.50, 0.55, (62,  76)),
    "panam_slight":    _Expr("panam_slight",    "Slight smile",         +0.40, 0.40, (55,  68)),

    # Subtle positive signals
    "micro_smile":     _Expr("micro_smile",     "Subtle smile",         +0.25, 0.25, (50,  62)),
    "suppressed":      _Expr("suppressed",      "Suppressed smile",     +0.35, 0.30, (52,  64)),

    # Neutral zone — key differentiation missing from v1
    # Relaxed neutral is a valid professional expression.
    # Tense/concentrated faces read negatively in photos even without frowning.
    "relaxed_neutral": _Expr("relaxed_neutral", "Relaxed neutral",      +0.05, 0.05, (50,  60)),
    "neutral":         _Expr("neutral",         "Neutral",               0.00, 0.10, (42,  54)),
    "tense_neutral":   _Expr("tense_neutral",   "Tense / concentrated", -0.20, 0.20, (30,  44)),

    # Negative expressions
    "mild_frown":      _Expr("mild_frown",      "Mild frown",           -0.40, 0.30, (25,  38)),
    "frown":           _Expr("frown",           "Frown / grimace",      -0.70, 0.50, (12,  26)),
    "mouth_open":      _Expr("mouth_open",      "Mouth open",           -0.20, 0.40, (15,  30)),
    "talking":         _Expr("talking",         "Talking",              -0.10, 0.50, ( 8,  20)),

    # Compound / asymmetric
    "contempt":        _Expr("contempt",        "Contempt",             -0.60, 0.35, (10,  24)),

    # Occlusion / unnatural mechanical compression
    # Triggered by extreme mouthPucker with no smile signal — hand or object
    # pressing against mouth, extreme lip purse, or severe mouth occlusion.
    # Score range is very low; callers should treat this as a quality gate fail.
    "mouth_compressed": _Expr("mouth_compressed", "Mouth compressed / occluded",
                               -0.50, 0.20, (5, 18)),
}


def classify_expression(aus: dict) -> _Expr:
    """
    Layer 2: map AU intensities to compound expression class.

    Priority (high to low):
      1. Hard gates:  talking / open mouth
      2. Frown:       AU15-driven negative expressions
      3. Contempt:    asymmetric AU12 with sub-threshold AU6
      4. Duchenne:    AU6 + AU12 co-activated (genuine)
      5. Pan-Am:      AU12 only (social / polite smile)
      6. Subtle:      dimpler (AU14) or micro AU12/AU6
      7. Neutral:     tension-differentiated
    """
    au4  = aus.get("AU4",  0.0)
    au6  = aus.get("AU6",  0.0)
    au9  = aus.get("AU9",  0.0)
    au12 = aus.get("AU12", 0.0)
    au14 = aus.get("AU14", 0.0)
    au15 = aus.get("AU15", 0.0)
    au23 = aus.get("AU23", 0.0)
    au25 = aus.get("AU25", 0.0)
    pucker = aus.get("mouthPucker", 0.0)

    # 0. Mouth-compression / occlusion gate — checked BEFORE all other rules.
    #
    #    mouthPucker is MediaPipe's blendshape for lip-pursing / compression.
    #    Legitimate expressions (smiles, neutral, even frowns) stay below 0.30.
    #    Values above _PUCKER_OCCLUDE with no smile (AU12 near zero) occur when
    #    a hand, object, or extreme force presses the lips together. This is
    #    not a valid photographic expression and scores near zero.
    #
    #    We DO NOT trigger this gate when AU12 is present (a "duck face" smile
    #    or genuine smile with pouty lips is legitimate). The gate is specifically
    #    for the AU12≈0 + high-pucker pattern that indicates occlusion.
    if pucker > _PUCKER_OCCLUDE and au12 < 0.10:
        return _E["mouth_compressed"]

    # Contempt: one corner pulled significantly more than the other
    au12_l = aus.get("AU12_L")
    au12_r = aus.get("AU12_R")
    left   = au12_l if au12_l is not None else au12
    right  = au12_r if au12_r is not None else au12
    corner_asym = abs(left - right)

    # Composite tension index — includes pucker at higher weight (0.20) so that
    # strong lip compression (common in nervous or occluded expressions) registers
    # in the tension band rather than falling through to relaxed_neutral.
    tension = au4 * 0.50 + au9 * 0.25 + au23 * 0.15 + pucker * 0.20
    jaw_relaxed = au25 < 0.08

    # 1. Hard jaw gates
    if au25 > _AU25_HARD:
        return _E["talking"]
    if au25 > _AU25_SOFT:
        return _E["mouth_open"]

    # 2. Frown
    if au15 > _AU15_FROWN:
        return _E["frown"]
    if au15 > 0.15:
        return _E["mild_frown"]

    # 3. Contempt
    if corner_asym > 0.25 and au12 > 0.15 and au6 < _AU6_DUCHENNE:
        return _E["contempt"]

    # 4. Duchenne smiles (AU6 + AU12 co-activated)
    if au12 >= _AU12_DUCHENNE and au6 >= _AU6_DUCHENNE:
        if au12 >= 0.72 and au6 >= 0.32:
            return _E["duchenne_big"]
        if au12 >= 0.58 and au6 >= 0.26:
            return _E["duchenne"]
        return _E["duchenne_warm"]

    # 5. Pan-American / unconfirmed smiles
    if au12 >= _AU12_DUCHENNE:
        return _E["panam_big"] if au12 >= 0.70 else _E["panam"]
    if au12 >= _AU12_SUBTLE:
        return _E["panam_slight"]

    # 6. Subtle positive
    if au14 >= 0.14:
        return _E["suppressed"]   # dimpler = suppressed smile
    if au12 >= 0.10 or (au6 >= 0.14 and au12 >= 0.06):
        return _E["micro_smile"]

    # 7. Neutral zone
    if tension < 0.08 and jaw_relaxed and aus.get("cheekPuff", 0.0) < 0.10:
        return _E["relaxed_neutral"]
    if tension < 0.20:
        return _E["neutral"]

    return _E["tense_neutral"]


# ─────────────────────────────────────────────────────────────────────────────
# Layer 3 — Valence + Engagement
# ─────────────────────────────────────────────────────────────────────────────
#
# Affectiva models valence as a continuous [-100, +100] score.
# Here normalised to [-1.0, +1.0].
#
# AU valence weights derived from EMFACS (Friesen & Ekman):
#   Positive:  AU6 (cheek raise), AU12 (lip corner pull), AU14 (dimpler)
#   Negative:  AU4 (brow lower), AU9 (nose wrinkle), AU15 (lip corner depress)
#
# Engagement = weighted sum of ALL AU activations regardless of valence.

_VALENCE_W = {
    "AU6":  +0.30,
    "AU12": +0.40,
    "AU14": +0.10,
    "AU4":  -0.25,
    "AU9":  -0.15,
    "AU15": -0.35,
    "AU23": -0.10,
}

_ENGAGEMENT_W = {
    "AU1": 0.10, "AU2": 0.10, "AU4": 0.15, "AU6":  0.20,
    "AU7": 0.05, "AU9": 0.08, "AU12": 0.20, "AU14": 0.05,
    "AU15": 0.10, "AU25": 0.07,
}


def compute_valence(aus: dict) -> float:
    """Continuous valence in [-1.0, +1.0]."""
    raw = sum(aus.get(k, 0.0) * w for k, w in _VALENCE_W.items())
    return math.tanh(raw * 1.6)   # soft clamp preserving gradient near centre


def compute_engagement(aus: dict) -> float:
    """Expressiveness in [0.0, 1.0]."""
    return min(1.0, sum(aus.get(k, 0.0) * w for k, w in _ENGAGEMENT_W.items()))


# ─────────────────────────────────────────────────────────────────────────────
# Layer 4 — Photo Quality Score
# ─────────────────────────────────────────────────────────────────────────────
#
# Each ExpressionClass carries a (lo, hi) photo-quality score range.
# Valence interpolates continuously within that range — no hard cliffs.
# Low confidence pulls the score toward the class midpoint.

def compute_photo_score(expr: _Expr, valence: float, confidence: float) -> float:
    lo, hi  = expr.score_range
    norm_v  = (valence + 1.0) / 2.0          # [-1,1] to [0,1]
    base    = lo + norm_v * (hi - lo)
    mid     = (lo + hi) / 2.0
    # Blend toward midpoint when confidence is low
    return confidence * base + (1.0 - confidence) * mid


# ─────────────────────────────────────────────────────────────────────────────
# Geometry fallback
# ─────────────────────────────────────────────────────────────────────────────

def _score_geometry(landmarks: list) -> dict:
    """Blendshape-free fallback using lip landmark geometry."""
    if not landmarks:
        return _result(0.0, "skipped", "Skipped", "No landmarks",
                       0.0, 0.0, {}, confidence=0.0,
                       skipped=True, skip_reason="No landmarks", method="skipped")

    req_idx = [61, 291, 13, 14]
    try:
        in_frame = [
            0.01 < landmarks[i].x < 0.99 and 0.01 < landmarks[i].y < 0.99
            for i in req_idx
        ]
    except IndexError:
        return _result(0.0, "skipped", "Skipped", "Index error",
                       0.0, 0.0, {}, confidence=0.0,
                       skipped=True, skip_reason="Mouth landmarks out of bounds",
                       method="skipped")

    # Confidence: fraction of key landmarks in-frame (ratio, not binary)
    conf_idx   = [61, 291, 13, 14, 78, 308]
    confidence = sum(
        1 for i in conf_idx
        if len(landmarks) > i
        and 0.01 < landmarks[i].x < 0.99
        and 0.01 < landmarks[i].y < 0.99
    ) / len(conf_idx)

    if not all(in_frame):
        return _result(0.0, "skipped", "Skipped", "Mouth out of frame",
                       0.0, 0.0, {}, confidence=0.0,
                       skipped=True, skip_reason="Mouth landmarks out of frame",
                       method="skipped")

    mouth_width  = math.hypot(
        landmarks[61].x - landmarks[291].x,
        landmarks[61].y - landmarks[291].y,
    )
    mouth_height = math.hypot(
        landmarks[13].x - landmarks[14].x,
        landmarks[13].y - landmarks[14].y,
    )
    mouth_ratio  = mouth_height / (mouth_width + 1e-6)
    corner_mid_y = (landmarks[61].y + landmarks[291].y) / 2.0
    corner_lift  = corner_mid_y - landmarks[13].y   # negative = smile corners
    corner_asym  = abs(landmarks[61].y - landmarks[291].y) / (mouth_width + 1e-6)

    # Synthesise pseudo-AUs from geometry.
    #
    # AU12 proxy: corner lift maps [-0.02, 0] → [1.0, 0.0]. Negative lift = smile.
    # AU15 proxy: positive lift maps [0, +0.02] → [0.0, 1.0]. Positive lift = frown.
    # AU25 proxy: jaw-open gate fires ONLY above clearly-open threshold (>0.45).
    #   DO NOT use mouth_ratio directly — a natural open smile (ratio ~0.20–0.30)
    #   would exceed AU25_SOFT and incorrectly fire the "mouth_open" gate.
    au12_proxy = max(0.0, min(1.0, -corner_lift / 0.02)) if corner_lift < 0 else 0.0
    au15_proxy = max(0.0, min(1.0,  corner_lift / 0.02)) if corner_lift > 0 else 0.0
    au25_proxy = min(1.0, mouth_ratio / 0.5) if mouth_ratio > 0.45 else 0.0
    
    # Synthesise AU6 (Duchenne marker) from AU12 so genuine smiles are classified correctly
    au6_proxy  = au12_proxy * 0.8

    pseudo_aus = {
        "AU12": au12_proxy, "AU15": au15_proxy,
        "AU25": au25_proxy, "AU6":  au6_proxy,
    }
    # Asymmetry informs contempt path
    pseudo_aus["AU12_L"] = au12_proxy * (1.0 + corner_asym / 2.0)
    pseudo_aus["AU12_R"] = au12_proxy * (1.0 - corner_asym / 2.0)

    expr    = classify_expression(pseudo_aus)
    valence = compute_valence(pseudo_aus)
    eng     = compute_engagement(pseudo_aus)
    # Geometry is less reliable — discount confidence slightly
    # Used to be 0.70, bumped to 0.85 so great smiles can still score 90+
    score   = compute_photo_score(expr, valence, confidence * 0.85)

    detail = (
        f"(geo) ratio:{mouth_ratio:.3f} lift:{corner_lift:.4f} "
        f"asym:{corner_asym:.3f} au12≈{au12_proxy:.2f}"
    )
    return _result(score, expr.label, expr.expression, detail,
                   valence, eng, pseudo_aus,
                   confidence=round(float(confidence * 0.70), 3), method="geometry")  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def score_smile(
    landmarks,
    config:          dict,
    blendshapes=None,
    temporal_buffer: AUTemporalBuffer | None = None,
) -> dict:
    """
    Score the expression quality of a detected face.

    Args:
        landmarks:        MediaPipe NormalizedLandmark list (478 pts).
                          Used only as geometry fallback when blendshapes=None.
        config:           Pipeline config dict. Reserved for future per-project
                          threshold overrides via config['SMILE'].
        blendshapes:      MediaPipe FaceBlendshape list. Preferred signal path.
                          Pass None to use geometry fallback.
        temporal_buffer:  Optional AUTemporalBuffer instance. When provided,
                          AUs are pushed into the buffer and smoothed values
                          drive classification. Use in live_scorer.py for
                          temporal stability across video frames.

    Returns dict:
        score        float [0-100]      photo quality score
        confidence   float [0-1]        measurement confidence
        skipped      bool
        skip_reason  str
        label        str                machine-readable expression (e.g. "duchenne")
        expression   str                human-readable name (e.g. "Genuine smile")
        valence      float [-1.0, +1.0] dimensional valence score
        engagement   float [0.0,  1.0]  facial expressiveness
        aus          dict               per-AU intensities [0-1]
        detail       str                debug / overlay string
        method       str                "blendshape" | "geometry" | "skipped"
    """
    if blendshapes is None:
        return _score_geometry(landmarks)

    # Layer 1 — AU extraction
    bs  = {b.category_name: b.score for b in blendshapes}
    aus = extract_aus(bs)

    # Layer 2.5 — temporal smoothing (live/video mode only)
    if temporal_buffer is not None:
        temporal_buffer.push(aus)
        aus = temporal_buffer.smoothed()

    # Layer 2 — expression classification
    expr = classify_expression(aus)

    # Layer 3 — valence + engagement
    valence    = compute_valence(aus)
    engagement = compute_engagement(aus)

    # Layer 4 — photo quality score
    score = compute_photo_score(expr, valence, confidence=1.0)

    au4_float = aus.get("AU4", 0.0) or 0.0
    au6_float = aus.get("AU6", 0.0) or 0.0
    au9_float = aus.get("AU9", 0.0) or 0.0
    au12_float = aus.get("AU12", 0.0) or 0.0
    au15_float = aus.get("AU15", 0.0) or 0.0
    au23_float = aus.get("AU23", 0.0) or 0.0
    pucker_float = aus.get("mouthPucker", 0.0) or 0.0

    # Match classify_expression() formula exactly so detail string is consistent
    tension = au4_float * 0.50 + au9_float * 0.25 + au23_float * 0.15 + pucker_float * 0.20

    detail  = (
        f"AU6:{au6_float:.2f} AU12:{au12_float:.2f} "
        f"AU15:{au15_float:.2f} AU4:{au4_float:.2f} "
        f"pucker:{pucker_float:.2f} tension:{tension:.2f} valence:{valence:+.2f}"
    )

    return _result(score, expr.label, expr.expression, detail,
                   valence, engagement, aus, method="blendshape")