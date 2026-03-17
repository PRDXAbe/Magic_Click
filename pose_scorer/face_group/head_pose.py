import cv2       # type: ignore
import numpy as np  # type: ignore
import math

# ─────────────────────────────────────────────────────────────────────────────
# solvePnP 3D Model Points (mm, average face geometry)
# ─────────────────────────────────────────────────────────────────────────────
MODEL_3D_POINTS = np.array([
    (  0.0,    0.0,   0.0),   # lm[1]   nose tip
    (  0.0, -330.0, -65.0),   # lm[152] chin
    (-225.0,  170.0,-135.0),  # lm[263] left eye outer corner
    ( 225.0,  170.0,-135.0),  # lm[33]  right eye outer corner
    (-150.0, -150.0,-125.0),  # lm[287] left mouth corner
    ( 150.0, -150.0,-125.0),  # lm[57]  right mouth corner
], dtype=np.float64)

_LANDMARK_IDS = [1, 152, 263, 33, 287, 57]

# Landmark indices used only by the chin geometry path
_LM_NOSE_TIP    = 1
_LM_CHIN        = 152
_LM_EYE_L_OUT   = 263   # left eye outer corner  (from subject's perspective)
_LM_EYE_R_OUT   = 33    # right eye outer corner
_LM_NOSE_BASE   = 94    # base of nose — extra horizontal anchor


# ─────────────────────────────────────────────────────────────────────────────
# Camera matrix
# ─────────────────────────────────────────────────────────────────────────────

def build_camera_matrix(crop_w: int, crop_h: int, fov_deg: float = 75.0) -> np.ndarray:
    """
    FOV-based camera matrix. MUST use crop dimensions (scaled_w, scaled_h).
    Using naive focal=image_w causes systematic pitch error of 10-15 degrees.
    RULE 12 compliance: solvePnP camera matrix uses CROP dimensions.
    """
    fov_rad = np.radians(fov_deg)
    focal = (crop_w / 2.0) / np.tan(fov_rad / 2.0)
    return np.array([
        [focal,     0, crop_w / 2.0],
        [    0, focal, crop_h / 2.0],
        [    0,     0,           1.0],
    ], dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Chin Geometry: orientation signal
# ─────────────────────────────────────────────────────────────────────────────
#
# Principle:
#   When the head turns, the chin is displaced horizontally relative to the
#   bilateral symmetry axis of the face (midpoint of the two eye outer corners).
#   Because the chin sits low on the face, it moves further laterally than the
#   nose or forehead, making it the most sensitive landmark for yaw detection
#   using purely 2D geometry.
#
#   chin_offset = (chin.x - eye_mid.x) / eye_span
#
#   In normalised image space [0, 1]:
#     chin_offset ≈  0.00  →  facing camera
#     chin_offset > +0.05  →  head turned right  (chin shifts right in image)
#     chin_offset < -0.05  →  head turned left   (chin shifts left in image)
#
#   Note: "right in image" = subject's left when facing camera. Follows the same
#   sign convention as solvePnP yaw (positive = subject's right, viewed from
#   camera = leftward shift in image). We negate to match.
#
# Robustness:
#   We use TWO anchor points (eye midpoint + nose base) to compute the symmetry
#   axis. If nose_base is unavailable, eye midpoint alone is used.
#   The result is normalised by eye span so face size doesn't affect the signal.

def _chin_orientation(landmarks: list) -> dict:
    """
    Compute chin-based orientation signal from face landmarks.

    Returns:
        direction   str    "forward" | "right" | "left"
        chin_offset float  signed normalised offset, negative=left, positive=right
        yaw_estimate float  rough yaw estimate in degrees (for fallback fusion)
        reliable    bool   False if eye span is too small to trust (e.g. face too small)
    """
    try:
        eye_l = landmarks[_LM_EYE_L_OUT]
        eye_r = landmarks[_LM_EYE_R_OUT]
        chin  = landmarks[_LM_CHIN]
        nose  = landmarks[_LM_NOSE_TIP]
    except IndexError:
        return {"direction": "unknown", "chin_offset": 0.0,
                "yaw_estimate": 0.0, "reliable": False}

    # Horizontal symmetry axis: midpoint of the two eye outer corners
    eye_mid_x = (eye_l.x + eye_r.x) / 2.0

    # Face width normaliser: distance between eye outer corners
    eye_span = abs(eye_r.x - eye_l.x)
    if eye_span < 0.04:
        # Face too small or nearly profile — geometry unreliable
        return {"direction": "unknown", "chin_offset": 0.0,
                "yaw_estimate": 0.0, "reliable": False}

    # Primary signal: chin offset from symmetry axis, normalised by face width
    chin_offset = (chin.x - eye_mid_x) / eye_span

    # Secondary signal: nose offset (corroborates chin direction)
    nose_offset = (nose.x - eye_mid_x) / eye_span

    # Agreement check: chin and nose should shift in the same direction.
    # If they disagree, confidence in the geometry reading drops.
    same_direction = (chin_offset * nose_offset) >= 0

    # Empirically derived thresholds (frontal webcam setup):
    #   0.00–0.05  natural face asymmetry / small pose variation → forward
    #   0.05–0.12  noticeable turn → slight turn (still acceptable)
    #   0.12+      clear profile turn → reject
    # Chin offset maps to yaw angle approximately as:
    #   yaw_deg ≈ chin_offset * 55  (empirical, based on average 3D face geometry)
    #   At 0.18 offset → ~10° yaw; at 0.55 offset → ~30° yaw
    yaw_estimate = chin_offset * 55.0   # positive = yaw right (subject's right)

    abs_offset = abs(chin_offset)
    if abs_offset < 0.05:
        direction = "forward"
    elif chin_offset > 0:
        direction = "right"
    else:
        direction = "left"

    return {
        "direction":    direction,
        "chin_offset":  round(chin_offset, 4),
        "nose_offset":  round(nose_offset, 4),
        "yaw_estimate": round(yaw_estimate, 1),
        "reliable":     same_direction,   # False if chin/nose disagree
    }


def _chin_score_from_offset(abs_offset: float) -> float:
    """
    Convert absolute normalised chin offset to a yaw-equivalent score
    using the same score bands as solvePnP yaw.
    """
    if abs_offset < 0.10:    return 100.0
    elif abs_offset < 0.18:  return 85.0
    elif abs_offset < 0.30:  return 60.0
    elif abs_offset < 0.40:  return 35.0
    else:                    return 10.0


# ─────────────────────────────────────────────────────────────────────────────
# Chin Vertical: pitch signal (chin-down / chin-up detection)
# ─────────────────────────────────────────────────────────────────────────────
#
# Principle:
#   When the head pitches DOWNWARD (chin toward chest), the chin landmark
#   rises toward eye level in image space. The vertical distance between
#   the eye midpoint and chin shrinks dramatically — much more than solvePnP
#   captures at extreme angles, because the 3D model fit degrades when
#   the face is severely foreshortened.
#
#   chin_drop_ratio = (chin.y - eye_mid.y) / eye_span
#
#   In normalised image space (y=0 top, y=1 bottom):
#     Frontal view:         chin_drop_ratio ≈ 1.2 – 2.2  (chin far below eyes)
#     Mild downward pitch:  chin_drop_ratio ≈ 0.8 – 1.2
#     Moderate pitch:       chin_drop_ratio ≈ 0.5 – 0.8
#     Severe chin-down:     chin_drop_ratio < 0.5         → hard reject zone
#
#   This is normalised by eye_span so it is scale-invariant (same ratio
#   whether the face fills the frame or is small in the corner).
#
#   Secondary corroboration: nose-to-chin vertical distance.
#   In frontal view nose is well above chin; on extreme downward pitch
#   both landmarks converge vertically.
#
# This signal catches the exact failure shown in the test images:
#   head pitched ~50-60° forward, solvePnP reports ~15-20° (inaccurate),
#   but chin_drop_ratio drops to ~0.3-0.4 — unambiguously severe pitch.

def _chin_vertical_pitch(landmarks: list) -> dict:
    """
    Compute chin vertical position signal for pitch cross-validation.

    Returns:
        pitch_state     str    "frontal" | "mild_down" | "moderate_down" | "severe_down"
                               | "mild_up" | "severe_up" | "unknown"
        chin_drop_ratio float  (chin.y - eye_mid.y) / eye_span  (larger = more frontal)
        pitch_estimate  float  rough pitch estimate in degrees (positive = down)
        hard_gate       bool   True if pitch is so extreme the photo must be rejected
    """
    try:
        eye_l = landmarks[_LM_EYE_L_OUT]
        eye_r = landmarks[_LM_EYE_R_OUT]
        chin  = landmarks[_LM_CHIN]
        nose  = landmarks[_LM_NOSE_TIP]
    except IndexError:
        return {"pitch_state": "unknown", "chin_drop_ratio": 1.0,
                "pitch_estimate": 0.0, "hard_gate": False}

    eye_span = abs(eye_r.x - eye_l.x)
    if eye_span < 0.04:
        return {"pitch_state": "unknown", "chin_drop_ratio": 1.0,
                "pitch_estimate": 0.0, "hard_gate": False}

    eye_mid_y  = (eye_l.y + eye_r.y) / 2.0
    chin_drop  = (chin.y - eye_mid_y) / eye_span   # positive = chin below eyes (good)
    nose_drop  = (nose.y - eye_mid_y) / eye_span   # corroboration

    # Pitch estimate: empirically, chin_drop ≈ 1.5 at 0° pitch.
    # Each 0.1 drop from 1.5 corresponds to roughly 7-8° of downward pitch.
    # pitch_estimate > 0 = downward (chin toward chest)
    pitch_estimate = (1.5 - chin_drop) * 70.0   # degrees

    if chin_drop < 0.30:
        pitch_state = "severe_down"
        hard_gate   = True     # unambiguous chin-to-chest — must reject
    elif chin_drop < 0.55:
        pitch_state = "moderate_down"
        hard_gate   = False
    elif chin_drop < 0.85:
        pitch_state = "mild_down"
        hard_gate   = False
    elif chin_drop < 2.5:
        pitch_state = "frontal"
        hard_gate   = False
    else:
        # chin_drop > 2.5 = chin unusually far below eyes = chin-up / looking up
        pitch_state = "severe_up" if chin_drop > 3.2 else "mild_up"
        hard_gate   = chin_drop > 3.2

    return {
        "pitch_state":    pitch_state,
        "chin_drop_ratio": round(chin_drop, 3),
        "nose_drop_ratio": round(nose_drop, 3),
        "pitch_estimate":  round(pitch_estimate, 1),  # type: ignore,
        "hard_gate":       hard_gate,
    }


def _pitch_score_from_chin(chin_drop_ratio: float) -> float:
    """
    Pitch score derived from chin vertical position.
    Mirrors the solvePnP pitch scoring scale.
    """
    if chin_drop_ratio >= 0.85:   return 100.0   # frontal
    elif chin_drop_ratio >= 0.65: return 85.0    # mild pitch
    elif chin_drop_ratio >= 0.50: return 60.0    # moderate — still acceptable
    elif chin_drop_ratio >= 0.38: return 25.0    # severe — penalise hard
    else:                         return 5.0     # extreme — near-zero


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate conversion utility
# ─────────────────────────────────────────────────────────────────────────────

def _to_global(lm, crop_meta: dict) -> tuple[float, float]:
    """
    Convert a face landmark from crop-local normalised [0,1] space to
    global image normalised [0,1] space.

    Face landmarks from FaceLandmarker are normalised to the crop region
    (and upscaled if face_min_size was applied). Pose landmarks from
    PoseLandmarker are normalised to the full original image. To compare
    chin with shoulder positions, both must be in the same coordinate space.

    This is the inverse of make_face_crop + resize in crop.py:
        pixel_in_crop   = lm.xy * scaled_w/h
        pixel_pre_scale = pixel_in_crop / scale
        pixel_global    = pixel_pre_scale + (offset_x, offset_y)
        normalised      = pixel_global / (original_w, original_h)
    """
    px = lm.x * crop_meta['scaled_w'] / crop_meta['scale'] + crop_meta['offset_x']
    py = lm.y * crop_meta['scaled_h'] / crop_meta['scale'] + crop_meta['offset_y']
    return px / crop_meta['original_w'], py / crop_meta['original_h']


# ─────────────────────────────────────────────────────────────────────────────
# Chin–Shoulder Lateral Tilt: roll / lateral head tilt signal
# ─────────────────────────────────────────────────────────────────────────────
#
# Principle:
#   When the head tilts laterally (ear toward shoulder), the chin moves
#   asymmetrically closer to one shoulder. The bilateral symmetry property
#   of the face means that in a neutral head position, the chin sits
#   equidistant from both shoulders. Any asymmetry directly indicates tilt.
#
#   We measure two quantities:
#
#   1. LATERAL ASYMMETRY (chin closer to one shoulder):
#      asymmetry = |dist_chin_left - dist_chin_right| / shoulder_span
#        ≈ 0.00–0.12  →  centered, no meaningful tilt
#        ≈ 0.12–0.25  →  slight tilt (noticeable in photos)
#        ≈ 0.25–0.40  →  moderate tilt (clearly off-axis)
#        >  0.40       →  severe tilt (head on shoulder)
#
#   2. CHIN HEIGHT ABOVE SHOULDER LINE (chin-up detection):
#      chin_height_ratio = (shoulder_mid_y - chin_y) / shoulder_span
#      Positive = chin above shoulder line (normal).
#      In normal upright posture this is roughly 0.5–1.5.
#        > 2.2  →  chin raised too high (looking up / chin-up posture)
#        < 0.20 →  chin at or below shoulder line (severe forward pitch,
#                  corroborates _chin_vertical_pitch hard gate)
#
#   Coordinate space: all values in global normalised [0,1] image space.
#   Face chin landmark is converted from crop-local space via _to_global().
#   Pose shoulder landmarks are already in global space from PoseLandmarker.
#
# Pose landmark indices (MediaPipe PoseLandmarker):
#   11 = left shoulder  (subject's left, image right)
#   12 = right shoulder (subject's right, image left)

_POSE_LM_SHOULDER_L = 11
_POSE_LM_SHOULDER_R = 12
_SHOULDER_VIS_MIN   = 0.50   # minimum visibility score to trust a shoulder


def _chin_shoulder_tilt(
    face_landmarks: list,
    pose_landmarks: list,
    crop_meta: dict,
) -> dict:
    """
    Measure lateral head tilt and chin height using chin–shoulder geometry.

    Args:
        face_landmarks: FaceLandmarker output (crop-local normalised)
        pose_landmarks: PoseLandmarker output (global normalised)
        crop_meta:      Face crop metadata for coordinate conversion

    Returns dict:
        available       bool    False if shoulders not visible / data insufficient
        tilt_score      float   0–100 score for lateral tilt (100 = perfectly centered)
        height_score    float   0–100 score for chin height / chin-up (100 = normal)
        direction       str     "center" | "left" | "right"
        asymmetry       float   normalised |left_dist - right_dist| / shoulder_span
        chin_height_ratio float  (shoulder_mid_y - chin_gy) / shoulder_span
        detail          str     human-readable debug string
    """
    # ── Validate pose shoulders ───────────────────────────────────────────────
    try:
        sh_l = pose_landmarks[_POSE_LM_SHOULDER_L]
        sh_r = pose_landmarks[_POSE_LM_SHOULDER_R]
    except (IndexError, TypeError):
        return {"available": False, "tilt_score": 100.0, "height_score": 100.0,
                "direction": "unknown", "asymmetry": 0.0, "chin_height_ratio": 1.0,
                "detail": "shoulders:unavailable"}

    vis_l = getattr(sh_l, 'visibility', 1.0)
    vis_r = getattr(sh_r, 'visibility', 1.0)
    if vis_l < _SHOULDER_VIS_MIN or vis_r < _SHOULDER_VIS_MIN:
        return {"available": False, "tilt_score": 100.0, "height_score": 100.0,
                "direction": "unknown", "asymmetry": 0.0, "chin_height_ratio": 1.0,
                "detail": f"shoulders:low_visibility({vis_l:.2f},{vis_r:.2f})"}

    shoulder_span = math.hypot(sh_r.x - sh_l.x, sh_r.y - sh_l.y)
    if shoulder_span < 0.05:
        # Shoulders too close together — person may be at extreme angle
        return {"available": False, "tilt_score": 100.0, "height_score": 100.0,
                "direction": "unknown", "asymmetry": 0.0, "chin_height_ratio": 1.0,
                "detail": f"shoulders:too_close(span:{shoulder_span:.3f})"}

    # ── Convert chin to global space ──────────────────────────────────────────
    try:
        chin_lm = face_landmarks[_LM_CHIN]
    except IndexError:
        return {"available": False, "tilt_score": 100.0, "height_score": 100.0,
                "direction": "unknown", "asymmetry": 0.0, "chin_height_ratio": 1.0,
                "detail": "chin:landmark_missing"}

    chin_gx, chin_gy = _to_global(chin_lm, crop_meta)

    # ── Signal 1: lateral asymmetry ───────────────────────────────────────────
    dist_l = math.hypot(chin_gx - sh_l.x, chin_gy - sh_l.y)
    dist_r = math.hypot(chin_gx - sh_r.x, chin_gy - sh_r.y)

    dist_l_norm = dist_l / shoulder_span
    dist_r_norm = dist_r / shoulder_span
    asymmetry   = abs(dist_l_norm - dist_r_norm)

    if dist_l_norm < dist_r_norm:
        direction = "left"    # chin closer to left shoulder → tilting left
    elif dist_r_norm < dist_l_norm:
        direction = "right"   # chin closer to right shoulder → tilting right
    else:
        direction = "center"

    if asymmetry < 0.12:    tilt_score = 100.0   # centered — no reduction
    elif asymmetry < 0.20:  tilt_score = 82.0    # subtle tilt — acceptable
    elif asymmetry < 0.30:  tilt_score = 60.0    # noticeable tilt
    elif asymmetry < 0.40:  tilt_score = 35.0    # significant tilt
    elif asymmetry < 0.55:  tilt_score = 15.0    # severe tilt
    else:                   tilt_score = 5.0     # head on shoulder

    # ── Signal 2: chin height above shoulder line ─────────────────────────────
    # Image y-axis: 0=top, 1=bottom. Shoulder_mid_y > chin_gy means chin
    # is ABOVE the shoulder line (normal). Higher ratio = chin further up.
    sh_mid_y          = (sh_l.y + sh_r.y) / 2.0
    chin_height_ratio = (sh_mid_y - chin_gy) / shoulder_span   # positive = above

    if chin_height_ratio < 0.10:
        # Chin at or below shoulder line — extreme forward pitch (cross-check)
        height_score = 10.0
    elif chin_height_ratio < 0.30:
        height_score = 30.0    # chin very low — severe forward pitch
    elif chin_height_ratio < 0.50:
        height_score = 65.0    # slightly low — mild forward pitch
    elif chin_height_ratio <= 2.00:
        height_score = 100.0   # normal range — no reduction
    elif chin_height_ratio <= 2.50:
        height_score = 75.0    # chin raised — mild chin-up posture
    elif chin_height_ratio <= 3.20:
        height_score = 50.0    # chin-up — noticeable
    else:
        height_score = 25.0    # extreme chin-up

    detail = (
        f"tilt:{direction}(asym:{asymmetry:.3f}) "
        f"height:{chin_height_ratio:.2f} "
        f"dists:L{dist_l_norm:.2f}/R{dist_r_norm:.2f}"
    )
    return {
        "available":         True,
        "tilt_score":        tilt_score,
        "height_score":      height_score,
        "direction":         direction,
        "asymmetry":         round(asymmetry, 4),  # type: ignore,
        "chin_height_ratio": round(chin_height_ratio, 3),
        "detail":            detail,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main scoring function
# ─────────────────────────────────────────────────────────────────────────────

def score_head_pose(
    landmarks:      list,
    crop_meta:      dict,
    config:         dict,
    pose_landmarks: list | None = None,
) -> dict:
    """
    Head pose scorer with chin-geometry cross-validation for yaw, pitch, AND
    lateral tilt (chin–shoulder distance).

    Pipeline:
      1. Validate key landmark coordinates
      2. Chin horizontal geometry → yaw direction + estimate
      3. Chin vertical geometry   → pitch state + hard gate (chin-down detection)
      4. solvePnP                 → precise yaw / pitch / roll angles
      5. Fuse signals:
           Yaw:   solvePnP × chin horizontal — blend on disagreement
           Pitch: solvePnP × chin vertical   — take WORST score (conservative)
                  Hard gate from chin vertical overrides everything
           Roll:  chin–shoulder lateral tilt (when pose_landmarks available)
                  → falls back to solvePnP roll when not available
           Chin-up: chin height above shoulder line adjusts pitch score

    Args:
        landmarks:      FaceLandmarker output (crop-local normalised)
        crop_meta:      Face crop metadata dict (from make_face_crop)
        config:         Pipeline config dict
        pose_landmarks: PoseLandmarker output (global normalised). Optional.
                        When provided, enables chin–shoulder lateral tilt scoring.
                        Pass as the full landmark list; indices 11/12 used.

    RULE 11: head_pose module runs before gaze_direction.
    RULE 12: camera matrix uses CROP dimensions and FOV-based focal length.
    """

    # ── Step 1: Landmark validation ──────────────────────────────────────────
    try:
        coords_valid = all(
            0.0 <= landmarks[i].x <= 1.0 and 0.0 <= landmarks[i].y <= 1.0
            for i in _LANDMARK_IDS
        )
    except IndexError:
        coords_valid = False

    if not coords_valid:
        return {
            "score": 0.0, "confidence": 0.0, "detail": "",
            "skipped": True, "skip_reason": "Key landmarks out of frame",
            "yaw_deg": 90.0, "pitch_deg": 0.0, "roll_deg": 0.0,
            "chin_direction": "unknown", "chin_offset": 0.0,
        }

    # ── Step 2: Chin horizontal geometry (yaw) ────────────────────────────────
    chin_geo       = _chin_orientation(landmarks)
    chin_direction = chin_geo["direction"]
    chin_offset    = chin_geo["chin_offset"]
    chin_yaw_est   = chin_geo["yaw_estimate"]
    chin_reliable  = chin_geo["reliable"]
    chin_yaw_score = _chin_score_from_offset(abs(chin_offset))

    # ── Step 3: Chin vertical geometry (pitch) ────────────────────────────────
    chin_vert        = _chin_vertical_pitch(landmarks)
    chin_drop_ratio  = chin_vert["chin_drop_ratio"]
    pitch_state      = chin_vert["pitch_state"]
    chin_pitch_est   = chin_vert["pitch_estimate"]
    chin_pitch_score = _pitch_score_from_chin(chin_drop_ratio)
    pitch_hard_gate  = chin_vert["hard_gate"]

    # Hard gate: chin-down is so extreme the photo is unusable regardless of
    # what solvePnP reports. Catch it before running expensive PnP solve.
    if pitch_hard_gate and chin_drop_ratio < 0.30:
        return {
            "score":      5.0,
            "confidence": 0.90,
            "detail": (
                f"Yaw:--deg Pitch:severe_down Roll:--deg "
                f"chin:{chin_direction}(offset:{chin_offset:.3f}) "
                f"chin_drop:{chin_drop_ratio:.3f} HARD_GATE:severe_chin_down"
            ),
            "skipped":     False,
            "skip_reason": "",
            "yaw_deg":         chin_yaw_est,
            "pitch_deg":       chin_pitch_est,
            "roll_deg":        0.0,
            "chin_direction":  chin_direction,
            "chin_offset":     chin_offset,
            "pitch_state":     pitch_state,
            "chin_drop_ratio": chin_drop_ratio,
            # Lateral tilt not evaluated on hard gate — defaults
            "lateral_tilt_direction": "unknown",
            "lateral_tilt_asymmetry": 0.0,
            "chin_height_ratio":      chin_drop_ratio,
        }

    # ── Step 4: solvePnP ──────────────────────────────────────────────────────
    scaled_w = crop_meta['scaled_w']
    scaled_h = crop_meta['scaled_h']

    pts2d = np.array([
        [landmarks[i].x * scaled_w, landmarks[i].y * scaled_h]
        for i in _LANDMARK_IDS
    ], dtype=np.float64)

    cam  = build_camera_matrix(scaled_w, scaled_h, config['CAMERA']['fov_degrees'])
    dist = np.zeros((4, 1))

    ok, rvec, _ = cv2.solvePnP(MODEL_3D_POINTS, pts2d, cam, dist,
                                flags=cv2.SOLVEPNP_ITERATIVE)
    pnp_failed = not ok

    if not pnp_failed:
        rmat, _    = cv2.Rodrigues(rvec)
        angles, *_ = cv2.RQDecomp3x3(rmat)
        pitch, yaw, roll = [float(a) for a in angles]
        if abs(roll) > 90:
            roll = roll - 180 * np.sign(roll)

    # ── Step 5a: Yaw score (solvePnP × chin horizontal fusion) ───────────────
    if pnp_failed:
        yaw   = chin_yaw_est
        pitch = chin_pitch_est
        roll  = 0.0
        final_y_score = chin_yaw_score
        fusion_note   = "pnp_failed:chin_fallback"
        confidence    = 0.75
    else:
        abs_yaw = abs(yaw)
        if abs_yaw < 10:   pnp_y_score = 100.0
        elif abs_yaw < 20: pnp_y_score = 85.0
        elif abs_yaw < 35: pnp_y_score = 60.0
        elif abs_yaw < 45: pnp_y_score = 35.0
        else:              pnp_y_score = 10.0

        yaw_disagreement = abs(abs_yaw - abs(chin_yaw_est))
        if not chin_reliable or yaw_disagreement <= 15:
            final_y_score = pnp_y_score
            fusion_note   = f"chin:{chin_direction}"
        else:
            final_y_score = pnp_y_score * 0.70 + chin_yaw_score * 0.30
            fusion_note   = f"chin:{chin_direction}(yaw_diff:{yaw_disagreement:.0f}deg)"

        confidence = 1.0

    # ── Step 5b: Pitch score (solvePnP × chin vertical fusion) ───────────────
    #
    # Direction-aware fusion strategy:
    #
    #   chin_drop_ratio is specifically calibrated to detect CHIN-DOWN poses
    #   (drop_ratio < 0.85 = chin moving toward chest). When solvePnP reports
    #   POSITIVE pitch (chin-up), chin_drop_ratio stays in its "frontal" zone
    #   (0.85–2.5) and gives score=100 — NOT because the pitch is fine, but
    #   because chin_drop_ratio simply has no resolution for mild chin-up.
    #   Applying min() in this case discards the solvePnP signal wrongly.
    #
    #   Rule:
    #     - pitch < 0 (chin-down) OR chin_drop_ratio < 0.75 → min() applies
    #       (conservative: chin_drop catches solvePnP underreporting chin-down)
    #     - pitch >= 0 (chin-up) AND chin_drop in frontal zone → trust solvePnP
    #       (chin_drop has no meaningful chin-up signal in this range)
    #
    #   Asymmetric bands: mild chin-UP (12-20°) is less problematic than chin-DOWN.
    #   Chin-up is common in standing/presentation photos; chin-down reads as
    #   submissive or distracted. Separate scoring reflects this reality.
    if not pnp_failed:
        abs_pitch = abs(pitch)
        if pitch >= 0:   # chin-up — less disruptive, more forgiving bands
            if abs_pitch < 5:    pnp_p_score = 100.0
            elif abs_pitch < 12: pnp_p_score = 88.0
            elif abs_pitch < 20: pnp_p_score = 75.0   # was 65.0 — too harsh for mild chin-up
            elif abs_pitch < 30: pnp_p_score = 50.0
            elif abs_pitch < 45: pnp_p_score = 25.0
            else:                pnp_p_score = 8.0
        else:            # chin-down — more problematic, stricter bands
            if abs_pitch < 5:    pnp_p_score = 100.0
            elif abs_pitch < 12: pnp_p_score = 85.0
            elif abs_pitch < 20: pnp_p_score = 65.0
            elif abs_pitch < 30: pnp_p_score = 40.0
            elif abs_pitch < 45: pnp_p_score = 20.0
            else:                pnp_p_score = 5.0
    else:
        pnp_p_score = chin_pitch_score

    # Apply chin_drop override only when it's actually informative:
    #   - chin is going DOWN (negative pitch) → chin_drop IS the right signal
    #   - chin_drop_ratio itself < 0.75 → geometry is actively detecting a problem
    # For chin-UP with frontal chin_drop, skip the min() to avoid spurious penalty.
    chin_down_signal = (pitch < 0) or (chin_drop_ratio < 0.75)
    if chin_down_signal:
        final_p_score = min(pnp_p_score, chin_pitch_score)
    else:
        final_p_score = pnp_p_score   # chin_drop not informative for chin-up

    # Detail note for pitch fusion decision
    pitch_fusion_note = ""
    if not pnp_failed:
        if chin_down_signal and chin_pitch_score < pnp_p_score:
            pitch_fusion_note = f" pitch_override:chin_down(drop:{chin_drop_ratio:.2f})"
        elif not chin_down_signal:
            pitch_fusion_note = " pitch:chin_up_pnp_only"

    # ── Step 5c: Roll / lateral tilt score ───────────────────────────────────
    #
    # When pose_landmarks are available, chin–shoulder distance replaces the
    # solvePnP roll score entirely. The chin–shoulder signal is more reliable
    # for lateral tilt because:
    #   a) It measures the head–body relationship directly (not just head orientation)
    #   b) solvePnP roll is susceptible to face geometry asymmetry noise
    #   c) The chin-height signal also cross-validates the pitch score here
    #
    # When pose_landmarks are unavailable, solvePnP roll is used unchanged.

    chin_shoulder_note = ""
    tilt_data = None

    if pose_landmarks is not None:
        tilt_data = _chin_shoulder_tilt(face_landmarks=landmarks,
                                        pose_landmarks=pose_landmarks,
                                        crop_meta=crop_meta)
        if tilt_data["available"]:
            # Replace roll score with lateral tilt score
            r_score = tilt_data["tilt_score"]

            # Chin height above shoulder line: cross-validates pitch.
            # Take the worst of the three pitch signals — we are conservative.
            final_p_score = min(final_p_score, tilt_data["height_score"])

            chin_shoulder_note = " | " + tilt_data["detail"]
        else:
            # Pose landmarks present but unreliable (low visibility etc.)
            # Fall back to solvePnP roll below
            abs_roll = abs(roll)
            if abs_roll < 5:    r_score = 100.0
            elif abs_roll < 10: r_score = 80.0
            elif abs_roll < 20: r_score = 55.0
            else:               r_score = 25.0
            chin_shoulder_note = " | " + tilt_data["detail"]
    else:
        # No pose landmarks — use solvePnP roll as before
        abs_roll = abs(roll)
        if abs_roll < 5:    r_score = 100.0
        elif abs_roll < 10: r_score = 80.0
        elif abs_roll < 20: r_score = 55.0
        else:               r_score = 25.0

    # ── Final composite: yaw 50% / pitch 30% / roll (or tilt) 20% ────────────
    combined = final_y_score * 0.5 + final_p_score * 0.3 + r_score * 0.2

    result = {
        "score":      round(float(combined), 1),  # type: ignore,
        "confidence": confidence,
        "detail": (
            f"Yaw:{yaw:.1f}deg Pitch:{pitch:.1f}deg Roll:{roll:.1f}deg "
            f"chin:{chin_direction}(offset:{chin_offset:.3f}) "
            f"drop:{chin_drop_ratio:.3f}({pitch_state}) "
            f"{fusion_note}{pitch_fusion_note}"
            f"{chin_shoulder_note}"
        ),
        "skipped":     False,
        "skip_reason": "",
        # Passed downstream — stripped before final output
        "yaw_deg":         yaw,
        "pitch_deg":       pitch,
        "roll_deg":        roll,
        "chin_direction":  chin_direction,
        "chin_offset":     chin_offset,
        "pitch_state":     pitch_state,
        "chin_drop_ratio": chin_drop_ratio,
    }

    # Attach lateral tilt fields when available — used by visualiser overlay
    if tilt_data and tilt_data["available"]:
        result["lateral_tilt_direction"] = tilt_data["direction"]
        result["lateral_tilt_asymmetry"] = tilt_data["asymmetry"]
        result["chin_height_ratio"]      = tilt_data["chin_height_ratio"]

    return result