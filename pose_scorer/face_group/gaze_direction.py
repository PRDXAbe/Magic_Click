def score_gaze_direction(landmarks: list, abs_yaw: float, abs_pitch: float, config: dict) -> dict:
    """
    RULE 11: gaze_direction RUNS AFTER head_pose — abs_yaw and abs_pitch passed in.
    Yaw gate:   hard skip if |yaw|   > gaze_yaw_gate   (30°) — iris unreliable at high angles.
    Pitch gate: hard skip if |pitch| > gaze_pitch_skip  (40°) — iris completely unreliable.
    Pitch penalty: linear 0→25pt penalty in soft zone   (20-40°) — leaning/tilting.
    Coordinates are crop-local normalised [0,1].
    """
    gaze_yaw_gate      = config['FACE']['gaze_yaw_gate']                     # 30°
    gaze_pitch_skip    = config['FACE'].get('gaze_pitch_skip',   40.0)        # 40°
    gaze_pitch_penalty = config['FACE'].get('gaze_pitch_penalty', 20.0)       # 20°

    if abs_yaw > gaze_yaw_gate:
        return {
            "score": 0.0, "confidence": 0.0, "detail": "", "skipped": True,
            "skip_reason": f"Yaw {abs_yaw:.1f}deg > gate {gaze_yaw_gate}deg, iris unreliable"
        }

    if abs_pitch > gaze_pitch_skip:
        return {
            "score": 0.0, "confidence": 0.0, "detail": "", "skipped": True,
            "skip_reason": f"Pitch {abs_pitch:.1f}deg > {gaze_pitch_skip}deg, iris unreliable"
        }

    right_iris = [468, 469, 470, 471]
    left_iris  = [473, 474, 475, 476, 477]

    # FaceLandmarker does not populate .presence/.visibility (always 0.0 in protobuf).
    # Instead, check that iris landmarks exist and have valid non-zero coordinates.
    # Iris landmarks 468-477 are only in the FULL model (not lite).
    try:
        iris_coords = [(landmarks[i].x, landmarks[i].y) for i in right_iris + left_iris]
    except IndexError:
        return {
            "score": 0.0, "confidence": 0.0, "detail": "", "skipped": True,
            "skip_reason": "Iris landmarks out of range — likely using lite model"
        }

    iris_valid = all(
        0.0 < x < 1.0 and 0.0 < y < 1.0
        for x, y in iris_coords
    )
    if not iris_valid:
        return {
            "score": 0.0, "confidence": 0.0, "detail": "", "skipped": True,
            "skip_reason": "Iris landmarks not present (lite model or all-zero coords)"
        }

    def center(ids):
        return (
            sum(landmarks[i].x for i in ids) / len(ids),
            sum(landmarks[i].y for i in ids) / len(ids)
        )

    r_iris = center(right_iris)
    l_iris = center(left_iris)

    # Eye midpoints and dimensions (33/133 right, 263/362 left)
    r_eye_mid_x = (landmarks[33].x  + landmarks[133].x) / 2.0
    l_eye_mid_x = (landmarks[263].x + landmarks[362].x) / 2.0
    r_eye_mid_y = (landmarks[159].y + landmarks[145].y) / 2.0   # top/bottom of right eye
    l_eye_mid_y = (landmarks[386].y + landmarks[374].y) / 2.0   # top/bottom of left eye

    r_eye_width  = abs(landmarks[33].x  - landmarks[133].x)
    l_eye_width  = abs(landmarks[263].x - landmarks[362].x)
    r_eye_height = abs(landmarks[159].y - landmarks[145].y)
    l_eye_height = abs(landmarks[386].y - landmarks[374].y)

    # Horizontal offset (normalised by eye width)
    r_h_off = abs(r_iris[0] - r_eye_mid_x) / (r_eye_width  + 1e-6)
    l_h_off = abs(l_iris[0] - l_eye_mid_x) / (l_eye_width  + 1e-6)
    h_offset = (r_h_off + l_h_off) / 2.0

    # Vertical offset (normalised by eye height)
    r_v_off = abs(r_iris[1] - r_eye_mid_y) / (r_eye_height + 1e-6)
    l_v_off = abs(l_iris[1] - l_eye_mid_y) / (l_eye_height + 1e-6)
    v_offset = (r_v_off + l_v_off) / 2.0

    if config.get('DEBUG', False):
        print(f"DEBUG gaze: r_eye_h={r_eye_height:.4f} l_eye_h={l_eye_height:.4f} r_eye_w={r_eye_width:.4f}")

    # Weight horizontal more since vertical eye height is often very small and normalisation inflates v_off
    offset = (h_offset * 0.7 + v_offset * 0.3)

    if offset < 0.10:   score = 100.0
    elif offset < 0.20: score = 80.0
    elif offset < 0.35: score = 50.0
    else:               score = 20.0

    # Soft penalty for moderate pitch (leaning/tilting but still scoreable)
    detail_suffix = ""
    if abs_pitch > gaze_pitch_penalty:
        pitch_excess  = abs_pitch - gaze_pitch_penalty           # how far into penalty zone
        penalty_range = gaze_pitch_skip - gaze_pitch_penalty     # width of zone (20°)
        penalty       = (pitch_excess / penalty_range) * 12.0    # max 12pt penalty (was 25 — reduced to avoid double-penalty with head_pose)
        score         = max(0.0, score - penalty)
        detail_suffix = f" pitch_penalty:{penalty:.1f}"

    return {
        "score": score,
        "confidence": 1.0,
        "detail": f"h_off:{h_offset:.3f} v_off:{v_offset:.3f}{detail_suffix}",
        "skipped": False,
        "skip_reason": ""
    }
