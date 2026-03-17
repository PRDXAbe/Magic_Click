def iris_plausible(landmarks, inner_corner, outer_corner, upper_lid, lower_lid, iris_idx):
    """
    Iris center must lie within the eye corner + lid bounding box.
    Catches occlusion/hallucination — EAR stays high but iris position is implausible.
    """
    try:
        iris_x  = landmarks[iris_idx].x
        iris_y  = landmarks[iris_idx].y
        inner_x = landmarks[inner_corner].x
        outer_x = landmarks[outer_corner].x
        upper_y = landmarks[upper_lid].y
        lower_y = landmarks[lower_lid].y
        TOL = 0.02
        x_ok = (min(inner_x, outer_x) - TOL) <= iris_x <= (max(inner_x, outer_x) + TOL)
        y_ok = (min(upper_y, lower_y) - TOL) <= iris_y <= (max(upper_y, lower_y) + TOL)
        return x_ok and y_ok
    except IndexError:
        return False


def score_eye_openness(landmarks, config) -> dict:
    if not landmarks:
        return {"score": 0.0, "confidence": 0.0, "skipped": True, "detail": "", "skip_reason": "No landmarks"}

    req_r = [33, 133, 159, 145, 160, 144]
    req_l = [263, 362, 386, 374, 387, 373]

    try:
        coords_ok_r = all(0.0 <= landmarks[i].x <= 1.0 and 0.0 <= landmarks[i].y <= 1.0 for i in req_r)
        coords_ok_l = all(0.0 <= landmarks[i].x <= 1.0 and 0.0 <= landmarks[i].y <= 1.0 for i in req_l)
    except IndexError:
        coords_ok_r, coords_ok_l = False, False
    conf = 1.0 if (coords_ok_r and coords_ok_l) else 0.0

    def calc_ear(idx_list):
        p1, p4 = landmarks[idx_list[0]], landmarks[idx_list[1]]
        p2, p6 = landmarks[idx_list[2]], landmarks[idx_list[3]]
        p3, p5 = landmarks[idx_list[4]], landmarks[idx_list[5]]
        w  = ((p1.x - p4.x)**2 + (p1.y - p4.y)**2)**0.5
        h1 = ((p2.x - p6.x)**2 + (p2.y - p6.y)**2)**0.5
        h2 = ((p3.x - p5.x)**2 + (p3.y - p5.y)**2)**0.5
        return (h1 + h2) / (2.0 * w + 1e-9)

    if conf < 0.5:
        return {"score": 0.0, "confidence": 0.0, "skipped": True, "detail": "", "skip_reason": "Eye landmarks out of frame"}

    r_ear = calc_ear(req_r)
    l_ear = calc_ear(req_l)
    avg_ear = (r_ear + l_ear) / 2.0
    ear_min = min(r_ear, l_ear)
    asym    = abs(l_ear - r_ear)

    # ── Iris plausibility — catches hand/object occlusion ─────────────────────
    # Right eye: inner=133, outer=33, upper=159, lower=145, iris=468
    # Left eye:  inner=362, outer=263, upper=386, lower=374, iris=473
    r_iris_ok = iris_plausible(landmarks, 133, 33, 159, 145, 468)
    l_iris_ok = iris_plausible(landmarks, 362, 263, 386, 374, 473)
    if config.get('DEBUG', False):
        print(f"[IRIS DEBUG] R_ok={r_iris_ok} L_ok={l_iris_ok} "
              f"iris_r={landmarks[468].x:.3f} iris_l={landmarks[473].x:.3f} "
              f"r_corners=({landmarks[33].x:.3f},{landmarks[133].x:.3f}) "
              f"l_corners=({landmarks[263].x:.3f},{landmarks[362].x:.3f}) "
              f"asym={abs(l_ear-r_ear):.3f} l={l_ear:.3f} r={r_ear:.3f}")


    if not r_iris_ok or not l_iris_ok:
        return {
            "score": 0.0,
            "confidence": conf,
            "detail": f"EAR L:{l_ear:.2f} R:{r_ear:.2f} iris_occluded:{'R' if not r_iris_ok else 'L'}",
            "skipped": False,
            "skip_reason": ""
        }

    # ── Hard close detection — either eye closed = score 0 ───────────────────
    # Triggers face quality gate → FACE_QUALITY_REJECTED
    CLOSED_THRESHOLD = 0.15
    if ear_min < CLOSED_THRESHOLD:
        min_conf = config['CONFIDENCE']['min_to_score']
        if conf < min_conf:
            return {"score": 0.0, "confidence": conf, "skipped": True, "detail": "", "skip_reason": f"Confidence {conf:.2f} < {min_conf}"}
        return {
            "score": 0.0,
            "confidence": conf,
            "detail": f"EAR L:{l_ear:.2f} R:{r_ear:.2f} asym:{asym:.2f}",
            "skipped": False,
            "skip_reason": ""
        }

    # ── Asymmetry gate — winking/squinting ───────────────────────────────────
    ASYM_THRESHOLD = 0.042  # was 0.05 — catches partial single-eye occlusion
    if asym > ASYM_THRESHOLD:
        return {
            "score": 0.0,
            "confidence": conf,
            "detail": f"EAR L:{l_ear:.2f} R:{r_ear:.2f} asym:{asym:.2f}",
            "skipped": False,
            "skip_reason": ""
        }

    # ── Normal scoring — both eyes open and symmetric ────────────────────────
    # EAR bands: MediaPipe consistently underestimates EAR vs raw geometry,
    # so thresholds are slightly lower than canonical.
    # The 0.20-0.25 band is raised to 85 (was 80): EAR=0.22-0.25 represents
    # naturally relaxed eyes and the previous 80 was unnecessarily harsh.
    if avg_ear > 0.25:    base = 100.0
    elif avg_ear >= 0.20: base = 85.0    # was 80.0 — natural relaxed eye opening
    elif avg_ear >= 0.15: base = 55.0    # was 50.0 — proportional adjustment
    else:                 base = 20.0

    # Small continuous asymmetry penalty (cosmetic only — hard gates above catch real cases)
    soft_asym_penalty = min(asym * 30.0, 15.0)
    score = max(0.0, base - soft_asym_penalty)

    min_conf = config['CONFIDENCE']['min_to_score']
    if conf < min_conf:
        return {"score": 0.0, "confidence": conf, "skipped": True, "detail": "", "skip_reason": f"Confidence {conf:.2f} < {min_conf}"}

    return {
        "score": score,
        "confidence": conf,
        "detail": f"EAR L:{l_ear:.2f} R:{r_ear:.2f} asym:{asym:.2f}",
        "skipped": False,
        "skip_reason": ""
    }