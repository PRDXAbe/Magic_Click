def score_shoulder_symmetry(landmarks, config) -> dict:
    if not landmarks:
        return {"score": 0.0, "confidence": 0.0, "skipped": True, "detail": "", "skip_reason": "No landmarks"}

    req_idx = [11, 12]
    for idx in req_idx:
        if landmarks[idx].visibility < 0.5:
            return {"score": 0.0, "confidence": 0.0, "skipped": True, "detail": "", "skip_reason": f"Landmark {idx} visibility < 0.5"}

    shoulder_dist = abs(landmarks[11].x - landmarks[12].x)  # horizontal distance
    if shoulder_dist < 0.01:
        return {"score": 0.0, "confidence": 0.0, "skipped": True,
                "detail": "", "skip_reason": "Shoulders too close"}

    raw_diff  = abs(landmarks[11].y - landmarks[12].y)
    norm_diff = raw_diff / (shoulder_dist + 1e-6)

    if   norm_diff < 0.05: score = 100.0
    elif norm_diff < 0.10: score = 85.0
    elif norm_diff < 0.15: score = 70.0
    elif norm_diff < 0.20: score = 50.0
    elif norm_diff < 0.30: score = 30.0
    else:                  score = 15.0

    # CHANGE: presence → visibility
    conf = min(landmarks[11].visibility, landmarks[12].visibility)
    min_conf = config['CONFIDENCE']['min_to_score']
    
    if conf < min_conf:
        return {"score": 0.0, "confidence": conf, "skipped": True, "detail": "", "skip_reason": f"Confidence {conf:.2f} < {min_conf}"}

    return {
        "score": score,
        "confidence": conf,
        "detail": f"norm_diff:{norm_diff:.3f} {'left_higher' if landmarks[11].y < landmarks[12].y else 'right_higher'}",
        "skipped": False,
        "skip_reason": ""
    }
