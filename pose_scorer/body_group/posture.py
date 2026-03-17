import math

def score_posture(landmarks, config) -> dict:
    """
    Spine angle + head alignment penalty.
    """
    if not landmarks:
        return {"score": 0.0, "confidence": 0.0, "skipped": True, "detail": "", "skip_reason": "No landmarks"}

    req_idx = [0, 11, 12, 23, 24] # nose, shoulders, hips
    for idx in req_idx:
        if landmarks[idx].visibility < 0.5:
            return {"score": 0.0, "confidence": 0.0, "skipped": True, "detail": "", "skip_reason": f"Landmark {idx} visibility < 0.5"}

    # Compute shoulder reference for distance-invariant measurement
    shoulder_dist = ((landmarks[11].x - landmarks[12].x)**2 +
                     (landmarks[11].y - landmarks[12].y)**2) ** 0.5

    if shoulder_dist < 0.02:
        return {"score": 0.0, "confidence": 0.0, "skipped": True,
                "detail": "", "skip_reason": "Shoulders too close — unreliable"}

    shoulder_mid_x = (landmarks[11].x + landmarks[12].x) / 2.0
    shoulder_mid_y = (landmarks[11].y + landmarks[12].y) / 2.0
    hip_mid_x      = (landmarks[23].x + landmarks[24].x) / 2.0

    # Lean normalised by shoulder width — distance invariant
    dx   = hip_mid_x - shoulder_mid_x
    lean = abs(dx) / (shoulder_dist + 1e-6)

    if   lean < 0.10: spine_score = 100.0
    elif lean < 0.25: spine_score = 80.0
    elif lean < 0.50: spine_score = 55.0
    else:             spine_score = 25.0

    nose_x = landmarks[0].x
    diff   = abs(nose_x - shoulder_mid_x) / (shoulder_dist + 1e-6)

    if   diff < 0.15: head_penalty = 0.0
    elif diff < 0.40: head_penalty = 10.0
    else:             head_penalty = 25.0

    final_score = max(0.0, spine_score - head_penalty)
    # CHANGE: presence → visibility (more reliable for body landmarks)
    conf = min(landmarks[i].visibility for i in [0, 11, 12, 23, 24])
    
    min_conf = config['CONFIDENCE']['min_to_score']
    if conf < min_conf:
        return {"score": 0.0, "confidence": conf, "skipped": True, "detail": "", "skip_reason": f"Confidence {conf:.2f} < {min_conf}"}

    return {
        "score": final_score,
        "confidence": conf,
        "detail": f"lean:{lean:.3f} head_offset:{diff:.3f}",
        "skipped": False,
        "skip_reason": ""
    }
