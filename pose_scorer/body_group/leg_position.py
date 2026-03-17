def score_leg_position(landmarks, config) -> dict:
    if not landmarks:
        return {"score": 0.0, "confidence": 0.0, "skipped": True, "detail": "", "skip_reason": "No landmarks"}

    req_idx = [27, 28] # left_ankle, right_ankle
    for idx in req_idx:
        if landmarks[idx].visibility < 0.5:
            return {"score": 0.0, "confidence": 0.0, "skipped": True, "detail": "", "skip_reason": "Ankles not visible (visibility<0.5)"}

    shoulder_dist = abs(landmarks[11].x - landmarks[12].x)
    if shoulder_dist < 0.01:
        return {"score": 0.0, "confidence": 0.0, "skipped": True,
                "detail": "", "skip_reason": "Shoulder reference too small"}

    ankle_sep = abs(landmarks[27].x - landmarks[28].x)
    stance    = ankle_sep / (shoulder_dist + 1e-6)
    
    print(f"DEBUG legs: lm27.x={landmarks[27].x:.3f} lm28.x={landmarks[28].x:.3f} stance={stance:.3f}")
    
    # Require stance to be relatively small for legs to be considered "crossed"
    # Note: normally 27.x > 28.x is uncrossed for non-mirrored images where person faces camera,
    # but we'll stick to the user's condition and constrain it with stance < 0.30.
    crossed   = (landmarks[27].x > landmarks[28].x) and (stance < 0.30)

    if crossed:
        score, txt = 50.0,  "crossed"
    elif stance < 0.30:
        score, txt = 70.0,  "feet_together"
    elif stance <= 0.60:
        score, txt = 100.0, "natural"
    elif stance <= 1.00:
        score, txt = 85.0,  "relaxed_wide"
    else:
        score, txt = 60.0,  "very_wide"

    # CHANGE: presence → visibility
    conf   = min(landmarks[27].visibility, landmarks[28].visibility)
    detail = f"stance:{stance:.3f} {txt}"

    min_conf = config['CONFIDENCE']['min_to_score']
    
    if conf < min_conf:
        return {"score": 0.0, "confidence": conf, "skipped": True, "detail": "", "skip_reason": f"Confidence {conf:.2f} < {min_conf}"}

    return {
        "score": score,
        "confidence": conf,
        "detail": detail,
        "skipped": False,
        "skip_reason": ""
    }
