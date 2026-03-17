def score_hand_position(landmarks, config) -> dict:
    if not landmarks:
        return {"score": 0.0, "confidence": 0.0, "skipped": True, "detail": "", "skip_reason": "No landmarks"}

    shoulder_dist = ((landmarks[11].x - landmarks[12].x)**2 +
                     (landmarks[11].y - landmarks[12].y)**2) ** 0.5
    if shoulder_dist < 0.02:
        return {"score": 0.0, "confidence": 0.0, "skipped": True,
                "detail": "", "skip_reason": "Shoulder reference too small"}

    # 15=left_wrist, 16=right_wrist, 23=left_hip, 24=right_hip
    def check_wrist(wrist_idx, hip_idx):
        if landmarks[wrist_idx].visibility < 0.5 or landmarks[hip_idx].visibility < 0.5:
            return None, None
        
        dx_norm = abs(landmarks[wrist_idx].x - landmarks[hip_idx].x) / (shoulder_dist + 1e-6)
        dy_norm = (landmarks[wrist_idx].y - landmarks[hip_idx].y) / (shoulder_dist + 1e-6)

        if dy_norm > -0.2 and dx_norm < 0.5:
            return 100.0, "natural_sides"
        elif dy_norm > -0.4 and dx_norm >= 0.5:
            return 85.0, "slightly_away"
        elif -0.8 < dy_norm <= -0.2 and dx_norm < 0.5:
            return 75.0, "hands_on_hips"
        elif -1.2 < dy_norm <= -0.4 and dx_norm < 0.8:
            return 70.0, "arms_crossed"
        elif dy_norm <= -1.2:
            return 50.0, "hands_raised"
        else:
            return 30.0, "covering"

    left_score, left_detail = check_wrist(15, 23)
    right_score, right_detail = check_wrist(16, 24)

    if left_score is not None and right_score is not None:
        score = (left_score + right_score) / 2.0
        detail = f"L:{left_detail} R:{right_detail}"
        conf = min(landmarks[15].visibility, landmarks[16].visibility)
    elif left_score is not None:
        score = left_score
        detail = left_detail
        conf = landmarks[15].visibility
    elif right_score is not None:
        score = right_score
        detail = right_detail
        conf = landmarks[16].visibility
    else:
        return {"score": 0.0, "confidence": 0.0, "skipped": True, "detail": "", "skip_reason": "Wrists not visible (visibility<0.5)"}

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
