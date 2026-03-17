import cv2

from .head_pose import score_head_pose
from .gaze_direction import score_gaze_direction
from .eye_openness import score_eye_openness
from .smile import score_smile

def run_face_group(face_crop_bgr, crop_meta, face_landmarker, config) -> dict:
    """
    STAGE 7: FACE GROUP
    BGR->RGB conversion
    Run MediaPipe Face on face crop
    Strict order: head_pose -> gaze -> eye -> smile
    """
    import mediapipe as mp
    
    rgb_image = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    
    results = face_landmarker.detect(mp_image)
    
    if not results.face_landmarks or len(results.face_landmarks) == 0:
        return {
            "_raw": None,
            "group_score": None,
            "group_confidence": 0.0,
            "modules": {
                "head_pose": {"score": 0.0, "confidence": 0.0, "skipped": True, "detail": "", "skip_reason": "No face detected in crop"},
                "gaze_direction": {"score": 0.0, "confidence": 0.0, "skipped": True, "detail": "", "skip_reason": "No face detected in crop"},
                "eye_openness": {"score": 0.0, "confidence": 0.0, "skipped": True, "detail": "", "skip_reason": "No face detected in crop"},
                "smile": {"score": 0.0, "confidence": 0.0, "skipped": True, "detail": "", "skip_reason": "No face detected in crop"},
            }
        }
        
    landmarks = results.face_landmarks[0]
    
    head_pose_res = score_head_pose(landmarks, crop_meta, config)
    yaw   = head_pose_res.get("yaw_deg",   90.0)
    pitch = head_pose_res.get("pitch_deg", 90.0)
    roll  = head_pose_res.get("roll_deg",   0.0)

    # Roll gate — hard reject before any other scoring
    roll_gate = config['FACE'].get('roll_gate', 45.0)
    if abs(roll) > roll_gate:
        return {
            "_raw": results,
            "group_score": None,
            "group_confidence": 0.0,
            "rejected": True,
            "reject_reason": f"Excessive head roll {roll:.1f}deg > {roll_gate}deg",
            "modules": {
                k: {"score": 0.0, "confidence": 0.0, "skipped": True,
                    "detail": "", "skip_reason": "Roll gate triggered"}
                for k in ["head_pose", "gaze_direction", "eye_openness", "smile"]
            },
        }

    gaze_res  = score_gaze_direction(landmarks, abs(yaw), abs(pitch), config)
    eye_res   = score_eye_openness(landmarks, config)
    blendshapes = results.face_blendshapes[0] if results.face_blendshapes else None
    smile_res = score_smile(landmarks, config, blendshapes=blendshapes)
    
    module_results = {
        "head_pose":      {k: v for k, v in head_pose_res.items() if k not in ("yaw_deg", "pitch_deg", "roll_deg")},
        "gaze_direction": gaze_res,
        "eye_openness":   eye_res,
        "smile":          smile_res,
    }

    # Face quality gate — 3 hard-reject conditions
    eye  = module_results["eye_openness"]
    gaze = module_results["gaze_direction"]

    # Hard reject 1: gaze skipped → subject not facing camera (yaw > 30° or pitch > 40°)
    # At these angles a portrait is unusable — a human critic would discard it.
    gaze_skipped = gaze.get("skipped", False)

    # Hard reject 2: eyes confirmed closed (EAR < 0.10, not a confidence failure)
    # eye_openness scores 0 only when avg_ear < 0.10 (fully closed).
    # Low confidence returns skipped=True (not score=0), so those are NOT caught here.
    eyes_confirmed_closed = (
        not eye.get("skipped", True) and eye.get("score", 100) == 0.0
    )

    # Hard reject 3: face completely undetectable — all modules skipped
    all_failed = all(res.get("skipped", True) for res in module_results.values())

    if gaze_skipped or eyes_confirmed_closed or all_failed:
        if gaze_skipped:
            reject_reason = f"Gaze skipped — subject not facing camera ({gaze.get('skip_reason', '')})"
        elif eyes_confirmed_closed:
            reject_reason = "Eyes confirmed closed (EAR < 0.10)"
        else:
            reject_reason = "All face modules failed — face undetectable"
        return {
            "_raw": results,
            "group_score": None,
            "group_confidence": 0.0,
            "rejected": True,
            "reject_reason": reject_reason,
            "modules": module_results,
        }


    # ── Combined pitch + roll penalty — applied once at group level ──────────
    abs_pitch           = abs(pitch)
    abs_roll_val        = abs(roll)
    pitch_penalty_start = config['FACE'].get('pitch_penalty_start', 15.0)
    roll_penalty_start  = config['FACE'].get('roll_penalty_start',  15.0)
    pitch_penalty_skip  = config['FACE'].get('gaze_pitch_skip', 40.0)

    pitch_factor = 1.0
    if abs_pitch >= pitch_penalty_start:
        pitch_range  = pitch_penalty_skip - pitch_penalty_start
        pitch_factor = max(0.50, 1.0 - ((abs_pitch - pitch_penalty_start) / (pitch_range + 1e-6)) * 1.20)

    roll_factor = 1.0
    if abs_roll_val >= roll_penalty_start:
        roll_range  = 45.0 - roll_penalty_start
        roll_factor = max(0.50, 1.0 - ((abs_roll_val - roll_penalty_start) / (roll_range + 1e-6)) * 1.20)

    combined_factor = pitch_factor * roll_factor

    # Smile rescue: Duchenne smile partially softens pitch penalty
    smile_score = smile_res.get('score', 0)
    if smile_score >= 85.0 and abs_pitch < 25.0 and combined_factor < 1.0:
        combined_factor = combined_factor * 0.60 + 1.0 * 0.40  # soften by 40%

    if config.get('DEBUG', False):
        print(f"[FACE PENALTY] pitch={abs_pitch:.1f} roll={abs_roll_val:.1f} "
              f"pf={pitch_factor:.3f} rf={roll_factor:.3f} combined={combined_factor:.3f}"
              f" smile_rescue={'YES' if smile_score >= 85.0 and abs_pitch < 25.0 and combined_factor < 1.0 else 'no'}")

    valid_confs = [res["confidence"] for res in module_results.values() if not res["skipped"]]
    group_conf = sum(valid_confs) / len(valid_confs) if valid_confs else 0.0

    # Calculate group_score using FACE_WEIGHTS
    face_weights = config.get('FACE_WEIGHTS', {
        "head_pose": 0.25, "gaze_direction": 0.25,
        "eye_openness": 0.30, "smile": 0.20
    })
    num, den = 0.0, 0.0
    low_thresh = config['CONFIDENCE']['low_threshold']
    low_factor = config['CONFIDENCE']['low_weight_factor']
    for name, res in module_results.items():
        if res.get('skipped', True):
            continue
        w = face_weights.get(name, 0.0)
        c = res['confidence']
        ec = c if c >= low_thresh else c * low_factor
        if ec > 0:
            num += res['score'] * w * ec
            den += w * ec

    group_score = round(num / (den + 1e-9), 1) if den > 0 else None

    # Apply combined factor once at group level
    if group_score is not None and combined_factor < 1.0:
        group_score = round(group_score * combined_factor, 1)
        if config.get('DEBUG', False):
            print(f"[FACE PENALTY GROUP] group_score={group_score}")

    return {
        "_raw": results,
        "group_score": group_score,
        "group_confidence": round(group_conf, 2),
        "modules": module_results
    }
