import cv2

from .body_orientation import score_body_orientation
from .posture import score_posture
from .shoulder_symmetry import score_shoulder_symmetry
from .hand_position import score_hand_position
from .leg_position import score_leg_position

def run_body_group(bgr_image, pose_landmarker, config) -> dict:
    """
    STAGE 6: BODY GROUP
    BGR->RGB conversion
    Run MediaPipe Pose on full image
    """
    import mediapipe as mp

    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    results = pose_landmarker.detect(mp_image)

    if not results.pose_landmarks or len(results.pose_landmarks) == 0:
        return {
            "_raw": None,
            "group_score": None,
            "group_confidence": 0.0,
            "modules": {
                "body_orientation": {"score": 0.0, "confidence": 0.0, "skipped": True, "detail": "", "skip_reason": "No pose detected"},
                "posture":          {"score": 0.0, "confidence": 0.0, "skipped": True, "detail": "", "skip_reason": "No pose detected"},
                "shoulder_symmetry":{"score": 0.0, "confidence": 0.0, "skipped": True, "detail": "", "skip_reason": "No pose detected"},
                "hand_position":    {"score": 0.0, "confidence": 0.0, "skipped": True, "detail": "", "skip_reason": "No pose detected"},
                "leg_position":     {"score": 0.0, "confidence": 0.0, "skipped": True, "detail": "", "skip_reason": "No pose detected"},
            }
        }

    pose       = results.pose_landmarks[0]
    world_pose = results.pose_world_landmarks[0] if results.pose_world_landmarks else None

    module_results = {
        "body_orientation":  score_body_orientation(pose, world_pose, config),
        "posture":           score_posture(pose, config),
        "shoulder_symmetry": score_shoulder_symmetry(pose, config),
        "hand_position":     score_hand_position(pose, config),
        "leg_position":      score_leg_position(pose, config),
    }

    # ── Lean penalty — parse normalized lean ratio from posture detail ─────────
    # posture.py emits: "lean:0.123 head_offset:0.045"
    posture_detail = module_results["posture"].get("detail", "")
    try:
        lean = float(posture_detail.split("lean:")[1].split(" ")[0])
    except (IndexError, ValueError):
        lean = 0.0

    lean_threshold = config.get('BODY', {}).get('lean_penalty_threshold', 0.30)
    if config.get('DEBUG', False):
        print(f"[LEAN DEBUG] detail='{posture_detail}' lean={lean:.3f} threshold={lean_threshold:.3f} fires={lean >= lean_threshold}")
    if lean >= lean_threshold:
        penalty_range  = 0.60 - lean_threshold
        penalty_factor = max(0.75, 1.0 - ((lean - lean_threshold) / (penalty_range + 1e-6)) * 0.25)
        for mod in module_results.values():
            if not mod.get("skipped", False):
                mod["score"] = round(mod["score"] * penalty_factor, 1)

    # ── Inline group score ─────────────────────────────────────────────────────
    body_weights = config.get('BODY_WEIGHTS', {
        "body_orientation": 0.30, "posture": 0.25, "shoulder_symmetry": 0.20,
        "hand_position": 0.15, "leg_position": 0.10
    })
    low_thresh = config['CONFIDENCE']['low_threshold']
    low_factor = config['CONFIDENCE']['low_weight_factor']

    num = den = 0.0
    for name, res in module_results.items():
        is_skipped = res.get('skipped', True)
        
        if is_skipped and name != "leg_position":
            continue
            
        w  = body_weights.get(name, 0.0)
        
        if is_skipped and name == "leg_position":
            # Apply 0 score penalty with full effective confidence
            score = 0.0
            ec = 1.0
            # Update the dict so the visualizer shows the 0.0 penalty instead of "SKIP"
            res['score'] = 0.0
            res['skipped'] = False
            res['detail'] = "penalized (not visible)"
        else:
            score = res['score']
            c  = res['confidence']
            ec = c if c >= low_thresh else c * low_factor
            
        if ec > 0:
            num += score * w * ec
            den += w * ec

    group_score = round(num / (den + 1e-9), 1) if den > 0 else None
    valid_confs = [res["confidence"] for res in module_results.values() if not res["skipped"]]

    return {
        "_raw":             results,
        "group_score":      group_score,
        "group_confidence": round(sum(valid_confs) / len(valid_confs), 2) if valid_confs else 0.0,
        "modules":          module_results,
    }
