# config.py — complete template, all values here
import os as _os

# Load .env from project root so BLUR_THRESHOLD and other env vars are available.
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(dotenv_path=_os.path.join(_os.path.dirname(__file__), "..", ".env"))
except ImportError:
    pass

DEBUG = True     # set True to enable debug prints across the pipeline

MODELS = {
    "person_detector":  "models/yolo26n.pt",
    "face_detector":    "models/yolo26n-face.pt",
    "face_landmarker":  "models/face_landmarker.task",      # FULL model, not lite
    "pose_landmarker":  "models/pose_landmarker_full.task",
}

DETECTION = {
    "person_conf":             0.6,
    "face_conf":               0.4,
    "person_pad_top":          0.25,   # asymmetric — head at top
    "person_pad_bottom":       0.10,
    "person_pad_side":         0.10,
    "face_pad":                0.20,   # symmetric
    "face_min_size":           512,    # upscale target for MediaPipe crop
    "min_person_height_ratio": 0.35,   # CALIBRATE — print ratios on your camera first
    # Run: print(f"ratio={(py2-py1)/image_h:.3f}") on 20 images before enabling this gate
}

# BLUR_THRESHOLD is read from .env (BLUR_THRESHOLD=40).
# IP cameras: 30-50. Webcam/DSLR: 80-100.
_blur_threshold = float(_os.environ.get("BLUR_THRESHOLD", "85.0"))

PREFLIGHT = {
    "blur_threshold":   _blur_threshold,
    "min_resolution":   (480, 480),
}

FRAME = {
    "max_horizontal_offset": 0.20,
    "edge_threshold":        0.02,
    "reject_top":    True,
    "reject_left":   True,
    "reject_right":  True,
    "reject_bottom": False,
}

CONFIDENCE = {
    "min_to_score":      0.55,
    "low_threshold":     0.75,
    "low_weight_factor": 0.60,
}

CAMERA = {
    "fov_degrees": 75.0,        # mobile/IP camera horizontal FOV
}

FACE = {
    "gaze_yaw_gate":        30.0,   # skip gaze if |yaw| > this (degrees)
    "gaze_pitch_skip":      40.0,   # hard skip — iris completely unreliable above this
    "gaze_pitch_penalty":   25.0,   # soft zone — score penalty applied above this (was 20 — widened to reduce double-penalty with head_pose)
    "roll_gate":            45.0,   # hard reject if |roll| > this (degrees)
    "pitch_penalty_start":  15.0,   # face pitch penalty begins at this angle (was 20)
    "roll_penalty_start":   15.0,   # face roll penalty begins at this angle (NEW)
}

BODY = {
    "min_orientation_score":  10,    # was 40 — only reject score 5 (back to camera); 3/4 turns now scored
    "lean_penalty_threshold": 0.30,  # normalized lean ratio (NOT degrees) — penalty above this
}

GROUP_WEIGHTS = {"frame_offset": 0.10, "face_group": 0.75, "body_group": 0.15}
FACE_WEIGHTS  = {"head_pose": 0.20, "gaze_direction": 0.20, "eye_openness": 0.25, "smile": 0.35}
BODY_WEIGHTS  = {"body_orientation": 0.30, "posture": 0.25, "shoulder_symmetry": 0.20,
                 "hand_position": 0.15, "leg_position": 0.10}

SCORE_BANDS = [(90,100,"Excellent"),(75,90,"Good"),(60,75,"Acceptable"),(40,60,"Poor"),(0,40,"Very Poor")]

def validate():
    assert abs(sum(GROUP_WEIGHTS.values()) - 1.0) < 0.001, "GROUP_WEIGHTS must sum to 1"
    assert abs(sum(FACE_WEIGHTS.values())  - 1.0) < 0.001, "FACE_WEIGHTS must sum to 1"
    assert abs(sum(BODY_WEIGHTS.values())  - 1.0) < 0.001, "BODY_WEIGHTS must sum to 1"
