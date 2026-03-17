import numpy as np

def score_body_orientation(landmarks, world_landmarks, config) -> dict:
    """
    Body Orientation using chest normal Z vector.
    Prefers pose_world_landmarks for Z depth.
    11 = left_shoulder, 12 = right_shoulder
    23 = left_hip, 24 = right_hip
    """
    # TWO SEPARATE SOURCES — never mix them
    lms_3d   = world_landmarks if world_landmarks else landmarks  # geometry only
    lms_conf = landmarks   # always normalised — only source with visibility+presence

    if not lms_3d:
        return {"score": 0.0, "confidence": 0.0, "skipped": True,
                "detail": "", "skip_reason": "No landmarks"}

    # Visibility from NORMALISED landmarks only
    for idx in [11, 12, 23, 24]:
        if lms_conf[idx].visibility < 0.5:
            return {"score": 0.0, "confidence": 0.0, "skipped": True,
                    "detail": "", "skip_reason": f"Landmark {idx} not visible"}

    # Confidence from NORMALISED landmarks using visibility (not presence)
    conf = min(lms_conf[idx].visibility for idx in [11, 12, 23, 24])
    min_conf = config['CONFIDENCE']['min_to_score']
    if conf < min_conf:
        return {"score": 0.0, "confidence": conf, "skipped": True,
                "detail": "", "skip_reason": f"Confidence {conf:.2f} < {min_conf}"}

    # Geometry from lms_3d (world or normalised fallback)
    p11 = np.array([lms_3d[11].x, lms_3d[11].y, lms_3d[11].z])
    p12 = np.array([lms_3d[12].x, lms_3d[12].y, lms_3d[12].z])
    p23 = np.array([lms_3d[23].x, lms_3d[23].y, lms_3d[23].z])
    p24 = np.array([lms_3d[24].x, lms_3d[24].y, lms_3d[24].z])

    v_shoulder = p11 - p12 
    mid_shoulder = (p11 + p12) / 2.0
    mid_hip = (p23 + p24) / 2.0
    v_spine = mid_shoulder - mid_hip 

    normal = np.cross(v_spine, v_shoulder)  # correct winding: normal points toward camera
    norm_len = np.linalg.norm(normal)
    if norm_len < 1e-9:
        return {"score": 0.0, "confidence": 0.0, "skipped": True, "detail": "", "skip_reason": "Degenerate chest normal"}
        
    normal = normal / norm_len
    nz = normal[2]


    # Map to score
    if nz > 0.85:
        score = 100.0
    elif nz >= 0.70:
        score = 85.0
    elif nz >= 0.50:
        score = 60.0
    elif nz >= 0.20:
        score = 30.0
    elif nz >= 0.00:
        score = 10.0
    else:
        score = 5.0

    source = "world" if world_landmarks else "normalised"
    return {
        "score": score,
        "confidence": round(conf, 3),
        "detail": f"normal_z:{nz:.2f} src:{source}",
        "skipped": False,
        "skip_reason": ""
    }
