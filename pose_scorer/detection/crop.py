import cv2
import numpy as np
# mediapipe imported lazily inside map_landmarks_to_global to avoid import errors

def make_person_crop(frame: np.ndarray, person_bbox: list[int], config: dict) -> tuple[np.ndarray, tuple[int, int]]:
    x1, y1, x2, y2 = person_bbox
    h, w = frame.shape[:2]
    bw = x2 - x1
    bh = y2 - y1

    # Asymmetric: more padding on top where head sits
    pad_top    = int(bh * config['DETECTION']['person_pad_top'])
    pad_bottom = int(bh * config['DETECTION']['person_pad_bottom'])
    pad_side   = int(bw * config['DETECTION']['person_pad_side'])

    nx1 = max(0, x1 - pad_side)
    ny1 = max(0, y1 - pad_top)
    nx2 = min(w, x2 + pad_side)
    ny2 = min(h, y2 + pad_bottom)

    return frame[ny1:ny2, nx1:nx2].copy(), (nx1, ny1)

def make_face_crop(frame: np.ndarray, face_bbox: list[int], config: dict) -> tuple[np.ndarray, dict]:
    """
    STAGE 5: FACE CROP
    Symmetric padding and upscale to minimum 512px.
    """
    pad = config['DETECTION']['face_pad']
    min_size = config['DETECTION']['face_min_size']
    
    x1, y1, x2, y2 = face_bbox
    h, w = frame.shape[:2]
    bw, bh = x2 - x1, y2 - y1

    pad_x = int(bw * pad)
    pad_y = int(bh * pad)
    nx1 = max(0, x1 - pad_x)
    ny1 = max(0, y1 - pad_y)
    nx2 = min(w, x2 + pad_x)
    ny2 = min(h, y2 + pad_y)

    crop = frame[ny1:ny2, nx1:nx2].copy()
    ch, cw = crop.shape[:2]

    scale = max(1.0, min_size / max(ch, cw))
    if scale > 1.0:
        crop = cv2.resize(crop, (int(cw * scale), int(ch * scale)),
                          interpolation=cv2.INTER_LANCZOS4)

    scaled_h, scaled_w = crop.shape[:2]
    crop_meta = {
        "offset_x": nx1,   "offset_y": ny1,
        "crop_w":   cw,    "crop_h":   ch,       # pre-upscale
        "scale":    scale,
        "scaled_w": scaled_w, "scaled_h": scaled_h,  # what MediaPipe sees
        "original_w": w,   "original_h": h,
    }
    return crop, crop_meta

def map_landmarks_to_global(landmarks: list, crop_meta: dict) -> list:
    from mediapipe.framework.formats import landmark_pb2
    mapped = []
    for lm in landmarks:
        abs_x = (lm.x * crop_meta['scaled_w']) / crop_meta['scale'] + crop_meta['offset_x']
        abs_y = (lm.y * crop_meta['scaled_h']) / crop_meta['scale'] + crop_meta['offset_y']

        new_lm = landmark_pb2.NormalizedLandmark()
        new_lm.x          = abs_x / crop_meta['original_w']
        new_lm.y          = abs_y / crop_meta['original_h']
        new_lm.z          = lm.z           # do not scale
        new_lm.visibility = lm.visibility
        new_lm.presence   = lm.presence    # preserve exactly
        mapped.append(new_lm)
    return mapped
