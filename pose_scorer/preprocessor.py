import cv2
import numpy as np
from PIL import Image, ExifTags

_ORIENTATION_TAG = next(k for k, v in ExifTags.TAGS.items() if v == "Orientation")
_ROTATIONS = {3: 180, 6: 270, 8: 90}

def load_image_correctly(path: str) -> np.ndarray:
    """Load image respecting EXIF orientation. Returns BGR numpy array."""
    pil = Image.open(path)
    try:
        exif = pil._getexif()
        if exif:
            orientation = exif.get(_ORIENTATION_TAG)
            if orientation in _ROTATIONS:
                pil = pil.rotate(_ROTATIONS[orientation], expand=True)
    except Exception:
        pass  # no EXIF or unreadable — use raw pixels
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def prepare_image(path: str, config: dict) -> tuple[np.ndarray, dict]:
    """
    STAGE 1: LOAD IMAGE
    PIL load with EXIF rotation correction
    validate min resolution
    blur check via Laplacian variance
    """
    try:
        bgr_image = load_image_correctly(path)
    except Exception as e:
        return None, {"status": "PREFLIGHT_FAIL", "skip_reason": f"Load error: {e}"}

    h, w = bgr_image.shape[:2]
    min_w, min_h = config['PREFLIGHT']['min_resolution']
    if w < min_w or h < min_h:
        return bgr_image, {
            "status": "PREFLIGHT_FAIL", 
            "skip_reason": f"Resolution {w}x{h} below minimum {min_w}x{min_h}"
        }

    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_threshold = config['PREFLIGHT']['blur_threshold']

    if blur_score < blur_threshold:
        return bgr_image, {
            "status": "PREFLIGHT_FAIL",
            "skip_reason": f"Blur score {blur_score:.1f} < {blur_threshold}",
            "blur_score": round(blur_score, 1),
            "resolution": [w, h]
        }

    return bgr_image, {
        "status": "SCORED",  # Meaning preflight passed
        "blur_score": round(blur_score, 1),
        "resolution": [w, h]
    }
