import numpy as np

def detect_person(bgr_image: np.ndarray, person_detector, config: dict) -> tuple[list[int] | None, dict]:
    """
    STAGE 2: PERSON DETECTION
    YOLO26n-person on full image.
    0 persons -> HARD FAIL 1
    2+ persons -> HARD FAIL 2
    """
    conf = config['DETECTION']['person_conf']
    results = person_detector(bgr_image, verbose=False, conf=conf, classes=[0])
    
    boxes = results[0].boxes
    if len(boxes) == 0:
        return None, {"status": "NO_PERSON_DETECTED", "skip_reason": "0 persons detected"}
    elif len(boxes) > 1:
        return None, {"status": "MULTIPLE_PERSONS_DETECTED", "skip_reason": f"{len(boxes)} persons detected"}
        
    box = boxes[0].xyxy[0].cpu().numpy().astype(int).tolist()
    person_conf = float(boxes[0].conf[0].cpu().numpy())
    return box, {"status": "SCORED", "person_conf": round(person_conf, 2)}

def detect_face(bgr_image: np.ndarray, person_bbox: list[int], face_detector, config: dict) -> tuple[list[int] | None, dict]:
    """
    STAGE 3: FACE DETECTION
    YOLO26n-face on padded person crop.
    0 faces -> HARD FAIL 3
    """
    from pose_scorer.detection.crop import make_person_crop
    
    person_crop, offset = make_person_crop(bgr_image, person_bbox, config)
    conf = config['DETECTION']['face_conf']
    results = face_detector(person_crop, verbose=False, conf=conf)
    
    boxes = results[0].boxes
    if len(boxes) == 0:
        return None, {"status": "NO_FACE_DETECTED", "skip_reason": "0 faces detected in person crop"}
        
    best_idx = np.argmax(boxes.conf.cpu().numpy())
    local_box = boxes[best_idx].xyxy[0].cpu().numpy().astype(int)
    
    # Map back to full frame
    global_box = [
        int(local_box[0] + offset[0]),   # x1
        int(local_box[1] + offset[1]),   # y1
        int(local_box[2] + offset[0]),   # x2
        int(local_box[3] + offset[1])    # y2
    ]
    face_conf = float(boxes[best_idx].conf[0].cpu().numpy())
    
    return global_box, {"status": "SCORED", "face_conf": round(face_conf, 2)}
