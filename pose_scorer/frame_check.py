def check_face(face_bbox: list[int], image_w: int, image_h: int, config: dict) -> dict:
    """
    STAGE 4: FRAME CHECK
    Check face centre horizontal offset from frame centre
    Check edge proximity (left/right/top within edge_threshold)
    Single person mode: any failure = HARD FAIL 4.
    """
    x1, y1, x2, y2 = face_bbox
    frame_cfg = config['FRAME']
    
    edge_threshold = frame_cfg['edge_threshold']
    max_horizontal_offset = frame_cfg['max_horizontal_offset']
    
    cx = (x1 + x2) / 2 / image_w
    cy = (y1 + y2) / 2 / image_h
    
    violations = []
    
    if frame_cfg['reject_top'] and (y1 / image_h) < edge_threshold:
        violations.append("top_edge")
    if frame_cfg['reject_left'] and (x1 / image_w) < edge_threshold:
        violations.append("left_edge")
    if frame_cfg['reject_right'] and (x2 / image_w) > (1 - edge_threshold):
        violations.append("right_edge")
        
    # Spec says BOTTOM flag only, never fail
    bottom_flag = False
    if (y2 / image_h) > (1 - edge_threshold):
        bottom_flag = True
        
    offset_x = abs(cx - 0.5)
    if offset_x > max_horizontal_offset:
        violations.append(f"horizontal_offset_{offset_x:.2f}")
        
    status = "FAIL" if len(violations) > 0 else "PASS"
    offset_score = max(0.0, 100.0 * (1.0 - offset_x / max_horizontal_offset)) if status == "PASS" else 0.0
    
    return {
        "status": status,
        "offset_score": round(offset_score, 2),
        "face_centre": {"x": round(cx, 2), "y": round(cy, 2)},
        "offset_from_centre": {"x": round(offset_x, 2), "y": None},
        "violations": violations,
        "bottom_flag": bottom_flag
    }
