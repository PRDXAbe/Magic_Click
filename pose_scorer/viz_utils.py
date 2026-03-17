import cv2
import numpy as np
import os
import mediapipe as mp

def draw_pose(img, landmarks):
    if not landmarks: return img
    h, w = img.shape[:2]
    pts = {}
    
    # Render ONLY points active in scoring functions:
    # 0 (nose), 11/12 (shoulders), 15/16 (wrists), 23/24 (hips), 27/28 (ankles)
    indices = [0, 11, 12, 15, 16, 23, 24, 27, 28]
    for i in indices:
        lm = landmarks[i]
        pts[i] = (int(lm.x * w), int(lm.y * h))
        cv2.circle(img, pts[i], 6, (0, 255, 255), -1)  # Yellow dots
        
    if all(i in pts for i in [11, 12, 23, 24]):
        # Shoulder and hip widths
        cv2.line(img, pts[11], pts[12], (255, 0, 0), 2)
        cv2.line(img, pts[23], pts[24], (255, 0, 0), 2)
        
        # Spine representation
        mid_s = ((pts[11][0] + pts[12][0]) // 2, (pts[11][1] + pts[12][1]) // 2)
        mid_h = ((pts[23][0] + pts[24][0]) // 2, (pts[23][1] + pts[24][1]) // 2)
        cv2.line(img, mid_s, mid_h, (0, 165, 255), 3)  # Orange spine
        
        # Head alignment
        if 0 in pts:
            cv2.line(img, pts[0], mid_s, (0, 165, 255), 2)
            
    # Hands vs Hips check references
    if 15 in pts and 23 in pts:
        cv2.line(img, pts[15], pts[23], (255, 0, 255), 2) # Purple hand ref
    if 16 in pts and 24 in pts:
        cv2.line(img, pts[16], pts[24], (255, 0, 255), 2) 
        
    # Ankle stance check references
    if 27 in pts and 28 in pts:
        cv2.line(img, pts[27], pts[28], (0, 255, 0), 3) # Green stance width
        
    return img

def draw_face(img, landmarks, config=None):
    if not landmarks: return img
    h, w = img.shape[:2]
    
    # Render ONLY points active in scoring modules:
    face_idx = [
        1, 152, 263, 33, 287, 57,      # head pose PnP
        33, 133, 159, 145,             # right eye bounds
        263, 362, 386, 374,            # left eye bounds
        78, 308, 13, 14                # smile bounds
    ]
    face_idx = list(set(face_idx))
    iris_idx = list(range(468, 472)) + list(range(473, 478))
    
    for i in face_idx:
        lm = landmarks[i]
        cv2.circle(img, (int(lm.x * w), int(lm.y * h)), 4, (0, 255, 0), -1) # Green dots
        
    for i in iris_idx:
        if i < len(landmarks):
            lm = landmarks[i]
            cv2.circle(img, (int(lm.x * w), int(lm.y * h)), 3, (0, 255, 255), -1) # Yellow iris
            
    def line(ids, color, thick=2):
        if all(i < len(landmarks) for i in ids):
            for i in range(len(ids) - 1):
                p1 = (int(landmarks[ids[i]].x * w), int(landmarks[ids[i]].y * h))
                p2 = (int(landmarks[ids[i+1]].x * w), int(landmarks[ids[i+1]].y * h))
                cv2.line(img, p1, p2, color, thick)
    
    # Eye crosshairs
    line([33, 133], (255, 0, 0))
    line([159, 145], (255, 0, 0))
    line([263, 362], (255, 0, 0))
    line([386, 374], (255, 0, 0))
    # Mouth crosshairs
    line([78, 308], (0, 165, 255))
    line([13, 14], (0, 165, 255))

    # Draw head pose axes
    if config is not None:
        try:
            MODEL_3D_POINTS = np.array([
                (  0.0,    0.0,   0.0),   # lm[1]   nose tip
                (  0.0, -330.0, -65.0),   # lm[152] chin
                (-225.0,  170.0,-135.0),  # lm[263] left eye outer corner
                ( 225.0,  170.0,-135.0),  # lm[33]  right eye outer corner
                (-150.0, -150.0,-125.0),  # lm[287] left mouth corner
                ( 150.0, -150.0,-125.0),  # lm[57]  right mouth corner
            ], dtype=np.float64)
            
            pts2d = np.array([
                [landmarks[1].x * w, landmarks[1].y * h],
                [landmarks[152].x * w, landmarks[152].y * h],
                [landmarks[263].x * w, landmarks[263].y * h],
                [landmarks[33].x * w, landmarks[33].y * h],
                [landmarks[287].x * w, landmarks[287].y * h],
                [landmarks[57].x * w, landmarks[57].y * h]
            ], dtype=np.float64)
            
            fov_deg = config.get('CAMERA', {}).get('fov_degrees', 75.0)
            fov_rad = np.radians(fov_deg)
            focal = (w / 2.0) / np.tan(fov_rad / 2.0)
            cam = np.array([
                [focal,     0, w / 2.0],
                [    0, focal, h / 2.0],
                [    0,     0,     1.0],
            ], dtype=np.float64)
            dist = np.zeros((4, 1))

            ok, rvec, tvec = cv2.solvePnP(MODEL_3D_POINTS, pts2d, cam, dist, flags=cv2.SOLVEPNP_ITERATIVE)
            if ok:
                cv2.drawFrameAxes(img, cam, dist, rvec, tvec, 200, 3)
        except Exception:
            pass

    return img

def create_pipeline_viz(bgr_image, result, face_landmarker, pose_landmarker, config):
    """
    Creates a crisp 4-panel visualization with annotations.
    """
    fname = result['image']
    full_h, full_w = bgr_image.shape[:2]
    
    # ── PANEL 1: Person Detection ─────────────────────────────────────────────
    p1 = bgr_image.copy()
    det = result.get('detection', {})
    if 'person_bbox' in det:
        b = det['person_bbox']
        cv2.rectangle(p1, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
        cv2.putText(p1, f"Person {det.get('person_conf', 0):.2f}", (b[0], b[1]-10), 0, 0.7, (0, 255, 0), 2)

    # ── PANEL 2: Face Detection on Person Crop ──────────────────────────────
    from pose_scorer.detection.crop import make_person_crop
    p2_img = None
    if 'person_bbox' in det:
        p_crop, p_off = make_person_crop(bgr_image, det['person_bbox'], config)
        p2_img = p_crop.copy()
        if 'face_bbox' in det:
            fb = det['face_bbox']
            lx1, ly1 = fb[0] - p_off[0], fb[1] - p_off[1]
            lx2, ly2 = fb[2] - p_off[0], fb[3] - p_off[1]
            cv2.rectangle(p2_img, (lx1, ly1), (lx2, ly2), (255, 0, 0), 2)
            cv2.putText(p2_img, f"Face {det.get('face_conf', 0):.2f}", (lx1, ly1-5), 0, 0.5, (255, 0, 0), 1)
    
    if p2_img is None:
        p2_img = np.zeros((400, 300, 3), dtype=np.uint8)
        cv2.putText(p2_img, "No Person", (50, 200), 0, 1, (0, 0, 255), 2)

    # ── PANEL 3: Face Landmarks on Face Crop ────────────────────────────────
    from pose_scorer.detection.crop import make_face_crop
    p3_img = None
    if 'face_bbox' in det:
        f_crop, _ = make_face_crop(bgr_image, det['face_bbox'], config)
        p3_img = f_crop.copy()
        rgb_f = cv2.cvtColor(p3_img, cv2.COLOR_BGR2RGB)
        mp_f = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_f)
        fl_res = face_landmarker.detect(mp_f)
        if fl_res.face_landmarks:
            draw_face(p3_img, fl_res.face_landmarks[0], config)
    
    if p3_img is None:
        p3_img = np.zeros((512, 512, 3), dtype=np.uint8)
        cv2.putText(p3_img, "No Face", (100, 250), 0, 1, (0, 0, 255), 2)

    # ── PANEL 4: Body Pose on Full ──────────────────────────────────────────
    p4 = bgr_image.copy()
    rgb_b = cv2.cvtColor(p4, cv2.COLOR_BGR2RGB)
    mp_b = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_b)
    pl_res = pose_landmarker.detect(mp_b)
    if pl_res.pose_landmarks:
        draw_pose(p4, pl_res.pose_landmarks[0])

    # ── ASSEMBLY ─────────────────────────────────────────────────────────────
    # Standardize height
    target_h = 600
    def rsz(im):
        h, w = im.shape[:2]
        return cv2.resize(im, (int(target_h * w / h), target_h))

    p1_r, p2_r, p3_r, p4_r = rsz(p1), rsz(p2_img), rsz(p3_img), rsz(p4)
    row = np.hstack([p1_r, p2_r, p3_r, p4_r])

    # Header Overlay
    header_h = 240
    header = np.zeros((header_h, row.shape[1], 3), dtype=np.uint8)
    header[:] = (20, 23, 32) # Dark background
    
    # Text colors
    white = (240, 240, 240)
    accent = (248, 126, 91) # custom accent BGR
    green = (142, 207, 62)
    red = (113, 113, 248)
    
    # Column 1
    cv2.putText(header, f"FILE: {fname}", (30, 50), 0, 0.9, white, 2)
    status_col = green if result['status'] == 'SCORED' else red
    cv2.putText(header, f"STATUS: {result['status']}", (30, 100), 0, 0.9, status_col, 2)
    cv2.putText(header, f"FINAL SCORE: {result.get('final_score') or '—'}", (30, 150), 0, 1.4, accent, 3)
    if result.get('score_band'):
        cv2.putText(header, f"BAND: {result['score_band']}", (30, 200), 0, 0.9, accent, 2)

    # Column 2: Face Group breakdown
    fx = 350
    if result.get('face_group'):
        fg = result['face_group']
        cv2.putText(header, f"FACE GROUP: {fg.get('group_score', '—')}", (fx, 50), 0, 0.7, (255, 200, 200), 2)
        for i, (m_name, m_res) in enumerate(fg['modules'].items()):
            val = f"{m_res['score']:.1f}" if not m_res['skipped'] else "SKIP"
            det = m_res.get('detail', m_res.get('skip_reason', ''))
            c = (180,180,180) if m_res['skipped'] else (255,255,255)
            cv2.putText(header, f"{m_name[:14]:<14}: {val:>5} | {det}", (fx + 20, 85 + i*30), 0, 0.5, c, 1)

    # Column 3: Body Group breakdown
    bx = 1000
    if result.get('body_group'):
        bg = result['body_group']
        cv2.putText(header, f"BODY GROUP: {bg.get('group_score', '—')}", (bx, 50), 0, 0.7, (200, 255, 200), 2)
        for i, (m_name, m_res) in enumerate(bg['modules'].items()):
            val = f"{m_res['score']:.1f}" if not m_res['skipped'] else "SKIP"
            det = m_res.get('detail', m_res.get('skip_reason', ''))
            c = (180,180,180) if m_res['skipped'] else (255,255,255)
            cv2.putText(header, f"{m_name[:17]:<17}: {val:>5} | {det}", (bx + 20, 85 + i*25), 0, 0.5, c, 1)

    # Final Canvas
    canvas = np.vstack([header, row])
    return canvas
