import os
import argparse
import base64
import cv2
import numpy as np
from ultralytics import YOLO

from pose_scorer import config as cfg
from pose_scorer.preprocessor import prepare_image
from pose_scorer.detection.yolo_detector import detect_person, detect_face
from pose_scorer.frame_check import check_face
from pose_scorer.detection.crop import make_face_crop
from pose_scorer.body_group import run_body_group
from pose_scorer.face_group import run_face_group
from pose_scorer.aggregator import aggregate
from pose_scorer.reporter import build_result, output_reports

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

def init_detectors():
    face_opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=cfg.MODELS['face_landmarker']),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.4,
        min_face_presence_confidence=0.4,
        min_tracking_confidence=0.4,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
    )
    face_lm = mp_vision.FaceLandmarker.create_from_options(face_opts)

    pose_opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=cfg.MODELS['pose_landmarker']),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.4,
        min_pose_presence_confidence=0.4,
        min_tracking_confidence=0.4,
        output_segmentation_masks=False,
    )
    pose_lm = mp_vision.PoseLandmarker.create_from_options(pose_opts)
    
    return face_lm, pose_lm

def _create_fail_result(image_name, status, reject_reason, preflight={}, detection={}, frame_check={}, face_group={}, body_group={}):
    return build_result(
        image_name=image_name,
        status=status,
        reject_reason=reject_reason,
        final_score=None,
        score_band="",
        preflight=preflight,
        detection=detection,
        frame_check=frame_check,
        face_group=face_group,
        body_group=body_group
    )

def _generate_debug_image(bgr_image, person_bbox, face_bbox, crop_meta, raw_face, raw_pose, result) -> str:
    """
    Return a base64-encoded JPEG annotated with:
      • Person bbox (green)  • Face bbox (orange)
      • Pose skeleton (cyan) • Face mesh contours (yellow/teal)
      • Score / module overlay panel (top-left)
    """
    debug = bgr_image.copy()
    img_h, img_w = debug.shape[:2]

    max_w = 1200
    dscale = min(1.0, max_w / img_w)
    if dscale < 1.0:
        debug = cv2.resize(debug, (int(img_w * dscale), int(img_h * dscale)), interpolation=cv2.INTER_AREA)
    dh, dw = debug.shape[:2]

    def sc(v):        return int(round(v * dscale))
    def sc_pt(x, y): return (max(0, min(dw-1, sc(x))), max(0, min(dh-1, sc(y))))

    # ── Person bbox (green) ──────────────────────────────────────────────────
    if person_bbox:
        x1, y1, x2, y2 = person_bbox
        cv2.rectangle(debug, sc_pt(x1, y1), sc_pt(x2, y2), (40, 220, 80), 2)
        cv2.putText(debug, f"Person {result['detection'].get('person_conf', 0):.2f}",
                    (sc(x1), max(14, sc(y1) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 220, 80), 1, cv2.LINE_AA)

    # ── Face bbox (orange) ───────────────────────────────────────────────────
    if face_bbox:
        x1, y1, x2, y2 = face_bbox
        cv2.rectangle(debug, sc_pt(x1, y1), sc_pt(x2, y2), (30, 140, 255), 2)
        cv2.putText(debug, f"Face {result['detection'].get('face_conf', 0):.2f}",
                    (sc(x1), max(14, sc(y1) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 140, 255), 1, cv2.LINE_AA)

    # ── Pose skeleton ─────────────────────────────────────────────────────────
    if raw_pose and raw_pose.pose_landmarks:
        pose = raw_pose.pose_landmarks[0]
        key_ids = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
        pts = {}
        for i in key_ids:
            lm = pose[i]
            pts[i] = (int(lm.x * dw), int(lm.y * dh))
        connections = [(11,12),(11,13),(13,15),(12,14),(14,16),
                       (11,23),(12,24),(23,24),
                       (23,25),(25,27),(24,26),(26,28)]
        for a, b in connections:
            if a in pts and b in pts:
                cv2.line(debug, pts[a], pts[b], (0, 190, 190), 2)
        for i, p in pts.items():
            cv2.circle(debug, p, 5, (0, 230, 230), -1)
            cv2.circle(debug, p, 5, (0, 80, 80), 1)        # dark ring

    # ── Face mesh contours ────────────────────────────────────────────────────
    if raw_face and raw_face.face_landmarks and crop_meta:
        lms = raw_face.face_landmarks[0]
        sw    = crop_meta['scaled_w']
        sh    = crop_meta['scaled_h']
        sc_f  = crop_meta['scale']
        ox    = crop_meta['offset_x']
        oy    = crop_meta['offset_y']

        def lm_px(lm):
            ax = lm.x * sw / sc_f + ox
            ay = lm.y * sh / sc_f + oy
            return (max(0, min(dw-1, int(ax * dscale))),
                    max(0, min(dh-1, int(ay * dscale))))

        def draw_contour(ids, color, closed=True, thickness=1):
            pts_c = [lm_px(lms[i]) for i in ids if i < len(lms)]
            if len(pts_c) < 2:
                return
            arr = np.array(pts_c, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(debug, [arr], isClosed=closed, color=color, thickness=thickness)

        # Face contour groups (MediaPipe 478-landmark indices)
        FACE_OVAL   = [10,338,297,332,284,251,389,356,454,323,361,288,
                       397,365,379,378,400,377,152,148,176,149,150,136,
                       172,58,132,93,234,127,162,21,54,103,67,109]
        R_EYE       = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
        L_EYE       = [263,249,390,373,374,380,381,382,362,398,384,385,386,387,388,466]
        NOSE        = [168,6,197,195,5,4,1,19,94,2,164,0,11,12,13,14,15,16,17,18]
        LIPS_OUTER  = [61,185,40,39,37,0,267,269,270,409,291,375,321,405,314,17,84,181,91,146]
        LIPS_INNER  = [78,95,88,178,87,14,317,402,318,324,308,415,310,311,312,13,82,81,80,191]
        R_BROW      = [46,53,52,65,55,70,63,105,66,107]
        L_BROW      = [276,283,282,295,285,300,293,334,296,336]

        draw_contour(FACE_OVAL,  (100, 200, 100), closed=True,  thickness=1)
        draw_contour(R_EYE,      (255, 215, 40),  closed=True,  thickness=1)
        draw_contour(L_EYE,      (255, 215, 40),  closed=True,  thickness=1)
        draw_contour(R_BROW,     (200, 180, 60),  closed=False, thickness=1)
        draw_contour(L_BROW,     (200, 180, 60),  closed=False, thickness=1)
        draw_contour(LIPS_OUTER, (255, 140, 80),  closed=True,  thickness=1)
        draw_contour(LIPS_INNER, (220, 100, 60),  closed=True,  thickness=1)
        draw_contour(NOSE,       (180, 180, 80),  closed=False, thickness=1)

        # Key anchor dots
        for i in [1, 33, 133, 263, 362, 61, 291, 0]:
            if i < len(lms):
                cv2.circle(debug, lm_px(lms[i]), 3, (255, 215, 40), -1)

        # Iris rings (only valid coords)
        for grp_ids in [[468, 469, 470, 471], [473, 474, 475, 476, 477]]:
            iris_pts = [lm_px(lms[i]) for i in grp_ids
                        if i < len(lms) and 0 < lms[i].x < 1 and 0 < lms[i].y < 1]
            if len(iris_pts) >= 2:
                arr = np.array(iris_pts, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(debug, [arr], isClosed=True, color=(0, 230, 255), thickness=2)

    # ── Score / module overlay panel ─────────────────────────────────────────
    pw, ph = min(310, dw), min(200, dh)
    region = debug[0:ph, 0:pw].copy()
    dark   = np.full_like(region, (8, 10, 16))
    debug[0:ph, 0:pw] = cv2.addWeighted(dark, 0.78, region, 0.22, 0)

    status = result.get('status', '')
    score  = result.get('final_score')
    band   = result.get('score_band', '')
    ok_col = (40, 220, 80) if status == 'SCORED' else (100, 100, 255)

    cv2.putText(debug, (f"{score:.1f}  {band}" if score is not None else status),
                (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, ok_col, 2, cv2.LINE_AA)

    y = 48
    for k, v in ((result.get('face_group') or {}).get('modules', {})).items():
        low_conf = v.get('confidence', 1.0) < 0.55
        val = 'SKIP' if (v.get('skipped') or low_conf) else f"{v['score']:.0f}"
        col = (100, 100, 100) if (v.get('skipped') or low_conf) else (175, 210, 255)
        cv2.putText(debug, f"{k.replace('_',' ')[:13]}: {val}",
                    (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, col, 1, cv2.LINE_AA)
        y += 16

    y = 48
    for k, v in ((result.get('body_group') or {}).get('modules', {})).items():
        low_conf = v.get('confidence', 1.0) < 0.55
        val = 'SKIP' if (v.get('skipped') or low_conf) else f"{v['score']:.0f}"
        col = (100, 100, 100) if (v.get('skipped') or low_conf) else (175, 255, 175)
        cv2.putText(debug, f"{k.replace('_',' ')[:13]}: {val}",
                    (158, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, col, 1, cv2.LINE_AA)
        y += 16

    _, buf = cv2.imencode('.jpg', debug, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return 'data:image/jpeg;base64,' + base64.b64encode(buf).decode()

def _print_debug(image_name: str, result: dict) -> None:
    status = result.get('status', '')
    score  = result.get('final_score')
    print(f"\n{'━'*60}")
    print(f"IMAGE: {image_name}")
    if status != 'SCORED':
        print(f"STATUS: {status}")
        print(f"REASON: {result.get('reject_reason', '')}")
        print('━'*60)
        return

    print(f"STATUS: SCORED  |  FINAL: {score}  |  BAND: {result.get('score_band')}")
    print('━'*60)

    det = result.get('detection', {})
    pf  = result.get('preflight', {})
    fc  = result.get('frame_check', {})
    print(f"[CAPTURE]  blur:{pf.get('blur_score')}  res:{pf.get('resolution')}")
    print(f"[DETECT]   person_conf:{det.get('person_conf')}  face_conf:{det.get('face_conf')}")
    print(f"[FRAME]    offset:{fc.get('offset_from_centre',{}).get('x')}  score:{fc.get('offset_score')}  {fc.get('status')}")

    fg = result.get('face_group', {})
    print(f"\n[FACE GROUP — score:{fg.get('group_score')}]")
    for name, mod in fg.get('modules', {}).items():
        if mod.get('skipped'):
            print(f"  {name:<18} SKIP  ({mod.get('skip_reason','')})")
        else:
            print(f"  {name:<18} {mod['score']:>5.1f}  {mod.get('detail','')}")

    bg = result.get('body_group', {})
    print(f"\n[BODY GROUP — score:{bg.get('group_score')}]")
    for name, mod in bg.get('modules', {}).items():
        if mod.get('skipped'):
            print(f"  {name:<18} SKIP  ({mod.get('skip_reason','')})")
        else:
            print(f"  {name:<18} {mod['score']:>5.1f}  {mod.get('detail','')}")
    print('━'*60)


def hand_on_face(body_raw, face_bbox, image_w, image_h, config=None):
    """
    Check if any fingertip landmark is INSIDE the face bounding box.
    Checks all three fingertip pairs tracked by MediaPipe Pose:
      17/18 = left/right pinky tip
      19/20 = left/right index tip
      21/22 = left/right thumb tip
    Triggers when ANY visible fingertip is physically on the face.
    """
    if body_raw is None or not body_raw.pose_landmarks:
        return False
    lms = body_raw.pose_landmarks[0]
    fx1 = face_bbox[0] / image_w
    fy1 = face_bbox[1] / image_h
    fx2 = face_bbox[2] / image_w
    fy2 = face_bbox[3] / image_h
    # All fingertip landmarks: pinky(17,18), index(19,20), thumb(21,22)
    FINGERTIP_IDS = [17, 18, 19, 20, 21, 22]
    if config and config.get('DEBUG', False):
        for idx in FINGERTIP_IDS:
            lm = lms[idx]
            in_box = fx1 <= lm.x <= fx2 and fy1 <= lm.y <= fy2
            print(f"[HAND_ON_FACE] lm{idx} x={lm.x:.3f} y={lm.y:.3f} "
                  f"vis={lm.visibility:.2f} in_bbox={in_box}")
    for idx in FINGERTIP_IDS:   # all fingertips — no wrist/palm
        lm = lms[idx]
        if lm.visibility < 0.3:
            continue
        if fx1 <= lm.x <= fx2 and fy1 <= lm.y <= fy2:   # no margin
            return True
    return False


def hand_near_face(body_raw, face_bbox, image_w, image_h, config=None):
    """
    Return a penalty factor (0.40–1.0) based on wrist proximity to face.
    Unlike hand_on_face (fingertip-inside-bbox hard gate), this catches
    hands NEAR the face (e.g. resting on top of the head) that don't
    literally overlap the face bounding box.
    """
    if body_raw is None or not body_raw.pose_landmarks:
        return 1.0  # no penalty
    lms = body_raw.pose_landmarks[0]
    fx1 = face_bbox[0] / image_w
    fy1 = face_bbox[1] / image_h
    fx2 = face_bbox[2] / image_w
    fy2 = face_bbox[3] / image_h
    face_h = fy2 - fy1
    margin = face_h * 1.5  # check 1.5x face heights around the box

    face_cx = (fx1 + fx2) / 2
    face_cy = (fy1 + fy2) / 2

    WRIST_IDS = [15, 16]  # left wrist, right wrist
    worst_factor = 1.0
    for idx in WRIST_IDS:
        lm = lms[idx]
        if lm.visibility < 0.3:
            continue
        # Check if wrist is within expanded face region
        in_expanded = (fx1 - margin <= lm.x <= fx2 + margin and
                       fy1 - margin <= lm.y <= fy2 + margin)
        if in_expanded:
            dist = ((lm.x - face_cx)**2 + (lm.y - face_cy)**2)**0.5
            max_dist = margin
            factor = max(0.40, min(1.0, dist / max_dist))
            worst_factor = min(worst_factor, factor)
            if config and config.get('DEBUG', False):
                print(f"[HAND_NEAR_FACE] lm{idx} x={lm.x:.3f} y={lm.y:.3f} "
                      f"dist={dist:.3f} factor={factor:.2f}")
    return worst_factor


def score_image(image_path: str, person_detector, face_detector, face_landmarker, pose_landmarker, debug_print=False, config_dict=None):
    image_name = os.path.basename(image_path)
    
    # 1
    # cfg is a module, not a dict. We should properly handle config format
    if config_dict is None:
        config_dict = {
            'PREFLIGHT': cfg.PREFLIGHT,
            'DETECTION': cfg.DETECTION,
            'FRAME': cfg.FRAME,
            'CONFIDENCE': cfg.CONFIDENCE,
            'CAMERA': cfg.CAMERA,
            'FACE': cfg.FACE,
            'BODY': cfg.BODY,
            'FACE_WEIGHTS': cfg.FACE_WEIGHTS,
            'BODY_WEIGHTS': cfg.BODY_WEIGHTS,
            'GROUP_WEIGHTS': cfg.GROUP_WEIGHTS,
            'SCORE_BANDS': cfg.SCORE_BANDS,
            'DEBUG': cfg.DEBUG
        }

    bgr_image, preflight = prepare_image(image_path, config_dict)
    if preflight['status'] != "SCORED":
        return _create_fail_result(image_name, preflight['status'], preflight['skip_reason'], preflight=preflight)
        
    h, w = bgr_image.shape[:2]
    
    # 2
    person_bbox, person_det_res = detect_person(bgr_image, person_detector, config_dict)
    if person_bbox is None:
        return _create_fail_result(image_name, person_det_res['status'], person_det_res['skip_reason'], preflight=preflight, detection=person_det_res)

    # 2b — person size gate: reject if person is too far from camera
    _px1, _py1, _px2, _py2 = person_bbox
    person_height_ratio = (_py2 - _py1) / h
    min_ratio = config_dict['DETECTION'].get('min_person_height_ratio', 0.35)
    if person_height_ratio < min_ratio:
        return _create_fail_result(
            image_name, "PERSON_TOO_SMALL",
            f"Person height {person_height_ratio:.2f} < minimum {min_ratio} (person too far from camera)",
            preflight, person_det_res
        )

    # 3
    face_bbox, face_det_res = detect_face(bgr_image, person_bbox, face_detector, config_dict)
    detection_res = {"person_bbox": person_bbox, "person_conf": person_det_res['person_conf']}
    
    if face_bbox is None:
        return _create_fail_result(image_name, face_det_res['status'], face_det_res['skip_reason'], preflight, detection_res)
    
    detection_res["face_bbox"] = face_bbox
    detection_res["face_conf"] = face_det_res['face_conf']
    
    # 4
    frame_res = check_face(face_bbox, w, h, config_dict)
    if frame_res['status'] == "FAIL":
        return _create_fail_result(image_name, "FACE_FRAME_REJECTED", "Frame bounds check failed", preflight, detection_res, frame_res)
        
    # 5
    face_crop_bgr, crop_meta = make_face_crop(bgr_image, face_bbox, config_dict)
    
    # 6
    body_res = run_body_group(bgr_image, pose_landmarker, config_dict)

    # 6b — orientation gate: reject side-on persons before face scoring
    orientation_mod = body_res.get('modules', {}).get('body_orientation', {})
    if not orientation_mod.get('skipped', True):
        min_orient = config_dict.get('BODY', {}).get('min_orientation_score', 40)
        if orientation_mod['score'] < min_orient:
            return _create_fail_result(
                image_name, "BODY_ORIENTATION_REJECTED",
                f"Person not facing camera (orientation score {orientation_mod['score']}, "
                f"detail: {orientation_mod.get('detail', '')})",
                preflight, detection_res, frame_res
            )

    # 6c — hand-on-face occlusion gate
    _body_raw_for_check = body_res.get('_raw')
    if face_bbox and hand_on_face(_body_raw_for_check, face_bbox, w, h, config=config_dict):
        return _create_fail_result(
            image_name, "FACE_OCCLUDED",
            "Hand on face detected",
            preflight, detection_res, frame_res
        )

    # 6d — hand-near-face proximity penalty (graduated, not hard reject)
    proximity_factor = hand_near_face(_body_raw_for_check, face_bbox, w, h, config=config_dict)
    if proximity_factor < 1.0 and body_res.get('group_score') is not None:
        body_res['group_score'] = round(body_res['group_score'] * proximity_factor, 1)

    # 7
    face_res = run_face_group(face_crop_bgr, crop_meta, face_landmarker, config_dict)

    # 7b — face quality gate: eyes closed + gaze skipped
    if face_res.get('rejected'):
        return _create_fail_result(
            image_name, "FACE_OCCLUDED",
            face_res['reject_reason'],
            preflight, detection_res, frame_res
        )

    # Extract raw landmark results for debug image (non-serialisable — strip before output)
    raw_pose = body_res.pop('_raw', None)
    raw_face = face_res.pop('_raw', None)

    # 8
    final_score, score_band = aggregate(frame_res, face_res, body_res, config_dict)
    
    # ── Reject if face group produced no score ────────────────────────────
    if face_res.get('group_score') is None:
        return _create_fail_result(
            image_name, "FACE_SCORE_FAILED",
            "Face group returned no score — all modules failed or skipped",
            preflight, detection_res, frame_res
        )

    if final_score is None:
        return _create_fail_result(
            image_name, "SCORING_FAILED",
            "Aggregation returned no score",
            preflight, detection_res, frame_res
        )
    
    # 9
    result = build_result(image_name, "SCORED", None, final_score, score_band, preflight, detection_res, frame_res, face_res, body_res)
    result['debug_image'] = _generate_debug_image(
        bgr_image, person_bbox, face_bbox, crop_meta,
        raw_face, raw_pose, result
    )
    if debug_print:
        _print_debug(image_name, result)
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    
    cfg.validate()
    print("Loading models...")
    person_detector = YOLO(cfg.MODELS['person_detector'])
    face_detector = YOLO(cfg.MODELS['face_detector'])
    face_lm, pose_lm = init_detectors()
    
    results = []
    
    if os.path.isfile(args.input):
        paths = [args.input]
        src_dir = os.path.dirname(args.input)
    else:
        paths = [os.path.join(args.input, f) for f in os.listdir(args.input) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        src_dir = args.input
        
    for p in paths:
        if not args.debug:
            print(f"Scoring {os.path.basename(p)}...")
        res = score_image(p, person_detector, face_detector, face_lm, pose_lm, debug_print=args.debug)
        res.pop('debug_image', None)  # strip large base64 blob from CLI JSON output
        results.append(res)
        
    output_reports(results, args.output, src_dir)
    print(f"Done. Wrote results to {args.output}")

if __name__ == "__main__":
    main()
