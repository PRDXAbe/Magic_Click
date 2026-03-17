import os
import argparse
import cv2
import sys
import json
from tqdm import tqdm
from ultralytics import YOLO

sys.path.insert(0, os.getcwd())

from pose_scorer import config as cfg
from pose_scorer.scorer import init_detectors, score_image
from pose_scorer.viz_utils import create_pipeline_viz
from pose_scorer.preprocessor import prepare_image

def main():
    parser = argparse.ArgumentParser(description="Terminal Folder Scorer with Visualization")
    parser.add_argument("--input",  required=True, help="Path to folder containing images")
    parser.add_argument("--output", required=True, help="Path to save visualizations")
    parser.add_argument("--debug",  action="store_true", help="Print debug information")
    parser.add_argument("--no-viz",        action="store_true", help="Skip saving viz images (faster, JSON only)")
    parser.add_argument("--no-preflight",   action="store_true", help="Skip blur and resolution preflight checks")
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print(f"Error: {args.input} is not a directory.")
        return

    os.makedirs(args.output, exist_ok=True)

    print("--- Initializing Best Photo Picker Pipeline ---")
    cfg.validate()

    print("Loading models...")
    try:
        person_det = YOLO(cfg.MODELS['person_detector'])
        face_det   = YOLO(cfg.MODELS['face_detector'])
        face_lm, pose_lm = init_detectors()
    except Exception as e:
        print(f"Initialization error: {e}")
        return

    config_dict = {
        'PREFLIGHT':     cfg.PREFLIGHT,
        'DETECTION':     cfg.DETECTION,
        'FRAME':         cfg.FRAME,
        'CONFIDENCE':    cfg.CONFIDENCE,
        'CAMERA':        cfg.CAMERA,
        'FACE':          cfg.FACE,
        'BODY':          cfg.BODY,
        'FACE_WEIGHTS':  cfg.FACE_WEIGHTS,
        'BODY_WEIGHTS':  cfg.BODY_WEIGHTS,
        'GROUP_WEIGHTS': cfg.GROUP_WEIGHTS,
        'SCORE_BANDS':   cfg.SCORE_BANDS,
        'DEBUG':         cfg.DEBUG,
    }

    if args.no_preflight:
        config_dict['PREFLIGHT'] = {'blur_threshold': 0.0, 'min_resolution': (1, 1)}
        print("[INFO] Preflight checks disabled (--no-preflight).")

    valid_exts = ('.jpg', '.jpeg', '.png', '.webp')
    img_files  = [f for f in os.listdir(args.input) if f.lower().endswith(valid_exts)]

    if not img_files:
        print(f"No images found in {args.input}")
        return

    print(f"Processing {len(img_files)} images...")
    print("-" * 95)
    print(f"{'Image Name':<30} | {'Status':<25} | {'Score':<6} | {'Band':<12} | {'Face':<6} | {'Body':<6}")
    print("-" * 95)

    results = []

    for fname in tqdm(img_files, desc="Scoring", unit="img", leave=True):
        img_path = os.path.join(args.input, fname)
        try:
            result = score_image(img_path, person_det, face_det, face_lm, pose_lm,
                                 debug_print=args.debug, config_dict=config_dict)

            # ── Terminal output ────────────────────────────────────────────
            score_str = f"{result['final_score']:.1f}" if result['final_score'] is not None else "—"
            band_str  = result.get('score_band', '—')
            fg        = result.get('face_group') or {}
            bg        = result.get('body_group') or {}
            f_score   = f"{fg['group_score']:.1f}" if fg.get('group_score') is not None else "—"
            b_score   = f"{bg['group_score']:.1f}" if bg.get('group_score') is not None else "—"
            status    = result.get('status', '—')
            reject    = f" ({result['reject_reason']})" if result.get('reject_reason') else ""
            print(f"{fname[:30]:<30} | {(status + reject)[:25]:<25} | {score_str:<6} | {band_str:<12} | {f_score:<6} | {b_score:<6}")

            # ── Viz image ─────────────────────────────────────────────────
            if not args.no_viz:
                fixed_image, _ = prepare_image(img_path, config_dict)
                if fixed_image is not None:
                    viz = create_pipeline_viz(fixed_image, result, face_lm, pose_lm, config_dict)
                    cv2.imwrite(os.path.join(args.output, f"viz_{fname}"), viz)

            # ── Collect for JSON — strip non-serialisable blobs ───────────
            r_clean = {k: v for k, v in result.items()
                       if k not in ('debug_image', 'raw_pose', 'raw_face', 'crop_meta')}
            results.append(r_clean)

        except Exception as e:
            print(f"\nError processing {fname}: {e}")

    print("-" * 95)

    # ── Write JSON report ──────────────────────────────────────────────────
    report_path = os.path.join(args.output, "results.json")
    with open(report_path, 'w') as fp:
        json.dump(results, fp, indent=2)

    scored = [r for r in results if r.get('status') == 'SCORED']
    print(f"\nSummary:")
    print(f"  Total processed : {len(results)}")
    print(f"  Scored          : {len(scored)}")
    print(f"  Rejected        : {len(results) - len(scored)}")
    if scored:
        avg = sum(r['final_score'] for r in scored) / len(scored)
        print(f"  Avg score       : {avg:.1f}")
    print(f"\nJSON report : {report_path}")
    if not args.no_viz:
        print(f"Viz images  : {args.output}")

if __name__ == "__main__":
    main()
