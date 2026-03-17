import json
import csv
import shutil
import os
from typing import List, Dict

def build_result(
    image_name: str,
    status: str,
    reject_reason: str | None,
    final_score: float | None,
    score_band: str,
    preflight: dict,
    detection: dict,
    frame_check: dict,
    face_group: dict,
    body_group: dict
) -> dict:
    
    return {
        "image": image_name,
        "status": status,
        "reject_reason": reject_reason,
        "final_score": final_score,
        "score_band": score_band,
        "rank": None, 
        "preflight": preflight,
        "detection": detection,
        "frame_check": frame_check,
        "face_group": face_group,
        "body_group": body_group
    }

def output_reports(results: List[Dict], output_dir: str, source_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    scored_results = [r for r in results if r['final_score'] is not None and r['status'] == 'SCORED']
    scored_results.sort(key=lambda x: x['final_score'], reverse=True)
    
    for i, res in enumerate(scored_results):
        res['rank'] = i + 1
        
    json_path = os.path.join(output_dir, "results.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    csv_path = os.path.join(output_dir, "ranked_summary.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Rank", "Image", "Final Score", "Score Band", "Status", "Reject Reason"])
        
        all_sorted = scored_results + [r for r in results if r['final_score'] is None or r['status'] != 'SCORED']
        for r in all_sorted:
            writer.writerow([
                r.get('rank', '-'),
                r['image'],
                f"{r['final_score']:.2f}" if r['final_score'] is not None else "-",
                r.get('score_band', '-'),
                r['status'],
                r.get('reject_reason', '') or ''
            ])
            
    if scored_results:
        best_image = scored_results[0]['image']
        src_path = os.path.join(source_dir, best_image)
        if os.path.exists(src_path):
            ext = os.path.splitext(best_image)[1]
            dest_path = os.path.join(output_dir, f"best_photo{ext}")
            shutil.copy2(src_path, dest_path)
