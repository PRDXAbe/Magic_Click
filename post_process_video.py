import os
import sys
import argparse
import subprocess
import shutil

# Import existing functions
from extract_frames import extract_frames
from filter_scored_images import filter_scored_images

def main():
    parser = argparse.ArgumentParser(description="Post-process a recorded video: extract, score, and filter.")
    parser.add_argument("--video", required=True, help="Path to the recorded video .mp4 file")
    args = parser.parse_args()

    video_path = args.video
    if not os.path.exists(video_path):
        print(f"Error: video {video_path} not found")
        return
        
    basename = os.path.splitext(os.path.basename(video_path))[0]
    
    # 1. Extract Frames
    session_raw_dir = os.path.join("captured_shots", "raw", f"{basename}_raw")
    print(f"\n[STEP 1] Extracting frames to {session_raw_dir}...")
    os.makedirs(session_raw_dir, exist_ok=True)
    extract_frames(video_path, session_raw_dir)
    
    # 2. Score Folder
    session_scored_dir = os.path.join("captured_shots", "score", f"{basename}_scored")
    print(f"\n[STEP 2] Scoring frames to {session_scored_dir}...")
    os.makedirs(session_scored_dir, exist_ok=True)
    
    # Call score_folder.py via subprocess to cleanly initialize the heavy models
    cmd = [
        sys.executable, "score_folder.py", 
        "--input", session_raw_dir, 
        "--output", session_scored_dir,
        "--no-viz" # Optional: disable visualization images to save time and disk space
    ]
    
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        print("\nError: Scoring process failed.")
        return
        
    # 3. Filter Images
    final_dir = os.path.join("captured_shots", "final", f"{basename}_final")
    report_path = os.path.join(session_scored_dir, "results.json")
    print(f"\n[STEP 3] Filtering scored images to {final_dir}...")
    filter_scored_images(
        input_dir=session_scored_dir,
        source_dir=session_raw_dir,
        output_dir=final_dir,
        report_path=report_path
    )
    
    # 4. Clean up original tracked video
    print(f"\n[STEP 4] Deleting original video {video_path}...")
    try:
        os.remove(video_path)
    except Exception as e:
        print(f"Warning: Could not delete video {video_path}: {e}")
        
    print(f"\n[DONE] Post-processing for {video_path} complete. Best images saved to {final_dir}.")

if __name__ == "__main__":
    main()
