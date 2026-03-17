import time
import subprocess
import threading
import sys
import os

from app.config import CAPTURED_VIDEOS_DIR

class JobWorkerService:
    def __init__(self):
        self.running = False
        self.thread = None
        
        # Make sure queue manager logic is reachable
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        if root_dir not in sys.path:
            sys.path.insert(0, root_dir)
            
        import queue_manager
        self.queue_manager = queue_manager
        
    def start(self):
        if self.running:
            return
            
        print("--- Starting Multi-Camera SJF Worker Background Service ---")
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        print("Job worker stopped.")
        
    def _run_loop(self):
        print("Polling database 'captured_videos/jobs.db' for PENDING jobs...")
        while self.running:
            try:
                job = self.queue_manager.get_shortest_job()
                
                if job is None:
                    # No jobs, wait and poll again
                    time.sleep(1.0)
                    continue
                
                job_id = job['id']
                video_path = job['video_path']
                frame_count = job['frame_count']
                
                pending_count = self.queue_manager.get_pending_count()
                print(f"\n[JOB WORKER] Picked Job #{job_id} | Frames: {frame_count} | Mode: SJF")
                print(f"[JOB WORKER] Pending Jobs Remaining: {pending_count}")
                print(f"[JOB WORKER] Processing: {video_path}")
                
                # We use the existing post_process_video.py logic for processing the mp4
                start_time = time.time()
                try:
                    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
                    post_proc_path = os.path.join(root_dir, "post_process_video.py")
                    
                    # Ensure post_process_video is run from the root directory so its relative paths (like 'captured_shots/') work
                    result = subprocess.run(
                        [sys.executable, post_proc_path, "--video", video_path],
                        check=True,
                        cwd=root_dir
                    )
                    
                    duration = time.time() - start_time
                    print(f"[JOB WORKER] Job #{job_id} Completed in {duration:.1f}s.")
                    self.queue_manager.mark_job_completed(job_id)

                except subprocess.CalledProcessError as e:
                    print(f"[JOB WORKER] Job #{job_id} FAILED!")
                    self.queue_manager.mark_job_failed(job_id)
                    
            except Exception as e:
                print(f"Worker Error: {e}")
                time.sleep(5.0)

job_worker_instance = JobWorkerService()
