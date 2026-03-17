import sqlite3
import os
import time

DB_PATH = os.path.join("captured_videos", "jobs.db")

def _get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
    # Give the connection dict-like rows
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with _get_conn() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_path TEXT UNIQUE NOT NULL,
                frame_count INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'PENDING',
                created_at REAL NOT NULL,
                started_at REAL,
                completed_at REAL
            )
        ''')
        conn.commit()

def add_job(video_path: str, frame_count: int) -> int:
    """Adds a new job to the queue."""
    with _get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            '''
            INSERT INTO jobs (video_path, frame_count, status, created_at)
            VALUES (?, ?, 'PENDING', ?)
            ''',
            (video_path, frame_count, time.time())
        )
        conn.commit()
        return cursor.lastrowid

def get_shortest_job():
    """Retrieves the pending job with the lowest frame count (SJF). Marks it as RUNNING."""
    with _get_conn() as conn:
        # Use a transaction to safely mark a job as RUNNING
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM jobs 
            WHERE status = 'PENDING' 
            ORDER BY frame_count ASC, created_at ASC 
            LIMIT 1
        ''')
        row = cursor.fetchone()
        
        if row is None:
            return None
            
        job_id = row['id']
        cursor.execute('''
            UPDATE jobs 
            SET status = 'RUNNING', started_at = ? 
            WHERE id = ? AND status = 'PENDING'
        ''', (time.time(), job_id))
        
        # If the row was actually updated, we successfully claimed it
        if cursor.rowcount > 0:
            conn.commit()
            return dict(row)
        return None

def mark_job_completed(job_id: int):
    with _get_conn() as conn:
        conn.execute('''
            UPDATE jobs 
            SET status = 'COMPLETED', completed_at = ? 
            WHERE id = ?
        ''', (time.time(), job_id))
        conn.commit()

def mark_job_failed(job_id: int):
    with _get_conn() as conn:
        conn.execute('''
            UPDATE jobs 
            SET status = 'FAILED', completed_at = ? 
            WHERE id = ?
        ''', (time.time(), job_id))
        conn.commit()

def get_pending_count() -> int:
    with _get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM jobs WHERE status = 'PENDING'")
        return cursor.fetchone()[0]

# Initialize tables when imported
init_db()
