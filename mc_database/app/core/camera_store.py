"""
Camera configuration persistent store
Reads and writes to data/cameras.json
"""
import os
import json
import uuid
from typing import List, Dict, Any, Optional

from app.config import CAMERAS_CONFIG_PATH

def init_store():
    """Create the json file with an empty list or default if it doesn't exist."""
    if not os.path.exists(CAMERAS_CONFIG_PATH):
        # By default, add the webcam 0
        default_cameras = [
            {
                "id": "cam_" + str(uuid.uuid4())[:8],
                "type": "wired",
                "source": "0",
                "label": "Built-in Webcam",
                "enabled": True
            }
        ]
        save_cameras(default_cameras)

def load_cameras() -> List[Dict[str, Any]]:
    """Load cameras from JSON."""
    if not os.path.exists(CAMERAS_CONFIG_PATH):
        init_store()
        
    try:
        with open(CAMERAS_CONFIG_PATH, 'r') as f:
            return json.load(f)
    except Exception:
        return []

def save_cameras(cameras: List[Dict[str, Any]]) -> None:
    """Save cameras to JSON."""
    os.makedirs(os.path.dirname(CAMERAS_CONFIG_PATH), exist_ok=True)
    with open(CAMERAS_CONFIG_PATH, 'w') as f:
        json.dump(cameras, f, indent=4)

def add_camera(cam_type: str, source: str, label: str) -> Dict[str, Any]:
    """Add a new camera to the store."""
    cameras = load_cameras()
    
    # Simple validation
    if cam_type == "wired" and source.isdigit():
        source = int(source) # type: ignore
        
    cam = {
        "id": "cam_" + str(uuid.uuid4())[:8],
        "type": cam_type,
        "source": source,
        "label": label,
        "enabled": True
    }
    
    cameras.append(cam)
    save_cameras(cameras)
    return cam

def remove_camera(cam_id: str) -> bool:
    """Remove a camera by ID. Returns True if removed."""
    cameras = load_cameras()
    original_count = len(cameras)
    cameras = [c for c in cameras if c.get('id') != cam_id]
    
    if len(cameras) < original_count:
        save_cameras(cameras)
        return True
    return False

def toggle_camera(cam_id: str, enabled: Optional[bool] = None) -> Optional[Dict[str, Any]]:
    """Toggle the enabled state of a camera, or set to a specific state."""
    cameras = load_cameras()
    for cam in cameras:
        if cam.get('id') == cam_id:
            if enabled is None:
                cam['enabled'] = not cam.get('enabled', True)
            else:
                cam['enabled'] = enabled
            save_cameras(cameras)
            return cam
    return None

# Initialize on module import
init_store()
