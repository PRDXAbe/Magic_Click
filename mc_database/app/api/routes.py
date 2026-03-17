"""
FastAPI Routes
New API structure: /search, /add, /name, /topn
Refactored to use FaceService.
"""
import os
import tempfile
import threading
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, UploadFile, Form, HTTPException, Body, Query, Request, File
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from app.services.face_service import FaceService
from app.core import camera_store

# Lazy-loaded scoring imports
_detectors = None
_lock = threading.Lock()

def get_detectors():
    global _detectors
    if _detectors is None:
        with _lock:
            if _detectors is None:
                # Import here so it doesn't slow down startup
                from ultralytics import YOLO
                
                # Append Temp_MCv2 path logically if needed, but since we are merging, 
                # we should just import pose_scorer directly if it's in PYTHONPATH or we move it.
                # Since we moved all files to the root of mc_database, let's just make sure 
                # pose_scorer is available. Wait, we moved all Temp_MCv2 files to Magic_Click root, 
                # and mc_database is a subfolder. Let's fix python path temporarily here.
                import sys
                import os
                root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
                if root_dir not in sys.path:
                    sys.path.insert(0, root_dir)
                    
                from pose_scorer import config as cfg
                from pose_scorer.scorer import init_detectors
                
                person_det = YOLO(cfg.MODELS['person_detector'])
                face_det   = YOLO(cfg.MODELS['face_detector'])
                face_lm, pose_lm = init_detectors()
                _detectors = {
                    'person': person_det,
                    'face':   face_det,
                    'face_lm': face_lm,
                    'pose_lm': pose_lm,
                    'cfg': cfg
                }
    return _detectors

api = APIRouter(prefix="/api")
face_service = FaceService()

# Pydantic Models
class BatchAddImage(BaseModel):
    img: str
    score: float = 0.0
    name: Optional[str] = None
    cropped: bool = False

class BatchAddRequest(BaseModel):
    images: List[BatchAddImage]

class CreatePersonRequest(BaseModel):
    name: str

class SearchRequest(BaseModel):
    img: Optional[str] = None
    cropped: bool = False

# Endpoints

@api.get('/health')
async def health():
    """Health check endpoint."""
    return face_service.get_health_stats()

@api.post('/search')
async def search_post(request: SearchRequest):
    """
    Search by face image (POST).
    """
    result = face_service.search_image(request.img, request.cropped)
    status_code = 200 if result.get('success', False) else 400
    if not result.get('success') and result.get('match') is False:
         status_code = 200 # No match is a valid result
    
    # Handle specific error cases that were returned as 500 in original code if appropriate, 
    # but 400 is generally safer for client errors. 
    # Original code returned 500 for 'Person data not found' logic error.
    if result.get('error') == 'Person data not found':
        status_code = 500
        
    return JSONResponse(result, status_code=status_code)

@api.get('/search')
async def search_get(
    id: Optional[str] = Query(None),
    name: Optional[str] = Query(None)
):
    """
    Search by ID or name (GET).
    """
    if not id and not name:
        return JSONResponse({'success': False, 'error': 'No search parameter provided (id or name)'}, status_code=400)
    
    result = face_service.get_person_by_id_or_name(id, name)
    status_code = 200 if result.get('success') else 404
    return JSONResponse(result, status_code=status_code)

@api.post('/add')
async def add_image(request: Request):
    """
    Add an image - auto-matches to existing person or creates new.
    Supports JSON (base64) and Multipart/Form-data (file upload).
    """
    content_type = request.headers.get('content-type', '')
    
    score = 0.0
    cropped = False
    img_data = None
    
    try:
        if 'application/json' in content_type:
            data = await request.json()
            img_data = data.get('img')
            score = float(data.get('score', 0.0))
            cropped = data.get('cropped', False)
            
        elif 'multipart/form-data' in content_type:
            form = await request.form()
            score = float(form.get('score', 0.0))
            cropped = str(form.get('cropped', '')).lower() in ('true', '1', 'yes')
            
            # Check for file
            upload_file = form.get('image')
            if upload_file is not None and hasattr(upload_file, 'read'):
                img_data = await upload_file.read()
            else:
                # Check for base64 string in form
                img_data = form.get('img')
                
        else:
            return JSONResponse({'success': False, 'error': 'Unsupported Content-Type'}, status_code=400)
            
        if not img_data:
             return JSONResponse({'success': False, 'error': 'No image provided'}, status_code=400)
             
        # Process
        result = face_service.process_and_add_image(img_data, score, cropped)
        status_code = 200 if result['success'] else 400
        return JSONResponse(result, status_code=status_code)
        
    except Exception as e:
        return JSONResponse({'success': False, 'error': str(e)}, status_code=400)

@api.post('/batch_add')
async def batch_add_images(request: BatchAddRequest):
    """
    Add multiple images in batch.
    """
    images = request.images
    
    if not images:
        return JSONResponse({'success': False, 'error': 'List of images required'}, status_code=400)
    
    results = []
    success_count = 0
    fail_count = 0
    
    print(f"DEBUG: Batch processing {len(images)} images...")
    
    for i, img_data in enumerate(images):
        result = face_service.process_and_add_image(img_data.img, img_data.score, img_data.cropped)
        result['index'] = i
        
        if result['success']:
            name = img_data.name
            # If name provided, try to assign it. This logic is slightly coupled 
            # but fits 'process_and_add' extension or calling assign_name separately.
            if name and result.get('person_id'):
                face_service.assign_name(result['person_id'], name)
                result['name_assigned'] = name
            success_count += 1
        else:
            print(f"DEBUG: Batch item {i} failed: {result.get('error')}")
            fail_count += 1
        
        results.append(result)
        
    return {
        'success': True,
        'total': len(images),
        'added': success_count,
        'failed': fail_count,
        'results': results
    }

@api.delete('/reset')
async def reset_database():
    """Clear all data from the database."""
    # This involves multiple services (storage, vector_db). 
    # For now, it's safer to keep this special administrative action here 
    # or add a 'reset_all' to service. 
    # Let's add it to service for consistency in next step or use dependency access here?
    # Better: Use dependencies directly as this is admin function, OR strictly use service.
    # Service doesn't have reset() yet. 
    # Implementation detail: I'll use the service deps pattern or imports. 
    # To follow Clean Code: add reset to service.
    try:
        from app.core.storage import get_storage
        from app.core.vector_db import get_vector_db
        storage = get_storage()
        vector_db = get_vector_db()
        storage.reset()
        vector_db.reset()
        
        return {
            'success': True,
            'message': 'Database successfully cleared'
        }
    except Exception as e:
        return JSONResponse({'success': False, 'error': str(e)}, status_code=500)

@api.post('/name')
async def assign_name(request: Request):
    """Assign a name to a person."""
    try:
        content_type = request.headers.get('content-type', '')
        final_pid = None
        final_name = None
        
        if 'application/json' in content_type:
            data = await request.json()
            final_pid = data.get('person_id')
            final_name = data.get('name')
        elif 'multipart/form-data' in content_type or 'application/x-www-form-urlencoded' in content_type:
            form = await request.form()
            final_pid = form.get('person_id')
            final_name = form.get('name')
    except Exception:
        return JSONResponse({'success': False, 'error': 'Invalid request body'}, status_code=400)
    
    if not final_pid or not final_name:
        return JSONResponse({'success': False, 'error': 'person_id and name required'}, status_code=400)
    
    result = face_service.assign_name(final_pid, final_name)
    status_code = 200 if result['success'] else 400
    if result.get('error') == 'Person not found':
        status_code = 404
        
    return JSONResponse(result, status_code=status_code)

@api.get('/topn')
async def get_top_images():
    """Get top 3 images globally by score."""
    return face_service.get_top_images(n=3)

@api.get('/person/{person_id}')
async def get_person(person_id: str):
    """Get person info by ID."""
    result = face_service.get_person_by_id_or_name(pid=person_id, name=None)
    
    if result.get('success'):
         return {
             'success': True,
             'person': result['person']
         }
    else:
        return JSONResponse(result, status_code=404)

@api.delete('/person/{person_id}')
async def delete_person(person_id: str):
    """Delete a person and all their images."""
    result = face_service.delete_person(person_id)
    status_code = 200 if result['success'] else 404
    return JSONResponse(result, status_code=status_code)

@api.get('/image/{image_id}')
async def get_image(image_id: str):
    """Serve an image file by ID (raw JPEG bytes)."""
    image_bytes = face_service.get_image_bytes(image_id)
    
    if not image_bytes:
        return JSONResponse({'success': False, 'error': 'Image not found'}, status_code=404)
    
    return Response(image_bytes, media_type='image/jpeg')


# ==========================================
# Camera Management Routes
# ==========================================

class CameraAddRequest(BaseModel):
    type: str
    source: str
    label: str

@api.get('/cameras')
async def list_cameras():
    """List all configured cameras."""
    cameras = camera_store.load_cameras()
    return {"success": True, "cameras": cameras}

@api.post('/cameras')
async def add_camera(request: CameraAddRequest):
    """Add a new camera."""
    if request.type not in ["wired", "ip"]:
        return JSONResponse({"success": False, "error": "Type must be 'wired' or 'ip'"}, status_code=400)
        
    cam = camera_store.add_camera(request.type, request.source, request.label)
    
    # Notify service to reload
    try:
        from app.core.camera_service import camera_service_instance
        if camera_service_instance:
            camera_service_instance.reload_cameras()
    except ImportError:
        pass
        
    return {"success": True, "camera": cam}

@api.delete('/cameras/{cam_id}')
async def remove_camera(cam_id: str):
    """Remove a camera."""
    success = camera_store.remove_camera(cam_id)
    
    # Notify service to reload
    if success:
        try:
            from app.core.camera_service import camera_service_instance
            if camera_service_instance:
                camera_service_instance.reload_cameras()
        except ImportError:
            pass
            
    return {"success": success}

@api.put('/cameras/{cam_id}/toggle')
async def toggle_camera(cam_id: str):
    """Toggle camera enabled state."""
    cam = camera_store.toggle_camera(cam_id)
    if not cam:
        return JSONResponse({"success": False, "error": "Camera not found"}, status_code=404)
        
    # Notify service to reload
    try:
        from app.core.camera_service import camera_service_instance
        if camera_service_instance:
            camera_service_instance.reload_cameras()
    except ImportError:
        pass
        
    return {"success": True, "camera": cam}

@api.get('/cameras/{cam_id}/stream')
async def get_camera_stream(cam_id: str):
    """MJPEG Stream for a specific camera."""
    try:
        from app.core.camera_service import camera_service_instance
        if not camera_service_instance:
            return JSONResponse({"success": False, "error": "Camera service not running"}, status_code=503)
            
        return StreamingResponse(
            camera_service_instance.generate_stream(cam_id),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
    except ImportError:
        return JSONResponse({"success": False, "error": "Service not initialized"}, status_code=500)


# ==========================================
# Image Scoring / Pipeline Routes
# ==========================================

@api.get('/score/status')
async def api_score_status():
    """Check if ML models are ready."""
    try:
        import sys
        import os
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        if root_dir not in sys.path:
            sys.path.insert(0, root_dir)
            
        from pose_scorer import config as cfg
        cfg.validate()
        from pathlib import Path
        models_ready = all(Path(p).exists() for p in cfg.MODELS.values())
        return {'status': 'ok', 'models_ready': models_ready}
    except Exception as e:
        return JSONResponse({'status': 'error', 'detail': str(e)}, status_code=500)

@api.post('/score')
async def api_score(images: List[UploadFile] = File(...)):
    """Score uploaded images using the ML pipeline."""
    if not images:
        return JSONResponse({'error': 'No images uploaded'}, status_code=400)

    try:
        det = get_detectors()
    except Exception as e:
        return JSONResponse({'error': f'Model load failed: {e}'}, status_code=503)

    import sys
    import os
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)
        
    from pose_scorer.scorer import score_image

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for f in images:
            if not f.filename:
                continue
            path = os.path.join(tmp, f.filename)
            
            # Save uploaded file
            contents = await f.read()
            with open(path, 'wb') as out_file:
                out_file.write(contents)

            try:
                result = score_image(
                    image_path      = path,
                    person_detector = det['person'],
                    face_detector   = det['face'],
                    face_landmarker = det['face_lm'],
                    pose_landmarker = det['pose_lm'],
                )
            except Exception as e:
                result = {
                    'image': f.filename, 'status': 'PIPELINE_ERROR',
                    'reject_reason': str(e)[:200],
                    'final_score': None, 'score_band': '', 'rank': None,
                    'preflight': {}, 'detection': {}, 'frame_check': {},
                    'face_group': {}, 'body_group': {}, 'debug_image': '',
                }

            # Generate lightweight debug image for rejections just like in Flask
            if isinstance(result, dict) and result.get('status') != 'SCORED' and 'debug_image' not in result:
                try:
                    import cv2, numpy as np, base64
                    img = cv2.imread(path)
                    if img is not None:
                        img_h, img_w = img.shape[:2]
                        scale = min(1.0, 900 / img_w)
                        if scale < 1.0:
                            img = cv2.resize(img, (int(img_w * scale), int(img_h * scale)))
                        dh, dw = img.shape[:2]
                        region = img[0:50, 0:min(340, img.shape[1])].copy()
                        dark = np.full_like(region, (8, 10, 16))
                        img[0:region.shape[0], 0:region.shape[1]] = cv2.addWeighted(dark, 0.80, region, 0.20, 0)
                        cv2.putText(img, result.get('status',''), (10, 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 255), 1, cv2.LINE_AA)
                        reason = result.get('reject_reason') or ''
                        cv2.putText(img, reason[:55], (10, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150, 150, 150), 1, cv2.LINE_AA)
                        
                        # Draw person bbox
                        det_info = result.get('detection') if isinstance(result, dict) else None
                        if isinstance(det_info, dict) and 'person_bbox' in det_info:
                            pb = det_info['person_bbox']
                            if isinstance(pb, list) and len(pb) == 4:
                                p = [int(v * scale) for v in pb]
                                cv2.rectangle(img, (p[0], p[1]), (p[2], p[3]), (40, 220, 80), 2)
                                
                        # Draw face bbox
                        if isinstance(det_info, dict) and 'face_bbox' in det_info:
                            fb = det_info['face_bbox']
                            if isinstance(fb, list) and len(fb) == 4:
                                f2 = [int(v * scale) for v in fb]
                                cv2.rectangle(img, (f2[0], f2[1]), (f2[2], f2[3]), (255, 120, 30), 2)
                                
                        _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        
                        if isinstance(result, dict):
                            result['debug_image'] = 'data:image/jpeg;base64,' + base64.b64encode(buf).decode()
                except Exception:
                    pass

            results.append(result)

    # Rank scored images
    scored = [r for r in results if r['status'] == 'SCORED' and r['final_score'] is not None]
    scored.sort(key=lambda x: x['final_score'], reverse=True)
    for i, r in enumerate(scored):
        r['rank'] = i + 1

    return results

