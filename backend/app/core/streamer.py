import os
import time
import cv2
import numpy as np
from typing import Optional
from colorama import Fore

# Force TCP transport for RTSP to prevent HEVC/H.264 packet loss
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

from app.core.tracker import LightweightFaceTracker, letterbox_resize
from app.core.face_engine import face_engine
from app.core.stranger import stranger_tracker
from app.core.camera_alert import camera_alert_manager
from app.core.attendance_service import process_attendance_and_learning_async
from app.database.tenant_manager import get_tenant_session
from app.model.models import Camera

def generate_video_stream(
    tenant_id: str, 
    camera_id: Optional[str] = None,
    current_port: Optional[str] = None,
    current_host: Optional[str] = None
):
    """
    Connects to tenant's DB, filters camera by port/instance_url (like Django),
    performs face recognition using tenant's face embeddings, and yields MJPEG chunks.
    """
    # 1. Fetch Camera Stream URL from Tenant's Database
    db = get_tenant_session(tenant_id)
    stream_url = None
    camera_name = "CCTV"
    cam = None
    try:
        if camera_id:
            cam = db.query(Camera).filter(
                (Camera.id == camera_id) | 
                (Camera.asset_no == camera_id) | 
                (Camera.unique_id == int(camera_id) if camera_id.isdigit() else False)
            ).first()
        else:
            query = db.query(Camera).filter(Camera.status == True)
            
            # Port-based filter (matching Django instance_url logic)
            if current_port:
                cam = query.filter(Camera.instance_url.like(f"%:{current_port}")).first()
            
            if not cam and current_host:
                cam = query.filter(Camera.instance_url == current_host).first()
                
            if not cam:
                cam = query.first()

        if cam and cam.stream_url:
            stream_url = cam.stream_url
            camera_name = cam.asset_no or cam.model_name or "CCTV"
    finally:
        db.close()

    if not stream_url:
        print(Fore.RED + f"[STREAM ERROR] No active camera stream found for Tenant '{tenant_id}'")
        return

    print(Fore.CYAN + f"[STREAM] Tenant '{tenant_id}' -> Connecting to camera: {camera_name} ({stream_url})")

    # Source detection (Webcam index 0 or URL/File)
    source = int(stream_url) if stream_url.isdigit() else stream_url
    cap = cv2.VideoCapture(source)

    cam_id_str = str(cam.id if cam else camera_id or "")

    if not cap.isOpened():
        camera_alert_manager.record_frame_failure(tenant_id, cam_id_str, camera_name, stream_url)
        print(Fore.RED + f"[STREAM ERROR] Cannot open stream: {stream_url}")
        return

    tracker = LightweightFaceTracker(max_disappeared=5, iou_threshold=0.3)
    recognition_cache = {}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                camera_alert_manager.record_frame_failure(tenant_id, cam_id_str, camera_name, stream_url)
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                time.sleep(0.033)
                continue

            camera_alert_manager.record_frame_success(tenant_id, cam_id_str, camera_name, stream_url)

            frame_resized = letterbox_resize(frame, 1280, 960)

            try:
                faces = face_engine.app.get(frame_resized)
            except Exception:
                faces = []

            detected_boxes = []
            valid_faces = []

            for face in faces:
                if len(detected_boxes) >= face_engine.MAX_FACES:
                    break
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

                if (x2 - x1) < 25 or (y2 - y1) < 25:
                    continue

                detected_boxes.append([x1, y1, x2, y2])
                valid_faces.append(face)

            tracked_results = tracker.update(detected_boxes)
            frame_matches = []

            for idx, face in enumerate(valid_faces):
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                track_id = tracked_results.get(idx, "ID-?")

                # Check Cache
                cached = recognition_cache.get(track_id)
                if cached and cached[0] is not None:
                    match, conf, confident, embed_type = cached
                    emb = face.embedding
                else:
                    emb = face.embedding
                    norm = np.linalg.norm(emb)
                    if norm <= 0:
                        continue
                    emb = emb / norm
                    # Match against Tenant's embeddings
                    match, conf, confident, embed_type = face_engine.recognize_face(emb, tenant_id=tenant_id)
                    if match is not None:
                        recognition_cache[track_id] = (match, conf, confident, embed_type)
                    else:
                        match, conf, confident, embed_type = None, conf, False, None

                # Draw Visuals on Frame
                if match:
                    person_pk, person_name = match[0], match[1]
                    if confident:
                        color = (0, 180, 0)
                        label = f"{person_name} {conf * 100:.1f}%"
                        frame_matches.append((person_pk, person_name, float(conf), emb, embed_type))
                    else:
                        color = (0, 165, 255)
                        label = f"[{track_id}] Low-conf {person_name}"

                    cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame_resized, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                else:
                    # STRANGER WORKFLOW: Frontal check, 1.5s confirmation, and auto-enrollment
                    status, elapsed = stranger_tracker.process_unknown_face(
                        face=face,
                        emb=emb,
                        track_id=track_id,
                        tenant_id=tenant_id,
                        camera_name=camera_name,
                        face_engine=face_engine
                    )

                    if status == "VERIFYING":
                        color = (0, 165, 255)  # Orange = Verifying window (hold for 1.5s)
                        label = f"[{track_id}] Verifying... ({elapsed:.1f}s)"
                    elif status == "CONFIRMED":
                        color = (0, 0, 255)    # Red = Confirmed Stranger (queued/enrolling)
                        label = f"[{track_id}] Stranger"
                        recognition_cache[track_id] = ((None, "Stranger"), 1.0, True, "normal")
                    else:
                        color = (0, 0, 255)    # Red = Unknown (profile angle or fast movement)
                        label = f"[{track_id}] Unknown"

                    cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame_resized, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Trigger Attendance for matched faces
            if frame_matches:
                ok_crop, crop_buf = cv2.imencode(".jpg", frame_resized)
                crop_bytes = crop_buf.tobytes() if ok_crop else None
                for person_pk, person_name, conf_val, emb_vec, etype in frame_matches:
                    process_attendance_and_learning_async(
                        person_pk, person_name, "", conf_val, emb_vec, crop_bytes, etype, camera_name, tenant_id=tenant_id
                    )

            ok, buf = cv2.imencode(".jpg", frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

    finally:
        cap.release()
        stranger_tracker.clean_camera(tenant_id, camera_name)
        print(Fore.YELLOW + f"[STREAM] Stopped stream for Tenant: {tenant_id} (Camera: {camera_name})")
