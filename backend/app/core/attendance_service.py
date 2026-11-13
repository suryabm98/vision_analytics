import os
import time
import pickle
import base64
import threading
import requests
from requests.auth import HTTPBasicAuth
from datetime import datetime
import pytz
import cv2
import numpy as np
from colorama import Fore, Style

from app.database.tenant_manager import get_tenant_session
from app.model.models import Person

FILE = "ATTENDANCE_SERVICE.py"
IST = pytz.timezone("Asia/Kolkata")

_last_seen = {}
_last_seen_lock = threading.Lock()
_detection_counters = {}

def process_attendance_and_learning_async(
    person_pk, 
    person_name, 
    emp_no, 
    conf, 
    raw_emb, 
    full_frame_bytes, 
    embed_type="normal", 
    camera_name="CCTV",
    tenant_id="MILKY_01"
):
    """Runs asynchronously in background thread to avoid blocking video stream."""
    thread = threading.Thread(
        target=_execute_task,
        args=(person_pk, person_name, emp_no, conf, raw_emb, full_frame_bytes, embed_type, camera_name, tenant_id),
        daemon=True
    )
    thread.start()

def _execute_task(person_pk, person_name, emp_no, conf, raw_emb, full_frame_bytes, embed_type, camera_name, tenant_id):
    try:
        cooldown_sec = float(os.getenv("ATTENDANCE_COOLDOWN_MINUTES", 1.0)) * 60.0
        now_ts = time.time()
        cooldown_key = (tenant_id, camera_name, person_pk)

        # 1. Cooldown Check per Tenant, Camera & Person
        with _last_seen_lock:
            last_ts = _last_seen.get(cooldown_key, 0)
            if now_ts - last_ts < cooldown_sec:
                return
            _last_seen[cooldown_key] = now_ts

        now = datetime.now(IST)
        today_date_str = str(now.date())
        time_str = now.strftime("%H:%M:%S")
        date_str = now.strftime("%a, %d %b %Y")

        # 2. Dynamic Filename Construction
        label_slug = (person_name or "unknown").lower().replace(" ", "_")
        counter_key = f"{tenant_id}_{label_slug}_{today_date_str}"
        _detection_counters[counter_key] = _detection_counters.get(counter_key, 0) + 1
        counter = _detection_counters[counter_key]
        custom_filename = f"{label_slug}-{now.strftime('%Y%m%d-%H%M%S')}-{counter}.jpeg"

        # 3. Burn OSD Bar into Frame Image in RAM
        formatted_image_b64 = None
        if full_frame_bytes:
            nparr = np.frombuffer(full_frame_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                h, w, _ = img.shape
                overlay_text = f"{date_str}  {time_str}  |  Camera: {camera_name}"
                font_scale = max(0.5, h / 720.0) * 0.7
                font_thickness = max(1, int(font_scale * 1.5))
                (text_w, text_h), baseline = cv2.getTextSize(overlay_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                bar_height = max(40, text_h + baseline + 20)

                cv2.rectangle(img, (0, h - bar_height), (w, h), (0, 0, 0), -1)
                cv2.putText(img, overlay_text, (20, h - baseline - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

                success, encoded = cv2.imencode(".jpg", img)
                if success:
                    raw_b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
                    formatted_image_b64 = f"data:image/jpeg;base64,{raw_b64}"

        # 4. Connect to Tenant's Database for Person UUID & Auto-Learning
        db = get_tenant_session(tenant_id)
        person_uuid = str(person_pk)
        try:
            person_obj = db.query(Person).filter(Person.unique_id == person_pk).first()
            if person_obj:
                person_uuid = person_obj.id

            # Auto-Learning in Tenant's Database (if conf >= AUTO_LEARN_THRESHOLD)
            auto_learn_thresh = float(os.getenv("AUTO_LEARN_THRESHOLD", 0.58))
            if conf >= auto_learn_thresh and person_obj and raw_emb is not None:
                norm_emb = raw_emb / (np.linalg.norm(raw_emb) + 1e-6)
                attr_name = "face_embedding_mask" if embed_type == "mask" else "face_embedding"
                existing_bytes = getattr(person_obj, attr_name)

                current_list = []
                if existing_bytes:
                    try:
                        loaded = pickle.loads(existing_bytes)
                        current_list = loaded if isinstance(loaded, list) else [loaded]
                    except Exception:
                        current_list = []

                if not any(np.dot(norm_emb, e) > 0.95 for e in current_list):
                    current_list.append(norm_emb)
                    if len(current_list) > 5:
                        current_list = current_list[-5:]
                    setattr(person_obj, attr_name, pickle.dumps(current_list))
                    db.commit()
                    print(Fore.MAGENTA + f"[{FILE}][{tenant_id}] [AUTO-LEARN] Added new angle vector for {person_name} ({embed_type}, total: {len(current_list)})")
        finally:
            db.close()

        # 5. POST Payload to External Attendance API
        api_url = os.getenv("ATTENDANCE_API_URL")
        if api_url:
            payload = {
                "image": formatted_image_b64,
                "filename": custom_filename,
                "file_name": custom_filename,
                "person_id": str(person_uuid),
                "camera_id": camera_name,
                "camera_unique_id": "",
                "person_no": emp_no,
                "date": today_date_str,
                "title": "IN"
            }
            api_user = os.getenv("API_USERNAME", "")
            api_pass = os.getenv("API_PASSWORD", "")
            try:
                res = requests.post(
                    api_url,
                    auth=HTTPBasicAuth(api_user, api_pass),
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=20
                )
                print(Fore.GREEN + Style.BRIGHT + f"[{FILE}][{tenant_id}] [API SENT] Status: {res.status_code} | Person: {person_name} | File: {custom_filename}")
            except Exception as req_err:
                print(Fore.RED + f"[{FILE}][{tenant_id}] [API ERROR] Failed to send payload: {req_err}")

    except Exception as e:
        print(Fore.RED + f"[{FILE}] [ERROR] Attendance task failed: {e}")
