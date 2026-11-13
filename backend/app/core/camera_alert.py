import os
import time
import threading
import requests
from datetime import datetime
from typing import Dict, Any, Optional
from colorama import Fore, Style

from app.database.tenant_manager import get_tenant_session
from app.model.models import Camera

FILE = "CAMERA_ALERT.py"

OFFLINE_FAILURE_COUNT_THRESHOLD = int(os.getenv("CAMERA_OFFLINE_THRESHOLD", 5))
ALERT_COOLDOWN_SECONDS = float(os.getenv("CAMERA_ALERT_COOLDOWN_SECONDS", 300.0)) # 5 mins
WEBHOOK_URL = os.getenv("CAMERA_ALERT_WEBHOOK_URL", "")


class CameraAlertManager:
    """
    Production-grade Camera Offline Detection & Multi-Channel Alert Manager.
    Monitors RTSP frame health, updates database 'is_working' flags, and dispatches webhook alerts.
    """
    def __init__(self):
        self.lock = threading.Lock()
        # Structure: { (tenant_id, camera_id): {
        #   "failures": int,
        #   "is_offline": bool,
        #   "last_alert_ts": float,
        #   "camera_name": str,
        #   "stream_url": str
        # } }
        self.camera_states: Dict[tuple, Dict[str, Any]] = {}

    def record_frame_success(self, tenant_id: str, camera_id: str, camera_name: str, stream_url: str):
        """Records a successful frame capture. Triggers recovery if camera was offline."""
        key = (tenant_id, camera_id)
        with self.lock:
            state = self.camera_states.setdefault(key, {
                "failures": 0,
                "is_offline": False,
                "last_alert_ts": 0.0,
                "camera_name": camera_name,
                "stream_url": stream_url
            })
            was_offline = state["is_offline"]
            state["failures"] = 0
            state["camera_name"] = camera_name
            state["stream_url"] = stream_url

            if was_offline:
                state["is_offline"] = False
                print(Fore.GREEN + Style.BRIGHT + f"[{FILE}][{tenant_id}] [RECOVERED] Camera '{camera_name}' (ID: {camera_id}) is BACK ONLINE!")
                self._dispatch_event_async("CAMERA_ONLINE", tenant_id, camera_id, camera_name, stream_url, "Camera connection restored successfully.")
                self._update_camera_db_status_async(tenant_id, camera_id, is_working=True)

    def record_frame_failure(self, tenant_id: str, camera_id: str, camera_name: str, stream_url: str):
        """Records a failed frame read. Triggers offline alert if failure threshold is reached."""
        key = (tenant_id, camera_id)
        now_ts = time.time()

        with self.lock:
            state = self.camera_states.setdefault(key, {
                "failures": 0,
                "is_offline": False,
                "last_alert_ts": 0.0,
                "camera_name": camera_name,
                "stream_url": stream_url
            })
            state["failures"] += 1
            state["camera_name"] = camera_name
            state["stream_url"] = stream_url

            if state["failures"] >= OFFLINE_FAILURE_COUNT_THRESHOLD:
                if not state["is_offline"]:
                    state["is_offline"] = True
                    state["last_alert_ts"] = now_ts

                    print(Fore.RED + Style.BRIGHT + "\n" + "=" * 65)
                    print(Fore.RED + Style.BRIGHT + f"🚨 [CRITICAL ALERT] Camera '{camera_name}' (ID: {camera_id}) in Tenant '{tenant_id}' is OFFLINE!")
                    print(Fore.RED + Style.BRIGHT + f"   RTSP URL: {stream_url}")
                    print(Fore.RED + Style.BRIGHT + "=" * 65 + "\n")

                    self._dispatch_event_async(
                        "CAMERA_OFFLINE", 
                        tenant_id, 
                        camera_id, 
                        camera_name, 
                        stream_url, 
                        f"Failed to read {state['failures']} consecutive frames. Camera stream unreachable."
                    )
                    self._update_camera_db_status_async(tenant_id, camera_id, is_working=False)
                else:
                    # Periodic repeat alert if cooldown expired
                    if now_ts - state["last_alert_ts"] >= ALERT_COOLDOWN_SECONDS:
                        state["last_alert_ts"] = now_ts
                        self._dispatch_event_async(
                            "CAMERA_STILL_OFFLINE", 
                            tenant_id, 
                            camera_id, 
                            camera_name, 
                            stream_url, 
                            f"Camera '{camera_name}' remains unreachable."
                        )

    def _update_camera_db_status_async(self, tenant_id: str, camera_id: str, is_working: bool):
        """Runs in background to update 'is_working' and 'last_working_on' in tenant's Camera table."""
        def _task():
            try:
                db = get_tenant_session(tenant_id)
                try:
                    cam = db.query(Camera).filter(
                        (Camera.id == camera_id) | 
                        (Camera.asset_no == camera_id) |
                        (Camera.unique_id == int(camera_id) if camera_id.isdigit() else False)
                    ).first()
                    if cam:
                        cam.is_working = is_working
                        cam.last_working_on = datetime.utcnow()
                        db.commit()
                finally:
                    db.close()
            except Exception as e:
                print(Fore.RED + f"[{FILE}][{tenant_id}] Error updating camera DB status: {e}")

        threading.Thread(target=_task, daemon=True).start()

    def _dispatch_event_async(self, event_type: str, tenant_id: str, camera_id: str, camera_name: str, stream_url: str, message: str):
        """Dispatches alert webhook to external system (Slack/Telegram/WhatsApp/Dashboard)."""
        def _task():
            payload = {
                "event": event_type,
                "tenant_id": tenant_id,
                "camera_id": camera_id,
                "camera_name": camera_name,
                "stream_url": stream_url,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "message": message
            }

            webhook_url = WEBHOOK_URL or os.getenv("CAMERA_ALERT_WEBHOOK_URL", "")
            if webhook_url:
                try:
                    res = requests.post(
                        webhook_url,
                        json=payload,
                        headers={"Content-Type": "application/json"},
                        timeout=5
                    )
                    print(Fore.CYAN + f"[{FILE}][{tenant_id}] Webhook alert sent: {event_type} (Status: {res.status_code})")
                except Exception as e:
                    print(Fore.RED + f"[{FILE}][{tenant_id}] Failed to dispatch webhook alert: {e}")

        threading.Thread(target=_task, daemon=True).start()


camera_alert_manager = CameraAlertManager()
