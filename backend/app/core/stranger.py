import os
import time
import uuid
import pickle
import threading
import numpy as np
from datetime import datetime
from typing import Optional, Tuple, Dict, Set, Any
from sqlalchemy import text
from colorama import Fore, Style

from app.database.tenant_manager import get_tenant_session
from app.model.models import Person, Camera, Users

FILE = "STRANGER.py"

_stranger_insert_lock = threading.Lock()

STRANGER_CONFIRM_SECONDS = float(os.getenv("STRANGER_CONFIRM_SECONDS", 1.5))
STRANGER_MIN_FRAMES = int(os.getenv("STRANGER_MIN_FRAMES", 3))
STRANGER_DUPLICATE_THRESHOLD = float(os.getenv("STRANGER_DUPLICATE_THRESHOLD", 0.45))


def is_clear_frontal_face(face) -> bool:
    """
    Validates if a detected face is frontal, clear, and high-quality for stranger enrollment.
    Rejects side-angle profiles, severe tilts, or blurry faces.
    """
    # 1. Pose check: pitch, yaw, roll within ±28 degrees
    if hasattr(face, "pose") and face.pose is not None:
        try:
            pitch, yaw, roll = face.pose
            if abs(yaw) > 28 or abs(pitch) > 28:
                return False
        except Exception:
            return False
    else:
        # Fallback landmark alignment: check eye horizontal level
        if not (hasattr(face, "kps") and face.kps is not None and len(face.kps) >= 5):
            return False
        left_eye, right_eye = face.kps[0], face.kps[1]
        dy = abs(float(left_eye[1]) - float(right_eye[1]))
        dx = abs(float(left_eye[0]) - float(right_eye[0])) + 1e-6
        tilt_ratio = dy / dx
        if tilt_ratio > 0.35:  # ~19° tilt max
            return False

    # 2. Eye distance check: both eyes must be clearly visible and sufficiently separated
    if hasattr(face, "kps") and face.kps is not None and len(face.kps) >= 2:
        left_eye, right_eye = face.kps[0], face.kps[1]
        eye_dist = np.linalg.norm(right_eye - left_eye)
        if eye_dist < 22:
            return False

    return True


def is_duplicate_stranger(emb: np.ndarray, tenant_id: str, face_engine, threshold: float = STRANGER_DUPLICATE_THRESHOLD) -> Tuple[bool, str]:
    """
    Checks if this stranger's face embedding already exists in the tenant's RAM matrix.
    Prevents creating duplicate Stranger rows for the same visitor across multiple tracks.
    """
    try:
        tenant_info = face_engine.tenant_data.get(tenant_id)
        if not tenant_info or tenant_info["matrix"] is None or len(tenant_info["matrix"]) == 0:
            return False, "None"

        matrix = tenant_info["matrix"]
        ids = tenant_info["ids"]

        sims = matrix @ emb
        max_sim = float(np.max(sims))

        if max_sim >= threshold:
            best_idx = int(np.argmax(sims))
            matched_name = ids[best_idx][1] if best_idx < len(ids) else "Unknown"
            print(Fore.CYAN + f"[{FILE}][{tenant_id}] [DUPLICATE CHECK] max_sim={max_sim:.4f} >= {threshold} with '{matched_name}' -> Already exists, skip insert.")
            return True, matched_name
    except Exception as e:
        print(Fore.RED + f"[{FILE}][{tenant_id}] [DUPLICATE CHECK ERROR] {e}")

    return False, "None"


def insert_stranger_async(tenant_id: str, raw_emb: np.ndarray, face_engine, camera_name: str = "CCTV"):
    """
    Executes asynchronous background database insertion into the specific tenant's 'person' table,
    and triggers an instant RAM matrix update in face_engine.
    """
    thread = threading.Thread(
        target=_execute_insert_stranger,
        args=(tenant_id, raw_emb, face_engine, camera_name),
        daemon=True
    )
    thread.start()


def _execute_insert_stranger(tenant_id: str, raw_emb: np.ndarray, face_engine, camera_name: str):
    try:
        norm = np.linalg.norm(raw_emb) + 1e-6
        norm_emb = raw_emb / norm
        emb_pickled = pickle.dumps([norm_emb])
        new_uuid = str(uuid.uuid4())

        db = get_tenant_session(tenant_id)
        try:
            # 1. Resolve created_by_id dynamically from Camera, Person, or Users table
            created_by = None
            cam = db.query(Camera).filter(Camera.created_by_id.isnot(None), Camera.created_by_id != "").first()
            if cam and cam.created_by_id:
                created_by = cam.created_by_id
            else:
                any_person = db.query(Person).filter(Person.created_by_id.isnot(None), Person.created_by_id != "").first()
                if any_person and any_person.created_by_id:
                    created_by = any_person.created_by_id
                else:
                    user_record = db.query(Users).first()
                    if user_record:
                        created_by = str(user_record.id)

            # 2. Resolve Stranger category_id
            stranger_cat_id = None
            try:
                cat_row = db.execute(text("SELECT id FROM category WHERE LOWER(name) = 'stranger' LIMIT 1")).fetchone()
                if cat_row:
                    stranger_cat_id = str(cat_row[0])
            except Exception:
                pass

            # 3. Thread-safe Insert into tenant's Person table
            with _stranger_insert_lock:
                new_person = Person(
                    id=new_uuid,
                    status=True,
                    created_by_id=created_by,
                    created_on=datetime.utcnow(),
                    name="",  # DB NOT NULL constraint; empty name represents Stranger
                    category_id=stranger_cat_id,
                    face_embedding=emb_pickled,
                    face_embedding_mask=None,
                    image=None,
                    is_active=False
                )
                db.add(new_person)
                db.commit()
                db.refresh(new_person)
                unique_id = new_person.unique_id

            print(Fore.GREEN + Style.BRIGHT + f"[{FILE}][{tenant_id}] [STRANGER ENROLLED] Inserted into DB: unique_id={unique_id}, uuid={new_uuid}, category_id={stranger_cat_id}")

            # 4. INSTANT RAM UPDATE: Add newly enrolled face to tenant's matrix immediately!
            face_engine.add_stranger_to_ram(tenant_id, norm_emb, unique_id)

        finally:
            db.close()

    except Exception as e:
        print(Fore.RED + Style.BRIGHT + f"[{FILE}][{tenant_id}] [STRANGER INSERT ERROR] {e}")


class StrangerTracker:
    """
    Production-grade Stranger Candidate Tracker.
    Tracks face candidates across frames and enforces the 1.5s frontal hold window before enrollment.
    """
    def __init__(self):
        self.lock = threading.Lock()
        # structure: { tenant_id: { camera_name: { track_id: candidate_dict } } }
        self.candidates: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}
        # structure: { tenant_id: { camera_name: Set[track_id] } }
        self.queued: Dict[str, Dict[str, Set[str]]] = {}

    def process_unknown_face(
        self, 
        face, 
        emb: np.ndarray, 
        track_id: str, 
        tenant_id: str, 
        camera_name: str, 
        face_engine
    ) -> Tuple[str, float]:
        """
        Processes an unknown face detection.
        Returns:
            status: 'VERIFYING' | 'CONFIRMED' | 'SKIPPED'
            elapsed: float seconds elapsed in verification window
        """
        now_ts = time.time()

        if not is_clear_frontal_face(face):
            return "SKIPPED", 0.0

        with self.lock:
            tenant_cams = self.candidates.setdefault(tenant_id, {})
            cam_candidates = tenant_cams.setdefault(camera_name, {})
            
            tenant_q = self.queued.setdefault(tenant_id, {})
            cam_queued = tenant_q.setdefault(camera_name, set())

            if track_id not in cam_candidates:
                cam_candidates[track_id] = {
                    "first_seen": now_ts,
                    "frames": 1,
                    "best_emb": emb,
                    "best_score": getattr(face, "det_score", 0.0)
                }
            else:
                cam_candidates[track_id]["frames"] += 1
                det_s = getattr(face, "det_score", 0.0)
                if det_s > cam_candidates[track_id]["best_score"]:
                    cam_candidates[track_id]["best_emb"] = emb
                    cam_candidates[track_id]["best_score"] = det_s

            candidate = cam_candidates[track_id]
            elapsed = now_ts - candidate["first_seen"]
            frame_cnt = candidate["frames"]

            # Check if 1.5s Hold Window and Minimum Frame count are satisfied
            if elapsed >= STRANGER_CONFIRM_SECONDS and frame_cnt >= STRANGER_MIN_FRAMES:
                if track_id not in cam_queued:
                    cam_queued.add(track_id)
                    best_emb = candidate["best_emb"]

                    # Check for duplicate stranger in current RAM matrix
                    is_dup, matched_name = is_duplicate_stranger(best_emb, tenant_id, face_engine)
                    if is_dup:
                        print(Fore.CYAN + f"[{FILE}][{tenant_id}] [STRANGER CACHED] Duplicate face track_id={track_id} -> matches existing '{matched_name}'")
                    else:
                        print(Fore.GREEN + Style.BRIGHT + f"[{FILE}][{tenant_id}] [STRANGER CONFIRMED] Held for {elapsed:.1f}s ({frame_cnt} frames). Enrolling track_id={track_id}...")
                        insert_stranger_async(tenant_id, best_emb, face_engine, camera_name)

                return "CONFIRMED", elapsed

            return "VERIFYING", elapsed

    def clean_camera(self, tenant_id: str, camera_name: str):
        """Cleans up candidate tracking when a camera stream stops."""
        with self.lock:
            if tenant_id in self.candidates and camera_name in self.candidates[tenant_id]:
                del self.candidates[tenant_id][camera_name]
            if tenant_id in self.queued and camera_name in self.queued[tenant_id]:
                del self.queued[tenant_id][camera_name]


stranger_tracker = StrangerTracker()
