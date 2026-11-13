import asyncio
import fractions
import time
import cv2
import numpy as np
from typing import Optional, Set
from colorama import Fore, Style
from av import VideoFrame

from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaRelay

from app.core.tracker import LightweightFaceTracker, letterbox_resize
from app.core.face_engine import face_engine
from app.core.stranger import stranger_tracker
from app.core.camera_alert import camera_alert_manager
from app.database.tenant_manager import get_tenant_session
from app.model.models import Camera

FILE = "WEBRTC_STREAMER.py"

relay = MediaRelay()
active_pcs: Set[RTCPeerConnection] = set()


class AIAnnotatedVideoTrack(VideoStreamTrack):
    """
    WebRTC Video Stream Track that performs real-time face detection,
    bounding box rendering, and stranger tracking with sub-200ms latency.
    """
    kind = "video"

    def __init__(self, tenant_id: str, camera_id: Optional[str] = None):
        super().__init__()
        self.tenant_id = tenant_id
        self.camera_id = camera_id
        self.cap = None
        self.camera_name = "CCTV"
        self.stream_url = None
        self.tracker = LightweightFaceTracker(max_disappeared=5, iou_threshold=0.3)
        self.recognition_cache = {}
        self.pts = 0
        self._init_camera()

    def _init_camera(self):
        db = get_tenant_session(self.tenant_id)
        try:
            if self.camera_id:
                cam = db.query(Camera).filter(
                    (Camera.id == self.camera_id) | 
                    (Camera.asset_no == self.camera_id) |
                    (Camera.unique_id == int(self.camera_id) if self.camera_id.isdigit() else False)
                ).first()
            else:
                cam = db.query(Camera).filter(Camera.status == True).first()

            if cam and cam.stream_url:
                self.stream_url = cam.stream_url
                self.camera_name = cam.asset_no or cam.model_name or "CCTV"
        finally:
            db.close()

        if self.stream_url:
            source = int(self.stream_url) if self.stream_url.isdigit() else self.stream_url
            self.cap = cv2.VideoCapture(source)
            if self.cap.isOpened():
                camera_alert_manager.record_frame_success(self.tenant_id, str(self.camera_id or ""), self.camera_name, self.stream_url)
                print(Fore.GREEN + Style.BRIGHT + f"[{FILE}][{self.tenant_id}] WebRTC Track opened camera: {self.camera_name} ({self.stream_url})")

    async def recv(self) -> VideoFrame:
        """Called by WebRTC stack to fetch each video frame."""
        self.pts += 1

        if self.cap is None or not self.cap.isOpened():
            # Return blank black frame if camera not ready
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "Camera Connecting / Offline...", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            frame = VideoFrame.from_ndarray(blank, format="bgr24")
            frame.pts = self.pts
            frame.time_base = fractions.Fraction(1, 30)
            return frame

        # Read camera frame in executor to avoid blocking asyncio event loop
        loop = asyncio.get_event_loop()
        ret, frame = await loop.run_in_executor(None, self.cap.read)

        if not ret:
            camera_alert_manager.record_frame_failure(self.tenant_id, str(self.camera_id or ""), self.camera_name, self.stream_url)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            await asyncio.sleep(0.033)
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "Reconnecting Stream...", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            av_frame = VideoFrame.from_ndarray(blank, format="bgr24")
            av_frame.pts = self.pts
            av_frame.time_base = fractions.Fraction(1, 30)
            return av_frame

        camera_alert_manager.record_frame_success(self.tenant_id, str(self.camera_id or ""), self.camera_name, self.stream_url)

        # Run Face Detection & Recognition
        frame_resized = letterbox_resize(frame, 1280, 960)
        try:
            faces = face_engine.app.get(frame_resized)
        except Exception:
            faces = []

        detected_boxes, valid_faces = [], []
        for face in faces:
            if len(detected_boxes) >= face_engine.MAX_FACES:
                break
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            if (x2 - x1) < 25 or (y2 - y1) < 25:
                continue
            detected_boxes.append([x1, y1, x2, y2])
            valid_faces.append(face)

        tracked_results = self.tracker.update(detected_boxes)

        for idx, face in enumerate(valid_faces):
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            track_id = tracked_results.get(idx, "ID-?")

            # Cache / Recognition
            cached = self.recognition_cache.get(track_id)
            if cached and cached[0] is not None:
                match, conf, confident, embed_type = cached
                emb = face.embedding
            else:
                emb = face.embedding
                norm = np.linalg.norm(emb)
                if norm <= 0:
                    continue
                emb = emb / norm
                match, conf, confident, embed_type = face_engine.recognize_face(emb, tenant_id=self.tenant_id)
                if match is not None:
                    self.recognition_cache[track_id] = (match, conf, confident, embed_type)
                else:
                    match, conf, confident, embed_type = None, conf, False, None

            if match:
                person_pk, person_name = match[0], match[1]
                color = (0, 180, 0) if confident else (0, 165, 255)
                label = f"{person_name} {conf * 100:.1f}%" if confident else f"[{track_id}] Low-conf {person_name}"
            else:
                status, elapsed = stranger_tracker.process_unknown_face(
                    face=face, emb=emb, track_id=track_id, tenant_id=self.tenant_id,
                    camera_name=self.camera_name, face_engine=face_engine
                )
                if status == "VERIFYING":
                    color, label = (0, 165, 255), f"[{track_id}] Verifying... ({elapsed:.1f}s)"
                elif status == "CONFIRMED":
                    color, label = (0, 0, 255), f"[{track_id}] Stranger"
                    self.recognition_cache[track_id] = ((None, "Stranger"), 1.0, True, "normal")
                else:
                    color, label = (0, 0, 255), f"[{track_id}] Unknown"

            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_resized, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Convert OpenCV BGR to PyAV VideoFrame for WebRTC transmission
        av_frame = VideoFrame.from_ndarray(frame_resized, format="bgr24")
        av_frame.pts = self.pts
        av_frame.time_base = fractions.Fraction(1, 30)
        return av_frame

    def stop(self):
        super().stop()
        if self.cap:
            self.cap.release()
            self.cap = None


async def handle_webrtc_offer(offer_sdp: str, offer_type: str, tenant_id: str, camera_id: Optional[str] = None) -> dict:
    """Handles an incoming WebRTC SDP Offer from the browser and returns an SDP Answer."""
    pc = RTCPeerConnection()
    active_pcs.add(pc)

    video_track = AIAnnotatedVideoTrack(tenant_id=tenant_id, camera_id=camera_id)
    pc.addTrack(video_track)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print(Fore.CYAN + f"[{FILE}][{tenant_id}] WebRTC Connection State: {pc.connectionState}")
        if pc.connectionState in ("failed", "closed"):
            video_track.stop()
            await pc.close()
            active_pcs.discard(pc)

    offer = RTCSessionDescription(sdp=offer_sdp, type=offer_type)
    await pc.setRemoteDescription(offer)

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    }
