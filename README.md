# Vision Analytics — Enterprise Multi-Tenant AI Platform

Vision Analytics is an enterprise-grade AI video analytics, facial recognition, and live streaming platform designed for high-density camera deployments (schools, universities, corporate offices, industrial campuses).

---

## 🚀 Key Features

### 1. ⚡ TensorRT (FP16) Acceleration
- Integrated **TensorRT Execution Provider** with FP16 half-precision and disk-based engine caching (`./trt_cache`).
- 3x faster inference throughput, 50% lower VRAM usage, with automatic graceful fallback to CUDA and CPU.

### 2. 🚨 Camera Offline Alert System
- Real-time heartbeat tracking for RTSP streams.
- Automatically marks `Camera.is_working = False` and updates `last_working_on` in tenant database upon disconnection.
- Multi-channel background alert dispatcher (Console Banner, Webhook JSON payload to WhatsApp/Telegram/Slack/Email).

### 3. 📡 WebRTC Ultra-Low Latency Streaming
- Built-in **aiortc WebRTC pipeline** (`/api/v1/stream/webrtc-offer`).
- Delivers real-time AI annotated video feeds to web browsers and mobile apps with **sub-200ms latency**.
- Dual-mode streaming: Supports both standard **MJPEG** and high-performance **WebRTC**.

### 4. 🏢 Central Common DB Multi-Tenant Architecture
- Backed by central `common_db.tenant` mapping UUIDs to dedicated client MySQL databases.
- Isolated in-memory embedding matrices per tenant (Zero cross-tenant data leakage).
- Automatic port-based camera filtering (Django parity).

### 5. 👤 Stranger / Unique Customer Auto-Enrollment
- **Frontal Pose Check**: Validates `pitch`, `yaw`, and `roll` within `±28°` and eye distance `>= 22px`.
- **1.5s Hold Window**: Enforces time-delay verification (`VERIFYING` orange box -> `CONFIRMED` red box).
- **Duplicate Prevention**: Cosine similarity check (`sim >= 0.45`) prevents duplicate stranger registrations.
- **Instant RAM Matrix Update**: Dynamically appends newly enrolled strangers to the in-memory matrix without server restarts.

---

## 🛠️ Quick Start

### 1. Activate Environment & Run Backend
```powershell
cd "D:\BM\Projects\All worked projects folder\Github_Projects\vision_analytics"
..\vision_streaming\streamingscript\Scripts\activate
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

### 2. Open Web Portal
Open `Frontend/index.html` in your web browser, enter your Tenant UUID, select your streaming protocol (MJPEG or WebRTC), and start streaming!

---

## 📦 Git Repository
- Remote: `https://github.com/suryabm98/vision_analytics.git`
- Branch: `main`
