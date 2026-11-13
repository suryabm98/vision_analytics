import os
import pickle
import threading
import numpy as np
import torch
from typing import Optional, Dict, Any
from insightface.app import FaceAnalysis
from colorama import Fore, Style

from app.database.tenant_manager import get_tenant_session, get_all_tenant_ids
from app.model.models import Person, VwPersonInfo

FILE = "FACE_ENGINE.py"

class FaceEngine:
    def __init__(self):
        self.app = None
        self.tenant_data: Dict[str, Dict[str, Any]] = {}  # {tenant_id: {"matrix": ndarray, "ids": list}}
        self.lock = threading.Lock()
        
        # Thresholds from .env with fallback
        self.SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.42))
        self.MIN_MATCH_MARGIN = float(os.getenv("MIN_MATCH_MARGIN", 0.05))
        self.MAX_FACES = int(os.getenv("MAX_FACES", 10))

    def load_model(self):
        """Loads InsightFace (buffalo_l) once across all tenants with TensorRT (FP16) support and CUDA/CPU fallback."""
        enable_trt = os.getenv("ENABLE_TENSORRT", "true").lower() in ("true", "1", "yes")
        has_cuda = torch.cuda.is_available()

        providers = []
        provider_options = []

        if enable_trt and has_cuda:
            trt_cache_dir = os.path.abspath(os.getenv("TRT_CACHE_DIR", "./trt_cache"))
            os.makedirs(trt_cache_dir, exist_ok=True)
            print(Fore.CYAN + Style.BRIGHT + f"[{FILE}] [TENSORRT] Configuring TensorRT (FP16) provider... (Cache: {trt_cache_dir})")
            
            providers.append('TensorrtExecutionProvider')
            provider_options.append({
                'device_id': 0,
                'trt_fp16_enable': True,
                'trt_max_workspace_size': int(os.getenv("TRT_MAX_WORKSPACE", 2147483648)),
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': trt_cache_dir,
                'trt_dump_subgraphs': False
            })

        if has_cuda:
            providers.append('CUDAExecutionProvider')
            provider_options.append({'device_id': 0})

        providers.append('CPUExecutionProvider')
        provider_options.append({})

        print(Fore.YELLOW + Style.BRIGHT + f"[{FILE}] Loading InsightFace (buffalo_l)... Priority Providers: {providers}")
        
        try:
            self.app = FaceAnalysis(name='buffalo_l', providers=providers, provider_options=provider_options)
            self.app.prepare(ctx_id=0 if has_cuda else -1, det_size=(480, 480))
            print(Fore.GREEN + Style.BRIGHT + f"[{FILE}] [OK] InsightFace (buffalo_l) loaded with TensorRT/CUDA acceleration!")
        except Exception as e:
            print(Fore.YELLOW + f"[{FILE}] [WARN] TensorRT/CUDA initialization fallback: {e}")
            fallback_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if has_cuda else ['CPUExecutionProvider']
            self.app = FaceAnalysis(name='buffalo_l', providers=fallback_providers)
            self.app.prepare(ctx_id=0 if has_cuda else -1, det_size=(480, 480))
            print(Fore.GREEN + Style.BRIGHT + f"[{FILE}] [OK] InsightFace loaded via Fallback Providers: {fallback_providers}")

    def refresh_embeddings(self, tenant_id: Optional[str] = None):
        """Fetches and unpickles embeddings for one tenant or all tenants."""
        tenant_ids = [tenant_id] if tenant_id else get_all_tenant_ids()

        for tid in tenant_ids:
            try:
                db = get_tenant_session(tid)
            except Exception as e:
                print(Fore.RED + f"[{FILE}] Error connecting to Tenant '{tid}' DB: {e}")
                continue

            try:
                records = db.query(Person).filter(
                    (Person.face_embedding.isnot(None)) | (Person.face_embedding_mask.isnot(None))
                ).all()
                embs, ids = [], []

                for p in records:
                    display_name = p.name.strip() if (p.name and p.name.strip()) else f"Stranger-[{p.unique_id}]"

                    # Normal embedding
                    if p.face_embedding:
                        try:
                            stored = pickle.loads(p.face_embedding)
                            stored_list = stored if isinstance(stored, list) else [stored]
                            for stored_emb in stored_list:
                                norm = np.linalg.norm(stored_emb)
                                if norm > 0:
                                    embs.append(stored_emb / norm)
                                    ids.append((p.unique_id, display_name, "normal"))
                        except Exception as e:
                            print(f"[{FILE}][{tid}] Normal embed error: {e}")

                    # Mask embedding
                    if p.face_embedding_mask:
                        try:
                            stored = pickle.loads(p.face_embedding_mask)
                            stored_list = stored if isinstance(stored, list) else [stored]
                            for stored_emb in stored_list:
                                norm = np.linalg.norm(stored_emb)
                                if norm > 0:
                                    embs.append(stored_emb / norm)
                                    ids.append((p.unique_id, display_name, "mask"))
                        except Exception as e:
                            print(f"[{FILE}][{tid}] Mask embed error: {e}")

                matrix = np.array(embs, dtype=np.float32) if embs else None
                with self.lock:
                    self.tenant_data[tid] = {
                        "matrix": matrix,
                        "ids": ids
                    }

                print(Fore.GREEN + Style.BRIGHT + f"[{FILE}] [EMBED REFRESH] Tenant '{tid}': Loaded {len(ids)} embeddings. Matrix shape: {matrix.shape if matrix is not None else 'None'}")
            except Exception as e:
                print(Fore.RED + f"[{FILE}] [EMBED REFRESH ERROR for {tid}] {e}")
            finally:
                db.close()

    def add_stranger_to_ram(self, tenant_id: str, norm_emb: np.ndarray, unique_id: int):
        """Instantly appends a newly enrolled stranger to the tenant's in-memory embedding matrix."""
        with self.lock:
            tenant_info = self.tenant_data.setdefault(tenant_id, {"matrix": None, "ids": []})
            display_name = f"Stranger-[{unique_id}]"
            norm_emb_reshaped = norm_emb.reshape(1, -1).astype(np.float32)

            if tenant_info["matrix"] is None or len(tenant_info["matrix"]) == 0:
                tenant_info["matrix"] = norm_emb_reshaped
                tenant_info["ids"] = [(unique_id, display_name, "normal")]
            else:
                tenant_info["matrix"] = np.vstack([tenant_info["matrix"], norm_emb_reshaped])
                tenant_info["ids"].append((unique_id, display_name, "normal"))

            print(Fore.CYAN + Style.BRIGHT + f"[{FILE}][{tenant_id}] [INSTANT RAM UPDATE] '{display_name}' added to RAM matrix (Total: {len(tenant_info['ids'])})")

    def start_background_sync(self, interval=30):
        """Periodically refreshes embeddings for all registered tenants."""
        def sync_worker():
            import time
            while True:
                time.sleep(interval)
                self.refresh_embeddings()
        threading.Thread(target=sync_worker, daemon=True).start()

    def recognize_face(self, current_emb, tenant_id: str):
        """Matches embedding against the specific tenant's matrix."""
        with self.lock:
            tenant_info = self.tenant_data.get(tenant_id)

        # Lazy load tenant embeddings if not yet in memory
        if not tenant_info:
            self.refresh_embeddings(tenant_id)
            with self.lock:
                tenant_info = self.tenant_data.get(tenant_id)

        if not tenant_info or tenant_info["matrix"] is None or len(tenant_info["matrix"]) == 0:
            return None, 0.0, False, None

        matrix = tenant_info["matrix"]
        ids = tenant_info["ids"]

        sims = matrix @ current_emb

        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        person_name = ids[best_idx][1] if best_idx < len(ids) else "N/A"

        # Margin check logic
        if len(sims) >= 2:
            top2_idx = np.argpartition(sims, -2)[-2:]
            top2_sorted = top2_idx[np.argsort(sims[top2_idx])[::-1]]
            best_idx, second_idx = top2_sorted[0], top2_sorted[1]
            margin = float(sims[best_idx] - sims[second_idx])
        else:
            best_idx = int(np.argmax(sims))
            margin = 1.0

        best_sim = float(sims[best_idx])

        # Dynamic threshold from .env
        threshold = float(os.getenv("SIMILARITY_THRESHOLD", self.SIMILARITY_THRESHOLD))
        margin_threshold = float(os.getenv("MIN_MATCH_MARGIN", self.MIN_MATCH_MARGIN))

        if best_sim >= threshold:
            confident = margin >= margin_threshold
            person_pk, person_name, embed_type = ids[best_idx]
            return (person_pk, person_name), best_sim, confident, embed_type

        return None, best_sim, False, None

face_engine = FaceEngine()
