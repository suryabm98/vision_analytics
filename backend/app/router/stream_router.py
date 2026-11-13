from typing import Optional
from pydantic import BaseModel
from fastapi import APIRouter, Query, HTTPException, Request
from fastapi.responses import StreamingResponse
from app.database.tenant_manager import get_tenant_database_name, get_tenant_session
from app.model.models import Camera, Person
from app.core.streamer import generate_video_stream

router = APIRouter(prefix="/api/v1", tags=["Multi-Tenant Face Recognition Stream"])

class TenantVerifyRequest(BaseModel):
    tenant_id: str

@router.post("/tenant/verify")
async def verify_tenant(payload: TenantVerifyRequest, request: Request):
    """Verifies Tenant UUID against Common DB and returns connected database & cameras filtered by current instance port."""
    tenant_uuid = payload.tenant_id.strip()
    db_name = get_tenant_database_name(tenant_uuid)
    if not db_name:
        raise HTTPException(
            status_code=404, 
            detail=f"Tenant ID '{tenant_uuid}' not found in Common DB 'tenant' table."
        )

    try:
        db = get_tenant_session(tenant_uuid)
        
        # Port-based filtering (like Django instance_url)
        host = request.headers.get("host", "")
        port = str(request.url.port or "")
        
        query = db.query(Camera).filter(Camera.status == True)
        if port:
            port_cams = query.filter(Camera.instance_url.like(f"%:{port}")).all()
            cameras = port_cams if port_cams else query.all()
        else:
            cameras = query.all()

        person_count = db.query(Person).filter(Person.face_embedding.isnot(None)).count()
        cam_list = [
            {
                "id": str(c.id), 
                "asset_no": c.asset_no or c.model_name or "Camera", 
                "type": c.type or "IN",
                "instance_url": c.instance_url
            } 
            for c in cameras
        ]
        db.close()
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Database connection failed for Tenant '{tenant_uuid}' (DB: '{db_name}'): {e}"
        )

    return {
        "status": "success",
        "tenant_id": tenant_uuid,
        "database_name": db_name,
        "tenant_name": db_name.replace("_", " ").title(),
        "registered_persons": person_count,
        "instance_port": port,
        "cameras": cam_list
    }

@router.get("/stream/video-feed")
async def live_video_feed(
    request: Request,
    tenant_id: str = Query(..., description="Unique Tenant UUID from Common DB"),
    camera_id: Optional[str] = Query(None, description="Optional specific Camera ID")
):
    """
    Looks up Tenant UUID in Common DB, connects to tenant's specific database,
    filters by running instance port (e.g. 8001 / 8002), performs real-time 
    face recognition against tenant's registered faces, and returns a live MJPEG stream.
    """
    tenant_uuid = tenant_id.strip()
    db_name = get_tenant_database_name(tenant_uuid)
    if not db_name:
        raise HTTPException(
            status_code=404, 
            detail=f"Tenant ID '{tenant_uuid}' not found in Common DB 'tenant' table."
        )

    host = request.headers.get("host", "")
    port = str(request.url.port or "")

    return StreamingResponse(
        generate_video_stream(
            tenant_id=tenant_uuid, 
            camera_id=camera_id,
            current_port=port,
            current_host=host
        ),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

class WebRTCOfferRequest(BaseModel):
    sdp: str
    type: str
    tenant_id: str
    camera_id: Optional[str] = None

@router.post("/stream/webrtc-offer")
async def webrtc_offer(payload: WebRTCOfferRequest):
    """
    Ultra-low latency WebRTC streaming endpoint.
    Takes browser SDP Offer, connects to tenant's camera, runs AI recognition,
    and returns SDP Answer for sub-200ms real-time video playback.
    """
    tenant_uuid = payload.tenant_id.strip()
    db_name = get_tenant_database_name(tenant_uuid)
    if not db_name:
        raise HTTPException(
            status_code=404,
            detail=f"Tenant ID '{tenant_uuid}' not found in Common DB 'tenant' table."
        )

    try:
        from app.core.webrtc_streamer import handle_webrtc_offer
        answer = await handle_webrtc_offer(
            offer_sdp=payload.sdp,
            offer_type=payload.type,
            tenant_id=tenant_uuid,
            camera_id=payload.camera_id
        )
        return answer
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"WebRTC negotiation failed: {e}")

