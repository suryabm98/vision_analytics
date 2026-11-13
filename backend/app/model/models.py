import uuid
import pickle
from datetime import datetime
from sqlalchemy import (
    Column, BigInteger, Integer, String, Boolean, 
    DateTime, Date, LargeBinary, Text
)
from app.database.database import Base


# 0. TENANT TABLE (In Common DB)
class Tenant(Base):
    """Common DB 'tenant' table: maps id (UUID) to database name (e.g. 'kathir-college')."""
    __tablename__ = "tenant"

    id = Column(String(36), primary_key=True)
    name = Column(String(255), nullable=False)


# 1. USERS TABLE
class Users(Base):
    __tablename__ = "users"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    name = Column(String(200), nullable=False)
    email = Column(String(200), unique=True, nullable=False)
    password = Column(String(255), nullable=False)


# 2. PERSON TABLE (Auto-learning & Stranger registration)
class Person(Base):
    __tablename__ = "person"

    unique_id = Column(BigInteger, primary_key=True, autoincrement=True)
    id = Column(String(36), nullable=False)
    status = Column(Boolean, default=True)
    created_by_id = Column(String(36), nullable=True)
    created_on = Column(DateTime, default=datetime.utcnow)
    updated_by_id = Column(String(255), nullable=True)
    updated_on = Column(DateTime, nullable=True)
    name = Column(String(255), nullable=False, default="")
    register_number = Column(String(255), nullable=True)
    category_id = Column(String(36), nullable=True)
    face_embedding = Column(LargeBinary, nullable=True)        # Pickled normal 512D embedding
    face_embedding_mask = Column(LargeBinary, nullable=True)   # Pickled mask 512D embedding
    image = Column(Text, nullable=True)
    is_active = Column(Boolean, default=False)

    def set_embedding(self, emb):
        self.face_embedding = pickle.dumps(emb)

    def get_embedding(self):
        if self.face_embedding:
            return pickle.loads(self.face_embedding)

    def set_embedding_mask(self, emb):
        self.face_embedding_mask = pickle.dumps(emb)

    def get_embedding_mask(self):
        if self.face_embedding_mask:
            return pickle.loads(self.face_embedding_mask)


# 3. VW_PERSON_INFO (Read-only view for fast embedding extraction)
class VwPersonInfo(Base):
    __tablename__ = "vw_person_info"

    id = Column(BigInteger, primary_key=True)                 # Person unique_id
    employee_no = Column(String(255), nullable=True)
    name = Column(String(255), nullable=False)
    designation_name = Column(String(255), nullable=True)
    department_name = Column(String(255), nullable=True)
    employee_photo = Column(Text, nullable=True)
    face_embedding = Column(LargeBinary, nullable=True)
    face_embedding_mask = Column(LargeBinary, nullable=True)


# 4. SECTION TABLE
class Section(Base):
    __tablename__ = "section"

    unique_id = Column(BigInteger, primary_key=True, autoincrement=True)
    id = Column(String(36), nullable=False)
    status = Column(Boolean, default=True)
    created_by_id = Column(String(36), nullable=False)
    created_on = Column(DateTime, default=datetime.utcnow)
    updated_by_id = Column(String(255), nullable=True)
    updated_on = Column(DateTime, nullable=True)
    floor_id = Column(String(36), nullable=True)
    name = Column(String(255), nullable=False)
    description = Column(String(255), nullable=True)
    display_order = Column(BigInteger, default=0)


# 5. CAMERA TABLE
class Camera(Base):
    __tablename__ = "camera"

    unique_id = Column(BigInteger, primary_key=True, autoincrement=True)
    id = Column(String(36), nullable=False)
    status = Column(Boolean, default=True)
    created_by_id = Column(String(36), nullable=False)
    created_on = Column(DateTime, default=datetime.utcnow)
    updated_by_id = Column(String(255), nullable=True)
    updated_on = Column(DateTime, nullable=True)
    section_id = Column(String(36), nullable=True)
    nvr_id = Column(String(36), nullable=True)
    asset_no = Column(String(255), nullable=True)
    type = Column(String(255), default="IN")                   # "IN" or "OUT"
    serial_number = Column(String(255), nullable=True)
    model_name = Column(String(255), nullable=True)
    ip_address = Column(String(255), nullable=True)
    port = Column(BigInteger, default=80)
    manufacturer = Column(String(255), nullable=True)
    vendor = Column(String(255), nullable=True)
    install_date = Column(Date, nullable=True)
    last_working_on = Column(DateTime, nullable=True)
    is_working = Column(Boolean, default=False)
    instance_url = Column(String(255), nullable=True)
    stream_url = Column(String(255), nullable=True)
    roi_x = Column(BigInteger, nullable=True)
    roi_y = Column(BigInteger, nullable=True)
    roi_height = Column(BigInteger, nullable=True)
    roi_width = Column(BigInteger, nullable=True)
    display_order = Column(BigInteger, default=0)


# 6. PERSON_FACE (Attendance Records Table)
class PersonFace(Base):
    __tablename__ = "person_face"

    unique_id = Column(BigInteger, primary_key=True, autoincrement=True)
    id = Column(String(36), default=lambda: str(uuid.uuid4()))
    status = Column(Boolean, default=True)
    created_by_id = Column(String(36), default="SYSTEM")
    created_on = Column(DateTime, default=datetime.utcnow)
    updated_by_id = Column(String(255), nullable=True)
    updated_on = Column(DateTime, nullable=True)
    person_id = Column(String(36), nullable=True)              # Person UUID (Person.id)
    camera_id = Column(String(36), nullable=True)              # Camera UUID (Camera.id)
    camera_unique_id = Column(BigInteger, nullable=True)
    person_no = Column(String(255), nullable=True)             # Employee No
    image = Column(Text, nullable=True)                        # Base64 Frame with OSD Bar
    date = Column(Date, nullable=True)
    title = Column(String(255), default="IN")                  # "IN" or "OUT"
