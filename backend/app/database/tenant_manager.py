import os
import threading
from typing import Dict, Any, Optional
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from colorama import Fore, Style
from dotenv import load_dotenv, find_dotenv

from app.model.models import Tenant

FILE = "TENANT_MANAGER.py"

load_dotenv(find_dotenv())

DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "Login123")
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = os.getenv("DB_PORT", "3306")
COMMON_DB_NAME = os.getenv("COMMON_DB_NAME", "common_db")

COMMON_DATABASE_URL = os.getenv(
    "COMMON_DATABASE_URL", 
    f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{COMMON_DB_NAME}"
)

_lock = threading.Lock()
_common_engine = None
_CommonSessionLocal = None

_tenant_engines: Dict[str, Any] = {}
_tenant_sessionmakers: Dict[str, sessionmaker] = {}
_tenant_db_names: Dict[str, str] = {}


def init_common_db():
    """Initializes connection to the Common DB and ensures the tenant table exists."""
    global _common_engine, _CommonSessionLocal
    with _lock:
        if _common_engine is not None:
            return

        print(Fore.CYAN + f"[{FILE}] Connecting to Common DB: {COMMON_DATABASE_URL}...")
        _common_engine = create_engine(
            COMMON_DATABASE_URL,
            pool_pre_ping=True,
            pool_recycle=3600,
            pool_size=5,
            max_overflow=10
        )
        _CommonSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_common_engine)

        # Ensure tenant table exists
        with _common_engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS tenant (
                    id VARCHAR(36) PRIMARY KEY,
                    name VARCHAR(255) NOT NULL
                );
            """))
            conn.commit()
        print(Fore.GREEN + Style.BRIGHT + f"[{FILE}] [OK] Common DB and 'tenant' table ready!")


def get_common_db_session() -> Session:
    """Returns a session for Common DB."""
    if _CommonSessionLocal is None:
        init_common_db()
    return _CommonSessionLocal()


def get_tenant_database_name(tenant_id: str) -> Optional[str]:
    """Queries Common DB's 'tenant' table to find the database name for given UUID."""
    tenant_id = tenant_id.strip()
    
    # Check cache first
    with _lock:
        if tenant_id in _tenant_db_names:
            return _tenant_db_names[tenant_id]

    db = get_common_db_session()
    try:
        tenant_record = db.query(Tenant).filter(Tenant.id == tenant_id).first()
        if tenant_record and tenant_record.name:
            db_name = tenant_record.name.strip()
            with _lock:
                _tenant_db_names[tenant_id] = db_name
            return db_name
        return None
    except Exception as e:
        print(Fore.RED + f"[{FILE}] Error querying tenant table in Common DB: {e}")
        return None
    finally:
        db.close()


def get_tenant_engine(tenant_id: str):
    """Retrieves or creates connection engine for that specific tenant's database."""
    tenant_id = tenant_id.strip()

    with _lock:
        if tenant_id in _tenant_engines:
            return _tenant_engines[tenant_id]

    db_name = get_tenant_database_name(tenant_id)
    if not db_name:
        raise ValueError(f"Unknown Tenant UUID: '{tenant_id}'. Not found in Common DB 'tenant' table.")

    # Construct dynamic database URL for this specific college/tenant database
    tenant_db_url = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{db_name}"

    with _lock:
        if tenant_id in _tenant_engines:
            return _tenant_engines[tenant_id]

        print(Fore.CYAN + f"[{FILE}] Initializing connection pool for Tenant UUID '{tenant_id}' -> Database: '{db_name}'...")
        engine = create_engine(
            tenant_db_url,
            pool_pre_ping=True,
            pool_recycle=3600,
            pool_size=5,
            max_overflow=10
        )
        _tenant_engines[tenant_id] = engine
        _tenant_sessionmakers[tenant_id] = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        print(Fore.GREEN + Style.BRIGHT + f"[{FILE}] [OK] Connected to Tenant Database: '{db_name}'")
        return engine


def get_tenant_session(tenant_id: str) -> Session:
    """Returns a new SQLAlchemy Session for the given tenant's database."""
    get_tenant_engine(tenant_id)
    with _lock:
        sm = _tenant_sessionmakers[tenant_id]
    return sm()


def get_all_active_tenants() -> list:
    """Fetches all tenants from Common DB."""
    db = get_common_db_session()
    try:
        tenants = db.query(Tenant).all()
        return [{"id": t.id, "name": t.name} for t in tenants]
    finally:
        db.close()


def get_all_tenant_ids() -> list:
    """Fetches all tenant UUIDs from Common DB."""
    db = get_common_db_session()
    try:
        tenants = db.query(Tenant).all()
        return [t.id for t in tenants]
    finally:
        db.close()
