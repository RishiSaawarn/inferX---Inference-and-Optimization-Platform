import json
import logging
import time
from typing import Dict, Any, List, Optional
import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extras import RealDictCursor
from app.core.config import settings

logger = logging.getLogger(__name__)

class ModelRegistry:
    def __init__(self):
        self.conn_str = settings.app_config.postgres.url if settings.app_config else ""
        self.pool = None
        if self.conn_str:
            try:
                self.pool = ThreadedConnectionPool(1, 20, self.conn_str)
                self._init_db()
            except Exception as e:
                logger.error(f"Failed to create connection pool: {e}")

    def _get_conn(self):
        if not self.pool:
            raise RuntimeError("Database pool not initialized")
        return self.pool.getconn()

    def _put_conn(self, conn):
        if self.pool and conn:
            self.pool.putconn(conn)

    def _init_db(self):
        retries = 3
        while retries > 0:
            conn = None
            try:
                conn = self._get_conn()
                with conn.cursor() as cur:
                    cur.execute('''
                        CREATE TABLE IF NOT EXISTS models (
                            id SERIAL PRIMARY KEY,
                            model_name VARCHAR(255) NOT NULL,
                            version VARCHAR(50) NOT NULL,
                            runtime VARCHAR(50) NOT NULL,
                            device VARCHAR(20) NOT NULL,
                            precision VARCHAR(20) NOT NULL,
                            artifact_path VARCHAR(512),
                            metadata JSONB,
                            status VARCHAR(50) DEFAULT 'INACTIVE' CHECK (status IN ('ACTIVE', 'INACTIVE', 'FAILED')),
                            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                            UNIQUE(model_name, version)
                        );
                        CREATE INDEX IF NOT EXISTS idx_model_status ON models(model_name, status);
                    ''')
                conn.commit()
                return
            except Exception as e:
                logger.warning(f"Database init failed (retries left: {retries-1}): {e}")
                retries -= 1
                time.sleep(2)
            finally:
                if conn:
                    self._put_conn(conn)
        logger.error("Failed to initialize database after retries.")

    def register_model(self, model_name: str, version: str, runtime: str, 
                       device: str, precision: str, metadata: Dict[str, Any], artifact_path: str = "") -> None:
        conn = None
        try:
            conn = self._get_conn()
            with conn.cursor() as cur:
                cur.execute('''
                    INSERT INTO models (model_name, version, runtime, device, precision, metadata, status, artifact_path)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (model_name, version) DO UPDATE SET
                        runtime = EXCLUDED.runtime,
                        device = EXCLUDED.device,
                        precision = EXCLUDED.precision,
                        metadata = EXCLUDED.metadata,
                        artifact_path = EXCLUDED.artifact_path
                ''', (model_name, version, runtime, device, precision, json.dumps(metadata), 'INACTIVE', artifact_path))
            conn.commit()
            logger.info(f"Registered model {model_name}:{version}")
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                self._put_conn(conn)

    def get_active_models(self, model_name: str) -> List[Dict[str, Any]]:
        conn = None
        try:
            conn = self._get_conn()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute('''
                    SELECT * FROM models WHERE model_name = %s AND status = 'ACTIVE'
                ''', (model_name,))
                rows = cur.fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"Failed to get active models: {e}")
            raise
        finally:
            if conn:
                self._put_conn(conn)
                
    def get_all_active_models(self) -> List[Dict[str, Any]]:
        conn = None
        try:
            conn = self._get_conn()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM models WHERE status = 'ACTIVE'")
                rows = cur.fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"Failed to get all active models: {e}")
            return []
        finally:
            if conn:
                self._put_conn(conn)

    def update_status(self, model_name: str, version: str, status: str) -> None:
        conn = None
        try:
            conn = self._get_conn()
            with conn.cursor() as cur:
                cur.execute('''
                    UPDATE models SET status = %s WHERE model_name = %s AND version = %s
                ''', (status, model_name, version))
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to update status: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                self._put_conn(conn)
