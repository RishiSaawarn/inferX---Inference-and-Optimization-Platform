import json
import logging
from typing import Any

import psycopg2
from psycopg2.extras import RealDictCursor

from app.core.config import settings

logger = logging.getLogger(__name__)


class ModelRegistry:
    def __init__(self):
        self.conn_str = settings.app_config.postgres.url if settings.app_config else ""
        self._init_db()

    def _init_db(self):
        if not self.conn_str:
            return
        try:
            with psycopg2.connect(self.conn_str) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS models (
                            id SERIAL PRIMARY KEY,
                            model_name VARCHAR(255) NOT NULL,
                            version VARCHAR(50) NOT NULL,
                            runtime VARCHAR(50) NOT NULL,
                            device VARCHAR(20) NOT NULL,
                            precision VARCHAR(20) NOT NULL,
                            metadata JSONB,
                            status VARCHAR(50) DEFAULT 'INACTIVE',
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            UNIQUE(model_name, version)
                        )
                    """
                    )
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")

    def register_model(
        self,
        model_name: str,
        version: str,
        runtime: str,
        device: str,
        precision: str,
        metadata: dict[str, Any],
    ) -> None:
        if not self.conn_str:
            return
        try:
            with psycopg2.connect(self.conn_str) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO models (model_name, version, runtime, device, precision, metadata, status)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (model_name, version) DO UPDATE SET
                            runtime = EXCLUDED.runtime,
                            device = EXCLUDED.device,
                            precision = EXCLUDED.precision,
                            metadata = EXCLUDED.metadata
                    """,
                        (
                            model_name,
                            version,
                            runtime,
                            device,
                            precision,
                            json.dumps(metadata),
                            "INACTIVE",
                        ),
                    )
                conn.commit()
            logger.info(f"Registered model {model_name}:{version}")
        except Exception as e:
            logger.error(f"Failed to register model: {e}")

    def get_active_models(self, model_name: str) -> list[dict[str, Any]]:
        if not self.conn_str:
            return []
        try:
            with psycopg2.connect(self.conn_str) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        """
                        SELECT * FROM models WHERE model_name = %s AND status = 'ACTIVE'
                    """,
                        (model_name,),
                    )
                    return cur.fetchall()
        except Exception as e:
            logger.error(f"Failed to get active models: {e}")
            return []

    def update_status(self, model_name: str, version: str, status: str) -> None:
        if not self.conn_str:
            return
        try:
            with psycopg2.connect(self.conn_str) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE models SET status = %s WHERE model_name = %s AND version = %s
                    """,
                        (status, model_name, version),
                    )
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to update status: {e}")
