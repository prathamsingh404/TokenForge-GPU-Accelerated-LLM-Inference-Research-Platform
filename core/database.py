# TokenForge GPU-Accelerated LLM Inference Platform
"""
SQLite persistence layer for experiment results.

Stores experiments, per-metric measurements, and GPU snapshots.
Async interface for the FastAPI dashboard, sync interface for
benchmark scripts.
"""

import sqlite3
import uuid
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

from core.config import get_config


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    phase TEXT NOT NULL,
    model_name TEXT NOT NULL,
    batch_size INTEGER,
    quantization TEXT,
    sequence_length INTEGER,
    extra_params TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    unit TEXT,
    recorded_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS gpu_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT REFERENCES experiments(id) ON DELETE CASCADE,
    gpu_util_percent REAL,
    vram_used_mb REAL,
    vram_total_mb REAL,
    temperature_c REAL,
    power_draw_w REAL,
    clock_mhz REAL,
    recorded_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_metrics_experiment ON metrics(experiment_id);
CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_gpu_experiment ON gpu_snapshots(experiment_id);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def generate_experiment_id() -> str:
    """Short, readable experiment ID: phase prefix + 8-char hex."""
    return uuid.uuid4().hex[:12]


class ExperimentDB:
    """
    Synchronous SQLite wrapper. Each benchmark script creates one of these,
    writes results, then closes. The dashboard reads via aiosqlite separately.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or get_config().db_path
        self._conn: Optional[sqlite3.Connection] = None

    def connect(self) -> "ExperimentDB":
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA_SQL)
        return self

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self.connect()

    def __exit__(self, *exc):
        self.close()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("Database not connected. Use `with ExperimentDB() as db:`")
        return self._conn

    # --- Experiments ---

    def create_experiment(
        self,
        name: str,
        phase: str,
        model_name: str,
        batch_size: Optional[int] = None,
        quantization: Optional[str] = None,
        sequence_length: Optional[int] = None,
        extra_params: Optional[dict] = None,
    ) -> str:
        exp_id = generate_experiment_id()
        self.conn.execute(
            """INSERT INTO experiments
               (id, name, phase, model_name, batch_size, quantization,
                sequence_length, extra_params, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                exp_id, name, phase, model_name, batch_size,
                quantization, sequence_length,
                json.dumps(extra_params) if extra_params else None,
                _now_iso(),
            ),
        )
        self.conn.commit()
        return exp_id

    def record_metric(
        self,
        experiment_id: str,
        metric_name: str,
        value: float,
        unit: str = "",
    ):
        self.conn.execute(
            """INSERT INTO metrics (experiment_id, metric_name, metric_value, unit, recorded_at)
               VALUES (?, ?, ?, ?, ?)""",
            (experiment_id, metric_name, value, unit, _now_iso()),
        )
        self.conn.commit()

    def record_metrics_batch(
        self,
        experiment_id: str,
        metrics: list[tuple[str, float, str]],
    ):
        """Record multiple metrics at once. Each tuple: (name, value, unit)."""
        rows = [
            (experiment_id, name, val, unit, _now_iso())
            for name, val, unit in metrics
        ]
        self.conn.executemany(
            """INSERT INTO metrics (experiment_id, metric_name, metric_value, unit, recorded_at)
               VALUES (?, ?, ?, ?, ?)""",
            rows,
        )
        self.conn.commit()

    def record_gpu_snapshot(
        self,
        experiment_id: Optional[str],
        gpu_util: float,
        vram_used_mb: float,
        vram_total_mb: float,
        temperature: float,
        power_draw: float,
        clock_mhz: float = 0.0,
    ):
        self.conn.execute(
            """INSERT INTO gpu_snapshots
               (experiment_id, gpu_util_percent, vram_used_mb, vram_total_mb,
                temperature_c, power_draw_w, clock_mhz, recorded_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                experiment_id, gpu_util, vram_used_mb, vram_total_mb,
                temperature, power_draw, clock_mhz, _now_iso(),
            ),
        )
        self.conn.commit()

    # --- Queries ---

    def get_experiment(self, exp_id: str) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT * FROM experiments WHERE id = ?", (exp_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_experiments(
        self,
        phase: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        if phase:
            rows = self.conn.execute(
                "SELECT * FROM experiments WHERE phase = ? ORDER BY created_at DESC LIMIT ?",
                (phase, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM experiments ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_metrics(self, experiment_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM metrics WHERE experiment_id = ? ORDER BY recorded_at",
            (experiment_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_gpu_snapshots(self, experiment_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM gpu_snapshots WHERE experiment_id = ? ORDER BY recorded_at",
            (experiment_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_comparison_data(
        self,
        phase: str,
        metric_name: str,
    ) -> list[dict]:
        """Get a specific metric across all experiments in a phase, for comparison charts."""
        rows = self.conn.execute(
            """SELECT e.id, e.name, e.model_name, e.batch_size, e.quantization,
                      m.metric_value, m.unit
               FROM experiments e
               JOIN metrics m ON e.id = m.experiment_id
               WHERE e.phase = ? AND m.metric_name = ?
               ORDER BY e.created_at""",
            (phase, metric_name),
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_experiment(self, exp_id: str):
        self.conn.execute("DELETE FROM experiments WHERE id = ?", (exp_id,))
        self.conn.commit()


if __name__ == "__main__":
    # Quick sanity check
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        tmp_db = Path(f.name)

    try:
        with ExperimentDB(tmp_db) as db:
            eid = db.create_experiment(
                name="test-run",
                phase="quantization",
                model_name="gpt2",
                batch_size=4,
                quantization="fp16",
            )
            db.record_metric(eid, "throughput", 142.5, "tokens/sec")
            db.record_metric(eid, "ttft", 0.85, "seconds")

            exp = db.get_experiment(eid)
            metrics = db.get_metrics(eid)
            print(f"Experiment: {exp['name']}")
            for m in metrics:
                print(f"  {m['metric_name']}: {m['metric_value']} {m['unit']}")
            print("Database self-test passed.")
    finally:
        tmp_db.unlink(missing_ok=True)
