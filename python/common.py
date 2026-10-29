
import os
import sys
import json
import time
import math
import numpy as np
from typing import Dict, Any, Tuple, Optional

SENSOR_DOWN = 0
SENSOR_FRONT = 1
SENSOR_LEFT = 2
SENSOR_RIGHT = 3
SENSOR_LIDAR2D = 4

SENSOR_NAMES = {
    SENSOR_DOWN: "depthDown",
    SENSOR_FRONT: "depthFront",
    SENSOR_LEFT: "depthLeft",
    SENSOR_RIGHT: "depthRight",
    SENSOR_LIDAR2D: "lidar2D"
}

DEFAULT_CRUISE_ALT = 5.0
DEFAULT_SPEED = 0.6
MAX_SPEED = 1.2
SIM_DT = 0.05
ARENA_MARGIN = 0.8
DRONE_BODY_FILTER_RADIUS = 0.4


def get_client(port: Optional[int] = None, host: str = "localhost"):
    try:
        from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    except ImportError:
        raise ImportError(
            "Paket 'coppeliasim_zmqremoteapi_client' belum terpasang. "
            "Pastikan modul telah terinstal di lingkungan Python."
        )

    if port is None:
        port = int(os.environ.get("ZMQ_PORT", 23000))

    client = RemoteAPIClient(host=host, port=port)
    sim = client.require("sim")
    return client, sim


def get_script_handle(sim, object_handle: int) -> int:
    try:
        return sim.getScript(sim.scripttype_childscript, object_handle)
    except Exception:
        try:
            return sim.getScript(sim.scripttype_simulation, object_handle)
        except Exception:
            return sim.getScriptAssociatedWithObject(object_handle)


def get_drone_handles(sim) -> Dict[str, Any]:
    handles: Dict[str, Any] = {}

    try:
        handles["drone"] = sim.getObject("/Quadcopter")
    except Exception as e:
        raise RuntimeError(f"Gagal menemukan objek '/Quadcopter': {e}")

    try:
        handles["base"] = sim.getObject("/Quadcopter/base")
    except Exception:
        handles["base"] = handles["drone"]

    try:
        handles["target"] = sim.getObject("/target")
    except Exception:
        try:
            handles["target"] = sim.getObject("/Quadcopter/target")
        except Exception as e:
            raise RuntimeError(f"Gagal menemukan target drone ('/target' atau '/Quadcopter/target'): {e}")

    try:
        handles["mount"] = sim.getObject("/Quadcopter/sensorMount")
    except Exception:
        handles["mount"] = None

    if handles["mount"] is not None:
        try:
            handles["sensor_script"] = get_script_handle(sim, handles["mount"])
        except Exception:
            handles["sensor_script"] = None
    else:
        handles["sensor_script"] = None

    try:
        h_path = sim.getObject("/Quadcopter/pathVisualizer")
        handles["path_script"] = get_script_handle(sim, h_path)
    except Exception:
        handles["path_script"] = None

    try:
        handles["arena"] = sim.getObject("/Arena")
    except Exception:
        handles["arena"] = None

    return handles


def unpack_points_and_sensors(
    packed_points: bytes,
    packed_sensor_ids: Optional[bytes] = None
) -> Tuple[np.ndarray, np.ndarray]:
    if not packed_points or len(packed_points) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.uint8)

    if isinstance(packed_points, (bytes, bytearray)):
        points = np.frombuffer(packed_points, dtype=np.float32).reshape(-1, 3).copy()
    elif isinstance(packed_points, (list, tuple)):
        points = np.array(packed_points, dtype=np.float32).reshape(-1, 3)
    else:
        points = np.empty((0, 3), dtype=np.float32)

    num_pts = len(points)
    if packed_sensor_ids and len(packed_sensor_ids) > 0:
        if isinstance(packed_sensor_ids, (bytes, bytearray)):
            s_ids = np.frombuffer(packed_sensor_ids, dtype=np.int32).astype(np.uint8).copy()
        elif isinstance(packed_sensor_ids, (list, tuple)):
            s_ids = np.array(packed_sensor_ids, dtype=np.uint8)
        else:
            s_ids = np.zeros(num_pts, dtype=np.uint8)
    else:
        s_ids = np.zeros(num_pts, dtype=np.uint8)

    if len(s_ids) != num_pts:
        min_len = min(len(points), len(s_ids))
        points = points[:min_len]
        s_ids = s_ids[:min_len]

    return points, s_ids


def filter_drone_body_points(
    points: np.ndarray,
    sensor_ids: np.ndarray,
    drone_pos: np.ndarray,
    radius: float = DRONE_BODY_FILTER_RADIUS
) -> Tuple[np.ndarray, np.ndarray]:
    if len(points) == 0:
        return points, sensor_ids

    dists = np.linalg.norm(points - drone_pos, axis=1)
    mask = dists >= radius
    return points[mask], sensor_ids[mask]


def save_raw_scan(
    filepath: str,
    points: np.ndarray,
    sensor_id: np.ndarray,
    t: np.ndarray,
    traj: np.ndarray,
    scan2d: np.ndarray,
    meta: Dict[str, Any]
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    points_f32 = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    sensor_id_u8 = np.asarray(sensor_id, dtype=np.uint8).ravel()
    t_f32 = np.asarray(t, dtype=np.float32).ravel()

    traj_f32 = np.asarray(traj, dtype=np.float32).reshape(-1, 8) if len(traj) > 0 else np.empty((0, 8), dtype=np.float32)
    scan2d_f32 = np.asarray(scan2d, dtype=np.float32).reshape(-1, 6) if len(scan2d) > 0 else np.empty((0, 6), dtype=np.float32)

    meta_str = json.dumps(meta, indent=2)

    tmp_path = filepath[:-4] + f"_tmp{os.getpid()}.npz"
    try:
        np.savez_compressed(
            tmp_path,
            points=points_f32,
            sensor_id=sensor_id_u8,
            t=t_f32,
            traj=traj_f32,
            scan2d=scan2d_f32,
            meta=meta_str
        )
        os.replace(tmp_path, filepath)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


def load_raw_scan(filepath: str) -> Dict[str, Any]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File raw_scan tidak ditemukan: {filepath}")

    with np.load(filepath, allow_pickle=True) as data:
        meta_raw = str(data["meta"])
        try:
            meta = json.loads(meta_raw)
        except Exception:
            meta = {"raw_meta": meta_raw}

        return {
            "points": data["points"],
            "sensor_id": data["sensor_id"],
            "t": data["t"],
            "traj": data["traj"],
            "scan2d": data["scan2d"],
            "meta": meta
        }
