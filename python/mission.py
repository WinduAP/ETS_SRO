
import os
import sys
import time
import json
import math
import select
import termios
import tty
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)


def proj(*parts):
    return os.path.join(ROOT_DIR, *parts)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from common import (
    get_client,
    get_drone_handles,
    unpack_points_and_sensors,
    filter_drone_body_points,
    save_raw_scan,
    SENSOR_NAMES,
    SENSOR_DOWN,
    SENSOR_FRONT,
    SENSOR_LEFT,
    SENSOR_RIGHT,
    SENSOR_LIDAR2D,
    DEFAULT_CRUISE_ALT,
    DEFAULT_SPEED,
    MAX_SPEED,
    SIM_DT,
    ARENA_MARGIN,
    DRONE_BODY_FILTER_RADIUS
)
from live_view import LiveVisualizer


def calculate_lawnmower_spacing(alt: float, fov_deg: float = 60.0, overlap: float = 0.40) -> Tuple[float, float]:
    fov_rad = math.radians(fov_deg)
    footprint_w = 2.0 * alt * math.tan(fov_rad / 2.0)
    spacing = footprint_w * (1.0 - overlap)
    return footprint_w, spacing


def plan_lawnmower(
    bounds_x: Tuple[float, float] = (-4.8, 4.8),
    bounds_y: Tuple[float, float] = (-4.8, 4.8),
    alt: float = DEFAULT_CRUISE_ALT,
    spacing: Optional[float] = None,
    overlap: float = 0.40
) -> List[Tuple[float, float, float, Optional[float]]]:
    footprint_w, calc_spacing = calculate_lawnmower_spacing(alt, fov_deg=60.0, overlap=overlap)
    if spacing is None:
        spacing = max(1.2, min(calc_spacing, 2.5))

    print(f"[Planner-Lawnmower] Alt: {alt:.1f}m, FOV: 60°, Footprint: {footprint_w:.2f}m, "
          f"Overlap: {overlap*100:.0f}%, Spacing: {spacing:.2f}m")

    x_min, x_max = bounds_x
    y_min, y_max = bounds_y

    y_coords = np.arange(y_min, y_max + 0.1, spacing)
    if y_coords[-1] < y_max:
        y_coords = np.append(y_coords, y_max)

    waypoints = []
    flip = False
    for y in y_coords:
        if not flip:
            waypoints.append((x_min, y, alt, None))
            waypoints.append((x_max, y, alt, None))
        else:
            waypoints.append((x_max, y, alt, None))
            waypoints.append((x_min, y, alt, None))
        flip = not flip

    return waypoints


def plan_spiral(
    center: Tuple[float, float] = (0.0, 0.0),
    max_radius: float = 4.8,
    alt: float = DEFAULT_CRUISE_ALT,
    num_loops: int = 5,
    num_points: int = 80
) -> List[Tuple[float, float, float, Optional[float]]]:
    waypoints = []
    thetas = np.linspace(0, num_loops * 2.0 * math.pi, num_points)
    radii = np.linspace(0.8, max_radius, num_points)

    for r, th in zip(radii, thetas):
        x = center[0] + r * math.cos(th)
        y = center[1] + r * math.sin(th)
        waypoints.append((x, y, alt, None))

    return waypoints


def plan_orbit(
    center: Tuple[float, float] = (0.0, 0.0),
    radius: float = 2.4,
    alt: float = 2.8,
    num_points: int = 36,
    clockwise: bool = False
) -> List[Tuple[float, float, float, Optional[float]]]:
    waypoints = []
    angles = np.linspace(0, 2.0 * math.pi, num_points, endpoint=False)
    if clockwise:
        angles = angles[::-1]

    for th in angles:
        x = center[0] + radius * math.cos(th)
        y = center[1] + radius * math.sin(th)
        yaw_to_center = math.atan2(center[1] - y, center[0] - x)
        waypoints.append((x, y, alt, yaw_to_center))

    waypoints.append(waypoints[0])
    return waypoints


def plan_smart(
    ground_truth_path: Optional[str] = None,
    cruise_alt: float = 4.8,
    bounds_x: Tuple[float, float] = (-4.8, 4.8),
    bounds_y: Tuple[float, float] = (-4.8, 4.8)
) -> Tuple[List[Tuple[float, float, float, Optional[float]]], Dict[str, Any]]:
    helipad = (-4.5, -4.5)
    ring_r, ring_alt = 5.4, 2.8
    wps: List[Tuple[float, float, float, Optional[float]]] = []
    wps.append((helipad[0], helipad[1], cruise_alt, 0.0))
    wps.extend(plan_lawnmower(bounds_x=bounds_x, bounds_y=bounds_y, alt=cruise_alt, spacing=2.0, overlap=0.45))
    wps.append((2.2, 1.3, 3.2, math.pi))
    wps.append((1.5, 1.3, 1.85, math.pi))
    wps.append((0.4, 1.35, 1.85, math.pi * 0.95))
    wps.append((-0.6, 1.45, 1.85, math.pi * 1.05))
    wps.append((-1.8, 1.35, 1.85, math.pi * 0.95))
    wps.append((-3.0, 1.30, 1.85, -math.pi * 0.85))
    wps.append((-3.5, 1.30, 2.4, -math.pi * 0.70))
    wps.append((ring_r, 0.0, cruise_alt, math.pi))
    wps.append((ring_r, 0.0, ring_alt, math.pi))
    wps.extend(plan_orbit((0.0, 0.0), radius=ring_r, alt=ring_alt, num_points=48))
    wps.append((ring_r, 0.0, cruise_alt, math.pi))
    wps.append((helipad[0], helipad[1], cruise_alt, 0.0))
    wps.append((helipad[0], helipad[1], 1.0, 0.0))
    wps.append((helipad[0], helipad[1], 0.45, 0.0))
    meta = {
        "plan_type": "smart",
        "description": "lawnmower 4.8 m + orbit keliling r=5.4 m pada 2.8 m (yaw ke pusat) + landing",
        "cruise_alt": cruise_alt,
        "ring_radius": ring_r,
        "ring_alt": ring_alt,
        "num_waypoints": len(wps),
    }
    return wps, meta


def interpolate_trajectory(
    waypoints: List[Tuple[float, float, float, Optional[float]]],
    speed: float = DEFAULT_SPEED,
    dt: float = SIM_DT
) -> np.ndarray:
    if len(waypoints) == 0:
        return np.empty((0, 4), dtype=np.float32)

    interpolated = []
    curr_pos = np.array(waypoints[0][:3], dtype=np.float64)
    curr_yaw = waypoints[0][3] if waypoints[0][3] is not None else 0.0
    interpolated.append([curr_pos[0], curr_pos[1], curr_pos[2], curr_yaw])

    for i in range(len(waypoints) - 1):
        p_start = np.array(waypoints[i][:3], dtype=np.float64)
        p_end = np.array(waypoints[i + 1][:3], dtype=np.float64)
        specified_yaw_end = waypoints[i + 1][3]

        segment_vec = p_end - p_start
        dist = float(np.linalg.norm(segment_vec))

        if dist < 1e-4:
            continue

        step_dist = max(0.01, speed * dt)
        num_steps = max(1, int(math.ceil(dist / step_dist)))

        if specified_yaw_end is not None:
            target_yaw = specified_yaw_end
        else:
            horiz_dist = math.hypot(segment_vec[0], segment_vec[1])
            if horiz_dist > 0.05:
                target_yaw = math.atan2(segment_vec[1], segment_vec[0])
            else:
                target_yaw = curr_yaw

        yaw_diff = (target_yaw - curr_yaw + math.pi) % (2.0 * math.pi) - math.pi

        for s in range(1, num_steps + 1):
            fraction = s / num_steps
            interp_pos = p_start + fraction * segment_vec
            interp_yaw = curr_yaw + fraction * yaw_diff
            interpolated.append([interp_pos[0], interp_pos[1], interp_pos[2], interp_yaw])

        curr_pos = p_end
        curr_yaw = target_yaw

    return np.array(interpolated, dtype=np.float32)


def query_sensors_direct(sim, handles: Dict[str, Any]) -> Tuple[bytes, bytes]:
    pts_world = []
    s_ids = []

    sensors_to_check = [
        ("/Quadcopter/sensorMount/depthDown", SENSOR_DOWN, 60.0),
        ("/Quadcopter/sensorMount/depthFront", SENSOR_FRONT, 60.0)
    ]

    for sensor_path, sid, fov_deg in sensors_to_check:
        try:
            h_s = sim.getObject(sensor_path)
            res, depth_packed = sim.getVisionSensorDepth(h_s, 1)
            if not depth_packed or len(depth_packed) == 0:
                continue

            matrix = sim.getObjectMatrix(h_s, -1)
            R = np.array([
                [matrix[0], matrix[1], matrix[2]],
                [matrix[4], matrix[5], matrix[6]],
                [matrix[8], matrix[9], matrix[10]]
            ])
            T = np.array([matrix[3], matrix[7], matrix[11]])

            depth_img = np.frombuffer(depth_packed, dtype=np.float32).reshape(res[1], res[0])
            h, w = res[1], res[0]
            fov_rad = math.radians(fov_deg)
            fx = (w / 2.0) / math.tan(fov_rad / 2.0)
            fy = (h / 2.0) / math.tan(fov_rad / 2.0)

            u_coords, v_coords = np.meshgrid(np.arange(0, w, 3), np.arange(0, h, 3))
            d_vals = depth_img[v_coords, u_coords].ravel()

            valid = (d_vals > 0.1) & (d_vals < 7.9)
            if not np.any(valid):
                continue

            u_val = u_coords.ravel()[valid]
            v_val = v_coords.ravel()[valid]
            d_val = d_vals[valid]

            xc = (u_val - w / 2.0) * d_val / fx
            yc = (v_val - h / 2.0) * d_val / fy
            zc = d_val

            pts_cam = np.column_stack([xc, yc, zc])
            pts_w = (R @ pts_cam.T).T + T

            pts_world.append(pts_w)
            s_ids.append(np.full(len(pts_w), sid, dtype=np.int32))
        except Exception:
            continue

    if len(pts_world) > 0:
        all_pts = np.vstack(pts_world).astype(np.float32).ravel()
        all_sids = np.concatenate(s_ids).astype(np.int32)
        packed_pts = all_pts.tobytes()
        packed_sids = all_sids.tobytes()
        return packed_pts, packed_sids

    return b"", b""


def run_manual_control(
    sim, client, handles: Dict[str, Any],
    out_path: str,
    scan_every: int = 2
):
    print()
    print("MODE KONTROL MANUAL DRONE (Terminal Teleoperation)")
    print("Kontrol Keyboard:")
    print("  [W] / [S]     : Maju / Mundur (+X / -X)")
    print("  [A] / [D]     : Geser Kiri / Kanan (+Y / -Y)")
    print("  [R] / [F]     : Naik / Turun (+Z / -Z)")
    print("  [Q] / [E]     : Putar Yaw Kiri / Kanan")
    print("  [SPACE]       : Hover / Rem di posisi saat ini")
    print("  [X] atau [ESC]: Simpan data dan keluar")
    print()

    h_drone = handles.get("base", handles["drone"])
    h_target = handles["target"]

    client.setStepping(True)
    sim.startSimulation()

    init_pos = sim.getObjectPosition(h_drone, -1)
    target_x = float(init_pos[0])
    target_y = float(init_pos[1])
    target_z = max(float(init_pos[2]), 1.5)
    target_yaw = 0.0

    sim.setObjectPosition(h_target, [target_x, target_y, target_z], -1)
    sim.setObjectOrientation(h_target, [0.0, 0.0, target_yaw], -1)

    accumulated_points = []
    accumulated_sensor_ids = []
    accumulated_t = []
    trajectory_records = []
    scan2d_records = []

    step_counter = 0
    start_sim_time = sim.getSimulationTime()

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        running = True
        step_size = 0.12
        yaw_step = math.radians(10.0)

        while running:
            rlist, _, _ = select.select([sys.stdin], [], [], 0.03)
            if rlist:
                ch = sys.stdin.read(1)
                if ch in ['x', 'X', '\x1b']:
                    running = False
                elif ch in ['w', 'W']:
                    target_x += step_size * math.cos(target_yaw)
                    target_y += step_size * math.sin(target_yaw)
                elif ch in ['s', 'S']:
                    target_x -= step_size * math.cos(target_yaw)
                    target_y -= step_size * math.sin(target_yaw)
                elif ch in ['a', 'A']:
                    target_x -= step_size * math.sin(target_yaw)
                    target_y += step_size * math.cos(target_yaw)
                elif ch in ['d', 'D']:
                    target_x += step_size * math.sin(target_yaw)
                    target_y -= step_size * math.cos(target_yaw)
                elif ch in ['r', 'R']:
                    target_z = min(6.0, target_z + step_size)
                elif ch in ['f', 'F']:
                    target_z = max(0.4, target_z - step_size)
                elif ch in ['q', 'Q']:
                    target_yaw = (target_yaw + yaw_step) % (2.0 * math.pi)
                elif ch in ['e', 'E']:
                    target_yaw = (target_yaw - yaw_step) % (2.0 * math.pi)
                elif ch == ' ':
                    actual = sim.getObjectPosition(h_drone, -1)
                    target_x, target_y, target_z = actual[0], actual[1], actual[2]

            sim.setObjectPosition(h_target, [target_x, target_y, target_z], -1)
            sim.setObjectOrientation(h_target, [0.0, 0.0, target_yaw], -1)

            client.step()
            sim_time = sim.getSimulationTime()
            step_counter += 1

            drone_pos = sim.getObjectPosition(h_drone, -1)
            drone_ori = sim.getObjectOrientation(h_drone, -1)
            target_dist = math.sqrt(
                (drone_pos[0] - target_x)**2 +
                (drone_pos[1] - target_y)**2 +
                (drone_pos[2] - target_z)**2
            )
            trajectory_records.append([
                sim_time, drone_pos[0], drone_pos[1], drone_pos[2],
                drone_ori[0], drone_ori[1], drone_ori[2], target_dist
            ])

            if step_counter % scan_every == 0:
                packed_pts = None
                packed_sids = None
                if handles.get("sensor_script") is not None:
                    try:
                        packed_pts, packed_sids = sim.callScriptFunction(
                            "getNewPoints", handles["sensor_script"]
                        )
                    except Exception:
                        pass

                if packed_pts is None or len(packed_pts) == 0:
                    packed_pts, packed_sids = query_sensors_direct(sim, handles)

                if packed_pts and len(packed_pts) > 0:
                    pts, sids = unpack_points_and_sensors(packed_pts, packed_sids)
                    pts, sids = filter_drone_body_points(pts, sids, np.array(drone_pos))
                    if len(pts) > 0:
                        accumulated_points.append(pts)
                        accumulated_sensor_ids.append(sids)
                        accumulated_t.append(np.full(len(pts), sim_time, dtype=np.float32))

            if step_counter % 10 == 0:
                total_pts = sum(len(p) for p in accumulated_points)
                sys.stdout.write(
                    f"\r[MANUAL] SimTime: {sim_time:.1f}s | "
                    f"Pose: ({drone_pos[0]:.2f}, {drone_pos[1]:.2f}, {drone_pos[2]:.2f}m) | "
                    f"Target: ({target_x:.2f}, {target_y:.2f}, {target_z:.2f}m) | "
                    f"Points: {total_pts:,}   "
                )
                sys.stdout.flush()

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        print("\n\n[MANUAL] Menghentikan simulasi...")
        sim.stopSimulation()
        while sim.getSimulationState() != sim.simulation_stopped:
            client.step()

    finalize_and_save(
        out_path=out_path,
        points_list=accumulated_points,
        sensor_ids_list=accumulated_sensor_ids,
        t_list=accumulated_t,
        traj_records=trajectory_records,
        scan2d_records=scan2d_records,
        meta_info={"plan_type": "manual", "duration": sim_time - start_sim_time}
    )


def run_mission(
    scene_path: Optional[str] = None,
    plan_type: str = "smart",
    cruise_alt: float = DEFAULT_CRUISE_ALT,
    speed: float = DEFAULT_SPEED,
    scan_every: int = 2,
    out_path: str = proj("data", "raw_scan.npz"),
    port: Optional[int] = None,
    live: bool = False,
    record_path: Optional[str] = None,
    on_step=None
) -> Dict[str, Any]:
    print()
    print("UTS 3D Mapping Drone - misi akuisisi")
    print(f"Mode Plan       : {plan_type}")
    print(f"Ketinggian      : {cruise_alt} m")
    print(f"Kecepatan       : {speed} m/s")
    print(f"Scan Tiap       : {scan_every} step ({scan_every * SIM_DT:.2f} s)")
    print(f"Output File     : {out_path}")
    print(f"Live Dashboard  : {'Aktif' if live or record_path else 'Mati'}")
    print()

    client, sim = get_client(port=port)
    print(f"[Koneksi] Terhubung ke CoppeliaSim (RPC Port: {client.socket.getsockopt(21)})")

    if scene_path and os.path.exists(scene_path):
        abs_scene = os.path.abspath(scene_path)
        print(f"[Scene] Memuat scene: {abs_scene} ...")
        if sim.getSimulationState() != sim.simulation_stopped:
            sim.stopSimulation()
            while sim.getSimulationState() != sim.simulation_stopped:
                time.sleep(0.1)
        sim.loadScene(abs_scene)
        time.sleep(1.0)

    client.setStepping(True)
    sim.startSimulation()
    for _ in range(5):
        client.step()

    handles = get_drone_handles(sim)
    h_drone = handles.get("base", handles["drone"])
    h_target = handles["target"]

    gt_path = proj("scene", "ground_truth.json")
    if plan_type == "smart":
        waypoints, plan_meta = plan_smart(gt_path, cruise_alt=cruise_alt)
    elif plan_type == "lawnmower":
        waypoints = plan_lawnmower(alt=cruise_alt, overlap=0.40)
        plan_meta = {"plan_type": "lawnmower", "cruise_alt": cruise_alt}
    elif plan_type == "spiral":
        waypoints = plan_spiral(alt=cruise_alt, max_radius=4.8)
        plan_meta = {"plan_type": "spiral", "cruise_alt": cruise_alt}
    elif plan_type == "orbit":
        waypoints = plan_orbit(center=(0.0, 0.0), radius=2.5, alt=cruise_alt)
        plan_meta = {"plan_type": "orbit", "cruise_alt": cruise_alt}
    else:
        raise ValueError(f"Plan tidak dikenal: {plan_type}")

    trajectory_setpoints = interpolate_trajectory(waypoints, speed=speed, dt=SIM_DT)
    total_steps = len(trajectory_setpoints)
    est_duration = total_steps * SIM_DT
    print(f"[Planner] Dihasilkan {len(waypoints)} waypoint -> {total_steps:,} setpoint step "
          f"(Estimasi durasi: {est_duration:.1f} s)")

    visualizer = None
    recorder = None
    if record_path:
        from recorder import MissionRecorder
        recorder = MissionRecorder(sim, record_path)
        recorder.title_card([("3D Mapping Area Kota Mini dengan Drone", 34),
                             ("UTS Sistem Robot Otonom - CoppeliaSim 4.10", 20),
                             ("I Made Windu Adi Pramana - 5022241125", 18),
                             ("Sensor: 4 kamera depth + LiDAR 2D 360 derajat", 16)], seconds=3)
        n_lawn = int(plan_meta.get("lawn_steps", 0))
    if live:
        visualizer = LiveVisualizer(
            title=f"Live 3D Mapping [{plan_type.upper()}] - CoppeliaSim",
            record_path=None,
            fps=12
        )

    accumulated_points = []
    accumulated_sensor_ids = []
    accumulated_t = []
    trajectory_records = []
    scan2d_records = []

    start_wall_time = time.time()
    start_sim_time = sim.getSimulationTime()

    takeoff_target = trajectory_setpoints[0]
    tx, ty, tz, tyaw = float(takeoff_target[0]), float(takeoff_target[1]), float(cruise_alt), float(takeoff_target[3])
    sim.setObjectPosition(h_target, [tx, ty, tz], -1)
    sim.setObjectOrientation(h_target, [0.0, 0.0, tyaw], -1)
    print(f"\n[Takeoff] Lepas landas menuju ketinggian {cruise_alt:.1f}m...")

    for _ in range(300):
        client.step()
        drone_pos = sim.getObjectPosition(h_drone, -1)
        if drone_pos[2] >= cruise_alt - 0.15:
            break

    print(f"[Takeoff] Drone stabil di Z = {drone_pos[2]:.2f}m. Memulai eksekusi lintasan!")

    target_idx = 0
    step_count = 0
    max_steps = total_steps * 3

    while target_idx < total_steps and step_count < max_steps:
        target_pt = trajectory_setpoints[target_idx]
        tx, ty, tz, tyaw = float(target_pt[0]), float(target_pt[1]), float(target_pt[2]), float(target_pt[3])

        sim.setObjectPosition(h_target, [tx, ty, tz], -1)
        sim.setObjectOrientation(h_target, [0.0, 0.0, tyaw], -1)

        client.step()
        sim_time = sim.getSimulationTime()
        step_count += 1

        drone_pos = sim.getObjectPosition(h_drone, -1)
        drone_ori = sim.getObjectOrientation(h_drone, -1)

        if on_step:
            on_step(sim, drone_pos, drone_ori, sim_time, target_idx / total_steps)

        target_dist = math.sqrt(
            (drone_pos[0] - tx)**2 +
            (drone_pos[1] - ty)**2 +
            (drone_pos[2] - tz)**2
        )

        if target_dist < 1.35 or target_idx >= total_steps - 5:
            target_idx += 1

        trajectory_records.append([
            sim_time, drone_pos[0], drone_pos[1], drone_pos[2],
            drone_ori[0], drone_ori[1], drone_ori[2], target_dist
        ])

        if step_count % scan_every == 0:
            packed_pts = None
            packed_sids = None

            if handles.get("sensor_script") is not None:
                try:
                    packed_pts, packed_sids = sim.callScriptFunction(
                        "getNewPoints", handles["sensor_script"]
                    )
                except Exception:
                    packed_pts = None

            if packed_pts is None or len(packed_pts) == 0:
                packed_pts, packed_sids = query_sensors_direct(sim, handles)

            if packed_pts and len(packed_pts) > 0:
                pts, sids = unpack_points_and_sensors(packed_pts, packed_sids)
                pts, sids = filter_drone_body_points(pts, sids, np.array(drone_pos))
                if len(pts) > 0:
                    accumulated_points.append(pts)
                    accumulated_sensor_ids.append(sids)
                    accumulated_t.append(np.full(len(pts), sim_time, dtype=np.float32))

            if handles.get("sensor_script") is not None:
                try:
                    packed_scan, pose = sim.callScriptFunction("getNewScan2D", handles["sensor_script"])
                    if packed_scan:
                        ar = np.array(sim.unpackFloatTable(packed_scan), dtype=np.float32).reshape(-1, 2)
                        rows = np.empty((len(ar), 6), np.float32)
                        rows[:, 0] = sim_time
                        rows[:, 1:3] = ar
                        rows[:, 3:6] = pose[:3]
                        scan2d_records.extend(rows.tolist())
                except Exception:
                    pass

        if recorder and step_count % 4 == 0:
            if accumulated_points and (step_count % 20 == 0 or not hasattr(recorder, "_pts")):
                recorder._pts = np.vstack(accumulated_points)
            sc = np.array(scan2d_records[-60000:], dtype=np.float32) if scan2d_records else np.empty((0, 6), np.float32)
            if len(sc):
                hit = sc[sc[:, 2] < 9.8]
                yaws = np.interp(hit[:, 0], [r[0] for r in trajectory_records], [r[6] for r in trajectory_records]) if len(hit) else []
                sxy = np.c_[hit[:, 3] + hit[:, 2] * np.cos(hit[:, 1] + yaws), hit[:, 4] + hit[:, 2] * np.sin(hit[:, 1] + yaws)] if len(hit) else np.empty((0, 2))
            else:
                sxy = np.empty((0, 2))
            pct = min(100.0, (target_idx + 1) / total_steps * 100.0)
            z_now = float(trajectory_setpoints[min(target_idx, total_steps - 1)][2])
            if z_now <= 2.0 and drone_pos[1] > 1.0:
                phase = "Canyon Weaving: Meliuk di antara Gedung Bertingkat & Silo"
            elif z_now > 4.0 and pct < 50:
                phase = "Lawnmower 4.8 m (ground + atap)"
            elif z_now <= 3.2 and pct < 90:
                phase = "Orbit keliling 2.8 m (badan objek)"
            elif pct >= 90:
                phase = "Pulang & mendarat"
            else:
                phase = "Transisi Koridor"
            mode = "follow"
            recorder.frame(drone_pos, drone_ori[2], getattr(recorder, "_pts", np.empty((0, 3))),
                           np.array(trajectory_records, dtype=np.float32), sxy, sim_time, pct, phase, mode)

        if visualizer and (step_count % 4 == 0 or target_idx >= total_steps - 1):
            curr_pts = np.vstack(accumulated_points) if len(accumulated_points) > 0 else np.empty((0, 3), dtype=np.float32)
            curr_sids = np.concatenate(accumulated_sensor_ids) if len(accumulated_sensor_ids) > 0 else np.empty((0,), dtype=np.uint8)
            curr_traj = np.array(trajectory_records, dtype=np.float32)
            curr_pose = np.array([drone_pos[0], drone_pos[1], drone_pos[2], drone_ori[2]])
            visualizer.update(curr_pts, curr_sids, curr_traj, curr_pose, sim_time)

        if step_count % 50 == 0 or target_idx >= total_steps - 1:
            total_pts = sum(len(p) for p in accumulated_points)
            pct = min(100.0, (target_idx + 1) / total_steps * 100.0)
            sys.stdout.write(
                f"\r[Progres {pct:5.1f}%] Waypoint {target_idx+1}/{total_steps} | "
                f"SimTime: {sim_time:.1f}s | Alt: {drone_pos[2]:.2f}m | "
                f"Titik: {total_pts:,}   "
            )
            sys.stdout.flush()

        if step_count > 0 and step_count % 1500 == 0:
            checkpoint_path = out_path.replace(".npz", f"_ckpt_{step_count}.npz")
            finalize_and_save(
                out_path=checkpoint_path,
                points_list=accumulated_points,
                sensor_ids_list=accumulated_sensor_ids,
                t_list=accumulated_t,
                traj_records=trajectory_records,
                scan2d_records=scan2d_records,
                meta_info={"checkpoint_step": step_count, "sim_time": sim_time}
            )

    print("\n\n[Misi] Selesai! Menghentikan simulasi CoppeliaSim...")
    sim.stopSimulation()
    while sim.getSimulationState() != sim.simulation_stopped:
        client.step()

    if visualizer:
        visualizer.close()
    if recorder:
        recorder.close()

    wall_duration = time.time() - start_wall_time
    total_sim_duration = sim_time - start_sim_time

    plan_meta.update({
        "wall_duration_sec": round(wall_duration, 2),
        "sim_duration_sec": round(total_sim_duration, 2),
        "speed_mps": speed,
        "scan_every_steps": scan_every,
        "total_steps": total_steps
    })

    stats = finalize_and_save(
        out_path=out_path,
        points_list=accumulated_points,
        sensor_ids_list=accumulated_sensor_ids,
        t_list=accumulated_t,
        traj_records=trajectory_records,
        scan2d_records=scan2d_records,
        meta_info=plan_meta
    )

    generate_debug_topview(out_path, proj("results", "debug_raw_topview.png"))

    return stats


def finalize_and_save(
    out_path: str,
    points_list: List[np.ndarray],
    sensor_ids_list: List[np.ndarray],
    t_list: List[np.ndarray],
    traj_records: List[List[float]],
    scan2d_records: List[List[float]],
    meta_info: Dict[str, Any]
) -> Dict[str, Any]:
    if len(points_list) > 0:
        all_points = np.vstack(points_list).astype(np.float32)
        all_sensor_ids = np.concatenate(sensor_ids_list).astype(np.uint8)
        all_t = np.concatenate(t_list).astype(np.float32)
    else:
        all_points = np.empty((0, 3), dtype=np.float32)
        all_sensor_ids = np.empty((0,), dtype=np.uint8)
        all_t = np.empty((0,), dtype=np.float32)

    all_traj = np.array(traj_records, dtype=np.float32) if len(traj_records) > 0 else np.empty((0, 8), dtype=np.float32)
    all_scan2d = np.array(scan2d_records, dtype=np.float32) if len(scan2d_records) > 0 else np.empty((0, 6), dtype=np.float32)

    traj_length = 0.0
    if len(all_traj) > 1:
        diffs = np.diff(all_traj[:, 1:4], axis=0)
        traj_length = float(np.sum(np.linalg.norm(diffs, axis=1)))

    sensor_breakdown = {}
    if len(all_sensor_ids) > 0:
        unique, counts = np.unique(all_sensor_ids, return_counts=True)
        for sid, cnt in zip(unique, counts):
            name = SENSOR_NAMES.get(int(sid), f"sensor_{sid}")
            sensor_breakdown[name] = int(cnt)

    min_bounds = all_points.min(axis=0).tolist() if len(all_points) > 0 else [0, 0, 0]
    max_bounds = all_points.max(axis=0).tolist() if len(all_points) > 0 else [0, 0, 0]

    coverage_pct = 0.0
    if len(all_points) > 0:
        arena_size = 12.0
        grid_res = 0.5
        grid_cells = int(arena_size / grid_res)
        xi = np.clip(((all_points[:, 0] + 6.0) / grid_res).astype(int), 0, grid_cells - 1)
        yi = np.clip(((all_points[:, 1] + 6.0) / grid_res).astype(int), 0, grid_cells - 1)
        occupied = np.zeros((grid_cells, grid_cells), dtype=bool)
        occupied[xi, yi] = True
        coverage_pct = float(np.mean(occupied) * 100.0)

    meta_info.update({
        "total_points": int(len(all_points)),
        "sensor_breakdown": sensor_breakdown,
        "trajectory_length_m": round(traj_length, 2),
        "point_cloud_bounds": {
            "min": [round(v, 3) for v in min_bounds],
            "max": [round(v, 3) for v in max_bounds]
        },
        "arena_coverage_pct": round(coverage_pct, 1),
        "saved_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    })

    save_raw_scan(
        filepath=out_path,
        points=all_points,
        sensor_id=all_sensor_ids,
        t=all_t,
        traj=all_traj,
        scan2d=all_scan2d,
        meta=meta_info
    )

    print()
    print("STATISTIK AKUISISI DATA RAW (raw_scan.npz)")
    print(f"File Output       : {out_path} ({os.path.getsize(out_path) / 1024 / 1024:.2f} MB)")
    print(f"Total Titik 3D    : {len(all_points):,} titik")
    print(f"Breakdown Sensor  : {sensor_breakdown}")
    print(f"Panjang Trajektori: {traj_length:.2f} m")
    print(f"Elevasi Z Terendah: {min_bounds[2]:.3f} m (Ground target z ≈ 0)")
    print(f"Elevasi Z Tertinggi: {max_bounds[2]:.3f} m")
    print(f"Cakupan Area Arena: {coverage_pct:.1f} %")
    print()

    return meta_info


def generate_debug_topview(npz_path: str, out_img: str):
    if not os.path.exists(npz_path):
        return

    os.makedirs(os.path.dirname(os.path.abspath(out_img)), exist_ok=True)
    matplotlib.use("Agg")

    data = np.load(npz_path, allow_pickle=True)
    points = data["points"]
    traj = data["traj"]

    fig, ax = plt.subplots(figsize=(9, 9), facecolor="#1e1e1e")
    ax.set_facecolor("#121212")
    ax.set_title(f"Debug Top-View Raw Point Cloud ({len(points):,} pts)", color="#ffffff", fontsize=13)

    if len(points) > 0:
        disp_pts = points[::max(1, len(points) // 50000)]
        sc = ax.scatter(
            disp_pts[:, 0], disp_pts[:, 1], c=disp_pts[:, 2],
            cmap="turbo", s=1.2, alpha=0.5, vmin=0.0, vmax=4.5
        )
        cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Elevasi Z (m)", color="#ffffff")
        cbar.ax.yaxis.set_tick_params(color="#ffffff")
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color="#ffffff")

    if len(traj) > 0:
        ax.plot(traj[:, 1], traj[:, 2], color="#00e5ff", linewidth=1.5, label="Drone Trajectory")
        ax.scatter([traj[0, 1]], [traj[0, 2]], color="#76ff03", s=70, marker="o", label="Start (Helipad)")
        ax.scatter([traj[-1, 1]], [traj[-1, 2]], color="#ff1744", s=70, marker="X", label="End")

    ax.set_xlim(-6.5, 6.5)
    ax.set_ylim(-6.5, 6.5)
    ax.set_xlabel("X (m)", color="#ffffff")
    ax.set_ylabel("Y (m)", color="#ffffff")
    ax.tick_params(colors="#ffffff")
    ax.grid(True, linestyle="--", alpha=0.3, color="#555555")
    ax.legend(facecolor="#222222", edgecolor="#444444", labelcolor="#ffffff", loc="upper right")

    plt.tight_layout()
    plt.savefig(out_img, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Debug] Plot top-view tersimpan: {out_img}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Misi drone: perencanaan jalur, kontrol, dan akuisisi data"
    )
    parser.add_argument("--scene", type=str, default=proj("scene", "drone_mapping_3d.ttt"), help="Path scene CoppeliaSim .ttt")
    parser.add_argument("--plan", type=str, default="smart", choices=["smart", "lawnmower", "spiral", "orbit"], help="Mode rencana lintasan")
    parser.add_argument("--alt", type=float, default=DEFAULT_CRUISE_ALT, help="Ketinggian jelajah drone (m)")
    parser.add_argument("--speed", type=float, default=DEFAULT_SPEED, help="Kecepatan drone (m/s)")
    parser.add_argument("--scan-every", type=int, default=2, help="Interval step polling sensor")
    parser.add_argument("--out", type=str, default=proj("data", "raw_scan.npz"), help="Path output raw_scan.npz")
    parser.add_argument("--port", type=int, default=None, help="Port ZMQ RPC CoppeliaSim (default dari env ZMQ_PORT atau 23000)")
    parser.add_argument("--live", action="store_true", help="Buka dashboard visualisasi real-time matplotlib")
    parser.add_argument("--record", "--record-video", dest="record", type=str, default=None, help="Simpan video rekaman misi ke file MP4")
    parser.add_argument("--manual", action="store_true", help="Jalankan mode kontrol manual keyboard terminal")

    args = parser.parse_args()

    if args.manual:
        c, s = get_client(port=args.port)
        h = get_drone_handles(s)
        run_manual_control(s, c, h, out_path=args.out, scan_every=args.scan_every)
    else:
        run_mission(
            scene_path=args.scene,
            plan_type=args.plan,
            cruise_alt=args.alt,
            speed=args.speed,
            scan_every=args.scan_every,
            out_path=args.out,
            port=args.port,
            live=args.live,
            record_path=args.record
        )
