#!/usr/bin/env python3

import os
import shutil
import sys
import json
import time
import argparse
import subprocess
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import label, binary_dilation

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def get_colormap(name):
    try:
        return matplotlib.colormaps[name]
    except Exception:
        return plt.get_cmap(name)


def generate_synthetic_ground_truth(arena_size=12.0):
    half = arena_size / 2.0
    obstacles = [
        {
            "name": "Gedung_Bertingkat_A",
            "type": "stacked_box",
            "pos": [-3.5, 3.5, 1.5],
            "size": [2.4, 2.4, 3.0],
            "height_top": 3.0,
            "color": [0.4, 0.45, 0.5],
            "desc": "Gedung Bertingkat 3 Lantai (Tinggi 3.0m)"
        },
        {
            "name": "Gedung_L_Shape",
            "type": "l_shape",
            "pos": [3.5, 3.5, 1.1],
            "size": [2.4, 2.4, 2.2],
            "height_top": 2.2,
            "color": [0.35, 0.5, 0.65],
            "desc": "Gedung Kantor Bentuk L (Tinggi 2.2m)"
        },
        {
            "name": "Gedung_Atap_Miring",
            "type": "sloped_box",
            "pos": [3.5, -3.5, 1.4],
            "size": [2.0, 2.0, 2.8],
            "height_top": 2.8,
            "color": [0.75, 0.4, 0.3],
            "desc": "Gedung Industri Atap Miring (Tinggi 2.8m)"
        },
        {
            "name": "Tangki_Silinder_A",
            "type": "cylinder",
            "pos": [-3.5, -1.0, 1.0],
            "size": [2.0, 2.0, 2.0],
            "height_top": 2.0,
            "color": [0.2, 0.6, 0.8],
            "desc": "Tangki Kimia Silinder Diameter 2.0m"
        },
        {
            "name": "Tangki_Silinder_B",
            "type": "cylinder",
            "pos": [-1.5, -3.5, 1.25],
            "size": [1.2, 1.2, 2.5],
            "height_top": 2.5,
            "color": [0.85, 0.7, 0.2],
            "desc": "Silo Material Ramping 2.5m"
        },
        {
            "name": "Menara_Radio_Telekomunikasi",
            "type": "tower",
            "pos": [0.0, 0.0, 1.9],
            "size": [0.6, 0.6, 3.8],
            "height_top": 3.8,
            "color": [0.85, 0.2, 0.2],
            "desc": "Menara Pemancar Tertinggi 3.8m"
        },
        {
            "name": "Kubah_Observatorium",
            "type": "dome",
            "pos": [-3.5, 1.0, 1.3],
            "size": [1.6, 1.6, 2.6],
            "height_top": 2.6,
            "color": [0.8, 0.85, 0.9],
            "desc": "Kubah Observatorium / Tangki Gas Bertekanan"
        },
        {
            "name": "Jembatan_Gerbang_3D",
            "type": "bridge",
            "pos": [0.0, 3.2, 1.1],
            "size": [2.0, 0.8, 2.2],
            "height_top": 2.2,
            "color": [0.55, 0.5, 0.45],
            "desc": "Jembatan Penyeberangan 3D (Kolong Bebas 1.8m)"
        },
        {
            "name": "Pohon_Taman_1",
            "type": "tree",
            "pos": [1.8, -1.5, 1.2],
            "size": [1.4, 1.4, 2.4],
            "height_top": 2.4,
            "color": [0.2, 0.65, 0.25],
            "desc": "Pohon Kanopi Lebar (2.4m)"
        },
        {
            "name": "Pohon_Taman_2",
            "type": "tree",
            "pos": [4.5, 0.2, 1.05],
            "size": [1.2, 1.2, 2.1],
            "height_top": 2.1,
            "color": [0.25, 0.7, 0.3],
            "desc": "Pohon Pinggir Area Timur (2.1m)"
        },
        {
            "name": "Pohon_Taman_3",
            "type": "tree",
            "pos": [-1.8, 1.8, 1.15],
            "size": [1.3, 1.3, 2.3],
            "height_top": 2.3,
            "color": [0.18, 0.6, 0.22],
            "desc": "Pohon Dekat Jembatan (2.3m)"
        },
        {
            "name": "Pohon_Taman_4",
            "type": "tree",
            "pos": [1.5, 1.2, 1.0],
            "size": [1.1, 1.1, 2.0],
            "height_top": 2.0,
            "color": [0.22, 0.68, 0.28],
            "desc": "Pohon Tengah Lapangan (2.0m)"
        },
        {
            "name": "Kontainer_Kargo_A",
            "type": "box",
            "pos": [1.5, -3.5, 0.25],
            "size": [2.2, 1.0, 0.5],
            "height_top": 0.5,
            "color": [0.8, 0.3, 0.2],
            "desc": "Kontainer Kargo Merah Rendah (0.5m)"
        },
        {
            "name": "Kontainer_Kargo_B",
            "type": "box",
            "pos": [-1.5, 3.5, 0.2],
            "size": [1.2, 0.8, 0.4],
            "height_top": 0.4,
            "color": [0.2, 0.4, 0.75],
            "desc": "Peti Kemas Biru Rendah (0.4m)"
        },
        {
            "name": "Ramp_Tanjakan_Miring",
            "type": "ramp",
            "pos": [3.2, -1.2, 0.4],
            "size": [2.0, 1.2, 0.8],
            "height_top": 0.8,
            "color": [0.65, 0.6, 0.55],
            "desc": "Ramp Tanjakan Miring (Gradasi 0 ke 0.8m)"
        },
        {
            "name": "Bukit_Kontur_Tanah",
            "type": "hill",
            "pos": [-1.2, -1.2, 0.35],
            "size": [2.2, 2.2, 0.7],
            "height_top": 0.7,
            "color": [0.45, 0.55, 0.35],
            "desc": "Gundukan Tanah Berkontur Alami (0.7m)"
        },
        {
            "name": "Pagar_Perimeter_Arena",
            "type": "fence",
            "pos": [0.0, 0.0, 0.2],
            "size": [11.6, 11.6, 0.4],
            "height_top": 0.4,
            "color": [0.3, 0.3, 0.3],
            "desc": "Pagar Batas Rendah Perimeter 0.4m"
        }
    ]

    sensors_meta = {
        "depthDown": {
            "type": "vision_sensor_depth",
            "res": [64, 64],
            "fov_deg": 60.0,
            "far_clip": 8.0,
            "orientation": "menghadap -Z dunia (pitch -90 deg)",
            "purpose": "Deteksi ground (lantai) dan puncak objek secara presisi top-down"
        },
        "depthFront": {
            "type": "vision_sensor_depth",
            "res": [64, 48],
            "fov_deg": 60.0,
            "pitch_deg": -35.0,
            "far_clip": 8.0,
            "orientation": "maju +X drone, pitch 35 deg ke bawah",
            "purpose": "Deteksi badan/dinding vertikal objek di depan lintasan"
        },
        "depthLeft": {
            "type": "vision_sensor_depth",
            "res": [48, 36],
            "fov_deg": 60.0,
            "pitch_deg": -35.0,
            "yaw_deg": 90.0,
            "far_clip": 6.0,
            "orientation": "sisi kiri drone, pitch 35 deg ke bawah",
            "purpose": "Deteksi fasad sisi kiri objek saat drone bermanuver"
        },
        "depthRight": {
            "type": "vision_sensor_depth",
            "res": [48, 36],
            "fov_deg": 60.0,
            "pitch_deg": -35.0,
            "yaw_deg": -90.0,
            "far_clip": 6.0,
            "orientation": "sisi kanan drone, pitch 35 deg ke bawah",
            "purpose": "Deteksi fasad sisi kanan objek saat drone bermanuver"
        },
        "lidar2D": {
            "type": "lidar_2d_planar",
            "num_beams": 360,
            "fov_deg": 360.0,
            "far_clip": 10.0,
            "orientation": "horizontal 360 deg pada bidang xy drone",
            "purpose": "Pemetaan okupansi 2D real-time dan perimeter obstacle"
        }
    }

    gt_data = {
        "arena": {
            "size_x": arena_size,
            "size_y": arena_size,
            "bounds_x": [-half, half],
            "bounds_y": [-half, half],
            "ground_z": 0.0
        },
        "obstacles": obstacles,
        "sensors": sensors_meta,
        "created_by": "generator data sintetis",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    return gt_data


def generate_synthetic_scan(gt_data, output_npz_path, num_target_points=280000):
    np.random.seed(42)
    print(f"Menghasilkan dataset sintetis ({num_target_points} titik)...")

    t_steps = 600
    total_time = 120.0
    t_arr = np.linspace(0, total_time, t_steps, dtype=np.float32)

    y_lines = [-4.5, -2.5, -0.5, 1.5, 3.5, 4.5]
    x_bounds = [-4.5, 4.5]

    traj_pts = []
    for t_sub in np.linspace(0, 8, 30):
        frac = t_sub / 8.0
        traj_pts.append([-5.0, -5.0, 0.1 + 4.4 * frac, 0.0, 0.0, 0.0])

    cur_x = -4.5
    for i, y_val in enumerate(y_lines):
        target_x = x_bounds[1] if (i % 2 == 0) else x_bounds[0]
        for frac in np.linspace(0, 1, 40):
            x = cur_x + (target_x - cur_x) * frac
            yaw = 0.0 if target_x > cur_x else np.pi
            pitch = -0.04
            traj_pts.append([x, y_val, 4.5, 0.0, pitch, yaw])
        cur_x = target_x
        if i < len(y_lines) - 1:
            next_y = y_lines[i + 1]
            for frac in np.linspace(0, 1, 15):
                y = y_val + (next_y - y_val) * frac
                traj_pts.append([cur_x, y, 4.5, 0.02, 0.0, np.pi/2])

    orbit_center = np.array([0.0, 0.0])
    orbit_radius = 2.2
    for theta in np.linspace(0, 2*np.pi, 80):
        ox = orbit_center[0] + orbit_radius * np.cos(theta)
        oy = orbit_center[1] + orbit_radius * np.sin(theta)
        yaw = np.arctan2(-oy, -ox)
        traj_pts.append([ox, oy, 3.2, 0.03, -0.05, yaw])

    for frac in np.linspace(0, 1, 30):
        x = traj_pts[-1][0] + (-5.0 - traj_pts[-1][0]) * frac
        y = traj_pts[-1][1] + (-5.0 - traj_pts[-1][1]) * frac
        z = 3.2 + (4.0 - 3.2) * (1 - frac) if frac < 0.5 else 4.0 - 3.9 * ((frac - 0.5) * 2)
        traj_pts.append([x, y, max(z, 0.1), 0.0, 0.0, -np.pi/4])

    traj_arr = np.array(traj_pts, dtype=np.float32)
    from scipy.interpolate import interp1d
    interp_func = interp1d(np.linspace(0, total_time, len(traj_arr)), traj_arr, axis=0)
    traj_resampled = interp_func(t_arr)

    target_dist = np.random.uniform(0.02, 0.08, size=(t_steps, 1)).astype(np.float32)
    traj_full = np.hstack([t_arr.reshape(-1, 1), traj_resampled, target_dist]).astype(np.float32)

    all_points = []
    all_sensor_ids = []
    all_times = []

    n_ground = int(num_target_points * 0.40)
    gx = np.random.uniform(-5.8, 5.8, n_ground).astype(np.float32)
    gy = np.random.uniform(-5.8, 5.8, n_ground).astype(np.float32)
    gz = 0.005 * np.sin(gx * 0.8) + 0.004 * np.cos(gy * 0.8) + np.random.normal(0, 0.012, n_ground).astype(np.float32)
    ground_pts = np.column_stack([gx, gy, gz])
    all_points.append(ground_pts)
    s_ids = np.zeros(n_ground, dtype=np.uint8)
    s_ids[np.random.rand(n_ground) > 0.85] = 1
    all_sensor_ids.append(s_ids)
    all_times.append(np.random.uniform(0, total_time, n_ground).astype(np.float32))

    obstacles = gt_data["obstacles"]
    n_per_obs = int((num_target_points * 0.54) / len(obstacles))

    for obs in obstacles:
        pos = np.array(obs["pos"], dtype=np.float32)
        size = np.array(obs["size"], dtype=np.float32)
        otype = obs["type"]

        if otype in ["box", "fence"]:
            nx = n_per_obs // 5
            pts_x1 = np.column_stack([
                np.full(nx, pos[0] - size[0]/2),
                np.random.uniform(pos[1] - size[1]/2, pos[1] + size[1]/2, nx),
                np.random.uniform(max(0, pos[2] - size[2]/2), pos[2] + size[2]/2, nx)
            ])
            pts_x2 = np.column_stack([
                np.full(nx, pos[0] + size[0]/2),
                np.random.uniform(pos[1] - size[1]/2, pos[1] + size[1]/2, nx),
                np.random.uniform(max(0, pos[2] - size[2]/2), pos[2] + size[2]/2, nx)
            ])
            pts_y1 = np.column_stack([
                np.random.uniform(pos[0] - size[0]/2, pos[0] + size[0]/2, nx),
                np.full(nx, pos[1] - size[1]/2),
                np.random.uniform(max(0, pos[2] - size[2]/2), pos[2] + size[2]/2, nx)
            ])
            pts_y2 = np.column_stack([
                np.random.uniform(pos[0] - size[0]/2, pos[0] + size[0]/2, nx),
                np.full(nx, pos[1] + size[1]/2),
                np.random.uniform(max(0, pos[2] - size[2]/2), pos[2] + size[2]/2, nx)
            ])
            pts_top = np.column_stack([
                np.random.uniform(pos[0] - size[0]/2, pos[0] + size[0]/2, nx),
                np.random.uniform(pos[1] - size[1]/2, pos[1] + size[1]/2, nx),
                np.full(nx, pos[2] + size[2]/2)
            ])
            obs_pts = np.vstack([pts_x1, pts_x2, pts_y1, pts_y2, pts_top])
            s_obs = np.random.choice([1, 2, 3], size=len(obs_pts))
            s_obs[4*nx:] = 0

        elif otype == "sloped_box":
            nx = n_per_obs
            ux = np.random.uniform(-size[0]/2, size[0]/2, nx)
            uy = np.random.uniform(-size[1]/2, size[1]/2, nx)
            uz = (pos[2] - size[2]/2) + (size[2]) * (ux + size[0]/2) / size[0]
            is_wall = np.random.rand(nx) < 0.4
            uz[is_wall] = np.random.uniform(0, uz[is_wall])
            obs_pts = np.column_stack([pos[0] + ux, pos[1] + uy, uz])
            s_obs = np.random.choice([0, 1, 2], size=len(obs_pts))

        elif otype in ["cylinder", "tower"]:
            radius = size[0] / 2.0
            height = size[2]
            n_side = int(n_per_obs * 0.75)
            theta = np.random.uniform(0, 2*np.pi, n_side)
            z_side = np.random.uniform(0, height, n_side)
            pts_side = np.column_stack([
                pos[0] + radius * np.cos(theta),
                pos[1] + radius * np.sin(theta),
                z_side
            ])
            n_top = n_per_obs - n_side
            r_top = radius * np.sqrt(np.random.uniform(0, 1, n_top))
            th_top = np.random.uniform(0, 2*np.pi, n_top)
            pts_top = np.column_stack([
                pos[0] + r_top * np.cos(th_top),
                pos[1] + r_top * np.sin(th_top),
                np.full(n_top, height)
            ])
            obs_pts = np.vstack([pts_side, pts_top])
            s_obs = np.random.choice([1, 2, 3, 4], size=len(obs_pts))
            s_obs[n_side:] = 0

        elif otype == "dome":
            radius = size[0] / 2.0
            cyl_h = size[2] - radius
            n_cyl = int(n_per_obs * 0.5)
            th_c = np.random.uniform(0, 2*np.pi, n_cyl)
            zc = np.random.uniform(0, cyl_h, n_cyl)
            pts_cyl = np.column_stack([pos[0] + radius * np.cos(th_c), pos[1] + radius * np.sin(th_c), zc])

            n_dome = n_per_obs - n_cyl
            u = np.random.uniform(0, 1, n_dome)
            v = np.random.uniform(0, 1, n_dome)
            phi = np.arccos(1 - u)
            theta = 2 * np.pi * v
            pts_dome = np.column_stack([
                pos[0] + radius * np.sin(phi) * np.cos(theta),
                pos[1] + radius * np.sin(phi) * np.sin(theta),
                cyl_h + radius * np.cos(phi)
            ])
            obs_pts = np.vstack([pts_cyl, pts_dome])
            s_obs = np.random.choice([0, 1, 2], size=len(obs_pts))

        elif otype == "tree":
            trunk_r = 0.15
            trunk_h = 1.0
            n_tr = int(n_per_obs * 0.2)
            th_t = np.random.uniform(0, 2*np.pi, n_tr)
            zt = np.random.uniform(0, trunk_h, n_tr)
            pts_tr = np.column_stack([pos[0] + trunk_r * np.cos(th_t), pos[1] + trunk_r * np.sin(th_t), zt])

            n_canopy = n_per_obs - n_tr
            canopy_r = size[0] / 2.0
            canopy_zc = pos[2]
            u = np.random.uniform(0, 1, n_canopy)
            v = np.random.uniform(0, 1, n_canopy)
            w = np.random.uniform(0.7, 1.0, n_canopy)
            phi = np.arccos(2*u - 1)
            theta = 2 * np.pi * v
            pts_canopy = np.column_stack([
                pos[0] + canopy_r * w * np.sin(phi) * np.cos(theta),
                pos[1] + canopy_r * w * np.sin(phi) * np.sin(theta),
                canopy_zc + canopy_r * w * np.cos(phi)
            ])
            obs_pts = np.vstack([pts_tr, pts_canopy])
            s_obs = np.random.choice([0, 1, 2, 3], size=len(obs_pts))

        elif otype == "ramp":
            nx = n_per_obs
            rx = np.random.uniform(-size[0]/2, size[0]/2, nx)
            ry = np.random.uniform(-size[1]/2, size[1]/2, nx)
            rz = (rx + size[0]/2) / size[0] * size[2]
            obs_pts = np.column_stack([pos[0] + rx, pos[1] + ry, rz])
            s_obs = np.random.choice([0, 1], size=len(obs_pts))

        elif otype == "hill":
            nx = n_per_obs
            r = np.random.exponential(0.6, nx)
            th = np.random.uniform(0, 2*np.pi, nx)
            hx = r * np.cos(th)
            hy = r * np.sin(th)
            hz = size[2] * np.exp(-(hx**2 + hy**2) / (2 * 0.7**2))
            obs_pts = np.column_stack([pos[0] + hx, pos[1] + hy, hz])
            s_obs = np.zeros(nx, dtype=np.uint8)

        else:
            obs_pts = np.random.uniform(pos - size/2, pos + size/2, (n_per_obs, 3))
            s_obs = np.zeros(n_per_obs, dtype=np.uint8)

        obs_pts += np.random.normal(0, 0.012, obs_pts.shape).astype(np.float32)

        all_points.append(obs_pts.astype(np.float32))
        all_sensor_ids.append(s_obs.astype(np.uint8))
        all_times.append(np.random.uniform(5, total_time - 5, len(obs_pts)).astype(np.float32))

    n_outliers = int(num_target_points * 0.035)
    n_under = n_outliers // 2
    out_under = np.column_stack([
        np.random.uniform(-5.5, 5.5, n_under),
        np.random.uniform(-5.5, 5.5, n_under),
        np.random.uniform(-1.5, -0.15, n_under)
    ])
    n_air = n_outliers - n_under
    out_air = np.column_stack([
        np.random.uniform(-5.5, 5.5, n_air),
        np.random.uniform(-5.5, 5.5, n_air),
        np.random.uniform(0.5, 4.8, n_air)
    ])
    outliers = np.vstack([out_under, out_air]).astype(np.float32)
    all_points.append(outliers)
    all_sensor_ids.append(np.random.choice([0, 1, 2, 3], size=len(outliers)).astype(np.uint8))
    all_times.append(np.random.uniform(0, total_time, len(outliers)).astype(np.float32))

    pts_combined = np.vstack(all_points).astype(np.float32)
    s_ids_combined = np.concatenate(all_sensor_ids).astype(np.uint8)
    times_combined = np.concatenate(all_times).astype(np.float32)

    n_scans = 2400
    scan_times = np.linspace(5, total_time - 5, n_scans, dtype=np.float32)
    scan_angles = np.random.uniform(-np.pi, np.pi, n_scans).astype(np.float32)
    t_idx = np.searchsorted(t_arr, scan_times).clip(0, t_steps - 1)
    drone_x = traj_full[t_idx, 1]
    drone_y = traj_full[t_idx, 2]
    drone_z = traj_full[t_idx, 3]
    scan_ranges = np.random.uniform(1.2, 7.5, n_scans).astype(np.float32)
    scan2d_full = np.column_stack([
        scan_times, scan_angles, scan_ranges, drone_x, drone_y, drone_z
    ]).astype(np.float32)

    meta_dict = {
        "generator": "data sintetis",
        "total_points": len(pts_combined),
        "total_sim_time": float(total_time),
        "arena_size": 12.0,
        "voxel_default": 0.10,
        "flight_mode": "smart_lawnmower_orbit",
        "sensors": gt_data["sensors"]
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_npz_path)), exist_ok=True)
    np.savez_compressed(
        output_npz_path,
        points=pts_combined,
        sensor_id=s_ids_combined,
        t=times_combined,
        traj=traj_full,
        scan2d=scan2d_full,
        meta=json.dumps(meta_dict)
    )
    print(f"Berhasil membuat file sintetis: {output_npz_path} ({len(pts_combined)} titik, {len(traj_full)} waypoint).")
    return output_npz_path


def crop_point_cloud(points, sensor_ids, bounds=(-6.0, 6.0, -6.0, 6.0, -0.10, 5.0)):
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    mask = (
        (points[:, 0] >= xmin) & (points[:, 0] <= xmax) &
        (points[:, 1] >= ymin) & (points[:, 1] <= ymax) &
        (points[:, 2] >= zmin) & (points[:, 2] <= zmax)
    )
    return points[mask], sensor_ids[mask], mask


def remove_statistical_outliers(points, sensor_ids, k=8, std_ratio=2.0, radius=0.25, min_radius_neighbors=3):
    t0 = time.time()
    n_pts = len(points)
    if n_pts == 0:
        return points, sensor_ids, np.zeros(0, dtype=bool)

    tree = cKDTree(points)
    dists, indices = tree.query(points, k=k+1, workers=-1)
    mean_dists = np.mean(dists[:, 1:], axis=1)

    mu = np.mean(mean_dists)
    sigma = np.std(mean_dists)
    threshold = mu + std_ratio * sigma

    sor_mask = mean_dists <= threshold

    valid_indices = np.where(sor_mask)[0]
    sub_pts = points[valid_indices]
    sub_tree = cKDTree(sub_pts)
    counts = sub_tree.query_ball_point(sub_pts, r=radius, return_length=True, workers=-1)
    r_mask = counts >= min_radius_neighbors

    final_mask = np.zeros(n_pts, dtype=bool)
    final_mask[valid_indices[r_mask]] = True

    dt = time.time() - t0
    n_removed = n_pts - np.sum(final_mask)
    print(f"Filter Outlier (SOR k={k}, sigma={std_ratio}): {n_pts:,} -> {np.sum(final_mask):,} "
          f"({n_removed:,} outlier dibuang, {dt:.2f}s)")
    return points[final_mask], sensor_ids[final_mask], final_mask


def voxel_downsample(points, voxel_size=0.10):
    t0 = time.time()
    if len(points) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.int32)

    v_idx = np.floor(points / voxel_size).astype(np.int64)

    v_min = v_idx.min(axis=0)
    shifted = v_idx - v_min
    max_dims = shifted.max(axis=0) + 1
    stride_y = int(max_dims[2])
    stride_x = int(max_dims[1] * stride_y)

    linear_keys = shifted[:, 0] * stride_x + shifted[:, 1] * stride_y + shifted[:, 2]

    unique_keys, inverse_indices, counts = np.unique(linear_keys, return_inverse=True, return_counts=True)
    n_voxels = len(unique_keys)

    cx = np.bincount(inverse_indices, weights=points[:, 0]) / counts
    cy = np.bincount(inverse_indices, weights=points[:, 1]) / counts
    cz = np.bincount(inverse_indices, weights=points[:, 2]) / counts
    centroids = np.column_stack([cx, cy, cz]).astype(np.float32)

    dt = time.time() - t0
    reduction_pct = (1.0 - (n_voxels / len(points))) * 100.0
    print(f"Voxel Downsampling (size={voxel_size}m): {len(points):,} -> {n_voxels:,} "
          f"({reduction_pct:.2f}% reduksi, {dt:.2f}s)")
    return centroids, counts, reduction_pct, dt


def segment_ground_ransac(points, distance_thresh=0.06, max_iterations=200):
    t0 = time.time()
    n_pts = len(points)
    low_mask = points[:, 2] < 0.35
    low_indices = np.where(low_mask)[0]

    if len(low_indices) < 3:
        is_ground = points[:, 2] <= 0.08
        return is_ground, ~is_ground, np.array([0, 0, 1, 0])

    best_inliers = np.zeros(n_pts, dtype=bool)
    best_count = 0
    best_plane = np.array([0.0, 0.0, 1.0, 0.0])

    for _ in range(max_iterations):
        sample_idx = np.random.choice(low_indices, size=3, replace=False)
        p1, p2, p3 = points[sample_idx]

        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-6:
            continue
        normal /= norm_len

        if normal[2] < 0:
            normal = -normal

        if normal[2] < 0.85:
            continue

        d = -np.dot(normal, p1)
        dists = np.abs(np.dot(points, normal) + d)
        inliers = (dists < distance_thresh) & (points[:, 2] < 0.40)
        count = np.sum(inliers)

        if count > best_count:
            best_count = count
            best_inliers = inliers
            best_plane = np.append(normal, d)

    is_ground = best_inliers
    is_obstacle = (~is_ground) & (points[:, 2] >= 0.06)

    dt = time.time() - t0
    print(f"RANSAC Ground Segmentation: {np.sum(is_ground):,} ground, {np.sum(is_obstacle):,} obstacle "
          f"(Plane: {best_plane[0]:.3f}x + {best_plane[1]:.3f}y + {best_plane[2]:.3f}z + {best_plane[3]:.3f} = 0, {dt:.2f}s)")
    return is_ground, is_obstacle, best_plane


LABEL_GRID = {}
STATS_EXTRA = {}


def cluster_3d_objects(obstacle_points, voxel_res=0.10, min_voxels_per_cluster=15, arena_half=6.0, wall_band=0.35):
    t0 = time.time()
    if len(obstacle_points) == 0:
        return [], np.zeros(0, dtype=np.int32)
    n = int(round(2 * arena_half / voxel_res))
    ix = np.clip(((obstacle_points[:, 0] + arena_half) / voxel_res).astype(int), 0, n - 1)
    iy = np.clip(((obstacle_points[:, 1] + arena_half) / voxel_res).astype(int), 0, n - 1)
    inner = (np.abs(obstacle_points[:, 0]) < arena_half - wall_band) & (np.abs(obstacle_points[:, 1]) < arena_half - wall_band)
    hits = np.zeros((n, n), np.int32)
    np.add.at(hits, (ix[inner], iy[inner]), 1)
    occ = hits >= 3
    lab2d, _ = label(occ, structure=np.ones((3, 3), int))
    point_labels = np.where(inner, lab2d[ix, iy], 0)
    LABEL_GRID.update(grid=lab2d, res=voxel_res, half=arena_half, pts=obstacle_points)
    cluster_info = []
    clean = np.zeros(len(obstacle_points), np.int32)
    new_id = 1
    order = np.argsort(point_labels)
    labs, starts, counts = np.unique(point_labels[order], return_index=True, return_counts=True)
    for lab, st, cnt in zip(labs, starts, counts):
        if lab == 0:
            continue
        idx = order[st:st + cnt]
        cells = np.count_nonzero(lab2d == lab)
        if cells < min_voxels_per_cluster:
            continue
        c_pts = obstacle_points[idx]
        c_min = c_pts.min(axis=0)
        c_max = np.r_[c_pts[:, :2].max(axis=0), np.percentile(c_pts[:, 2], 99.9)]
        c_min[2] = 0.0
        cluster_info.append({
            "id": int(new_id), "label2d": int(lab), "num_points": int(cnt),
            "center": ((c_min + c_max) / 2.0).tolist(), "bbox_min": c_min.tolist(), "bbox_max": c_max.tolist(),
            "size": (c_max - c_min).tolist(), "height": float(c_max[2]), "height_top": float(c_max[2]),
        })
        clean[idx] = new_id
        new_id += 1
    print(f"Segmentasi objek: {len(cluster_info)} klaster ({time.time() - t0:.2f} s)")
    return cluster_info, clean


def _cluster_at(clusters, x, y, search=0.6):
    g = LABEL_GRID.get("grid")
    if g is None:
        return None
    res, half = LABEL_GRID["res"], LABEL_GRID["half"]
    by_lab = {c["label2d"]: c for c in clusters}
    cx, cy = int((x + half) / res), int((y + half) / res)
    r = int(search / res)
    best, bd = None, 1e9
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            i, j = cx + dx, cy + dy
            if 0 <= i < g.shape[0] and 0 <= j < g.shape[1] and g[i, j] in by_lab:
                d = dx * dx + dy * dy
                if d < bd:
                    bd, best = d, by_lab[g[i, j]]
    return best


def evaluate_accuracy_vs_ground_truth(clusters, gt_data):
    grouped_gt = {}
    for obs in gt_data["obstacles"]:
        if obs.get("type") == "fence":
            continue
        name = obs["name"]
        root_name = name
        for sfx in ["_T1", "_T2", "_T3", "_Sayap1", "_Sayap2", "_Pilar_Utara", "_Pilar_Selatan", "_Gelagar_Atas", "_Batang", "_Daun"]:
            if root_name.endswith(sfx):
                root_name = root_name[:-len(sfx)]
                break

        pos = np.array(obs["pos"], dtype=np.float32)
        size = np.array(obs["size"], dtype=np.float32)
        top = float(obs.get("height_top", pos[2] + size[2]/2))
        bmin = np.array([pos[0] - size[0]/2, pos[1] - size[1]/2, max(0.0, pos[2] - size[2]/2)])
        bmax = np.array([pos[0] + size[0]/2, pos[1] + size[1]/2, top])

        if root_name not in grouped_gt:
            grouped_gt[root_name] = {
                "name": root_name,
                "type": obs.get("type", "box"),
                "bbox_min": bmin,
                "bbox_max": bmax,
                "height_top": top
            }
        else:
            grouped_gt[root_name]["bbox_min"] = np.minimum(grouped_gt[root_name]["bbox_min"], bmin)
            grouped_gt[root_name]["bbox_max"] = np.maximum(grouped_gt[root_name]["bbox_max"], bmax)
            grouped_gt[root_name]["height_top"] = max(grouped_gt[root_name]["height_top"], top)

    gt_obstacles = []
    for k, v in grouped_gt.items():
        center = (v["bbox_min"] + v["bbox_max"]) / 2.0
        size = v["bbox_max"] - v["bbox_min"]
        gt_obstacles.append({
            "name": k,
            "type": v["type"],
            "pos": center.tolist(),
            "size": size.tolist(),
            "height_top": float(v["height_top"]),
            "bbox_min": v["bbox_min"],
            "bbox_max": v["bbox_max"]
        })

    matched_results = []
    gt_obstacles = [g for g in gt_obstacles if g["type"] not in ("perimeter_wall", "fence")]

    for gt in gt_obstacles:
        gt_top = gt["height_top"]
        gt_bbox_min = gt["bbox_min"]
        gt_bbox_max = gt["bbox_max"]
        best_match = _cluster_at(clusters, gt["pos"][0], gt["pos"][1])
        if best_match is not None:
            shared = [o["name"] for o in gt_obstacles if o is not gt and _cluster_at(clusters, o["pos"][0], o["pos"][1]) is best_match]
        else:
            shared = []

        if best_match is not None:
            c_min = np.array(best_match["bbox_min"])
            c_max = np.array(best_match["bbox_max"])

            inter_min = np.maximum(gt_bbox_min, c_min)
            inter_max = np.minimum(gt_bbox_max, c_max)
            inter_dims = np.maximum(0.0, inter_max - inter_min)
            inter_vol = np.prod(inter_dims)

            vol_gt = np.prod(gt_bbox_max - gt_bbox_min)
            vol_c = np.prod(c_max - c_min)
            union_vol = vol_gt + vol_c - inter_vol
            iou = inter_vol / max(union_vol, 1e-6)

            P = LABEL_GRID["pts"]
            m = gt_bbox_max - gt_bbox_min
            inside = ((P[:, 0] > gt_bbox_min[0] + 0.1 * m[0]) & (P[:, 0] < gt_bbox_max[0] - 0.1 * m[0]) &
                      (P[:, 1] > gt_bbox_min[1] + 0.1 * m[1]) & (P[:, 1] < gt_bbox_max[1] - 0.1 * m[1]))
            local_top = float(np.percentile(P[inside, 2], 99.5)) if inside.sum() > 20 else best_match["height_top"]
            h_err = abs(local_top - gt_top)
            pos_err = np.linalg.norm(np.array(gt["pos"]) - np.array(best_match["center"]))

            matched_results.append({
                "gt_name": gt["name"],
                "gt_type": gt["type"],
                "gt_pos": gt["pos"],
                "gt_height_top": gt_top,
                "est_pos": best_match["center"],
                "est_height_top": local_top,
                "pos_error_m": float(pos_err),
                "height_error_m": float(h_err),
                "iou_3d": float(iou),
                "points_in_cluster": best_match["num_points"],
                "merged_with": shared,
                "status": "TERDETEKSI" if not shared else "TERDETEKSI (menyatu)"
            })
        else:
            matched_results.append({
                "gt_name": gt["name"],
                "gt_type": gt["type"],
                "gt_pos": gt["pos"],
                "gt_height_top": gt_top,
                "est_pos": None,
                "est_height_top": None,
                "pos_error_m": None,
                "height_error_m": None,
                "iou_3d": 0.0,
                "points_in_cluster": 0,
                "status": "TIDAK TERDETEKSI"
            })

    detected = [r for r in matched_results if r["status"].startswith("TERDETEKSI")]
    separate = [r for r in matched_results if r["status"] == "TERDETEKSI"]
    n_detected = len(detected)
    n_total = len(gt_obstacles)
    det_rate = (n_detected / n_total) * 100.0 if n_total > 0 else 0.0

    mean_pos_err = np.mean([r["pos_error_m"] for r in detected]) if detected else 0.0
    mean_h_err = np.mean([r["height_error_m"] for r in detected]) if detected else 0.0
    mean_iou = np.mean([r["iou_3d"] for r in separate]) if separate else 0.0
    med_h_err = np.median([r["height_error_m"] for r in detected]) if detected else 0.0

    summary = {
        "total_gt_obstacles": n_total,
        "detected_obstacles": n_detected,
        "detection_rate_pct": float(det_rate),
        "mean_position_error_m": float(mean_pos_err),
        "mean_height_error_m": float(mean_h_err),
        "mean_iou_3d": float(mean_iou),
        "median_height_error_m": float(med_h_err),
        "separate_objects": len(separate),
        "details": matched_results
    }
    print(f"Evaluasi Akurasi vs Ground Truth: {n_detected}/{n_total} objek terdeteksi ({det_rate:.1f}%), "
          f"Rata-rata Error Posisi={mean_pos_err*100:.1f} cm, Error Tinggi={mean_h_err*100:.1f} cm, Mean IoU={mean_iou:.3f}")
    return summary


def compute_dem_and_hillshade(points, arena_size=12.0, res=0.05, azimuth=315.0, altitude=45.0):
    half = arena_size / 2.0
    nx = int(arena_size / res)
    ny = int(arena_size / res)

    x_bins = np.linspace(-half, half, nx + 1)
    y_bins = np.linspace(-half, half, ny + 1)

    ix = np.clip(np.digitize(points[:, 0], x_bins) - 1, 0, nx - 1)
    iy = np.clip(np.digitize(points[:, 1], y_bins) - 1, 0, ny - 1)

    dem_grid = np.zeros((ny, nx), dtype=np.float32)
    linear_idx = iy * nx + ix
    dem_flat = np.zeros(ny * nx, dtype=np.float32)
    np.maximum.at(dem_flat, linear_idx, np.maximum(points[:, 2], 0.0))
    dem_grid = dem_flat.reshape((ny, nx))

    dy, dx = np.gradient(dem_grid, res, res)
    slope = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(-dy, dx)

    zenith_rad = np.radians(90.0 - altitude)
    azimuth_rad = np.radians(360.0 - azimuth + 90.0)

    shaded = (np.cos(zenith_rad) * np.cos(slope) +
              np.sin(zenith_rad) * np.sin(slope) * np.cos(azimuth_rad - aspect))
    shaded = np.clip((shaded + 1.0) / 2.0, 0.0, 1.0)

    return dem_grid, shaded, x_bins, y_bins


def compute_2d_occupancy_grid(scan2d, traj=None, arena_size=12.0, res=0.08, max_range=10.0):
    half = arena_size / 2.0
    n = int(round(arena_size / res))
    log_odds = np.zeros((n, n), dtype=np.float32)
    if len(scan2d) == 0:
        return 1.0 - 1.0 / (1.0 + np.exp(log_odds)), np.empty((0, 2))
    scan2d = scan2d[(scan2d[:, 5] > 2.4) & (scan2d[:, 5] < 3.2) & (scan2d[:, 2] > 0.5)]
    if len(scan2d) == 0:
        return 0.5 * np.ones((n, n), np.float32), np.empty((0, 2))
    t, ang, rng, dx, dy = scan2d[:, 0], scan2d[:, 1], scan2d[:, 2], scan2d[:, 3], scan2d[:, 4]
    yaw = np.interp(t, traj[:, 0], np.unwrap(traj[:, 6])) if traj is not None and len(traj) else 0.0
    th = ang + yaw
    hit = rng < max_range - 0.1
    r = np.minimum(rng, max_range)
    steps = np.arange(0.0, 1.0, res / max_range * 0.7)
    for k in range(0, len(scan2d), 20000):
        sl = slice(k, k + 20000)
        fx = dx[sl, None] + r[sl, None] * steps[None, :] * np.cos(th[sl, None])
        fy = dy[sl, None] + r[sl, None] * steps[None, :] * np.sin(th[sl, None])
        ok = (steps[None, :] * r[sl, None]) < (r[sl, None] - res)
        ix = ((fx + half) / res).astype(int); iy = ((fy + half) / res).astype(int)
        ok &= (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
        np.add.at(log_odds, (iy[ok], ix[ok]), -0.05)
    ex = dx + r * np.cos(th); ey = dy + r * np.sin(th)
    ix = ((ex + half) / res).astype(int); iy = ((ey + half) / res).astype(int)
    ok = hit & (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
    np.add.at(log_odds, (iy[ok], ix[ok]), 0.9)
    log_odds = np.clip(log_odds, -4.0, 4.0)
    known = np.zeros_like(log_odds, bool)
    prob = 1.0 - 1.0 / (1.0 + np.exp(log_odds))
    return prob, np.c_[ex[hit], ey[hit]]


def export_ascii_ply(filepath, points, colors_rgb):
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    n_pts = len(points)
    with open(filepath, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment UTS 3D Mapping Drone\n")
        f.write(f"element vertex {n_pts}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for i in range(n_pts):
            p = points[i]
            c = colors_rgb[i]
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
    print(f"Model PLY diekspor: {filepath} ({n_pts:,} vertex).")


def generate_publication_figures(raw_pts, raw_sensors, filtered_pts, voxel_dict,
                                  ground_mask, obs_mask, clusters, traj,
                                  dem_grid, hillshade, prob_grid, gt_data,
                                  accuracy_summary, outdir, scan_hits=None):
    os.makedirs(outdir, exist_ok=True)
    plt.rcParams.update({
        'font.sans-serif': 'DejaVu Sans',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'figure.autolayout': False
    })

    print(f"membuat figur...")

    fig = plt.figure(figsize=(10, 7), dpi=200)
    ax = fig.add_subplot(111, projection='3d')
    sub_idx = np.random.choice(len(raw_pts), size=min(35000, len(raw_pts)), replace=False)
    sub_raw = raw_pts[sub_idx]
    sub_sens = raw_sensors[sub_idx]
    scatter = ax.scatter(sub_raw[:, 0], sub_raw[:, 1], sub_raw[:, 2],
                         c=sub_sens, cmap='tab10', s=1.2, alpha=0.6)
    ax.plot(traj[:, 1], traj[:, 2], traj[:, 3], color='crimson', lw=1.5, label='Trajektori Drone')
    ax.set_title("Hasil Pemindaian Mentah (Raw Point Cloud Multi-Sensor)\nTotal: "
                 f"{len(raw_pts):,} Titik (Termasuk Outlier & Noise)", fontweight='bold')
    ax.set_xlabel("X (meter)")
    ax.set_ylabel("Y (meter)")
    ax.set_zlabel("Ketinggian Z (meter)")
    ax.set_zlim(-0.8, 5.0)
    plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1, label="Sensor ID (0=Down, 1=Front, 2=Left, 3=Right, 4=LiDAR)")
    ax.legend(loc='upper right')
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_raw.png"))
    plt.close(fig)

    fig = plt.figure(figsize=(10, 7), dpi=200)
    ax = fig.add_subplot(111, projection='3d')
    sub_idx_f = np.random.choice(len(filtered_pts), size=min(35000, len(filtered_pts)), replace=False)
    sub_f = filtered_pts[sub_idx_f]
    ax.scatter(sub_f[:, 0], sub_f[:, 1], sub_f[:, 2],
               c=sub_f[:, 2], cmap='viridis', s=1.2, alpha=0.7)
    ax.set_title("Point Cloud Setelah Pembersihan Outlier (SOR k=8, $\sigma$=2.0)\n"
                 f"Tersisa: {len(filtered_pts):,} Titik Bersih (Outlier Dieliminasi)", fontweight='bold')
    ax.set_xlabel("X (meter)")
    ax.set_ylabel("Y (meter)")
    ax.set_zlabel("Ketinggian Z (meter)")
    ax.set_zlim(0.0, 4.5)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_filtered.png"))
    plt.close(fig)

    vox_01 = voxel_dict[0.10]["points"]
    fig = plt.figure(figsize=(10, 7), dpi=200)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(vox_01[:, 0], vox_01[:, 1], vox_01[:, 2],
               c='steelblue', s=3.5, alpha=0.8, edgecolors='none')
    ax.set_title("Model Voxel Grid (Ukuran Grid = 0.10 m)\n"
                 f"Jumlah Voxel: {len(vox_01):,} Titik Centroid (Reduksi: {voxel_dict[0.10]['reduction']:.1f}%)", fontweight='bold')
    ax.set_xlabel("X (meter)")
    ax.set_ylabel("Y (meter)")
    ax.set_zlabel("Ketinggian Z (meter)")
    ax.set_zlim(0.0, 4.5)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_voxel.png"))
    plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=200, subplot_kw={'projection': '3d'})

    sc1 = ax1.scatter(vox_01[:, 0], vox_01[:, 1], vox_01[:, 2],
                      c=vox_01[:, 2], cmap='turbo', s=3.0, alpha=0.85, vmin=0.0, vmax=4.0)
    ax1.set_title("Color Mapping Kontinu (Colormap: Turbo)", fontweight='bold')
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (m)")
    cbar1 = plt.colorbar(sc1, ax=ax1, shrink=0.55, pad=0.1)
    cbar1.set_label("Ketinggian Z (meter)")

    bands = np.floor(vox_01[:, 2] / 0.5) * 0.5
    sc2 = ax2.scatter(vox_01[:, 0], vox_01[:, 1], vox_01[:, 2],
                      c=bands, cmap='jet', s=3.0, alpha=0.85, vmin=0.0, vmax=4.0)
    ax2.set_title("Color Mapping Pita Diskrit (Interval 0.5 m)", fontweight='bold')
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_zlabel("Z (m)")
    cbar2 = plt.colorbar(sc2, ax=ax2, shrink=0.55, pad=0.1)
    cbar2.set_label("Pita Ketinggian (meter)")

    plt.suptitle("Representasi Ketinggian Model 3D dengan Pewarnaan Khusus ", fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_voxel_height.png"))
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=200, subplot_kw={'projection': '3d'})
    v_sizes = [0.05, 0.10, 0.20]
    for ax, vs in zip(axes, v_sizes):
        pts_v = voxel_dict[vs]["points"]
        red_v = voxel_dict[vs]["reduction"]
        ax.scatter(pts_v[:, 0], pts_v[:, 1], pts_v[:, 2],
                   c=pts_v[:, 2], cmap='turbo', s=1.8 if vs==0.05 else (4.0 if vs==0.10 else 9.0),
                   alpha=0.8, vmin=0.0, vmax=4.0)
        ax.set_title(f"Resolusi Voxel: {vs*100:.0f} cm\n"
                     f"Titik: {len(pts_v):,} | Reduksi: {red_v:.1f}%", fontweight='bold')
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_zlim(0.0, 4.0)
    plt.suptitle("Analisis Komparatif Efek Resolusi Voxel Grid Terhadap Densitas dan Reduksi Titik", fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_compare_voxel_sizes.png"))
    plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), dpi=200)
    im1 = ax1.imshow(dem_grid, extent=[-6, 6, -6, 6], origin='lower', cmap='terrain')
    ax1.set_title("Digital Elevation Model (DEM 2D, Res: 5 cm)", fontweight='bold')
    ax1.set_xlabel("Koordinat X (meter)")
    ax1.set_ylabel("Koordinat Y (meter)")
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.75)
    cbar1.set_label("Ketinggian Maksimum Z (meter)")
    cs = ax1.contour(dem_grid, levels=7, extent=[-6, 6, -6, 6], colors='black', alpha=0.35, linewidths=0.7)
    ax1.clabel(cs, inline=True, fontsize=7, fmt='%.1fm')

    ax2.imshow(hillshade, extent=[-6, 6, -6, 6], origin='lower', cmap='gray')
    ax2.imshow(dem_grid, extent=[-6, 6, -6, 6], origin='lower', cmap='turbo', alpha=0.45)
    ax2.set_title("Hillshade Topografis (Pencahayaan Azimuth 315°)", fontweight='bold')
    ax2.set_xlabel("Koordinat X (meter)")
    ax2.set_ylabel("Koordinat Y (meter)")

    plt.suptitle("Pemodelan Elevasi 2D dan Visualisasi Relief Hillshade", fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_dem.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7), dpi=200)
    im = ax.imshow(prob_grid, extent=[-6, 6, -6, 6], origin='lower', cmap='Greys', vmin=0.0, vmax=1.0)
    ax.plot(traj[:, 1], traj[:, 2], color='dodgerblue', lw=0.8, alpha=0.6, label='Jalur drone')
    ring = (traj[:, 3] > 2.4) & (traj[:, 3] < 3.2)
    ax.plot(traj[ring, 1], traj[ring, 2], color='orange', lw=1.6, label='Orbit 2,8 m (bidang scan 2D)')
    if scan_hits is not None and len(scan_hits):
        ax.scatter(scan_hits[:, 0], scan_hits[:, 1], s=0.3, color='crimson', alpha=0.5, label='Pantulan LiDAR')
    for o in gt_data.get("obstacles", []):
        if o.get("height_top", 0) > 2.8 and o.get("type") != "perimeter_wall":
            cx, cy = o["pos"][:2]; sx, sy = o["size"][:2]
            ax.add_patch(plt.Rectangle((cx - sx / 2, cy - sy / 2), sx, sy, fill=False, ls='--', lw=1.0, ec='green'))
    ax.plot([], [], ls='--', color='green', label='Tapak GT objek > 2,8 m')
    ax.set_title("Peta okupansi 2D dari LiDAR 360 derajat (bidang z = 2,8 m)", fontweight='bold')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    cbar = plt.colorbar(im, ax=ax, shrink=0.75)
    cbar.set_label("Probabilitas terisi (0 = bebas, 1 = terisi)")
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_occupancy2d.png"))
    plt.close(fig)

    fig = plt.figure(figsize=(11, 8), dpi=200)
    ax = fig.add_subplot(111, projection='3d')
    sub_g = filtered_pts[ground_mask]
    if len(sub_g) > 10000:
        sub_g = sub_g[np.random.choice(len(sub_g), 10000, replace=False)]
    ax.scatter(sub_g[:, 0], sub_g[:, 1], sub_g[:, 2],
               c='#a0c49d', s=0.8, alpha=0.4, label='Ground Plane (RANSAC)')

    cmap = get_colormap('tab20')
    for i, c in enumerate(clusters):
        color = cmap(i % 20)
        c_min = np.array(c["bbox_min"])
        c_max = np.array(c["bbox_max"])

        for edge_z in [c_min[2], c_max[2]]:
            ax.plot([c_min[0], c_max[0], c_max[0], c_min[0], c_min[0]],
                    [c_min[1], c_min[1], c_max[1], c_max[1], c_min[1]],
                    [edge_z, edge_z, edge_z, edge_z, edge_z], color='black', lw=0.8, alpha=0.7)
        for corner_x, corner_y in [(c_min[0], c_min[1]), (c_max[0], c_min[1]), (c_max[0], c_max[1]), (c_min[0], c_max[1])]:
            ax.plot([corner_x, corner_x], [corner_y, corner_y], [c_min[2], c_max[2]], color='black', lw=0.8, alpha=0.7)

    sub_obs = filtered_pts[obs_mask]
    if len(sub_obs) > 25000:
        sub_obs = sub_obs[np.random.choice(len(sub_obs), 25000, replace=False)]
    ax.scatter(sub_obs[:, 0], sub_obs[:, 1], sub_obs[:, 2],
               c='crimson', s=1.5, alpha=0.7, label='Obstacle Terklaster')

    ax.set_title("Segmentasi Ground Plane & Ekstraksi Objek 3D Terpisah\n"
                 f"Berhasil Mengisolasi {len(clusters)} Objek Struktural", fontweight='bold')
    ax.set_xlabel("X (meter)")
    ax.set_ylabel("Y (meter)")
    ax.set_zlabel("Z (meter)")
    ax.legend(loc='upper right')
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_segmentation.png"))
    plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=200)
    ax1.plot(traj[:, 1], traj[:, 2], color='navy', lw=1.8, label='Lintasan Drone')
    ax1.scatter(traj[0, 1], traj[0, 2], color='green', s=70, marker='o', label='Start / Helipad')
    ax1.scatter(traj[-1, 1], traj[-1, 2], color='red', s=70, marker='X', label='End / Landing')
    for obs in gt_data["obstacles"]:
        if obs["type"] != "fence":
            ax1.scatter(obs["pos"][0], obs["pos"][1], color='gray', s=30, alpha=0.5)
    ax1.set_title("Lintasan 2D Boustrophedon + Orbit Menara", fontweight='bold')
    ax1.set_xlabel("X (meter)")
    ax1.set_ylabel("Y (meter)")
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='lower right')

    ax2.plot(traj[:, 0], traj[:, 3], color='darkorange', lw=2.0)
    ax2.set_title("Profil Ketinggian Jelajah Drone vs Waktu Simulasi", fontweight='bold')
    ax2.set_xlabel("Waktu Simulasi (detik)")
    ax2.set_ylabel("Ketinggian Drone Z (meter)")
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.suptitle("Analisis Kinematika Navigasi dan Kontrol Autonomous Quadcopter", fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_trajectory.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4.5), dpi=200)
    ax.axis('off')

    stages = [
        "1. Akuisisi Data\nMulti-Sensor\n(Raw Point Cloud)",
        "2. Filtering & Crop\nBoundary Filter +\nSOR (k=8, $\sigma$=2.0)",
        "3. Downsampling\nVoxel Grid Hashing\n(Ukuran: 0.10 m)",
        "4. Segmentasi\nRANSAC Ground vs\n3D Connected Comp.",
        "5. Pemetaan 2D\nDEM + Hillshade &\nLog-Odds Grid",
        "6. Evaluasi Akurasi\n& Laporan PDF\n(IoU vs Ground Truth)"
    ]
    box_colors = ['#E2E8F0', '#CBD5E1', '#93C5FD', '#67E8F9', '#86EFAC', '#FDE047']

    for idx, (stage, col) in enumerate(zip(stages, box_colors)):
        x = 0.04 + idx * 0.158
        ax.text(x + 0.065, 0.5, stage, ha='center', va='center', fontsize=8.5, fontweight='bold',
                transform=ax.transAxes, color='#0F172A',
                bbox=dict(boxstyle='round,pad=0.6', facecolor=col, edgecolor='#1E293B', lw=1.5))
        if idx < len(stages) - 1:
            ax.annotate("", xy=(x + 0.155, 0.5), xytext=(x + 0.13, 0.5),
                        arrowprops=dict(arrowstyle="->", lw=2.0, color='#1E293B'),
                        xycoords=ax.transAxes)

    ax.set_title("Alur post-processing point cloud",
                 fontsize=13, fontweight='bold', y=0.92)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_pipeline.png"))
    plt.close(fig)

    res_g, half = 0.10, 6.0
    ng = int(2 * half / res_g)
    gx = np.clip(((raw_pts[:, 0] + half) / res_g).astype(int), 0, ng - 1)
    gy = np.clip(((raw_pts[:, 1] + half) / res_g).astype(int), 0, ng - 1)
    top = np.full((ng, ng), -1.0, np.float32)
    np.maximum.at(top, (gx, gy), raw_pts[:, 2])
    zt = top[gx, gy]
    cls = np.where(raw_pts[:, 2] < 0.08, 0, np.where(raw_pts[:, 2] >= zt - 0.10, 2, 1))
    names = ["depthDown (bawah)", "depthFront (depan 35°)", "depthLeft (kiri 35°)", "depthRight (kanan 35°)"]
    fig = plt.figure(figsize=(18, 8), dpi=200)
    shares = []
    for sid in range(4):
        m = raw_sensors == sid
        ps = raw_pts[m]
        ax = fig.add_subplot(2, 4, sid + 1, projection='3d')
        if len(ps) > 6000:
            ps = ps[np.random.choice(len(ps), 6000, replace=False)]
        ax.scatter(ps[:, 0], ps[:, 1], ps[:, 2], c=ps[:, 2], cmap='turbo', s=0.8, vmin=0, vmax=4.3, linewidths=0)
        ax.set_title(f"{names[sid]}\n{np.sum(m):,} titik".replace(",", "."), fontsize=10, fontweight='bold')
        ax.set_zlim(0, 4.5)
        ax.view_init(elev=25, azim=-60)
        c = cls[m]
        shares.append([np.mean(c == k) * 100 if len(c) else 0 for k in range(3)])
    ax = fig.add_subplot(2, 1, 2)
    shares = np.array(shares)
    left = np.zeros(4)
    for k, (lab, col) in enumerate([("ground", "#7f8c8d"), ("badan objek", "#2e86c1"), ("puncak/atap", "#e74c3c")]):
        ax.barh(names, shares[:, k], left=left, color=col, label=lab)
        for i, v in enumerate(shares[:, k]):
            if v > 4:
                ax.text(left[i] + v / 2, i, f"{v:.1f}%".replace(".", ","), ha='center', va='center', color='white', fontsize=10)
        left += shares[:, k]
    ax.set_xlim(0, 100)
    ax.set_xlabel("Proporsi titik (%)")
    ax.invert_yaxis()
    ax.legend(loc='lower right', ncol=3)
    ax.set_title("Pembagian titik tiap sensor ke ground, badan, dan puncak objek", fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_sensor_breakdown.png"))
    plt.close(fig)
    STATS_EXTRA["sensor_class_share_pct"] = {names[i]: shares[i].round(2).tolist() for i in range(4)}

    print(f"figur selesai")


def generate_360_video(voxel_pts, clusters, traj, outdir, video_name="hasil_3d_mapping.mp4",
                       raw_pts=None, raw_sensors=None, filtered_pts=None, stats=None, fps=30, seconds_per_stage=9):
    video_path = os.path.join(outdir, video_name)
    ffmpeg_bin = shutil.which("ffmpeg") or os.path.expanduser("~/.local/bin/ffmpeg")
    W, H = 1920, 1080
    proc = subprocess.Popen([ffmpeg_bin, "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
                             "-pix_fmt", "yuv420p", "-crf", "19", video_path], stdin=subprocess.PIPE)
    rng = np.random.default_rng(0)
    def sub(a, n):
        return a if a is None or len(a) <= n else a[rng.choice(len(a), n, replace=False)]
    raw_s = sub(raw_pts, 30000) if raw_pts is not None else sub(voxel_pts, 30000)
    raw_id = None
    if raw_pts is not None and raw_sensors is not None and len(raw_pts) > 30000:
        idx = rng.choice(len(raw_pts), 30000, replace=False)
        raw_s, raw_id = raw_pts[idx], raw_sensors[idx]
    filt_s = sub(filtered_pts if filtered_pts is not None else voxel_pts, 30000)
    vox_s = sub(voxel_pts, 26000)
    cmap = get_colormap('turbo')
    norm = Normalize(vmin=0.0, vmax=4.3)
    sensor_cols = np.array([[0.12, 0.47, 0.71], [1.0, 0.5, 0.05], [0.17, 0.63, 0.17], [0.84, 0.15, 0.16], [0.58, 0.4, 0.74]])
    stages = [
        ("1/6  Point cloud mentah dari 4 kamera depth", "warna = sensor asal (biru bawah, oranye depan, hijau kiri, merah kanan), garis = jalur drone"),
        ("2/6  Setelah crop dan statistical outlier removal", "titik melayang dan noise di luar arena dibuang"),
        ("3/6  Voxel grid downsampling 0,10 m", "satu titik centroid per voxel"),
        ("4/6  Color mapping ketinggian (colormap turbo)", "biru = lantai, hijau-kuning = badan objek, merah = puncak tertinggi"),
        ("5/6  Segmentasi ground (RANSAC) dan klaster objek", "kotak kuning = bounding box tiap objek terdeteksi"),
        ("6/6  Hasil akhir peta 3D", "tampak atas dan orbit penutup"),
    ]
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor='white')
    ax = fig.add_axes([0.0, 0.02, 0.78, 0.86], projection='3d')
    cax = fig.add_axes([0.80, 0.25, 0.015, 0.5])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Ketinggian z [m]", fontsize=13)
    t_title = fig.text(0.02, 0.95, "", fontsize=24, fontweight='bold')
    t_sub = fig.text(0.02, 0.91, "", fontsize=15, color='#444444')
    t_side = fig.text(0.80, 0.80, "", fontsize=13, family='monospace', va='top')
    side = ""
    if stats:
        side = "\n".join(f"{k:<14}{v}" for k, v in stats.items())
    n_stage = int(fps * seconds_per_stage)
    total = n_stage * len(stages)
    t0 = time.time()
    for f in range(total):
        st, k = divmod(f, n_stage)
        u = k / n_stage
        azim = -60 + 360.0 * f / total * 1.5
        elev = 28 + 8 * np.sin(2 * np.pi * f / total)
        ax.cla()
        if st == 0:
            c = sensor_cols[np.clip(raw_id, 0, 4)] if raw_id is not None else 'steelblue'
            ax.scatter(raw_s[:, 0], raw_s[:, 1], raw_s[:, 2], c=c, s=0.8, linewidths=0)
            ax.plot(traj[:, 1], traj[:, 2], traj[:, 3], color='black', lw=1.0)
        elif st == 1:
            ax.scatter(filt_s[:, 0], filt_s[:, 1], filt_s[:, 2], c='#5b6b7b', s=0.8, linewidths=0)
        elif st == 2:
            ax.scatter(vox_s[:, 0], vox_s[:, 1], vox_s[:, 2], c='#7d8fa3', s=4, marker='s', linewidths=0)
        else:
            ax.scatter(vox_s[:, 0], vox_s[:, 1], vox_s[:, 2], c=vox_s[:, 2], cmap=cmap, norm=norm, s=4, marker='s', linewidths=0)
            if st == 4:
                for cl in clusters:
                    lo, hi = np.array(cl["bbox_min"]), np.array(cl["bbox_max"])
                    xs = [lo[0], hi[0], hi[0], lo[0], lo[0]]
                    ys = [lo[1], lo[1], hi[1], hi[1], lo[1]]
                    ax.plot(xs, ys, [hi[2]] * 5, color='gold', lw=1.2)
                    ax.plot(xs, ys, [lo[2]] * 5, color='gold', lw=0.6)
                    for x_, y_ in zip(xs[:4], ys[:4]):
                        ax.plot([x_, x_], [y_, y_], [lo[2], hi[2]], color='gold', lw=0.6)
            if st == 5:
                elev = 28 + (88 - 28) * min(1.0, u * 2) if u < 0.6 else 88 - (88 - 35) * (u - 0.6) / 0.4
                ax.plot(traj[:, 1], traj[:, 2], traj[:, 3], color='black', lw=0.7, alpha=0.5)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim(-6, 6); ax.set_ylim(-6, 6); ax.set_zlim(0, 5)
        ax.set_box_aspect((12, 12, 5))
        ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
        cax.set_visible(st >= 3)
        t_title.set_text(stages[st][0])
        t_sub.set_text(stages[st][1])
        t_side.set_text(side)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        proc.stdin.write(np.ascontiguousarray(buf).tobytes())
        if (f + 1) % 150 == 0:
            print(f"  frame {f + 1}/{total} ({time.time() - t0:.0f} s)")
    plt.close(fig)
    proc.stdin.close()
    proc.wait()
    print(f"Video hasil: {video_path} ({os.path.getsize(video_path) / 1e6:.1f} MB)")


def run_pipeline(input_npz, gt_json, voxel_size, outdir, make_video=False, is_synthetic=False):
    t_start = time.time()
    os.makedirs(outdir, exist_ok=True)
    stats_path = os.path.join(outdir, "stats.json")

    if is_synthetic or not os.path.exists(input_npz) or not os.path.exists(gt_json):
        print(f"Data masukan ({input_npz}) atau GT ({gt_json}) belum tersedia.")
        print(f"Mengaktifkan mode generator data sintetis sesuai kontrak...")
        gt_data = generate_synthetic_ground_truth(arena_size=12.0)
        os.makedirs(os.path.dirname(os.path.abspath(gt_json)), exist_ok=True)
        with open(gt_json, 'w') as f:
            json.dump(gt_data, f, indent=2)
        generate_synthetic_scan(gt_data, input_npz, num_target_points=260000)
    else:
        print(f"Memuat data ground truth yang sudah ada: {gt_json}")
        with open(gt_json, 'r') as f:
            gt_data = json.load(f)

    print(f"Memuat raw scan dari: {input_npz}")
    raw_data = np.load(input_npz, allow_pickle=True)
    raw_points = raw_data["points"].astype(np.float32)
    raw_sensors = raw_data["sensor_id"].astype(np.uint8)
    traj = raw_data["traj"].astype(np.float32)
    scan2d = raw_data["scan2d"].astype(np.float32) if "scan2d" in raw_data else np.zeros((0, 6))

    n_raw_initial = len(raw_points)

    cropped_pts, cropped_sensors, crop_mask = crop_point_cloud(raw_points, raw_sensors)

    filtered_pts, filtered_sensors, sor_mask = remove_statistical_outliers(cropped_pts, cropped_sensors)

    voxel_benchmark = {}
    for vs in [0.05, 0.10, 0.20]:
        cents, counts, red, dt = voxel_downsample(filtered_pts, voxel_size=vs)
        voxel_benchmark[vs] = {
            "points": cents,
            "counts": counts,
            "reduction": float(red),
            "num_voxels": int(len(cents)),
            "elapsed_sec": float(dt)
        }

    selected_voxel = voxel_benchmark[voxel_size]["points"]

    ground_mask, obs_mask, ground_plane = segment_ground_ransac(filtered_pts)

    clusters, cluster_labels = cluster_3d_objects(filtered_pts[obs_mask], voxel_res=0.10)
    accuracy_eval = evaluate_accuracy_vs_ground_truth(clusters, gt_data)

    dem_grid, hillshade, _, _ = compute_dem_and_hillshade(filtered_pts, arena_size=12.0)
    prob_grid, scan_hits = compute_2d_occupancy_grid(scan2d, traj, arena_size=12.0)

    norm = Normalize(vmin=0.0, vmax=3.8)
    cmap = get_colormap('turbo')
    raw_sub_idx = np.random.choice(len(filtered_pts), size=min(60000, len(filtered_pts)), replace=False)
    raw_colors = (cmap(norm(filtered_pts[raw_sub_idx, 2]))[:, :3] * 255).astype(np.uint8)
    export_ascii_ply(os.path.join(outdir, "map_raw.ply"), filtered_pts[raw_sub_idx], raw_colors)

    vox_colors = (cmap(norm(selected_voxel[:, 2]))[:, :3] * 255).astype(np.uint8)
    export_ascii_ply(os.path.join(outdir, "map_voxel.ply"), selected_voxel, vox_colors)

    generate_publication_figures(
        raw_pts=raw_points,
        raw_sensors=raw_sensors,
        filtered_pts=filtered_pts,
        voxel_dict=voxel_benchmark,
        ground_mask=ground_mask,
        obs_mask=obs_mask,
        clusters=clusters,
        traj=traj,
        dem_grid=dem_grid,
        hillshade=hillshade,
        prob_grid=prob_grid,
        gt_data=gt_data,
        accuracy_summary=accuracy_eval,
        outdir=outdir,
        scan_hits=scan_hits
    )

    if make_video:
        vstats = {
            "Titik mentah": f"{len(raw_points):,}".replace(",", "."),
            "Setelah filter": f"{len(filtered_pts):,}".replace(",", "."),
            "Voxel 0,10 m": f"{len(selected_voxel):,}".replace(",", "."),
            "Reduksi": f"{(1 - len(selected_voxel) / max(1, len(raw_points))) * 100:.2f} %".replace(".", ","),
            "Objek": str(len(clusters)),
        }
        generate_360_video(selected_voxel, clusters, traj, outdir, raw_pts=raw_points,
                           raw_sensors=raw_sensors, filtered_pts=filtered_pts, stats=vstats)

    total_elapsed = time.time() - t_start
    stats = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_processing_time_sec": float(total_elapsed),
        "raw_points_count": int(n_raw_initial),
        "cropped_points_count": int(len(cropped_pts)),
        "filtered_points_count": int(len(filtered_pts)),
        "cropped_removed_count": int(n_raw_initial - len(cropped_pts)),
        "outliers_removed_count": int(len(cropped_pts) - len(filtered_pts)),
        "outlier_pct": float((len(cropped_pts) - len(filtered_pts)) / max(1, len(cropped_pts)) * 100.0),
        "voxel_reduction_benchmark": {
            "0.05m": {
                "num_voxels": voxel_benchmark[0.05]["num_voxels"],
                "reduction_pct": voxel_benchmark[0.05]["reduction"],
                "time_sec": voxel_benchmark[0.05]["elapsed_sec"]
            },
            "0.10m": {
                "num_voxels": voxel_benchmark[0.10]["num_voxels"],
                "reduction_pct": voxel_benchmark[0.10]["reduction"],
                "time_sec": voxel_benchmark[0.10]["elapsed_sec"]
            },
            "0.20m": {
                "num_voxels": voxel_benchmark[0.20]["num_voxels"],
                "reduction_pct": voxel_benchmark[0.20]["reduction"],
                "time_sec": voxel_benchmark[0.20]["elapsed_sec"]
            }
        },
        "selected_voxel_size_m": float(voxel_size),
        "selected_voxel_count": int(len(selected_voxel)),
        "selected_voxel_reduction_pct": float(voxel_benchmark[voxel_size]["reduction"]),
        "ground_plane": {
            "num_ground_points": int(np.sum(ground_mask)),
            "num_obstacle_points": int(np.sum(obs_mask)),
            "plane_equation": ground_plane.tolist()
        },
        "clusters_count": int(len(clusters)),
        "accuracy": accuracy_eval,
        "trajectory": {
            "total_waypoints": int(len(traj)),
            "total_time_sec": float(traj[-1, 0]) if len(traj) > 0 else 0.0,
            "max_altitude_m": float(np.max(traj[:, 3])) if len(traj) > 0 else 0.0,
            "flight_length_m": float(np.sum(np.linalg.norm(np.diff(traj[:, 1:4], axis=0), axis=1))) if len(traj) > 1 else 0.0
        },
        "arena": gt_data.get("arena", {"size_x": 12.0, "size_y": 12.0})
    }

    stats.update(STATS_EXTRA)
    stats["scan2d_rays"] = int(len(scan2d))
    stats["scan2d_hits"] = int(np.sum(scan2d[:, 2] < 9.8)) if len(scan2d) else 0
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"stats: {stats_path}")
    print(f"    - Raw Points: {stats['raw_points_count']:,}")
    print(f"    - Filtered Points: {stats['filtered_points_count']:,} ({stats['outliers_removed_count']:,} outlier dibuang)")
    print(f"    - Voxel (0.10m): {stats['selected_voxel_count']:,} ({stats['selected_voxel_reduction_pct']:.2f}% reduksi)")
    print(f"    - Objek Terdeteksi: {accuracy_eval['detected_obstacles']}/{accuracy_eval['total_gt_obstacles']} "
          f"({accuracy_eval['detection_rate_pct']:.1f}%), Mean IoU: {accuracy_eval['mean_iou_3d']:.3f}")
    print(f"    - Total Waktu Eksekusi: {total_elapsed:.2f} detik\n")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Post-processing point cloud hasil mapping drone")
    parser.add_argument("--input", default="data/raw_scan.npz", help="Jalur berkas input raw_scan.npz")
    parser.add_argument("--gt", default="scene/ground_truth.json", help="Jalur berkas ground_truth.json")
    parser.add_argument("--voxel", type=float, default=0.10, help="Ukuran grid voxel downsampling (meter, default: 0.10)")
    parser.add_argument("--outdir", default="results", help="Direktori keluaran hasil (default: results)")
    parser.add_argument("--video", action="store_true", help="Generate video animasi 360 derajat MP4 via ffmpeg")
    parser.add_argument("--synthetic", action="store_true", help="Hasilkan dataset sintetis secara eksplisit")
    parser.add_argument("--show", action="store_true", help="Tampilkan plot secara interaktif (bila ada GUI)")

    args = parser.parse_args()

    input_path = args.input
    gt_path = args.gt
    outdir = args.outdir

    in_drone_mapping = (os.path.basename(os.getcwd()) == "drone_mapping") or (os.path.exists("python") and os.path.exists("scene"))
    if in_drone_mapping:
        if input_path.startswith("drone_mapping/"):
            input_path = input_path[len("drone_mapping/"):]
        if gt_path.startswith("drone_mapping/"):
            gt_path = gt_path[len("drone_mapping/"):]
        if outdir.startswith("drone_mapping/"):
            outdir = outdir[len("drone_mapping/"):]
    else:
        if not input_path.startswith("drone_mapping/"):
            input_path = os.path.join("drone_mapping", input_path)
        if not gt_path.startswith("drone_mapping/"):
            gt_path = os.path.join("drone_mapping", gt_path)
        if not outdir.startswith("drone_mapping/"):
            outdir = os.path.join("drone_mapping", outdir)

    run_pipeline(
        input_npz=input_path,
        gt_json=gt_path,
        voxel_size=args.voxel,
        outdir=outdir,
        make_video=args.video,
        is_synthetic=args.synthetic
    )


if __name__ == "__main__":
    main()
