
import os
import sys
import time
import json
import argparse
import subprocess
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Optional, List, Dict, Any

from common import (
    load_raw_scan,
    SENSOR_NAMES,
    SENSOR_DOWN,
    SENSOR_FRONT,
    SENSOR_LEFT,
    SENSOR_RIGHT,
    SENSOR_LIDAR2D
)


class LiveVisualizer:
    def __init__(
        self,
        title: str = "UTS 3D Drone Mapping - Live Acquisition Dashboard",
        max_display_points: int = 40000,
        record_path: Optional[str] = None,
        fps: int = 10,
        headless: bool = False
    ):
        self.title = title
        self.max_display_points = max_display_points
        self.record_path = record_path
        self.fps = fps
        self.headless = headless or ("DISPLAY" not in os.environ and "WAYLAND_DISPLAY" not in os.environ)

        if self.headless and not self.record_path:
            self.enabled = False
            return
        self.enabled = True

        if self.headless:
            matplotlib.use("Agg")

        plt.style.use("dark_background")
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle(self.title, fontsize=14, fontweight="bold", color="#00e5ff")

        gs = GridSpec(2, 2, width_ratios=[1.2, 1.0], height_ratios=[1.3, 0.7], hspace=0.28, wspace=0.22)

        self.ax_3d = self.fig.add_subplot(gs[0, 0], projection="3d")
        self.ax_3d.set_title("3D Point Cloud (Z-Turbo Colored) & Flight Path", fontsize=11, color="#00e5ff")
        self.ax_3d.set_xlabel("X (m)", labelpad=4)
        self.ax_3d.set_ylabel("Y (m)", labelpad=4)
        self.ax_3d.set_zlabel("Z (m)", labelpad=4)
        self.ax_3d.set_xlim(-6.5, 6.5)
        self.ax_3d.set_ylim(-6.5, 6.5)
        self.ax_3d.set_zlim(0, 5.5)

        self.ax_2d = self.fig.add_subplot(gs[0, 1])
        self.ax_2d.set_title("2D Top-Down Floor Map & LiDAR Coverage", fontsize=11, color="#76ff03")
        self.ax_2d.set_xlabel("X (m)")
        self.ax_2d.set_ylabel("Y (m)")
        self.ax_2d.set_xlim(-6.5, 6.5)
        self.ax_2d.set_ylim(-6.5, 6.5)
        self.ax_2d.set_aspect("equal")
        self.ax_2d.grid(True, linestyle="--", alpha=0.3)

        self.ax_alt = self.fig.add_subplot(gs[1, 0])
        self.ax_alt.set_title("Drone Altitude Profile z(t)", fontsize=11, color="#ffd600")
        self.ax_alt.set_xlabel("Sim Time (s)")
        self.ax_alt.set_ylabel("Altitude Z (m)")
        self.ax_alt.set_ylim(0, 6.0)
        self.ax_alt.grid(True, linestyle="--", alpha=0.3)

        self.ax_pts = self.fig.add_subplot(gs[1, 1])
        self.ax_pts.set_title("Point Count & Sensor Breakdown", fontsize=11, color="#ff4081")
        self.ax_pts.set_xlabel("Sim Time (s)")
        self.ax_pts.set_ylabel("Total Points")
        self.ax_pts.grid(True, linestyle="--", alpha=0.3)

        self.ffmpeg_proc = None
        if self.record_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.record_path)), exist_ok=True)
            self._init_ffmpeg()

        self.last_update_wall_time = 0.0
        self.update_interval = 0.4

        if not self.headless:
            plt.ion()
            plt.show(block=False)

    def _init_ffmpeg(self):
        self.fig.canvas.draw()
        w, h = self.fig.canvas.get_width_height()
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{w}x{h}",
            "-pix_fmt", "rgba",
            "-r", str(self.fps),
            "-i", "-",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "22",
            "-pix_fmt", "yuv420p",
            self.record_path
        ]
        try:
            self.ffmpeg_proc = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print(f"[LiveVisualizer] Perekam video aktif: {self.record_path} ({w}x{h} @ {self.fps}fps)")
        except Exception as e:
            print(f"[LiveVisualizer] Peringatan: Gagal menjalankan ffmpeg ({e}). Perekaman dinonaktifkan.")
            self.ffmpeg_proc = None

    def update(
        self,
        points: np.ndarray,
        sensor_ids: np.ndarray,
        traj: np.ndarray,
        current_pose: np.ndarray,
        sim_time: float,
        force: bool = False
    ):
        if not self.enabled:
            return

        now = time.time()
        if not force and (now - self.last_update_wall_time) < self.update_interval:
            return
        self.last_update_wall_time = now

        num_points = len(points)
        if num_points > self.max_display_points:
            stride = int(np.ceil(num_points / self.max_display_points))
            disp_pts = points[::stride]
            disp_sids = sensor_ids[::stride]
        else:
            disp_pts = points
            disp_sids = sensor_ids

        self.ax_3d.cla()
        self.ax_3d.set_title(
            f"3D Point Cloud ({num_points:,} pts) | t = {sim_time:.1f} s",
            fontsize=11, color="#00e5ff"
        )
        self.ax_3d.set_xlabel("X (m)", labelpad=2)
        self.ax_3d.set_ylabel("Y (m)", labelpad=2)
        self.ax_3d.set_zlabel("Z (m)", labelpad=2)
        self.ax_3d.set_xlim(-6.5, 6.5)
        self.ax_3d.set_ylim(-6.5, 6.5)
        self.ax_3d.set_zlim(0, 5.5)

        if len(disp_pts) > 0:
            sc = self.ax_3d.scatter(
                disp_pts[:, 0], disp_pts[:, 1], disp_pts[:, 2],
                c=disp_pts[:, 2], cmap="turbo", s=1.2, alpha=0.6, vmin=0.0, vmax=4.5
            )

        if len(traj) > 1:
            self.ax_3d.plot(
                traj[:, 1], traj[:, 2], traj[:, 3],
                color="#00e5ff", linewidth=1.5, label="Trajectory"
            )

        self.ax_3d.scatter(
            [current_pose[0]], [current_pose[1]], [current_pose[2]],
            color="#ff1744", s=60, marker="X", label="Drone"
        )

        self.ax_2d.cla()
        self.ax_2d.set_title("2D Top-Down Map & Flight Track", fontsize=11, color="#76ff03")
        self.ax_2d.set_xlabel("X (m)")
        self.ax_2d.set_ylabel("Y (m)")
        self.ax_2d.set_xlim(-6.5, 6.5)
        self.ax_2d.set_ylim(-6.5, 6.5)
        self.ax_2d.set_aspect("equal")
        self.ax_2d.grid(True, linestyle="--", alpha=0.3)

        arena_rect = plt.Rectangle((-6, -6), 12, 12, fill=False, edgecolor="#ff9100", linestyle=":", linewidth=1.5)
        self.ax_2d.add_patch(arena_rect)

        if len(disp_pts) > 0:
            self.ax_2d.scatter(
                disp_pts[:, 0], disp_pts[:, 1],
                c=disp_pts[:, 2], cmap="turbo", s=1.0, alpha=0.35, vmin=0.0, vmax=4.5
            )

        if len(traj) > 1:
            self.ax_2d.plot(traj[:, 1], traj[:, 2], color="#00e5ff", linewidth=1.2)

        yaw = current_pose[3] if len(current_pose) > 3 else 0.0
        dx = 0.8 * np.cos(yaw)
        dy = 0.8 * np.sin(yaw)
        self.ax_2d.arrow(
            current_pose[0], current_pose[1], dx, dy,
            head_width=0.3, head_length=0.3, fc="#ff1744", ec="#ff1744"
        )
        self.ax_2d.scatter([current_pose[0]], [current_pose[1]], color="#ffea00", s=40)

        self.ax_alt.cla()
        self.ax_alt.set_title(f"Altitude Profile | Current Z: {current_pose[2]:.2f} m", fontsize=11, color="#ffd600")
        self.ax_alt.set_xlabel("Sim Time (s)")
        self.ax_alt.set_ylabel("Z (m)")
        self.ax_alt.set_ylim(0, 6.0)
        self.ax_alt.grid(True, linestyle="--", alpha=0.3)

        if len(traj) > 0:
            self.ax_alt.plot(traj[:, 0], traj[:, 3], color="#ffd600", linewidth=1.5)
            self.ax_alt.set_xlim(0, max(sim_time + 1.0, 10.0))

        self.ax_pts.cla()
        self.ax_pts.set_title(f"Total Acquisition: {num_points:,} Points", fontsize=11, color="#ff4081")
        self.ax_pts.set_xlabel("Sensor")
        self.ax_pts.set_ylabel("Titik Terkumpul")
        self.ax_pts.grid(True, linestyle="--", alpha=0.3, axis="y")

        sensor_counts = {name: 0 for name in SENSOR_NAMES.values()}
        if len(sensor_ids) > 0:
            unique, counts = np.unique(sensor_ids, return_counts=True)
            for sid, cnt in zip(unique, counts):
                if sid in SENSOR_NAMES:
                    sensor_counts[SENSOR_NAMES[sid]] = cnt

        names = list(sensor_counts.keys())
        values = list(sensor_counts.values())
        colors = ["#2979ff", "#00e676", "#ff9100", "#d500f9", "#ff1744"]

        bars = self.ax_pts.bar(names, values, color=colors, alpha=0.85)
        for bar in bars:
            h_val = bar.get_height()
            if h_val > 0:
                self.ax_pts.text(
                    bar.get_x() + bar.get_width() / 2.0, h_val + 50,
                    f"{int(h_val):,}", ha="center", va="bottom", fontsize=8, color="#ffffff"
                )

        self.fig.canvas.draw()
        if not self.headless:
            plt.pause(0.001)

        if self.ffmpeg_proc and self.ffmpeg_proc.stdin:
            try:
                rgba_buf = self.fig.canvas.buffer_rgba()
                self.ffmpeg_proc.stdin.write(rgba_buf)
            except Exception as e:
                print(f"[LiveVisualizer] Gagal menulis frame video: {e}")

    def close(self):
        if self.ffmpeg_proc:
            try:
                if self.ffmpeg_proc.stdin:
                    self.ffmpeg_proc.stdin.close()
                self.ffmpeg_proc.wait(timeout=30)
                print(f"[LiveVisualizer] Rekaman video berhasil disimpan: {self.record_path}")
            except Exception as e:
                print(f"[LiveVisualizer] Kesalahan menutup ffmpeg: {e}")
                try:
                    self.ffmpeg_proc.kill()
                except Exception:
                    pass
            self.ffmpeg_proc = None

        if not self.headless:
            try:
                plt.close(self.fig)
            except Exception:
                pass


def replay_from_file(npz_file: str, record_path: Optional[str] = None):
    print(f"[Replay] Memuat data dari {npz_file}...")
    data = load_raw_scan(npz_file)
    points = data["points"]
    sensor_ids = data["sensor_id"]
    t_pts = data["t"]
    traj = data["traj"]

    print(f"[Replay] Total titik: {len(points):,}, Trajectory steps: {len(traj):,}")

    viz = LiveVisualizer(
        title=f"Replay Mapping 3D - {os.path.basename(npz_file)}",
        record_path=record_path,
        fps=15
    )

    step_indices = np.linspace(0, len(traj) - 1, min(len(traj), 150), dtype=int)
    for idx in step_indices:
        curr_t = traj[idx, 0]
        curr_pose = [traj[idx, 1], traj[idx, 2], traj[idx, 3], traj[idx, 6]]
        sub_traj = traj[:idx + 1]

        mask_pts = t_pts <= curr_t
        sub_pts = points[mask_pts]
        sub_sids = sensor_ids[mask_pts]

        viz.update(sub_pts, sub_sids, sub_traj, curr_pose, curr_t, force=True)
        time.sleep(0.03)

    viz.close()
    print("[Replay] Pemutaran ulang selesai.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Visualizer & Replayer untuk 3D Drone Mapping")
    parser.add_argument("--in", dest="input_file", type=str, default="data/raw_scan.npz", help="File raw_scan.npz untuk replay")
    parser.add_argument("--record", type=str, default=None, help="Path output video mp4 (misal: results/mapping_live.mp4)")
    args = parser.parse_args()

    if os.path.exists(args.input_file):
        replay_from_file(args.input_file, args.record)
    else:
        print(f"File {args.input_file} tidak ditemukan. Silakan jalankan mission.py terlebih dahulu.")
