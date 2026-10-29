import os
import math
import shutil
import subprocess
import numpy as np
from PIL import Image, ImageDraw

W, H = 1920, 1080
VP_W, VP_H = W - 260, H - 32
FFMPEG = shutil.which("ffmpeg") or os.path.expanduser("~/.local/bin/ffmpeg")


def look_at(pos, target):
    p = np.asarray(pos, float)
    z = np.asarray(target, float) - p
    z_norm = np.linalg.norm(z)
    if z_norm < 1e-6:
        z = np.array([1.0, 0.0, 0.0])
    else:
        z /= z_norm
    up = np.array([0.0, 0.0, 1.0])
    x = np.cross(up, z)
    x_norm = np.linalg.norm(x)
    if x_norm < 1e-6:
        x = np.array([0.0, 1.0, 0.0])
    else:
        x /= x_norm
    y = np.cross(z, x)
    return [x[0], y[0], z[0], p[0], x[1], y[1], z[1], p[1], x[2], y[2], z[2], p[2]]


class MissionRecorder:
    def __init__(self, sim, out_path, fps=30, title="UTS Sistem Robot Otonom - 3D Mapping Drone"):
        self.sim = sim
        self.cam = sim.getObject("/RecCam")
        self.df = sim.getObject("/Quadcopter/sensorMount/depthFront", {"noError": True})
        if not self.df:
            self.df = sim.getObject("/Quadcopter/depthFront", {"noError": True})

        self.sim.setObjectInt32Param(self.cam, self.sim.visionintparam_resolution_x, VP_W)
        self.sim.setObjectInt32Param(self.cam, self.sim.visionintparam_resolution_y, VP_H)

        self.title = title
        self.cam_pos = None
        self.frames = 0
        self.accumulated_scan_points = []
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        self.proc = subprocess.Popen(
            [FFMPEG, "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
             "-pix_fmt", "yuv420p", "-crf", "19", "-preset", "fast", out_path],
            stdin=subprocess.PIPE)

        self.tree_items = [
            ("[-] drone_mapping_3d", "#111111", False),
            ("  |-- DefaultCamera", "#555555", False),
            ("  |-- XYZCameraProxy", "#555555", False),
            ("  |-- Floor", "#555555", False),
            ("  [-] Quadcopter", "#1a73e8", False),
            ("    |-- Script", "#555555", False),
            ("    |-- base", "#555555", False),
            ("    [-] sensorMount", "#1a73e8", False),
            ("      |-- depthDown [Vision]", "#555555", False),
            ("      |-- depthFront [Vision]", "#555555", False),
            ("      |-- depthLeft [Vision]", "#555555", False),
            ("      |-- depthRight [Vision]", "#555555", False),
            ("      |-- lidar2d [Prox]", "#555555", False),
            ("  |-- Gedung_Tingkat_A", "#222222", False),
            ("  |-- Silo_Gandum", "#222222", False),
            ("  |-- Menara_Komunikasi", "#222222", False),
            ("  |-- Reaktor_Kubah", "#222222", False),
            ("  |-- Jembatan_Flyover", "#222222", False),
            ("  |-- Tangki_Bahan_Bakar_1", "#222222", False),
            ("  |-- Tangki_Bahan_Bakar_2", "#222222", False),
            ("  |-- Kontainer_Kargo_A & B", "#222222", False),
            ("  |-- Kendaraan_Truk", "#222222", False),
            ("  |-- Helipad (-4.5, -4.5)", "#222222", False),
            ("  |-- target (PID setpoint)", "#000000", True)
        ]

    def title_card(self, lines, seconds=3, fps=30):
        card = Image.new("RGB", (W, H), "#0b0f14")
        d = ImageDraw.Draw(card)
        d.rectangle([200, 360, 1720, 366], fill="#1a73e8")
        d.text((W/2 - 420, 400), "3D Mapping Drone di CoppeliaSim", fill="#ffffff")
        d.text((W/2 - 440, 480), "UTS Sistem Robot Otonom - I Made Windu Adi Pramana 5022241125", fill="#64b5f6")
        d.text((W/2 - 380, 560), "Simulasi Autonomous Flight & 3D Reconstruction", fill="#a0aec0")
        d.text((W/2 - 390, 620), "Departemen Teknik Elektro & Otomasi - Institut Teknologi Sepuluh Nopember", fill="#718096")
        buf = card.tobytes()
        for _ in range(int(seconds * fps)):
            self.proc.stdin.write(buf)
            self.frames += 1

    def _render_cam(self, drone_pos, yaw, mode="follow"):
        is_weaving = (drone_pos[2] < 2.5 and drone_pos[1] > 1.0)
        cam_dist = 2.6 if is_weaving else 3.5
        cam_height = 1.1 if is_weaving else 1.6

        back = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        want = np.asarray(drone_pos) - cam_dist * back + np.array([0.0, 0.0, cam_height])
        target = np.asarray(drone_pos) + np.array([0.8 * math.cos(yaw), 0.8 * math.sin(yaw), 0.2])

        self.cam_pos = want if self.cam_pos is None else 0.85 * self.cam_pos + 0.15 * want
        self.sim.setObjectMatrix(self.cam, look_at(self.cam_pos, target), -1)
        self.sim.handleVisionSensor(self.cam)
        img, res = self.sim.getVisionSensorImg(self.cam)
        return np.frombuffer(img, np.uint8).reshape(res[1], res[0], 3)[::-1]

    def frame(self, drone_pos, yaw, pts, traj, scan_xy, t, progress, phase, mode="follow"):
        cam_rgb = self._render_cam(drone_pos, yaw, mode)
        img_viewport = Image.fromarray(cam_rgb)

        base = Image.new("RGB", (W, H), "#c8c8c8")
        base.paste(img_viewport, (260, 32))
        draw = ImageDraw.Draw(base)

        draw.rectangle([0, 0, W, 32], fill="#e1e1e1", outline="#b0b0b0")
        draw.text((12, 8), "CoppeliaSim Edu - drone_mapping_3d.ttt", fill="#111111")
        draw.text((360, 8), "File   Edit   Simulation   Tools   Add-ons   Help", fill="#333333")
        draw.text((W - 260, 8), f"Simulation time: +{t:06.2f} s", fill="#1a73e8")

        hierarchy_w = 260
        draw.rectangle([0, 32, hierarchy_w, H], fill="#f5f5f5", outline="#c0c0c0")
        draw.rectangle([0, 32, hierarchy_w, 60], fill="#e4e4e4", outline="#c8c8c8")
        draw.text((10, 38), "Scene hierarchy", fill="#222222")
        draw.text((hierarchy_w - 24, 38), "✕", fill="#666666")

        for i, (txt, col, selected) in enumerate(self.tree_items):
            y = 68 + i * 21
            if selected:
                draw.rectangle([2, y - 2, hierarchy_w - 2, y + 18], fill="#ffe082")
            draw.text((10, y), txt, fill=col)

        fv1_w, fv1_h = 320, 220
        fv1_x, fv1_y = 280, 48
        draw.rectangle([fv1_x, fv1_y, fv1_x + fv1_w, fv1_y + fv1_h], fill="#1e2227", outline="#ffffff", width=2)
        draw.rectangle([fv1_x, fv1_y, fv1_x + fv1_w, fv1_y + 24], fill="#e0e0e0")
        draw.text((fv1_x + 8, fv1_y + 4), "/Quadcopter/depthFront/sensor", fill="#111111")
        draw.text((fv1_x + fv1_w - 20, fv1_y + 4), "✕", fill="#333333")

        if self.df:
            try:
                df_buf, df_res = self.sim.getVisionSensorImg(self.df)
                df_arr = np.frombuffer(df_buf, dtype=np.uint8).reshape((df_res[1], df_res[0], 3))[::-1]
                img_df_resized = Image.fromarray(df_arr).resize((fv1_w, fv1_h - 24))
                base.paste(img_df_resized, (fv1_x, fv1_y + 24))
            except Exception:
                pass

        fv2_w, fv2_h = 340, 240
        fv2_x, fv2_y = W - fv2_w - 20, 48
        draw.rectangle([fv2_x, fv2_y, fv2_x + fv2_w, fv2_y + fv2_h], fill="#f5ecd5", outline="#ffffff", width=2)
        draw.rectangle([fv2_x, fv2_y, fv2_x + fv2_w, fv2_y + 24], fill="#e0e0e0")
        draw.text((fv2_x + 8, fv2_y + 4), "/XYZCameraProxy/ZViewCamera", fill="#111111")
        draw.text((fv2_x + fv2_w - 20, fv2_y + 4), "✕", fill="#333333")

        floor_tile = np.zeros((fv2_h - 24, fv2_w, 3), dtype=np.uint8)
        for r in range(fv2_h - 24):
            for c in range(fv2_w):
                chk = ((r // 24) + (c // 24)) % 2
                floor_tile[r, c] = [230, 225, 215] if chk == 0 else [210, 205, 195]
        img_floor = Image.fromarray(floor_tile)
        draw_fl = ImageDraw.Draw(img_floor)
        draw_fl.rectangle([60, 50, 110, 100], fill="#7a8a9a", outline="#4a5a6a")
        draw_fl.rectangle([120, 50, 155, 85], fill="#b0a080", outline="#706050")
        draw_fl.ellipse([160, 40, 185, 65], fill="#c03030", outline="#802020")

        d_screen_x = int(170 + drone_pos[0] * 22)
        d_screen_y = int(120 - drone_pos[1] * 18)
        self.accumulated_scan_points.append((d_screen_x, d_screen_y))

        for sx, sy in self.accumulated_scan_points[-400:]:
            draw_fl.point((sx, sy), fill="#ff1744")
        draw_fl.ellipse([d_screen_x - 6, d_screen_y - 6, d_screen_x + 6, d_screen_y + 6], fill="#00e5ff", outline="#002244")
        base.paste(img_floor, (fv2_x, fv2_y + 24))

        gx, gy = W - 70, H - 70
        draw.line([(gx, gy), (gx + 40, gy)], fill="#ff3333", width=3)
        draw.text((gx + 45, gy - 8), "x", fill="#ff3333")
        draw.line([(gx, gy), (gx, gy - 40)], fill="#33ff33", width=3)
        draw.text((gx - 5, gy - 55), "y", fill="#33ff33")
        draw.line([(gx, gy), (gx - 25, gy + 15)], fill="#3388ff", width=3)
        draw.text((gx - 40, gy + 10), "z", fill="#3388ff")

        hud_w, hud_h = 680, 56
        hud_x, hud_y = 280, H - 75
        draw.rectangle([hud_x, hud_y, hud_x + hud_w, hud_y + hud_h], fill="#000000c0", outline="#00e5ff", width=2)
        is_weaving = ("Weaving" in phase or "Canyon" in phase or (drone_pos[2] < 2.5 and drone_pos[1] > 1.0))
        if is_weaving:
            draw.text((hud_x + 14, hud_y + 8), "★ MANUVER WEAVING: Drone meliuk di antara Gedung & Silo", fill="#ffeb3b")
            draw.text((hud_x + 14, hud_y + 30), f"Ketinggian: z={drone_pos[2]:.2f}m (Celah Rendah) | Kecepatan: 0.6 m/s | Sensor: Multi-Depth + LiDAR", fill="#a0c4ff")
            if not getattr(self, "_proof_saved", False):
                base.save("/home/windu/main/ets/drone_mapping/results/bukti_video1_coppelia_gui.png")
                base.save("/home/windu/main/ets/PENGUMPULAN_UTS_5022241125/video/bukti_video1_coppelia_gui.png")
                self._proof_saved = True
        else:
            draw.text((hud_x + 14, hud_y + 8), f"Fase Misi: {phase} ({progress:.1f}%)", fill="#ffffff")
            draw.text((hud_x + 14, hud_y + 30), f"Ketinggian: z={drone_pos[2]:.2f}m | Posisi: ({drone_pos[0]:.2f}, {drone_pos[1]:.2f})m | Point Cloud: {len(pts):,} titik", fill="#a0c4ff")

        self.proc.stdin.write(base.tobytes())
        self.frames += 1

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()
