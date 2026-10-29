#!/usr/bin/env python3

import os
import sys
import json
import math
import argparse
import time
import numpy as np
from PIL import Image

try:
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
except ImportError:
    print("Error: coppeliasim_zmqremoteapi_client tidak terpasang!")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Membangun scene CoppeliaSim untuk 3D mapping drone")
    parser.add_argument("--out", type=str, default="scene/drone_mapping_3d.ttt",
                        help="Path output file .ttt (default: scene/drone_mapping_3d.ttt)")
    parser.add_argument("--gt", type=str, default="scene/ground_truth.json",
                        help="Path output ground truth JSON (default: scene/ground_truth.json)")
    parser.add_argument("--port", type=int, default=None,
                        help="Port ZMQ RPC (default: env ZMQ_PORT atau 23001)")
    parser.add_argument("--res", type=int, default=64,
                        help="Resolusi sensor depth (default: 64)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed untuk variasi (default: 42)")
    parser.add_argument("--screenshot", type=str, default="results/scene_overview.png",
                        help="Path screenshot hasil render (default: results/scene_overview.png)")
    return parser.parse_args()


class SceneBuilder:
    def __init__(self, port, depth_res=64):
        self.port = port
        self.depth_res = depth_res
        print(f"[SceneBuilder] Menghubungkan ke CoppeliaSim di localhost:{port}...")
        self.client = RemoteAPIClient('localhost', port)
        self.sim = self.client.require('sim')
        ver = self.sim.getInt32Param(self.sim.intparam_program_version)
        print(f"[SceneBuilder] Terhubung! CoppeliaSim versi: {ver}")

        self.ground_truth = {
            "arena": {
                "size_x": 12.0,
                "size_y": 12.0,
                "floor_z": 0.0,
                "wall_height": 0.4
            },
            "helipad": {
                "pos": [-4.5, -4.5, 0.01],
                "radius": 0.8
            },
            "flight_corridor": {
                "min_alt": 4.8,
                "recommended_alt": 5.0,
                "max_obstacle_height": 4.2
            },
            "obstacles": [],
            "sensors": {
                "depthDown": {
                    "type": "vision_sensor_depth",
                    "orientation": "pointing -Z (down)",
                    "fov_deg": 60.0,
                    "resolution": [depth_res, depth_res],
                    "far_clip": 12.0,
                    "reason": "Mendeteksi kontur ground/lantai (z≈0) serta bidang atap/puncak seluruh objek dari atas secara akurat."
                },
                "depthFront": {
                    "type": "vision_sensor_depth",
                    "orientation": "facing +X (forward), pitched down -35 deg",
                    "fov_deg": 60.0,
                    "resolution": [depth_res, int(depth_res * 0.75)],
                    "far_clip": 12.0,
                    "reason": "Mendeteksi fasad vertikal dan dinding depan bangunan saat drone bergerak maju."
                },
                "depthLeft": {
                    "type": "vision_sensor_depth",
                    "orientation": "facing +Y (left), pitched down -35 deg",
                    "fov_deg": 60.0,
                    "resolution": [depth_res, int(depth_res * 0.75)],
                    "far_clip": 12.0,
                    "reason": "Mendeteksi fasad samping kiri gedung guna meminimalkan area bayangan/oklusi."
                },
                "depthRight": {
                    "type": "vision_sensor_depth",
                    "orientation": "facing -Y (right), pitched down -35 deg",
                    "fov_deg": 60.0,
                    "resolution": [depth_res, int(depth_res * 0.75)],
                    "far_clip": 12.0,
                    "reason": "Mendeteksi fasad samping kanan gedung untuk cakupan 3D bilateral."
                },
                "lidar2D": {
                    "type": "lidar_2d_360",
                    "components": ["s0_front", "s1_left", "s2_rear", "s3_right"],
                    "fov_deg": 360.0,
                    "resolution_per_sensor": [128, 1],
                    "far_clip": 10.0,
                    "reason": "Menyediakan pemindaian bidang horizontal 360° simultan untuk mapping okupansi 2D (Grade 2/BC) dan deteksi rintangan horizontal."
                }
            }
        }

    def reset_scene(self):
        print("[SceneBuilder] Membersihkan scene...")
        try:
            self.sim.stopSimulation()
            time.sleep(0.3)
        except Exception:
            pass
        self.sim.closeScene()
        try:
            default_floor = self.sim.getObject("/Floor")
            self.sim.removeModel(default_floor)
            print("  + Model default /Floor CoppeliaSim dibersihkan.")
        except Exception:
            pass
        print("[SceneBuilder] Scene baru siap.")

    def set_shape_properties(self, handle, is_static=True, is_respondable=True):
        props = (self.sim.objectspecialproperty_collidable +
                 self.sim.objectspecialproperty_measurable +
                 self.sim.objectspecialproperty_renderable +
                 self.sim.objectspecialproperty_detectable)
        self.sim.setObjectSpecialProperty(handle, props)
        if is_static:
            self.sim.setObjectInt32Param(handle, self.sim.shapeintparam_static, 1)
        if is_respondable:
            self.sim.setObjectInt32Param(handle, self.sim.shapeintparam_respondable, 1)

    def create_box(self, name, size, pos, color, parent=None, is_static=True):
        h = self.sim.createPrimitiveShape(self.sim.primitiveshape_cuboid, size)
        self.sim.setObjectAlias(h, name)
        self.sim.setObjectPosition(h, pos)
        self.sim.setShapeColor(h, None, self.sim.colorcomponent_ambient_diffuse, color)
        self.set_shape_properties(h, is_static=is_static, is_respondable=True)
        if parent:
            self.sim.setObjectParent(h, parent, True)
        return h

    def create_cylinder(self, name, diameter, height, pos, color, parent=None):
        h = self.sim.createPrimitiveShape(self.sim.primitiveshape_cylinder, [diameter, diameter, height])
        self.sim.setObjectAlias(h, name)
        self.sim.setObjectPosition(h, pos)
        self.sim.setShapeColor(h, None, self.sim.colorcomponent_ambient_diffuse, color)
        self.set_shape_properties(h, is_static=True, is_respondable=True)
        if parent:
            self.sim.setObjectParent(h, parent, True)
        return h

    def create_sphere(self, name, diameter, pos, color, parent=None):
        h = self.sim.createPrimitiveShape(self.sim.primitiveshape_spheroid, [diameter, diameter, diameter])
        self.sim.setObjectAlias(h, name)
        self.sim.setObjectPosition(h, pos)
        self.sim.setShapeColor(h, None, self.sim.colorcomponent_ambient_diffuse, color)
        self.set_shape_properties(h, is_static=True, is_respondable=True)
        if parent:
            self.sim.setObjectParent(h, parent, True)
        return h

    def create_dummy(self, name, pos, parent=None):
        d = self.sim.createDummy(0.05)
        self.sim.setObjectAlias(d, name)
        self.sim.setObjectPosition(d, pos)
        if parent:
            self.sim.setObjectParent(d, parent, True)
        return d

    def build_arena_and_floor(self):
        print("[SceneBuilder] Membangun arena dan lantai...")
        arena_dummy = self.sim.createDummy(0.2)
        self.sim.setObjectAlias(arena_dummy, "Arena")
        self.sim.setObjectPosition(arena_dummy, [0.0, 0.0, 0.0])
        self.arena = arena_dummy

        floor = self.create_box("Floor", [24.0, 24.0, 0.1], [0.0, 0.0, -0.05], [0.32, 0.33, 0.35], self.arena)

        helipad = self.create_cylinder("Helipad", 1.6, 0.02, [-4.5, -4.5, 0.01], [0.95, 0.82, 0.12], self.arena)
        self.create_box("H_Mark_1", [0.8, 0.15, 0.025], [-4.5, -4.5, 0.015], [0.15, 0.15, 0.15], helipad)
        self.create_box("H_Mark_2", [0.15, 0.5, 0.025], [-4.8, -4.5, 0.015], [0.15, 0.15, 0.15], helipad)
        self.create_box("H_Mark_3", [0.15, 0.5, 0.025], [-4.2, -4.5, 0.015], [0.15, 0.15, 0.15], helipad)

        wall_color = [0.45, 0.46, 0.48]
        self.create_box("Pagar_Barat", [0.15, 12.0, 0.4], [-6.0, 0.0, 0.2], wall_color, self.arena)
        self.create_box("Pagar_Timur", [0.15, 12.0, 0.4], [6.0, 0.0, 0.2], wall_color, self.arena)
        self.create_box("Pagar_Utara_1", [4.8, 0.15, 0.4], [-3.6, 6.0, 0.2], wall_color, self.arena)
        self.create_box("Pagar_Utara_2", [4.8, 0.15, 0.4], [3.6, 6.0, 0.2], wall_color, self.arena)
        self.create_box("Pagar_Selatan_1", [4.8, 0.15, 0.4], [-3.6, -6.0, 0.2], wall_color, self.arena)
        self.create_box("Pagar_Selatan_2", [4.8, 0.15, 0.4], [3.6, -6.0, 0.2], wall_color, self.arena)

    def record_gt(self, name, obj_type, pos, size, height_top, color):
        half_x = size[0] * 0.5
        half_y = size[1] * 0.5
        self.ground_truth["obstacles"].append({
            "name": name,
            "type": obj_type,
            "pos": [round(float(p), 4) for p in pos],
            "size": [round(float(s), 4) for s in size],
            "height_top": round(float(height_top), 4),
            "color": [round(float(c), 3) for c in color],
            "bounding_box": {
                "min": [round(pos[0] - half_x, 4), round(pos[1] - half_y, 4), 0.0],
                "max": [round(pos[0] + half_x, 4), round(pos[1] + half_y, 4), round(float(height_top), 4)]
            }
        })

    def build_obstacles(self):
        print("[SceneBuilder] Membangun 20 obstacle tematik kawasan industri...")

        g1_pos = [-2.5, 2.5]
        g1_d = self.create_dummy("Gedung_Tingkat_A", [g1_pos[0], g1_pos[1], 0.0], self.arena)
        self.create_box("Tier_1", [2.4, 2.0, 1.2], [g1_pos[0], g1_pos[1], 0.6], [0.25, 0.28, 0.32], g1_d)
        self.create_box("Tier_2", [1.8, 1.5, 1.0], [g1_pos[0], g1_pos[1], 1.7], [0.35, 0.40, 0.45], g1_d)
        self.create_box("Tier_3", [1.2, 1.0, 0.8], [g1_pos[0], g1_pos[1], 2.6], [0.55, 0.60, 0.65], g1_d)
        self.record_gt("Gedung_Tingkat_A", "stacked_building", [g1_pos[0], g1_pos[1], 1.5], [2.4, 2.0, 3.0], 3.0, [0.35, 0.40, 0.45])

        gl_d = self.create_dummy("Gedung_L_Shape", [2.4, -2.8, 0.0], self.arena)
        self.create_box("Wing_1", [2.8, 1.2, 1.8], [2.8, -2.2, 0.9], [0.18, 0.28, 0.45], gl_d)
        self.create_box("Wing_2", [1.2, 2.0, 1.8], [2.0, -3.4, 0.9], [0.18, 0.28, 0.45], gl_d)
        self.record_gt("Gedung_L_Shape", "l_shape_building", [2.4, -2.8, 0.9], [2.8, 2.8, 1.8], 1.8, [0.18, 0.28, 0.45])

        gm_d = self.create_dummy("Gedung_Atap_Miring", [-2.0, -1.2, 0.0], self.arena)
        self.create_box("Body", [2.2, 1.8, 1.5], [-2.0, -1.2, 0.75], [0.65, 0.28, 0.20], gm_d)
        b_roof = self.create_box("Roof", [2.3, 1.3, 0.15], [-2.0, -1.2, 1.85], [0.22, 0.22, 0.25], gm_d)
        self.sim.setObjectOrientation(b_roof, [0.0, 25.0 * math.pi / 180.0, 0.0])
        self.record_gt("Gedung_Atap_Miring", "pitched_roof_building", [-2.0, -1.2, 1.1], [2.3, 1.8, 2.2], 2.2, [0.65, 0.28, 0.20])

        tk1_d = self.create_dummy("Tangki_Bahan_Bakar_1", [3.5, 3.2, 0.0], self.arena)
        self.create_cylinder("Body", 2.0, 2.2, [3.5, 3.2, 1.1], [0.85, 0.86, 0.88], tk1_d)
        self.create_cylinder("Cap", 2.05, 0.15, [3.5, 3.2, 2.2], [0.85, 0.20, 0.15], tk1_d)
        self.record_gt("Tangki_Bahan_Bakar_1", "cylindrical_tank", [3.5, 3.2, 1.14], [2.05, 2.05, 2.28], 2.28, [0.85, 0.86, 0.88])

        self.create_cylinder("Tangki_Bahan_Bakar_2", 1.4, 1.8, [4.8, 1.8, 0.9], [0.20, 0.50, 0.55], self.arena)
        self.record_gt("Tangki_Bahan_Bakar_2", "cylindrical_tank", [4.8, 1.8, 0.9], [1.4, 1.4, 1.8], 1.8, [0.20, 0.50, 0.55])

        self.create_cylinder("Tangki_Bahan_Bakar_3", 1.0, 1.4, [4.6, 3.6, 0.7], [0.88, 0.62, 0.15], self.arena)
        self.record_gt("Tangki_Bahan_Bakar_3", "cylindrical_tank", [4.6, 3.6, 0.7], [1.0, 1.0, 1.4], 1.4, [0.88, 0.62, 0.15])

        mn_pos = [1.2, 4.2]
        mn_d = self.create_dummy("Menara_Komunikasi", [mn_pos[0], mn_pos[1], 0.0], self.arena)
        self.create_box("Base", [0.6, 0.6, 2.0], [mn_pos[0], mn_pos[1], 1.0], [0.30, 0.32, 0.35], mn_d)
        self.create_box("Mid", [0.4, 0.4, 1.2], [mn_pos[0], mn_pos[1], 2.6], [0.85, 0.20, 0.20], mn_d)
        self.create_cylinder("Spire", 0.12, 1.0, [mn_pos[0], mn_pos[1], 3.7], [0.92, 0.92, 0.95], mn_d)
        self.record_gt("Menara_Komunikasi", "radio_tower", [mn_pos[0], mn_pos[1], 2.1], [0.6, 0.6, 4.2], 4.2, [0.85, 0.20, 0.20])

        rk_d = self.create_dummy("Reaktor_Kubah", [-4.0, 0.5, 0.0], self.arena)
        self.create_cylinder("Cylinder", 1.6, 1.6, [-4.0, 0.5, 0.8], [0.78, 0.75, 0.70], rk_d)
        self.create_sphere("Dome", 1.5, [-4.0, 0.5, 1.6], [0.70, 0.45, 0.25], rk_d)
        self.record_gt("Reaktor_Kubah", "dome_reactor", [-4.0, 0.5, 1.175], [1.6, 1.6, 2.35], 2.35, [0.70, 0.45, 0.25])

        jb_d = self.create_dummy("Jembatan_Flyover", [0.0, 0.0, 0.0], self.arena)
        self.create_box("Pillar_West", [0.4, 0.8, 1.6], [-1.0, 0.0, 0.8], [0.45, 0.45, 0.48], jb_d)
        self.create_box("Pillar_East", [0.4, 0.8, 1.6], [1.0, 0.0, 0.8], [0.45, 0.45, 0.48], jb_d)
        self.create_box("Deck", [2.8, 1.0, 0.25], [0.0, 0.0, 1.725], [0.20, 0.38, 0.62], jb_d)
        self.record_gt("Jembatan_Flyover", "bridge_archway", [0.0, 0.0, 0.925], [2.8, 1.0, 1.85], 1.85, [0.20, 0.38, 0.62])

        self.create_box("Kontainer_Kargo_A", [2.4, 1.0, 0.9], [0.5, -3.8, 0.45], [0.10, 0.32, 0.70], self.arena)
        self.record_gt("Kontainer_Kargo_A", "shipping_container", [0.5, -3.8, 0.45], [2.4, 1.0, 0.9], 0.9, [0.10, 0.32, 0.70])

        self.create_box("Kontainer_Kargo_B", [2.4, 1.0, 0.9], [0.5, -2.6, 0.45], [0.88, 0.38, 0.12], self.arena)
        self.record_gt("Kontainer_Kargo_B", "shipping_container", [0.5, -2.6, 0.45], [2.4, 1.0, 0.9], 0.9, [0.88, 0.38, 0.12])

        tr_d = self.create_dummy("Kendaraan_Truk", [-1.325, -3.6, 0.0], self.arena)
        self.create_box("Cabin", [0.7, 0.8, 0.7], [-1.8, -3.6, 0.35], [0.92, 0.78, 0.12], tr_d)
        self.create_box("Bed", [1.2, 0.8, 0.5], [-0.85, -3.6, 0.25], [0.25, 0.25, 0.28], tr_d)
        self.record_gt("Kendaraan_Truk", "vehicle", [-1.325, -3.6, 0.35], [1.9, 0.8, 0.7], 0.7, [0.92, 0.78, 0.12])

        ramp = self.create_box("Ramp_Tanjakan", [2.0, 1.0, 0.15], [3.0, -0.5, 0.35], [0.60, 0.55, 0.48], self.arena)
        self.sim.setObjectOrientation(ramp, [0.0, -20.0 * math.pi / 180.0, 0.0])
        self.record_gt("Ramp_Tanjakan", "inclined_ramp", [3.0, -0.5, 0.35], [2.0, 1.0, 0.75], 0.75, [0.60, 0.55, 0.48])

        self.create_sphere("Bukit_Kontur", 1.8, [-3.5, -3.2, -0.25], [0.32, 0.48, 0.28], self.arena)
        self.record_gt("Bukit_Kontur", "terrain_mound", [-3.5, -3.2, 0.325], [1.8, 1.8, 0.65], 0.65, [0.32, 0.48, 0.28])

        p1_d = self.create_dummy("Pohon_Pinus_1", [-4.5, 4.0, 0.0], self.arena)
        self.create_cylinder("Trunk", 0.18, 1.0, [-4.5, 4.0, 0.5], [0.38, 0.24, 0.14], p1_d)
        self.create_sphere("Foliage", 1.3, [-4.5, 4.0, 1.6], [0.14, 0.48, 0.18], p1_d)
        self.record_gt("Pohon_Pinus_1", "pine_tree", [-4.5, 4.0, 1.2], [1.3, 1.3, 2.4], 2.4, [0.14, 0.48, 0.18])

        p2_d = self.create_dummy("Pohon_Rindang_2", [-1.0, 4.5, 0.0], self.arena)
        self.create_cylinder("Trunk", 0.22, 1.1, [-1.0, 4.5, 0.55], [0.38, 0.24, 0.14], p2_d)
        self.create_sphere("Foliage", 1.5, [-1.0, 4.5, 1.7], [0.18, 0.58, 0.22], p2_d)
        self.record_gt("Pohon_Rindang_2", "deciduous_tree", [-1.0, 4.5, 1.2], [1.5, 1.5, 2.25], 2.25, [0.18, 0.58, 0.22])

        p3_d = self.create_dummy("Pohon_Rindang_3", [4.5, -4.2, 0.0], self.arena)
        self.create_cylinder("Trunk", 0.20, 0.9, [4.5, -4.2, 0.45], [0.38, 0.24, 0.14], p3_d)
        self.create_sphere("Foliage", 1.3, [4.5, -4.2, 1.45], [0.22, 0.52, 0.20], p3_d)
        self.record_gt("Pohon_Rindang_3", "deciduous_tree", [4.5, -4.2, 1.05], [1.3, 1.3, 2.1], 2.1, [0.22, 0.52, 0.20])

        p4_d = self.create_dummy("Pohon_Pinus_4", [2.0, 1.8, 0.0], self.arena)
        self.create_cylinder("Trunk", 0.16, 0.8, [2.0, 1.8, 0.4], [0.38, 0.24, 0.14], p4_d)
        self.create_sphere("Foliage", 1.1, [2.0, 1.8, 1.4], [0.12, 0.42, 0.16], p4_d)
        self.record_gt("Pohon_Pinus_4", "pine_tree", [2.0, 1.8, 1.0], [1.1, 1.1, 2.0], 2.0, [0.12, 0.42, 0.16])

        silo_d = self.create_dummy("Silo_Gandum", [-0.6, 2.6, 0.0], self.arena)
        self.create_cylinder("Body", 1.2, 2.8, [-0.6, 2.6, 1.4], [0.76, 0.73, 0.66], silo_d)
        self.create_cylinder("Top", 1.3, 0.6, [-0.6, 2.6, 3.1], [0.58, 0.60, 0.64], silo_d)
        self.record_gt("Silo_Gandum", "grain_silo", [-0.6, 2.6, 1.7], [1.3, 1.3, 3.4], 3.4, [0.76, 0.73, 0.66])

        self.record_gt("Pagar_Keliling", "perimeter_wall", [0.0, 0.0, 0.2], [12.0, 12.0, 0.4], 0.4, [0.45, 0.46, 0.48])

    def load_drone_and_sensors(self):
        print("[SceneBuilder] Memuat model drone Quadcopter.ttm...")
        model_path = os.path.join(self.sim.getStringParam(self.sim.stringparam_application_path), "models", "robots", "mobile", "Quadcopter.ttm")
        drone_h = self.sim.loadModel(model_path)
        self.sim.setObjectAlias(drone_h, "Quadcopter")

        helipad_pos = [-4.5, -4.5, 0.20]
        self.sim.setObjectPosition(drone_h, helipad_pos)

        target_h = self.sim.getObject("/Quadcopter/target")
        self.sim.setObjectPosition(target_h, helipad_pos)

        base_h = self.sim.getObject("/Quadcopter/base")

        sensor_mount = self.sim.createDummy(0.08)
        self.sim.setObjectAlias(sensor_mount, "sensorMount")
        self.sim.setObjectParent(sensor_mount, base_h, False)
        self.sim.setObjectPosition(sensor_mount, [0.0, 0.0, -0.06], self.sim.handle_parent)
        self.sim.setObjectOrientation(sensor_mount, [0.0, 0.0, 0.0])

        casing = self.create_box("SensorCasing", [0.08, 0.08, 0.04], [0.0, 0.0, -0.06], [0.1, 0.1, 0.12], base_h)
        lens = self.create_cylinder("SensorLens", 0.03, 0.015, [0.0, 0.0, -0.085], [0.0, 0.5, 0.8], base_h)
        self.sim.setObjectPosition(casing, [0.0, 0.0, -0.06], self.sim.handle_parent)
        self.sim.setObjectPosition(lens, [0.0, 0.0, -0.085], self.sim.handle_parent)

        print("[SceneBuilder] Memasang 5 sensor (depthDown, depthFront, depthLeft, depthRight, lidar2D)...")
        res_xy = [self.depth_res, self.depth_res, 0, 0]
        fov_rad = 60.0 * math.pi / 180.0
        vs_down = self.sim.createVisionSensor(2, res_xy, [0.05, 12.0, fov_rad, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.sim.setObjectAlias(vs_down, "depthDown")
        self.sim.setObjectParent(vs_down, sensor_mount, False)
        self.sim.setObjectPosition(vs_down, [0.0, 0.0, -0.02], self.sim.handle_parent)
        self.sim.setObjectOrientation(vs_down, [math.pi, 0.0, 0.0])

        res_front = [self.depth_res, int(self.depth_res * 0.75), 0, 0]
        vs_front = self.sim.createVisionSensor(2, res_front, [0.05, 12.0, fov_rad, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.sim.setObjectAlias(vs_front, "depthFront")
        self.sim.setObjectParent(vs_front, sensor_mount, False)
        self.sim.setObjectPosition(vs_front, [0.06, 0.0, 0.0], self.sim.handle_parent)
        self.sim.setObjectOrientation(vs_front, [-math.pi, 55.0 * math.pi / 180.0, -math.pi * 0.5])

        vs_left = self.sim.createVisionSensor(2, res_front, [0.05, 12.0, fov_rad, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.sim.setObjectAlias(vs_left, "depthLeft")
        self.sim.setObjectParent(vs_left, sensor_mount, False)
        self.sim.setObjectPosition(vs_left, [0.0, 0.06, 0.0], self.sim.handle_parent)
        self.sim.setObjectOrientation(vs_left, [-125.0 * math.pi / 180.0, 0.0, -math.pi])

        vs_right = self.sim.createVisionSensor(2, res_front, [0.05, 12.0, fov_rad, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.sim.setObjectAlias(vs_right, "depthRight")
        self.sim.setObjectParent(vs_right, sensor_mount, False)
        self.sim.setObjectPosition(vs_right, [0.0, -0.06, 0.0], self.sim.handle_parent)
        self.sim.setObjectOrientation(vs_right, [125.0 * math.pi / 180.0, 0.0, 0.0])

        lidar_dummy = self.sim.createDummy(0.04)
        self.sim.setObjectAlias(lidar_dummy, "lidar2D")
        self.sim.setObjectParent(lidar_dummy, sensor_mount, False)
        self.sim.setObjectPosition(lidar_dummy, [0.0, 0.0, -0.03], self.sim.handle_parent)

        lidar_res = [128, 1, 0, 0]
        lidar_fov = 90.0 * math.pi / 180.0
        lidar_configs = [
            ("s0", [0.02, 0.0, 0.0], [0.0, math.pi * 0.5, math.pi * 0.5]),
            ("s1", [0.0, 0.02, 0.0], [-math.pi * 0.5, 0.0, -math.pi]),
            ("s2", [-0.02, 0.0, 0.0], [0.0, -math.pi * 0.5, -math.pi * 0.5]),
            ("s3", [0.0, -0.02, 0.0], [math.pi * 0.5, 0.0, 0.0])
        ]
        for s_name, s_pos, s_euler in lidar_configs:
            s_h = self.sim.createVisionSensor(2, lidar_res, [0.05, 10.0, lidar_fov, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            self.sim.setObjectAlias(s_h, s_name)
            self.sim.setObjectParent(s_h, lidar_dummy, False)
            self.sim.setObjectPosition(s_h, s_pos, self.sim.handle_parent)
            self.sim.setObjectOrientation(s_h, s_euler  )

        print("[SceneBuilder] Menyematkan script Lua ke scene...")
        script_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lua")

        sensor_lua_path = os.path.join(script_dir, "drone_sensor.lua")
        if os.path.exists(sensor_lua_path):
            with open(sensor_lua_path, "r", encoding="utf-8") as f:
                sensor_code = f.read()
            s_mount = self.sim.createScript(self.sim.scripttype_simulation, sensor_code, 0, 'lua')
            self.sim.setObjectParent(s_mount, sensor_mount, False)
            print("  + drone_sensor.lua disematkan ke /Quadcopter/sensorMount")

        path_visualizer = self.create_dummy("pathVisualizer", [0.0, 0.0, 0.0], drone_h)
        path_lua_path = os.path.join(script_dir, "drone_path.lua")
        if os.path.exists(path_lua_path):
            with open(path_lua_path, "r", encoding="utf-8") as f:
                path_code = f.read()
            s_path = self.sim.createScript(self.sim.scripttype_simulation, path_code, 0, 'lua')
            self.sim.setObjectParent(s_path, path_visualizer, False)
            print("  + drone_path.lua disematkan ke /Quadcopter/pathVisualizer")

        ctrl_lua_path = os.path.join(script_dir, "drone_controller.lua")
        if os.path.exists(ctrl_lua_path):
            with open(ctrl_lua_path, "r", encoding="utf-8") as f:
                ctrl_code = f.read()
            s_ctrl = self.sim.createScript(self.sim.scripttype_simulation, ctrl_code, 0, 'lua')
            self.sim.setObjectParent(s_ctrl, target_h, False)
            print("  + drone_controller.lua disematkan ke /Quadcopter/target")

        self.vs_down = vs_down
        self.vs_front = vs_front

    def setup_cinematic_views(self):
        print("[SceneBuilder] Mengatur kamera default dan OverviewCamera...")
        cam = self.sim.getObject("/DefaultCamera")
        cam_pos = [-8.0, -8.0, 7.5]
        cam_euler = [55.0 * math.pi / 180.0, 0.0, -45.0 * math.pi / 180.0]
        self.sim.setObjectPosition(cam, cam_pos, -1)
        self.sim.setObjectOrientation(cam, cam_euler, -1)

        vs_overview = self.sim.createVisionSensor(3, [1280, 720, 0, 0], [0.1, 50.0, 60.0 * math.pi / 180.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.sim.setObjectAlias(vs_overview, "RecCam")
        self.sim.setObjectParent(vs_overview, self.arena, True)
        self.sim.setObjectPosition(vs_overview, cam_pos, -1)
        self.sim.setObjectOrientation(vs_overview, cam_euler, -1)
        self.vs_overview = vs_overview

        if not self.sim.getBoolParam(self.sim.boolparam_headless):
            try:
                view1 = self.sim.floatingViewAdd(0.02, 0.72, 0.24, 0.24, 0)
                self.sim.adjustView(view1, self.vs_down, 64, "Depth Down")

                view2 = self.sim.floatingViewAdd(0.28, 0.72, 0.24, 0.24, 0)
                self.sim.adjustView(view2, self.vs_front, 64, "Depth Front")
                print("  + Floating views aktif: Depth Down & Depth Front")
            except Exception as e:
                print(f"  ! Catatan floating view: {e}")

    def capture_screenshot(self, out_path):
        print(f"[SceneBuilder] Mengambil screenshot scene ke {out_path}...")
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

        self.sim.setStepping(True)
        self.sim.startSimulation()
        self.sim.step()

        self.sim.handleVisionSensor(self.vs_overview)
        img_buf, [rx, ry] = self.sim.getVisionSensorImg(self.vs_overview)
        img_arr = np.frombuffer(img_buf, dtype=np.uint8).reshape((ry, rx, 3))
        img_arr = np.flipud(img_arr)
        img = Image.fromarray(img_arr)
        img.save(out_path)
        print(f"  + Screenshot tersimpan ({rx}x{ry}): {out_path}")

        self.sim.stopSimulation()
        while self.sim.getSimulationState() != self.sim.simulation_stopped:
            time.sleep(0.05)
        self.sim.setStepping(False)

    def save_scene(self, out_path):
        print(f"[SceneBuilder] Menyimpan scene ke {out_path}...")
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        abs_path = os.path.abspath(out_path)
        res = self.sim.saveScene(abs_path)
        if res == 1:
            size_mb = os.path.getsize(abs_path) / (1024 * 1024)
            print(f"  + Scene .ttt berhasil disimpan! Ukuran: {size_mb:.2f} MB")
        else:
            print(f"  ! Gagal menyimpan scene ke {abs_path}")

    def save_ground_truth(self, gt_path):
        print(f"[SceneBuilder] Menyimpan ground truth JSON ke {gt_path}...")
        os.makedirs(os.path.dirname(os.path.abspath(gt_path)), exist_ok=True)
        with open(gt_path, "w", encoding="utf-8") as f:
            json.dump(self.ground_truth, f, indent=2)
        print(f"  + Ground truth tersimpan: {len(self.ground_truth['obstacles'])} obstacles.")


def main():
    args = parse_args()
    port = args.port or int(os.environ.get("ZMQ_PORT", 23001))

    builder = SceneBuilder(port=port, depth_res=args.res)
    builder.reset_scene()
    builder.build_arena_and_floor()
    builder.build_obstacles()
    builder.load_drone_and_sensors()
    builder.setup_cinematic_views()
    builder.save_scene(args.out)
    builder.save_ground_truth(args.gt)
    builder.capture_screenshot(args.screenshot)

    print("\n[SceneBuilder] Scene selesai dibangun.")


if __name__ == "__main__":
    main()
