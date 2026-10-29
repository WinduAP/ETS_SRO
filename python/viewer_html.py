#!/usr/bin/env python3

import os
import sys
import json
import argparse
import numpy as np


def generate_viewer_html(stats_path, npz_path, gt_path, output_html):
    print(f"Menyiapkan 3D Web Viewer dari: {stats_path}")

    with open(stats_path, 'r') as f:
        stats = json.load(f)

    with open(gt_path, 'r') as f:
        gt_data = json.load(f)

    data = np.load(npz_path, allow_pickle=True)
    traj = data["traj"][:, 1:4].tolist() if "traj" in data else []

    ply_voxel_path = os.path.join(os.path.dirname(output_html), "map_voxel.ply")
    voxels = []
    if os.path.exists(ply_voxel_path):
        with open(ply_voxel_path, 'r') as f:
            lines = f.readlines()
        header_end = 0
        for idx, line in enumerate(lines):
            if line.strip() == "end_header":
                header_end = idx + 1
                break
        all_pts = []
        for line in lines[header_end:]:
            parts = line.strip().split()
            if len(parts) >= 3:
                all_pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
        all_pts = np.array(all_pts, dtype=np.float32)
        if len(all_pts) > 40000:
            step = len(all_pts) // 35000
            sampled_pts = all_pts[::step]
        else:
            sampled_pts = all_pts
        voxels = sampled_pts.tolist()
    else:
        raw_pts = data["points"]
        step = max(1, len(raw_pts) // 30000)
        voxels = raw_pts[::step].tolist()

    bboxes = []
    for obs in gt_data.get("obstacles", []):
        if obs.get("type") == "fence":
            continue
        p = obs["pos"]
        s = obs["size"]
        top_h = obs.get("height_top", s[2])
        bboxes.append({
            "name": obs.get("name", "Obstacle"),
            "type": obs.get("type", "box"),
            "desc": obs.get("desc", ""),
            "center": [p[0], p[1], top_h / 2.0],
            "size": [s[0], s[1], top_h],
            "height_top": top_h
        })

    voxels_json = json.dumps([[round(p[0], 3), round(p[1], 3), round(p[2], 3)] for p in voxels])
    traj_json = json.dumps([[round(p[0], 3), round(p[1], 3), round(p[2], 3)] for p in traj[::2]])
    bboxes_json = json.dumps(bboxes)
    stats_json = json.dumps({
        "raw_points": stats.get("raw_points_count", 0),
        "filtered_points": stats.get("filtered_points_count", 0),
        "voxel_count": stats.get("selected_voxel_count", len(voxels)),
        "reduction_pct": round(stats.get("selected_voxel_reduction_pct", 0.0), 2),
        "detected_count": stats.get("accuracy", {}).get("detected_obstacles", len(bboxes)),
        "total_gt": stats.get("accuracy", {}).get("total_gt_obstacles", len(bboxes)),
        "mean_iou": round(stats.get("accuracy", {}).get("mean_iou_3d", 0.0), 3),
        "mean_pos_err_cm": round(stats.get("accuracy", {}).get("mean_position_error_m", 0.0) * 100, 1),
        "mean_h_err_cm": round(stats.get("accuracy", {}).get("mean_height_error_m", 0.0) * 100, 1),
        "flight_length_m": round(stats.get("trajectory", {}).get("flight_length_m", 0.0), 1),
        "flight_time_s": round(stats.get("trajectory", {}).get("total_time_sec", 0.0), 1)
    })

    html_content = f"""<!DOCTYPE html>
<html lang="id">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>UTS Drone Mapping — 3D Interactive Point Cloud & Voxel Viewer</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <style>
    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      user-select: none;
    }}
    body {{
      background-color: #0b0f19;
      color: #f1f5f9;
      font-family: 'Inter', -apple-system, sans-serif;
      overflow: hidden;
      width: 100vw;
      height: 100vh;
    }}
    #canvas-container {{
      width: 100%;
      height: 100%;
      position: absolute;
      top: 0;
      left: 0;
      z-index: 1;
    }}
    /* Glassmorphism UI Panels */
    .glass-panel {{
      background: rgba(15, 23, 42, 0.78);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 12px;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.45);
      position: absolute;
      z-index: 10;
      pointer-events: auto;
    }}
    /* Top Header */
    #header {{
      top: 16px;
      left: 16px;
      padding: 14px 20px;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }}
    #header h1 {{
      font-size: 16px;
      font-weight: 700;
      letter-spacing: -0.01em;
      color: #38bdf8;
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    #header .badge {{
      background: #0284c7;
      color: #ffffff;
      font-size: 10px;
      font-weight: 600;
      padding: 2px 7px;
      border-radius: 20px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    #header p {{
      font-size: 12px;
      color: #94a3b8;
    }}
    /* Controls Panel */
    #controls {{
      top: 16px;
      right: 16px;
      width: 320px;
      max-height: calc(100vh - 32px);
      overflow-y: auto;
      padding: 18px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }}
    .ctrl-group {{
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}
    .ctrl-group label {{
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: #cbd5e1;
      display: flex;
      justify-content: space-between;
    }}
    .ctrl-group label span {{
      color: #38bdf8;
      font-family: 'JetBrains Mono', monospace;
    }}
    select, input[type="range"] {{
      width: 100%;
      background: rgba(30, 41, 59, 0.8);
      border: 1px solid rgba(255, 255, 255, 0.12);
      color: #f8fafc;
      padding: 6px 10px;
      border-radius: 6px;
      font-size: 12px;
      outline: none;
      cursor: pointer;
    }}
    select:focus {{
      border-color: #38bdf8;
    }}
    input[type="range"] {{
      -webkit-appearance: none;
      height: 6px;
      background: #334155;
      padding: 0;
    }}
    input[type="range"]::-webkit-slider-thumb {{
      -webkit-appearance: none;
      width: 14px;
      height: 14px;
      border-radius: 50%;
      background: #38bdf8;
      cursor: pointer;
    }}
    .checkbox-row {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }}
    .checkbox-item {{
      display: flex;
      align-items: center;
      gap: 6px;
      font-size: 12px;
      color: #e2e8f0;
      cursor: pointer;
    }}
    .checkbox-item input {{
      cursor: pointer;
      accent-color: #38bdf8;
    }}
    /* HUD Stats Panel */
    #hud-stats {{
      bottom: 16px;
      left: 16px;
      padding: 14px 18px;
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 12px 18px;
      font-family: 'JetBrains Mono', monospace;
    }}
    .stat-item {{
      display: flex;
      flex-direction: column;
      gap: 2px;
    }}
    .stat-label {{
      font-size: 10px;
      color: #64748b;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .stat-val {{
      font-size: 13px;
      font-weight: 600;
      color: #38bdf8;
    }}
    /* Colorbar Legend */
    #colorbar-panel {{
      bottom: 16px;
      right: 16px;
      padding: 10px 14px;
      display: flex;
      flex-direction: column;
      gap: 6px;
      font-size: 11px;
    }}
    #colorbar-gradient {{
      width: 160px;
      height: 12px;
      border-radius: 4px;
      background: linear-gradient(to right, #30123b, #4662d8, #35abf8, #1ae4b6, #72fe5e, #c8fa35, #faba39, #f15a24, #d92120, #7a0403);
    }}
    .colorbar-labels {{
      display: flex;
      justify-content: space-between;
      color: #94a3b8;
      font-family: 'JetBrains Mono', monospace;
      font-size: 10px;
    }}
    /* Tooltip modal */
    #tooltip {{
      position: absolute;
      display: none;
      padding: 8px 12px;
      background: rgba(15, 23, 42, 0.92);
      border: 1px solid #38bdf8;
      border-radius: 6px;
      font-size: 11px;
      pointer-events: none;
      z-index: 100;
      line-height: 1.4;
      max-width: 240px;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.6);
    }}
    /* Quick Reset View Button */
    .btn {{
      background: #0284c7;
      color: #ffffff;
      border: none;
      padding: 7px 12px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.2s;
    }}
    .btn:hover {{
      background: #0369a1;
    }}
    /* Mobile responsive */
    @media (max-width: 768px) {{
      #controls {{
        width: calc(100vw - 32px);
        max-height: 40vh;
        bottom: 16px;
        top: auto;
      }}
      #hud-stats {{
        display: none;
      }}
      #colorbar-panel {{
        display: none;
      }}
    }}
  </style>
  <!-- Three.js & OrbitControls dari cdnjs -->
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
  <div id="canvas-container"></div>

  <!-- Header -->
  <div id="header" class="glass-panel">
    <h1>
      <span>3D DRONE MAPPING VIEWER</span>
      <span class="badge">UTS Sistem Robot Otonom</span>
    </h1>
    <p>Autonomous Multi-Vision Depth + 2D LiDAR Fusion Reconstruction</p>
  </div>

  <!-- Controls Panel -->
  <div id="controls" class="glass-panel">
    <div class="ctrl-group">
      <label>Skema Warna (Colormap)</label>
      <select id="sel-colormap">
        <option value="turbo" selected>Turbo (Elevasi Kontinu)</option>
        <option value="jet">Jet / Pelangi Klasik</option>
        <option value="viridis">Viridis (Perseptual Seragam)</option>
        <option value="discrete">Pita Diskrit (Interval 0.5 m)</option>
      </select>
    </div>

    <div class="ctrl-group">
      <label>Ukuran Titik Voxel <span id="lbl-ptsize">3.5 px</span></label>
      <input type="range" id="rng-ptsize" min="1.0" max="8.0" step="0.5" value="3.5">
    </div>

    <div class="ctrl-group">
      <label>Irisan Ketinggian Max Z <span id="lbl-maxz">4.5 m</span></label>
      <input type="range" id="rng-maxz" min="0.1" max="4.5" step="0.1" value="4.5">
    </div>

    <div class="ctrl-group">
      <label>Irisan Ketinggian Min Z <span id="lbl-minz">0.0 m</span></label>
      <input type="range" id="rng-minz" min="0.0" max="3.5" step="0.1" value="0.0">
    </div>

    <div class="ctrl-group">
      <label>Lapisan Tampilan (Layers)</label>
      <div class="checkbox-row">
        <label class="checkbox-item"><input type="checkbox" id="chk-voxels" checked> Voxel Grid</label>
        <label class="checkbox-item"><input type="checkbox" id="chk-traj" checked> Jalur Drone</label>
        <label class="checkbox-item"><input type="checkbox" id="chk-bboxes" checked> Bounding Box</label>
        <label class="checkbox-item"><input type="checkbox" id="chk-ground" checked> Lantai Grid</label>
        <label class="checkbox-item"><input type="checkbox" id="chk-axes" checked> Sumbu XYZ</label>
        <label class="checkbox-item"><input type="checkbox" id="chk-rotate"> Rotasi Otomatis</label>
      </div>
    </div>

    <button id="btn-reset" class="btn">Reset Posisi Kamera</button>
  </div>

  <!-- HUD Stats Panel -->
  <div id="hud-stats" class="glass-panel">
    <div class="stat-item">
      <span class="stat-label">Titik Mentah</span>
      <span class="stat-val" id="hud-raw">-</span>
    </div>
    <div class="stat-item">
      <span class="stat-label">Titik Voxel</span>
      <span class="stat-val" id="hud-vox">-</span>
    </div>
    <div class="stat-item">
      <span class="stat-label">Reduksi Voxel</span>
      <span class="stat-val" id="hud-red">-</span>
    </div>
    <div class="stat-item">
      <span class="stat-label">Objek Terdeteksi</span>
      <span class="stat-val" id="hud-obj">-</span>
    </div>
    <div class="stat-item">
      <span class="stat-label">Rata-Rata IoU</span>
      <span class="stat-val" id="hud-iou">-</span>
    </div>
    <div class="stat-item">
      <span class="stat-label">Panjang Lintasan</span>
      <span class="stat-val" id="hud-traj">-</span>
    </div>
  </div>

  <!-- Colorbar Panel -->
  <div id="colorbar-panel" class="glass-panel">
    <span style="font-weight: 600; color: #cbd5e1;">Skala Elevasi Ketinggian</span>
    <div id="colorbar-gradient"></div>
    <div class="colorbar-labels">
      <span>0.0 m</span>
      <span>2.0 m</span>
      <span>4.0+ m</span>
    </div>
  </div>

  <!-- Tooltip -->
  <div id="tooltip"></div>

  <script>

    const RAW_VOXELS = {voxels_json};
    const TRAJECTORY = {traj_json};
    const BBOXES = {bboxes_json};
    const STATS = {stats_json};

    // Update HUD Stats
    document.getElementById('hud-raw').innerText = STATS.raw_points.toLocaleString();
    document.getElementById('hud-vox').innerText = STATS.voxel_count.toLocaleString();
    document.getElementById('hud-red').innerText = STATS.reduction_pct + '%';
    document.getElementById('hud-obj').innerText = STATS.detected_count + '/' + STATS.total_gt;
    document.getElementById('hud-iou').innerText = STATS.mean_iou;
    document.getElementById('hud-traj').innerText = STATS.flight_length_m + ' m';

    // 1. Setup Three.js Scene, Camera, Renderer
    const container = document.getElementById('canvas-container');
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0b0f19);

    const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 100);
    camera.position.set(-11, 12, 14);

    const renderer = new THREE.WebGLRenderer({{ antialias: true, powerPreference: 'high-performance' }});
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.target.set(0, 1.2, 0);

    // Pencahayaan
    const ambLight = new THREE.AmbientLight(0xffffff, 0.7);
    scene.add(ambLight);
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(10, 20, 15);
    scene.add(dirLight);

    // Sumbu Koordinat (CoppeliaSim: Z up -> di Three.js kita petakan (x, z, -y) atau gunakan grup rotasi)
    // Di Three.js sumbu Y adalah tegak ke atas, jadi koordinat (x, y_coppelia, z_coppelia) dipetakan ke (x, z, -y)
    const worldGroup = new THREE.Group();
    scene.add(worldGroup);

    // Grid Lantai Arena 12x12 m
    const gridHelper = new THREE.GridHelper(12, 24, 0x0284c7, 0x1e293b);
    gridHelper.position.y = 0.0;
    worldGroup.add(gridHelper);

    // Sumbu Asli
    const axesHelper = new THREE.AxesHelper(3.0);
    worldGroup.add(axesHelper);

    // 2. Color Palettes
    function colormapTurbo(t) {{
      t = Math.max(0, Math.min(1, t));
      const r = Math.sin(t * Math.PI - 0.2) * 0.5 + 0.5;
      const g = Math.sin(t * Math.PI + 0.3) * 0.8;
      const b = Math.cos(t * Math.PI * 0.8) * 0.5 + 0.5;
      // Aproksimasi Turbo akurat
      const kR = Math.max(0, Math.min(1, 1.5 * t - 0.1));
      const kG = Math.max(0, Math.min(1, 1.0 - Math.abs(2 * t - 1)));
      const kB = Math.max(0, Math.min(1, 1.0 - 1.8 * t));
      return new THREE.Color(kR, kG, kB);
    }}

    function colormapJet(t) {{
      t = Math.max(0, Math.min(1, t));
      const r = Math.max(0, Math.min(1, 1.5 - Math.abs(4 * t - 3)));
      const g = Math.max(0, Math.min(1, 1.5 - Math.abs(4 * t - 2)));
      const b = Math.max(0, Math.min(1, 1.5 - Math.abs(4 * t - 1)));
      return new THREE.Color(r, g, b);
    }}

    function colormapViridis(t) {{
      t = Math.max(0, Math.min(1, t));
      const r = 0.28 + 0.72 * Math.pow(t, 2);
      const g = 0.15 + 0.75 * t;
      const b = 0.45 + 0.35 * Math.sin(t * Math.PI);
      return new THREE.Color(r, g, b);
    }}

    // 3. Bangun Point Cloud Voxel
    const numPoints = RAW_VOXELS.length;
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(numPoints * 3);
    const colors = new Float32Array(numPoints * 3);

    for (let i = 0; i < numPoints; i++) {{
      const p = RAW_VOXELS[i];
      // Petakan CoppeliaSim (X, Y, Z) ke Three.js (X, Z, -Y)
      positions[i * 3 + 0] = p[0];
      positions[i * 3 + 1] = p[2];  // Z coppelia jadi Y Three.js
      positions[i * 3 + 2] = -p[1]; // Y coppelia jadi -Z Three.js

      const normZ = Math.min(1.0, Math.max(0.0, p[2] / 3.8));
      const c = colormapTurbo(normZ);
      colors[i * 3 + 0] = c.r;
      colors[i * 3 + 1] = c.g;
      colors[i * 3 + 2] = c.b;
    }}

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const material = new THREE.PointsMaterial({{
      size: 0.12,
      vertexColors: true,
      transparent: true,
      opacity: 0.9,
      sizeAttenuation: true
    }});

    const pointCloud = new THREE.Points(geometry, material);
    worldGroup.add(pointCloud);

    // 4. Bangun Trajektori Drone (LineStrip 3D)
    const trajGeo = new THREE.BufferGeometry();
    const trajPositions = [];
    for (let i = 0; i < TRAJECTORY.length; i++) {{
      const pt = TRAJECTORY[i];
      trajPositions.push(pt[0], pt[2], -pt[1]);
    }}
    trajGeo.setAttribute('position', new THREE.Float32BufferAttribute(trajPositions, 3));
    const trajMat = new THREE.LineBasicMaterial({{ color: 0xef4444, linewidth: 2 }});
    const trajLine = new THREE.Line(trajGeo, trajMat);
    worldGroup.add(trajLine);

    // 5. Bangun 3D Bounding Boxes
    const bboxGroup = new THREE.Group();
    const bboxMeshes = [];
    for (let i = 0; i < BBOXES.length; i++) {{
      const b = BBOXES[i];
      const boxGeo = new THREE.BoxGeometry(b.size[0], b.size[2], b.size[1]);
      const edges = new THREE.EdgesGeometry(boxGeo);
      const line = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({{ color: 0xfacc15, linewidth: 1.5 }}));
      // Posisi: b.center (X, Z_height, -Y)
      line.position.set(b.center[0], b.center[2], -b.center[1]);
      line.userData = b;
      bboxGroup.add(line);

      // Transparent solid box untuk raycasting klik
      const hitMat = new THREE.MeshBasicMaterial({{ visible: false }});
      const hitMesh = new THREE.Mesh(boxGeo, hitMat);
      hitMesh.position.copy(line.position);
      hitMesh.userData = b;
      bboxMeshes.push(hitMesh);
      bboxGroup.add(hitMesh);
    }}
    worldGroup.add(bboxGroup);

    // 6. Logika Interaksi UI & Kontrol
    function updateColors() {{
      const scheme = document.getElementById('sel-colormap').value;
      const minZ = parseFloat(document.getElementById('rng-minz').value);
      const maxZ = parseFloat(document.getElementById('rng-maxz').value);
      const colAttr = geometry.attributes.color;

      for (let i = 0; i < numPoints; i++) {{
        const z = RAW_VOXELS[i][2];
        if (z < minZ || z > maxZ) {{
          colAttr.setXYZ(i, 0, 0, 0); // Hide clipped points
          continue;
        }}
        const normZ = Math.min(1.0, Math.max(0.0, z / 3.8));
        let c;
        if (scheme === 'turbo') c = colormapTurbo(normZ);
        else if (scheme === 'jet') c = colormapJet(normZ);
        else if (scheme === 'viridis') c = colormapViridis(normZ);
        else if (scheme === 'discrete') {{
          const band = Math.floor(z / 0.5) * 0.5;
          c = colormapJet(band / 3.5);
        }}
        colAttr.setXYZ(i, c.r, c.g, c.b);
      }}
      colAttr.needsUpdate = true;
    }}

    document.getElementById('sel-colormap').addEventListener('change', updateColors);
    document.getElementById('rng-minz').addEventListener('input', (e) => {{
      document.getElementById('lbl-minz').innerText = parseFloat(e.target.value).toFixed(1) + ' m';
      updateColors();
    }});
    document.getElementById('rng-maxz').addEventListener('input', (e) => {{
      document.getElementById('lbl-maxz').innerText = parseFloat(e.target.value).toFixed(1) + ' m';
      updateColors();
    }});
    document.getElementById('rng-ptsize').addEventListener('input', (e) => {{
      const val = parseFloat(e.target.value);
      document.getElementById('lbl-ptsize').innerText = val.toFixed(1) + ' px';
      material.size = val * 0.035;
    }});

    document.getElementById('chk-voxels').addEventListener('change', (e) => {{ pointCloud.visible = e.target.checked; }});
    document.getElementById('chk-traj').addEventListener('change', (e) => {{ trajLine.visible = e.target.checked; }});
    document.getElementById('chk-bboxes').addEventListener('change', (e) => {{ bboxGroup.visible = e.target.checked; }});
    document.getElementById('chk-ground').addEventListener('change', (e) => {{ gridHelper.visible = e.target.checked; }});
    document.getElementById('chk-axes').addEventListener('change', (e) => {{ axesHelper.visible = e.target.checked; }});

    document.getElementById('btn-reset').addEventListener('click', () => {{
      camera.position.set(-11, 12, 14);
      controls.target.set(0, 1.2, 0);
      controls.update();
    }});

    // Raycaster untuk Tooltip Klik
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    const tooltip = document.getElementById('tooltip');

    window.addEventListener('pointerdown', (e) => {{
      mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
      mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;
      raycaster.setFromCamera(mouse, camera);
      const intersects = raycaster.intersectObjects(bboxMeshes);
      if (intersects.length > 0) {{
        const b = intersects[0].object.userData;
        tooltip.style.display = 'block';
        tooltip.style.left = (e.clientX + 14) + 'px';
        tooltip.style.top = (e.clientY + 14) + 'px';
        tooltip.innerHTML = `
          <strong style="color: #38bdf8; font-size: 12px;">${{b.name}}</strong><br>
          <span style="color: #94a3b8;">Tipe:</span> ${{b.type}}<br>
          <span style="color: #94a3b8;">Tinggi Puncak:</span> ${{b.height_top.toFixed(2)}} m<br>
          <span style="color: #94a3b8;">Ukuran (P x L):</span> ${{b.size[0].toFixed(1)}} x ${{b.size[1].toFixed(1)}} m<br>
          <span style="color: #64748b; font-size: 10px;">${{b.desc}}</span>
        `;
      }} else {{
        tooltip.style.display = 'none';
      }}
    }});

    // Resize Window
    window.addEventListener('resize', () => {{
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    }});

    // 7. Animation Loop
    function animate() {{
      requestAnimationFrame(animate);
      if (document.getElementById('chk-rotate').checked) {{
        worldGroup.rotation.y += 0.005;
      }}
      controls.update();
      renderer.render(scene, camera);
    }}
    animate();
  </script>
</body>
</html>
"""

    os.makedirs(os.path.dirname(os.path.abspath(output_html)), exist_ok=True)
    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(html_content)

    file_size_kb = os.path.getsize(output_html) / 1024
    print(f"viewer: {output_html} ({file_size_kb:.1f} KB)")
    return output_html


def main():
    parser = argparse.ArgumentParser(description="Membuat viewer 3D berbasis HTML")
    parser.add_argument("--stats", default="results/stats.json", help="Path ke stats.json")
    parser.add_argument("--npz", default="data/raw_scan.npz", help="Path ke raw_scan.npz")
    parser.add_argument("--gt", default="scene/ground_truth.json", help="Path ke ground_truth.json")
    parser.add_argument("--out", default="results/viewer_3d.html", help="Path berkas HTML keluaran")

    args = parser.parse_args()

    stats_p = args.stats
    npz_p = args.npz
    gt_p = args.gt
    out_p = args.out

    in_drone_mapping = (os.path.basename(os.getcwd()) == "drone_mapping") or (os.path.exists("python") and os.path.exists("scene"))
    if in_drone_mapping:
        if stats_p.startswith("drone_mapping/"): stats_p = stats_p[len("drone_mapping/"):]
        if npz_p.startswith("drone_mapping/"): npz_p = npz_p[len("drone_mapping/"):]
        if gt_p.startswith("drone_mapping/"): gt_p = gt_p[len("drone_mapping/"):]
        if out_p.startswith("drone_mapping/"): out_p = out_p[len("drone_mapping/"):]
    else:
        if not stats_p.startswith("drone_mapping/"): stats_p = os.path.join("drone_mapping", stats_p)
        if not npz_p.startswith("drone_mapping/"): npz_p = os.path.join("drone_mapping", npz_p)
        if not gt_p.startswith("drone_mapping/"): gt_p = os.path.join("drone_mapping", gt_p)
        if not out_p.startswith("drone_mapping/"): out_p = os.path.join("drone_mapping", out_p)

    generate_viewer_html(stats_p, npz_p, gt_p, out_p)


if __name__ == "__main__":
    main()
