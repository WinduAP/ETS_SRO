#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

COPPELIA_ROOT="${COPPELIASIM_ROOT:-$HOME/main/coppelia/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu22_04}"
export ZMQ_PORT="${ZMQ_PORT:-23000}"

ping_sim() {
    python3 -c "from coppeliasim_zmqremoteapi_client import RemoteAPIClient as C; C(port=$ZMQ_PORT).require('sim').getInt32Param(0)" 2>/dev/null
}

if ! ping_sim; then
    (cd "$COPPELIA_ROOT" && tail -f /dev/null | ./coppeliaSim.sh -h -vwarnings -GzmqRemoteApi.rpcPort="$ZMQ_PORT" >/tmp/coppelia_$ZMQ_PORT.log 2>&1) &
    SIM_PID=$!
    trap 'pkill -f "rpcPort=$ZMQ_PORT" || true' EXIT
    for _ in $(seq 30); do sleep 1; ping_sim && break; done
fi

python3 python/build_scene.py --port "$ZMQ_PORT" --out scene/drone_mapping_3d.ttt --gt scene/ground_truth.json
python3 python/mission.py --port "$ZMQ_PORT" --scene scene/drone_mapping_3d.ttt --plan smart \
    --out data/raw_scan.npz --record-video results/video1_proses_mapping.mp4
python3 python/postprocess.py --input data/raw_scan.npz --gt scene/ground_truth.json --outdir results --video
python3 python/viewer_html.py
echo "selesai"
