#!/usr/bin/env bash
set -euo pipefail

FPS=33
SIZE=1280x720
IN_PIX=uyvy422

ENC=h264_videotoolbox
BITRATE=6M

CAM1=0
CAM2=1
CAM3=2

# ---- preflight settings ----
HASH_W=32
HASH_H=32
PROBE_TIMEOUT_SEC=2
PROBE_RETRIES=5
PROBE_SLEEP=0.4

ts=$(date +"%Y%m%d_%H%M%S")
outdir="videos/$ts"
mkdir -p "$outdir"

echo "Recording to: $outdir"
echo "Cameras: $CAM1 $CAM2 $CAM3"

# ---- helpers ----
hash_cam() {
  local cam_idx="$1"

  # Use a supported framerate (15) so avfoundation can open the device.
  # Keep it short but give it time to deliver one frame.
  local out
  out="$(
    ffmpeg -hide_banner -loglevel error -nostdin \
      -f avfoundation \
      -framerate 15 \
      -video_size "$SIZE" \
      -i "${cam_idx}:none" \
      -analyzeduration 0 -probesize 32k \
      -t 0.8 -frames:v 1 \
      -vf "scale=${HASH_W}:${HASH_H},format=gray" \
      -f rawvideo - 2>/dev/null | md5 -q
  )" || true

  # md5 of empty input = failure
  if [[ "$out" == "d41d8cd98f00b204e9800998ecf8427e" ]]; then
    echo ""
  else
    echo "$out"
  fi
}

check_duplicates() {
  local h1 h2 h3
  h1="$(hash_cam "$CAM1" || true)"
  h2="$(hash_cam "$CAM2" || true)"
  h3="$(hash_cam "$CAM3" || true)"

  # Basic sanity: did we get hashes?
  if [[ -z "$h1" || -z "$h2" || -z "$h3" ]]; then
    echo "[WARN] Preflight: could not read all cameras (h1='${h1}', h2='${h2}', h3='${h3}')"
    return 2
  fi

  echo "[INFO] Preflight hashes:"
  echo "  cam1($CAM1) $h1"
  echo "  cam2($CAM2) $h2"
  echo "  cam3($CAM3) $h3"

  # Duplicate detection
  if [[ "$h1" == "$h2" || "$h1" == "$h3" || "$h2" == "$h3" ]]; then
    echo "[ERR] Preflight: duplicate camera feeds detected (two indices look identical)."
    return 1
  fi

  return 0
}

echo "Preflight: checking for duplicate feeds..."
ok=0
for attempt in $(seq 1 "$PROBE_RETRIES"); do
  echo "[INFO] Preflight attempt $attempt/$PROBE_RETRIES"
  if check_duplicates; then
    ok=1
    break
  fi
  sleep "$PROBE_SLEEP"
done

if [[ "$ok" -ne 1 ]]; then
  echo
  echo "[FATAL] Cameras appear duplicated or unavailable."
  echo "Try:"
  echo "  - unplug/replug the cams"
  echo "  - use a powered USB hub"
  echo "  - keep them on the same ports"
  echo "  - close apps that might use cameras (Zoom/OBS/Browser)"
  exit 1
fi

echo "Preflight OK ✅"
echo "Press ENTER to stop."

# ---- record with ONE ffmpeg process (reduces avfoundation race) ----
ffmpeg -hide_banner -loglevel info \
  -f avfoundation -framerate "$FPS" -video_size "$SIZE" -pix_fmt "$IN_PIX" -i "${CAM1}:none" \
  -f avfoundation -framerate "$FPS" -video_size "$SIZE" -pix_fmt "$IN_PIX" -i "${CAM2}:none" \
  -f avfoundation -framerate "$FPS" -video_size "$SIZE" -pix_fmt "$IN_PIX" -i "${CAM3}:none" \
  -map 0:v -c:v "$ENC" -b:v "$BITRATE" "$outdir/cam1.mp4" \
  -map 1:v -c:v "$ENC" -b:v "$BITRATE" "$outdir/cam2.mp4" \
  -map 2:v -c:v "$ENC" -b:v "$BITRATE" "$outdir/cam3.mp4" \
  >"$outdir/ffmpeg.log" 2>&1 &
ffmpeg_pid=$!

read -r

# Graceful stop (macOS-safe)
kill -INT "$ffmpeg_pid" 2>/dev/null || true
wait "$ffmpeg_pid" 2>/dev/null || true

echo "Stopped."
ls -lh "$outdir"