#!/usr/bin/env bash
set -euo pipefail

FPS=15
SIZE=1280x720
# NOTE: forcing IN_PIX can break/stall multi-cam on macOS.
# Leave empty to let avfoundation choose a stable format.
IN_PIX=""   # e.g. "uyvy422" if you *really* want; default "" (recommended)

ENC=h264_videotoolbox
BITRATE=6M

# Set any CAM* to -1 to disable it.
CAM1=0
CAM2=1
CAM3=-1
CAM4=-1
CAM5=-1

# ---- preflight settings ----
HASH_W=32
HASH_H=32
PROBE_RETRIES=5
PROBE_SLEEP=0.4

ts=$(date +"%Y%m%d_%H%M%S")
outdir="videos/$ts"
mkdir -p "$outdir"

# ---- collect enabled camera indices in order ----
CAMS=()
for v in "$CAM1" "$CAM2" "$CAM3" "$CAM4" "$CAM5"; do
  if [[ "${v}" != "-1" ]]; then
    CAMS+=("$v")
  fi
done

if [[ "${#CAMS[@]}" -eq 0 ]]; then
  echo "[FATAL] No cameras enabled. Set CAM1..CAM5 to indices (0,1,2,...) and use -1 to disable."
  exit 1
fi

echo "Recording to: $outdir"
echo "Cameras enabled (${#CAMS[@]}): ${CAMS[*]}"

# ---- helpers ----
hash_cam() {
  local cam_idx="$1"
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
  local -a hashes=()
  local cam_idx h

  for cam_idx in "${CAMS[@]}"; do
    h="$(hash_cam "$cam_idx" || true)"
    hashes+=("$h")
  done

  # Basic sanity: did we get hashes?
  local i
  for i in "${!CAMS[@]}"; do
    if [[ -z "${hashes[$i]}" ]]; then
      echo "[WARN] Preflight: could not read cam${i} (idx=${CAMS[$i]})"
      return 2
    fi
  done

  echo "[INFO] Preflight hashes:"
  for i in "${!CAMS[@]}"; do
    printf "  cam%d(%s) %s\n" "$((i+1))" "${CAMS[$i]}" "${hashes[$i]}"
  done

  # Duplicate detection (pairwise)
  local a b
  for a in "${!hashes[@]}"; do
    for b in "${!hashes[@]}"; do
      if (( b <= a )); then
        continue
      fi
      if [[ "${hashes[$a]}" == "${hashes[$b]}" ]]; then
        echo "[ERR] Preflight: duplicate camera feeds detected between cam$((a+1)) (idx=${CAMS[$a]}) and cam$((b+1)) (idx=${CAMS[$b]})."
        return 1
      fi
    done
  done

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

# ---- build ffmpeg args dynamically ----
ff_args=(
  -hide_banner -loglevel info
)

# Inputs
for cam_idx in "${CAMS[@]}"; do
  ff_args+=(
    -thread_queue_size 1024
    -f avfoundation
    -framerate "$FPS"
    -video_size "$SIZE"
  )
  if [[ -n "${IN_PIX}" ]]; then
    ff_args+=(-pix_fmt "$IN_PIX")
  fi
  ff_args+=(-i "${cam_idx}:none")
done

# Outputs (one file per input)
for i in "${!CAMS[@]}"; do
  out="cam$((i+1)).mp4"
  ff_args+=(
    -map "${i}:v"
    -c:v "$ENC"
    -b:v "$BITRATE"
    "$outdir/$out"
  )
done

# ---- record with ONE ffmpeg process (reduces avfoundation race) ----
ffmpeg "${ff_args[@]}" >"$outdir/ffmpeg.log" 2>&1 &
ffmpeg_pid=$!

read -r

# Graceful stop (macOS-safe)
kill -INT "$ffmpeg_pid" 2>/dev/null || true
wait "$ffmpeg_pid" 2>/dev/null || true

echo "Stopped."
ls -lh "$outdir"
