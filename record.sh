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



ts=$(date +"%Y%m%d_%H%M%S")
outdir="data/sessions/$ts"
mkdir -p "$outdir"

echo "Recording to: $outdir"
echo "Press ENTER to stop."

ffmpeg -hide_banner -loglevel info \
  -f avfoundation -framerate $FPS -video_size $SIZE -pix_fmt $IN_PIX -i "${CAM1}:none" \
  -c:v $ENC -b:v $BITRATE \
  "$outdir/cam1.mp4" >"$outdir/cam1.log" 2>&1 &
p1=$!

ffmpeg -hide_banner -loglevel info \
  -f avfoundation -framerate $FPS -video_size $SIZE -pix_fmt $IN_PIX -i "${CAM2}:none" \
  -c:v $ENC -b:v $BITRATE \
  "$outdir/cam2.mp4" >"$outdir/cam2.log" 2>&1 &
p2=$!

ffmpeg -hide_banner -loglevel info \
  -f avfoundation -framerate $FPS -video_size $SIZE -pix_fmt $IN_PIX -i "${CAM3}:none" \
  -c:v $ENC -b:v $BITRATE \
  "$outdir/cam3.mp4" >"$outdir/cam3.log" 2>&1 &
p3=$!

read -r

kill $p1 $p2 $p3 2>/dev/null || true
wait $p1 $p2 $p3 2>/dev/null || true

echo "Stopped."
ls -lh "$outdir"
