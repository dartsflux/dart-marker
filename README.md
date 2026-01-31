label command:

python label_dart_tip.py \
  --video ./videos/20260128_030440/cam1.mp4 --start-frame 1 \
  --auto-pause \
  --roi 150,120,1180,700 \
  --motion-frac 0.0025 \
  --stable-frac 0.0007 \
  --stable-frames 3 \
  --diff-px 16 --blur 7 --downscale 0.33 \
  --cooldown-frames 10 \
  --motion-debug

-------------
train board model
yolo task=pose mode=train \
  model=yolov8s-pose.pt \
  data=./datasets/yolo_board_pose7/dataset.yaml \
  imgsz=960 \
  epochs=200 \
  patience=20 \
  batch=16 \
  lr0=0.002 \
  freeze=10 \
  device=mps \
  name=boardcalib-8s

