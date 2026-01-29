label command:

python label_dart_tip.py \                                  
  --video ../../footage/data/sessions/20260128_024058/cam3.mp4 --start-frame 1 \
  --auto-pause \
  --roi 150,120,1180,700 \
  --motion-frac 0.0025 \
  --stable-frac 0.0007 \
  --stable-frames 3 \
  --diff-px 16 --blur 7 --downscale 0.33 \
  --cooldown-frames 10 \
  --motion-debug