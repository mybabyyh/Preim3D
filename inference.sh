CUDA_VISIBLE_DEVICES=0 python ./scripts/inference.py \
--images_dir=./data/test   --n_sample=2 \
--edit_attribute='inversion' \
--edit_direction_dir=./editings/interfacegan_directions/inversion3_tensor \
--save_dir=./experiment/test --ckpt=./pretrained/preim3d_ffhq.pt \
--eg3d_generator=./pretrained/eg3d_G_ema.pkl \
--video

CUDA_VISIBLE_DEVICES=0 python ./scripts/inference.py \
--images_dir=./data/test   --n_sample=2 \
--edit_attribute='age'  --edit_degree='10' \
--edit_direction_dir=./editings/interfacegan_directions/inversion3_tensor \
--save_dir=./experiment/test --ckpt=./pretrained/preim3d_ffhq.pt \
--eg3d_generator=./pretrained/eg3d_G_ema.pkl \
--video

CUDA_VISIBLE_DEVICES=0 python ./scripts/inference.py \
--images_dir=./data/test   --n_sample=2 \
--edit_attribute='smile'  --edit_degree='10' \
--edit_direction_dir=./editings/interfacegan_directions/inversion3_tensor \
--save_dir=./experiment/test --ckpt=./pretrained/preim3d_ffhq.pt \
--eg3d_generator=./pretrained/eg3d_G_ema.pkl

# CUDA_VISIBLE_DEVICES=0 python ./scripts/inference.py \
# --images_dir=./data/test   --n_sample=2 \
# --edit_attribute='eyeglass'  --edit_degree='10' \
# --edit_direction_dir=./editings/interfacegan_directions/inversion3_tensor \
# --save_dir=./experiment/test --ckpt=./pretrained/preim3d_ffhq.pt \
# --eg3d_generator=./pretrained/eg3d_G_ema.pkl

# CUDA_VISIBLE_DEVICES=0 python ./scripts/inference.py \
# --images_dir=./data/test   --n_sample=2 \
# --edit_attribute='goatee'  --edit_degree='10' \
# --edit_direction_dir=./editings/interfacegan_directions/inversion3_tensor \
# --save_dir=./experiment/test --ckpt=./pretrained/preim3d_ffhq.pt \
# --eg3d_generator=./pretrained/eg3d_G_ema.pkl

# CUDA_VISIBLE_DEVICES=0 python ./scripts/inference.py \
# --images_dir=./data/test   --n_sample=2 \
# --edit_attribute='male'  --edit_degree='10' \
# --edit_direction_dir=./editings/interfacegan_directions/inversion3_tensor \
# --save_dir=./experiment/test --ckpt=./pretrained/preim3d_ffhq.pt \
# --eg3d_generator=./pretrained/eg3d_G_ema.pkl

# CUDA_VISIBLE_DEVICES=0 python ./scripts/inference.py \
# --images_dir=./data/test   --n_sample=2 \
# --edit_attribute='wavyhair'  --edit_degree='10' \
# --edit_direction_dir=./editings/interfacegan_directions/inversion3_tensor \
# --save_dir=./experiment/test --ckpt=./pretrained/preim3d_ffhq.pt \
# --eg3d_generator=./pretrained/eg3d_G_ema.pkl

# CUDA_VISIBLE_DEVICES=0 python ./scripts/inference.py \
# --images_dir=./data/test   --n_sample=2 \
# --edit_attribute='grayhair'  --edit_degree='10' \
# --edit_direction_dir=./editings/interfacegan_directions/inversion3_tensor \
# --save_dir=./experiment/test --ckpt=./pretrained/preim3d_ffhq.pt \
# --eg3d_generator=./pretrained/eg3d_G_ema.pkl

# CUDA_VISIBLE_DEVICES=0 python ./scripts/inference.py \
# --images_dir=./data/test   --n_sample=2 \
# --edit_attribute='lipstick'  --edit_degree='10' \
# --edit_direction_dir=./editings/interfacegan_directions/inversion3_tensor \
# --save_dir=./experiment/test --ckpt=./pretrained/preim3d_ffhq.pt \
# --eg3d_generator=./pretrained/eg3d_G_ema.pkl

# CUDA_VISIBLE_DEVICES=0 python ./scripts/inference.py \
# --images_dir=./data/test   --n_sample=2 \
# --edit_attribute='makeup'  --edit_degree='10' \
# --edit_direction_dir=./editings/interfacegan_directions/inversion3_tensor \
# --save_dir=./experiment/test --ckpt=./pretrained/preim3d_ffhq.pt \
# --eg3d_generator=./pretrained/eg3d_G_ema.pkl

# CUDA_VISIBLE_DEVICES=0 python ./scripts/inference.py \
# --images_dir=./data/test   --n_sample=2 \
# --edit_attribute='blackhair'  --edit_degree='10' \
# --edit_direction_dir=./editings/interfacegan_directions/inversion3_tensor \
# --save_dir=./experiment/test --ckpt=./pretrained/preim3d_ffhq.pt \
# --eg3d_generator=./pretrained/eg3d_G_ema.pkl

# CUDA_VISIBLE_DEVICES=0 python ./scripts/inference.py \
# --images_dir=./data/test   --n_sample=2 \
# --edit_attribute='brownhair'  --edit_degree='10' \
# --edit_direction_dir=./editings/interfacegan_directions/inversion3_tensor \
# --save_dir=./experiment/test --ckpt=./pretrained/preim3d_ffhq.pt \
# --eg3d_generator=./pretrained/eg3d_G_ema.pkl

# CUDA_VISIBLE_DEVICES=0 python ./scripts/inference.py \
# --images_dir=./data/test   --n_sample=2 \
# --edit_attribute='mouthopen'  --edit_degree='10' \
# --edit_direction_dir=./editings/interfacegan_directions/inversion3_tensor \
# --save_dir=./experiment/test --ckpt=./pretrained/preim3d_ffhq.pt \
# --eg3d_generator=./pretrained/eg3d_G_ema.pkl