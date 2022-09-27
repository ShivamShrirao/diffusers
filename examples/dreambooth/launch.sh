export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="/home/ubuntu/diffusion_tests/data/shivam"
export CLASS_DIR="/home/ubuntu/diffusion_tests/data/guy"
export OUTPUT_DIR="/home/ubuntu/diffusion_tests/models/shivam"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME --use_auth_token \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation \
  --instance_prompt="photo of sks guy" \
  --class_prompt="photo of a guy" \
  --resolution=256 \
  --train_batch_size=1 \
  --sample_batch_size 1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=800 \
  --use_8bit_adam \
  --gradient_checkpointing \
  --mixed_precision bf16
