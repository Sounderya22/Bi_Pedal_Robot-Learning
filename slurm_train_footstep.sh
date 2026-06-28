#!/bin/bash
#SBATCH --job-name=cassie_footstep
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa6000:2
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=tron
#SBATCH --output=/fs/nexus-scratch/vvs22/slurm_logs/cassie_footstep_%j.out
#SBATCH --error=/fs/nexus-scratch/vvs22/slurm_logs/cassie_footstep_%j.err

echo "=== Job started at $(date) ==="
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader

# Initialize conda
source /nfshomes/vvs22/miniconda3/etc/profile.d/conda.sh
conda activate cassie

# Environment variables for headless MuJoCo
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin

# Start training
cd /fs/nexus-scratch/vvs22/Robot-Learning-Bipedal-Footstep-PID-MPC/exe
mpirun -np 16 python ../scripts/train.py \
    --train_name 'footstep_training_v7' \
    --rnd_seed 42 \
    --max_iters 500000 \
    --save_interval 100 \
    --restore_from 'footstep_training_v7_rnds42' \
    --restore_cont 1

echo "=== Job finished at $(date) ==="
