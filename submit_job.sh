#!/bin/bash -l
#SBATCH --job-name=llama3_70b
#SBATCH --nodes=2
#SBATCH --tasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=2:00:00
#SBATCH --account=OD-236007
#SBATCH --gres=gpu:2
 
module load python pytorch
source /scratch3/koo01a/virtual_envs/test_env/bin/activate

cd /home/koo01a/llama3-8b-medical
# HUGGINGFACE_HUB_CACHE="/scratch3/koo01a/huggingface/hub" HF_DATASETS_CACHE="/scratch3/koo01a/huggingface/hub" ACCELERATE_USE_FSDP=1 FSDP_CPU_RAM_EFFICIENT_LOADING=1 python -m torch.distributed.run --nproc_per_node=2 run_fsdp_qlora.py --config llama_3_70b_fsdp_qlora.yaml
#
HUGGINGFACE_HUB_CACHE="/scratch3/koo01a/huggingface/hub" HF_DATASETS_CACHE="/scratch3/koo01a/huggingface/hub" ACCELERATE_USE_FSDP=1 FSDP_CPU_RAM_EFFICIENT_LOADING=1 python $1
