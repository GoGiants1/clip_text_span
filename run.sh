#!/bin/bash

# SBATCH --job-name=mlvu-team8-project                    # Submit a job named "example"
# SBATCH --nodes=1                             # Using 1 node
# SBATCH --gres=gpu:1                          # Using 1 gpu
# SBATCH --time=0-02:00:00                     # 1 hour timelimit
# SBATCH --mem=10000MB                         # Using 10GB CPU Memory
# SBATCH --partition=class2                         # Using "b" partition 
# SBATCH --cpus-per-task=4                     # Using 4 maximum processor

source ${HOME}/.bashrc
source ${HOME}/anaconda3/bin/activate
conda activate prsclip

# 처음 실행시, 주석 해제 후, 다운로드.
# wget https://huggingface.co/datasets/GoGiants1/TMDBEval500/resolve/main/TMDBEval500.zip
# unzip -l TMDBEval500.zip # 목록 출력
# unzip TMDBEval500.zip

python compute_prs.py --dataset TMDBEval500 --device cuda:0 --model ViT-L-14 --pretrained laion2b_s32b_b82k --data_path ./TMDBEval500/images 
python compute_text_projection.py  --dataset TMDBEval500 --device cuda:0 --model ViT-L-14 --pretrained laion2b_s32b_b82k
python compute_ablations.py --model ViT-H-14
python compute_text_set_projection.py --device cuda:0 --model ViT-L-14 --pretrained laion2b_s32b_b82k --data_path text_descriptions/letter_search.txt
python compute_complete_text_set.py --device cuda:0 --model ViT-L-14 --texts_per_head 20 --num_of_last_layers 4 --text_descriptions image_descriptions_general
python compute_use_specific_heads.py --model ViT-L-14 --dataset TMDBEval500 

