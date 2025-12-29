#!/bin/bash
#SBATCH --job-name=train_others
#SBATCH -w inspur1
#SBATCH --gres=shard:2
#SBATCH -c 8
#SBATCH -t 144000

# train_flemme --config /home/guoqingzhang/vcg/resources/other/train_dgcnn_condition_config.yaml
# train_flemme --config /home/guoqingzhang/vcg/resources/other/train_pointnet2_condition_config.yaml
# train_flemme --config /home/guoqingzhang/vcg/resources/other/train_pointmamba2_condition_config.yaml
train_flemme --config /home/guoqingzhang/vcg/resources/other/train_pct2_condition_config.yaml
# train_flemme --config /home/guoqingzhang/vcg/resources/other/train_pvd_condition_config.yaml
# train_vcg --config /home/guoqingzhang/vcg/resources/other/train_edm_none_condition_config.yaml

