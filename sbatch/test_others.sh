#!/bin/bash
#SBATCH --job-name=test_others
#SBATCH -w inspur1
#SBATCH --gres=shard:2
#SBATCH -c 8
#SBATCH -t 144000

# test_flemme --config /home/guoqingzhang/meshage/resources/other/test_pointnet2_condition_config.yaml
# test_flemme --config /home/guoqingzhang/meshage/resources/other/test_dgcnn_condition_config.yaml
test_flemme --config /home/guoqingzhang/meshage/resources/other/test_pct2_condition_config.yaml
# test_flemme --config /home/guoqingzhang/meshage/resources/other/test_pointmamba2_condition_config.yaml
# test_meshage --config /home/guoqingzhang/meshage/resources/test_edm_none_condition_config.yaml
# test_flemme --config /home/guoqingzhang/meshage/resources/other/test_pvd_condition_config.yaml
