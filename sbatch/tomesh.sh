#!/bin/bash
#SBATCH --job-name=tomesh
#SBATCH -w inspur1
#SBATCH --gres=shard:2
#SBATCH -c 8
#SBATCH -t 144000

# test_vcg --config /home/guoqingzhang/vcg/resources/other/tomesh_dgcnn_config.yaml
# test_vcg --config /home/guoqingzhang/vcg/resources/other/tomesh_pointmamba2_config.yaml
# test_vcg --config /home/guoqingzhang/vcg/resources/other/tomesh_pointnet2_config.yaml
test_vcg --config /home/guoqingzhang/vcg/resources/other/tomesh_pct2_config.yaml
# test_vcg --config /home/guoqingzhang/vcg/resources/other/tomesh_pvd_config.yaml
# test_flemme --config /home/guoqingzhang/vcg/resources/other/test_pointmamba2_condition_config.yaml
# test_vcg --config /home/guoqingzhang/vcg/resources/other/test_edm_none_condition_config.yaml