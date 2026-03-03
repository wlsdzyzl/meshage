#!/bin/bash
#SBATCH --job-name=timer
#SBATCH -w inspur1
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH -t 144000

test_meshage --config /home/guoqingzhang/meshage/resources/other/test_timer_pvd_condition_config.yaml
test_meshage --config /home/guoqingzhang/meshage/resources/other/test_timer_edm_none_condition_config.yaml
test_meshage --config /home/guoqingzhang/meshage/resources/skcnn/test_timer_ldm_skcnn_condition_config.yaml
test_meshage --config /home/guoqingzhang/meshage/resources/other/tomesh_timer_config.yaml