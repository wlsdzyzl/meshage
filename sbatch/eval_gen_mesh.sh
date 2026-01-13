#!/bin/bash
#SBATCH --job-name=test_cnn
#SBATCH -w inspur1
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH -t 144000
# echo diffpcd
# eval_flemme --config /home/guoqingzhang/vcg/resources/other/eval_diffpcd_gen_mesh_condition_config.yaml
# echo edm-pcd
# eval_flemme --config /home/guoqingzhang/vcg/resources/other/eval_emd_none_gen_mesh_condition_config.yaml
# echo gem3d
# eval_flemme --config /home/guoqingzhang/vcg/resources/other/eval_gem3d_gen_mesh_condition_config.yaml
# echo skcnn
# eval_flemme --config /home/guoqingzhang/vcg/resources/skcnn/eval_ldm_skcnn_mesh_condition_config.yaml
# echo skcnn wo skc
# eval_flemme --config /home/guoqingzhang/vcg/resources/skcnn/eval_ldm_skcnn_mesh_condition_config_wo_skc.yaml
# echo spcnn
# eval_flemme --config /home/guoqingzhang/vcg/resources/spcnn/eval_ldm_spcnn_mesh_condition_config.yaml
# echo spcnn wo lpc
# eval_flemme --config /home/guoqingzhang/vcg/resources/spcnn/eval_ldm_spcnn_mesh_condition_config_wo_lpc.yaml
# echo skspcnn
# eval_flemme --config /home/guoqingzhang/vcg/resources/skspcnn/eval_ldm_skspcnn_mesh_condition_config.yaml
# echo skspcnn wo lpc
# eval_flemme --config /home/guoqingzhang/vcg/resources/skspcnn/eval_ldm_skspcnn_mesh_condition_config_wo_lpc.yaml
# echo sktrans
# eval_flemme --config /home/guoqingzhang/vcg/resources/sktrans/eval_ldm_sktrans_mesh_condition_config.yaml
# echo sktrans wo skc
# eval_flemme --config /home/guoqingzhang/vcg/resources/sktrans/eval_ldm_sktrans_mesh_condition_config_wo_skc.yaml
# echo sksptrans wo lpc
# eval_flemme --config /home/guoqingzhang/vcg/resources/sksptrans/eval_ldm_sksptrans_mesh_condition_config_wo_lpc.yaml
# echo skcnn-512
# eval_flemme --config /home/guoqingzhang/vcg/resources/skcnn/eval_ldm_skcnn_mesh_condition_config_512.yaml
echo skcnn-512 with LA OS
eval_flemme --config /home/guoqingzhang/vcg/resources/skcnn/eval_ldm_skcnn_mesh_condition_config_with_la_os_512.yaml
echo skcnn with LA
eval_flemme --config /home/guoqingzhang/vcg/resources/skcnn/eval_ldm_skcnn_mesh_condition_config_with_la.yaml
# echo skcnn-512 with LA
# eval_flemme --config /home/guoqingzhang/vcg/resources/skcnn/eval_ldm_skcnn_mesh_condition_config_with_la_512.yaml