#!/bin/bash
#SBATCH --job-name=eval_recon_mesh
#SBATCH -w inspur1
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH -t 144000

# echo pointnet++
# eval_flemme --config /home/guoqingzhang/vcg/resources/other/eval_pointnet2_mesh_condition_config.yaml
# echo dgcnn
# eval_flemme --config /home/guoqingzhang/vcg/resources/other/eval_dgcnn_mesh_condition_config.yaml
# echo pointmamba2
# eval_flemme --config /home/guoqingzhang/vcg/resources/other/eval_pointmamba2_mesh_condition_config.yaml
# echo diff_pcd
# eval_flemme --config /home/guoqingzhang/vcg/resources/other/eval_diffpcd_recon_mesh_condition_config.yaml
# echo vessel_diff
# eval_flemme --config /home/guoqingzhang/vcg/resources/eval_diffves_mesh_condition_config.yaml
# echo gem3d
# eval_flemme --config /home/guoqingzhang/vcg/resources/other/eval_gem3d_recon_mesh_condition_config.yaml
# echo skcnn
# eval_flemme --config /home/guoqingzhang/vcg/resources/skcnn/eval_skae_cnn_mesh_condition_config.yaml
# echo skcnn wo skc
# eval_flemme --config /home/guoqingzhang/vcg/resources/skcnn/eval_skae_cnn_mesh_condition_config_wo_skc.yaml
# echo spcnn
# eval_flemme --config /home/guoqingzhang/vcg/resources/spcnn/eval_spae_cnn_mesh_condition_config.yaml
# echo spcnn wo lpc
# eval_flemme --config /home/guoqingzhang/vcg/resources/spcnn/eval_spae_cnn_mesh_condition_config_wo_lpc.yaml
# echo skspcnn
# eval_flemme --config /home/guoqingzhang/vcg/resources/skspcnn/eval_skspae_cnn_mesh_condition_config.yaml
# echo skspcnn wo lpc
# eval_flemme --config /home/guoqingzhang/vcg/resources/skspcnn/eval_skspae_cnn_mesh_condition_config_wo_lpc.yaml
# echo sktrans
# eval_flemme --config /home/guoqingzhang/vcg/resources/sktrans/eval_skae_trans_mesh_condition_config.yaml
# echo sktrans wo skc
# eval_flemme --config /home/guoqingzhang/vcg/resources/sktrans/eval_skae_trans_mesh_condition_config_wo_skc.yaml
# echo sksptrans wo lpc
# eval_flemme --config /home/guoqingzhang/vcg/resources/sksptrans/eval_skspae_trans_mesh_condition_config_wo_lpc.yaml
# echo skcnn-512
# eval_flemme --config /home/guoqingzhang/vcg/resources/skcnn/eval_skae_cnn_mesh_condition_config_512.yaml
# echo skcnn-512 with LA OS
# eval_flemme --config /home/guoqingzhang/vcg/resources/skcnn/eval_skae_cnn_mesh_condition_config_512_with_la_os.yaml
echo skcnn with LA
eval_flemme --config /home/guoqingzhang/vcg/resources/skcnn/eval_skae_cnn_mesh_condition_config_with_la.yaml
echo skcnn-512 with LA
eval_flemme --config /home/guoqingzhang/vcg/resources/skcnn/eval_skae_cnn_mesh_condition_config_with_la_512.yaml