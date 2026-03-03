#!/bin/bash
#SBATCH --job-name=eval_vessel
#SBATCH -w inspur1
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH -t 144000

echo CoW
echo '## Recon'
# echo '### VessDiff'
# eval_flemme --config /home/guoqingzhang/meshage/resources/vessel/cow_vessel_diff/eval_diffvess_pcd_ae_config.yaml
# eval_flemme --config /home/guoqingzhang/meshage/resources/vessel/cow_vessel_diff/eval_diffvess_mesh_ae_config.yaml
# echo '### SpCNN'
# eval_flemme --config /home/guoqingzhang/meshage/resources/vessel/cow_vessel_diff/eval_spcnn_pcd_ae_config.yaml
# echo '### SpCNN with LPC LA'
# eval_flemme --config /home/guoqingzhang/meshage/resources/vessel/cow_vessel_diff/eval_spcnn_pcd_ae_config_with_la.yaml
# echo '### SkCNN with OS'
# eval_flemme --config /home/guoqingzhang/meshage/resources/vessel/cow_vessel_diff/eval_skcnn_pcd_ae_config_with_os.yaml
# echo '### SkCNN'
# eval_flemme --config /home/guoqingzhang/meshage/resources/vessel/cow_vessel_diff/eval_skcnn_pcd_ae_config.yaml
# eval_flemme --config /home/guoqingzhang/meshage/resources/vessel/cow_vessel_diff/eval_skcnn_mesh_ae_config.yaml
# echo '### SkCNN with SKC'
# eval_flemme --config /home/guoqingzhang/meshage/resources/vessel/cow_vessel_diff/eval_skcnn_pcd_ae_config_with_skc.yaml
# eval_flemme --config /home/guoqingzhang/meshage/resources/vessel/cow_vessel_diff/eval_skcnn_mesh_ae_config_with_skc.yaml
# echo '### SkSpCNN'
# eval_flemme --config /home/guoqingzhang/meshage/resources/vessel/cow_vessel_diff/eval_skspcnn_pcd_ae_config.yaml
# eval_flemme --config /home/guoqingzhang/meshage/resources/vessel/cow_vessel_diff/eval_skspcnn_mesh_ae_config.yaml
# echo '### SkCNN with SKC LA'
# eval_flemme --config /home/guoqingzhang/meshage/resources/vessel/cow_vessel_diff/eval_skcnn_pcd_ae_config_with_skc_la.yaml
# eval_flemme --config /home/guoqingzhang/meshage/resources/vessel/cow_vessel_diff/eval_skcnn_mesh_ae_config_with_skc_la.yaml
# echo '### SkCNN with SKC LA OS'
# eval_flemme --config /home/guoqingzhang/meshage/resources/vessel/cow_vessel_diff/eval_skcnn_pcd_ae_config_with_skc_la_os.yaml
# eval_flemme --config /home/guoqingzhang/meshage/resources/vessel/cow_vessel_diff/eval_skcnn_mesh_ae_config_with_skc_la_os.yaml
echo '## Gen'
# echo '### VessDiff'
# eval_flemme --config /home/guoqingzhang/meshage/resources/vessel/cow_vessel_diff/eval_diffvess_pcd_gen_config.yaml
# eval_flemme --config /home/guoqingzhang/meshage/resources/vessel/cow_vessel_diff/eval_diffvess_mesh_gen_config.yaml
# echo '### SpCNN'
# eval_flemme --config /home/guoqingzhang/meshage/resources/vessel/cow_vessel_diff/eval_spcnn_pcd_gen_config.yaml
# echo '### SpCNN with LPC LA'
# eval_flemme --config /home/guoqingzhang/meshage/resources/vessel/cow_vessel_diff/eval_spcnn_pcd_gen_config_with_la.yaml
# echo '### SkCNN with OS'
# eval_flemme --config /home/guoqingzhang/meshage/resources/vessel/cow_vessel_diff/eval_skcnn_pcd_gen_config_with_os.yaml
# echo '### SkCNN'
# eval_flemme --config /home/guoqingzhang/meshage/resources/vessel/cow_vessel_diff/eval_skcnn_pcd_gen_config.yaml
# eval_flemme --config /home/guoqingzhang/meshage/resources/vessel/cow_vessel_diff/eval_skcnn_mesh_gen_config.yaml
# echo '### SkCNN with SKC'
# eval_flemme --config /home/guoqingzhang/meshage/resources/vessel/cow_vessel_diff/eval_skcnn_pcd_gen_config_with_skc.yaml
# eval_flemme --config /home/guoqingzhang/meshage/resources/vessel/cow_vessel_diff/eval_skcnn_mesh_gen_config_with_skc.yaml
# echo '### SkSpCNN'
# eval_flemme --config /home/guoqingzhang/meshage/resources/vessel/cow_vessel_diff/eval_skspcnn_pcd_gen_config.yaml
# eval_flemme --config /home/guoqingzhang/meshage/resources/vessel/cow_vessel_diff/eval_skspcnn_mesh_gen_config.yaml
# echo '### SkCNN with SKC LA'
# eval_flemme --config /home/guoqingzhang/meshage/resources/vessel/cow_vessel_diff/eval_skcnn_pcd_gen_config_with_skc_la.yaml
# eval_flemme --config /home/guoqingzhang/meshage/resources/vessel/cow_vessel_diff/eval_skcnn_mesh_gen_config_with_skc_la.yaml
# echo '### SkCNN with SKC LA OS'
# eval_flemme --config /home/guoqingzhang/meshage/resources/vessel/cow_vessel_diff/eval_skcnn_pcd_gen_config_with_skc_la_os.yaml
# eval_flemme --config /home/guoqingzhang/meshage/resources/vessel/cow_vessel_diff/eval_skcnn_mesh_gen_config_with_skc_la_os.yaml

echo imageCAS
echo '## Recon'
# echo '### VessDiff'
# eval_flemme --config /home/guoqingzhang/meshage/resources/vessel/imagecas_vessel_diff/eval_diffvess_pcd_ae_config.yaml
# eval_flemme --config /home/guoqingzhang/meshage/resources/vessel/imagecas_vessel_diff/eval_diffvess_mesh_ae_config.yaml
# echo '### SpCNN'
# eval_flemme --config /home/guoqingzhang/meshage/resources/vessel/imagecas_vessel_diff/eval_spcnn_pcd_ae_config.yaml
# echo '### SpCNN with LPC LA'
# eval_flemme --config /home/guoqingzhang/meshage/resources/vessel/imagecas_vessel_diff/eval_spcnn_pcd_ae_config_with_la.yaml
# echo '### SkCNN with OS'
# eval_flemme --config /home/guoqingzhang/meshage/resources/vessel/imagecas_vessel_diff/eval_skcnn_pcd_ae_config_with_os.yaml
# echo '### SkCNN'
eval_flemme --config /home/guoqingzhang/meshage/resources/vessel/imagecas_vessel_diff/eval_skcnn_pcd_ae_config.yaml
eval_flemme --config /home/guoqingzhang/meshage/resources/vessel/imagecas_vessel_diff/eval_skcnn_mesh_ae_config.yaml
# echo '### SkCNN with SKC'
# eval_flemme --config /home/guoqingzhang/meshage/resources/vessel/imagecas_vessel_diff/eval_skcnn_pcd_ae_config_with_skc.yaml
# eval_flemme --config /home/guoqingzhang/meshage/resources/vessel/imagecas_vessel_diff/eval_skcnn_mesh_ae_config_with_skc.yaml
# echo '### SkSpCNN'
# eval_flemme --config /home/guoqingzhang/meshage/resources/vessel/imagecas_vessel_diff/eval_skspcnn_pcd_ae_config.yaml
# eval_flemme --config /home/guoqingzhang/meshage/resources/vessel/imagecas_vessel_diff/eval_skspcnn_mesh_ae_config.yaml
# echo '### SkCNN with SKC LA'
# eval_flemme --config /home/guoqingzhang/meshage/resources/vessel/imagecas_vessel_diff/eval_skcnn_pcd_ae_config_with_skc_la.yaml
# eval_flemme --config /home/guoqingzhang/meshage/resources/vessel/imagecas_vessel_diff/eval_skcnn_mesh_ae_config_with_skc_la.yaml
# echo '### SkCNN with SKC LA OS'
# eval_flemme --config /home/guoqingzhang/meshage/resources/vessel/imagecas_vessel_diff/eval_skcnn_pcd_ae_config_with_skc_la_os.yaml
# eval_flemme --config /home/guoqingzhang/meshage/resources/vessel/imagecas_vessel_diff/eval_skcnn_mesh_ae_config_with_skc_la_os.yaml
echo '## Gen'
# echo '### VessDiff'
# eval_flemme --config /home/guoqingzhang/meshage/resources/vessel/imagecas_vessel_diff/eval_diffvess_pcd_gen_config.yaml
# eval_flemme --config /home/guoqingzhang/meshage/resources/vessel/imagecas_vessel_diff/eval_diffvess_mesh_gen_config.yaml
# echo '### SpCNN'
# eval_flemme --config /home/guoqingzhang/meshage/resources/vessel/imagecas_vessel_diff/eval_spcnn_pcd_gen_config.yaml
# echo '### SpCNN with LPC LA'
# eval_flemme --config /home/guoqingzhang/meshage/resources/vessel/imagecas_vessel_diff/eval_spcnn_pcd_gen_config_with_la.yaml
# echo '### SkCNN with OS'
# eval_flemme --config /home/guoqingzhang/meshage/resources/vessel/imagecas_vessel_diff/eval_skcnn_pcd_gen_config_with_os.yaml
echo '### SkCNN'
eval_flemme --config /home/guoqingzhang/meshage/resources/vessel/imagecas_vessel_diff/eval_skcnn_pcd_gen_config.yaml
eval_flemme --config /home/guoqingzhang/meshage/resources/vessel/imagecas_vessel_diff/eval_skcnn_mesh_gen_config.yaml
# echo '### SkCNN with SKC'
# eval_flemme --config /home/guoqingzhang/meshage/resources/vessel/imagecas_vessel_diff/eval_skcnn_pcd_gen_config_with_skc.yaml
# eval_flemme --config /home/guoqingzhang/meshage/resources/vessel/imagecas_vessel_diff/eval_skcnn_mesh_gen_config_with_skc.yaml
# echo '### SkSpCNN'
# eval_flemme --config /home/guoqingzhang/meshage/resources/vessel/imagecas_vessel_diff/eval_skspcnn_pcd_gen_config.yaml
# eval_flemme --config /home/guoqingzhang/meshage/resources/vessel/imagecas_vessel_diff/eval_skspcnn_mesh_gen_config.yaml
# echo '### SkCNN with SKC LA'
# eval_flemme --config /home/guoqingzhang/meshage/resources/vessel/imagecas_vessel_diff/eval_skcnn_pcd_gen_config_with_skc_la.yaml
# eval_flemme --config /home/guoqingzhang/meshage/resources/vessel/imagecas_vessel_diff/eval_skcnn_mesh_gen_config_with_skc_la.yaml
# echo '### SkCNN with SKC LA OS'
# eval_flemme --config /home/guoqingzhang/meshage/resources/vessel/imagecas_vessel_diff/eval_skcnn_pcd_gen_config_with_skc_la_os.yaml
# eval_flemme --config /home/guoqingzhang/meshage/resources/vessel/imagecas_vessel_diff/eval_skcnn_mesh_gen_config_with_skc_la_os.yaml