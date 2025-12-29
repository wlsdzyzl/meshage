#!/bin/bash
#SBATCH --job-name=eval_recon_pcd
#SBATCH -w inspur1
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH -t 144000

echo CoW
echo '## Recon'
echo '### VessDiff'
eval_flemme --config /home/wlsdzyzl/project/vcg/resources/vessel/cow_vessel_diff/eval_diffvess_pcd_ae_config.yaml
eval_flemme --config /home/wlsdzyzl/project/vcg/resources/vessel/cow_vessel_diff/eval_diffvess_mesh_ae_config.yaml
echo '### SkCNN'
eval_flemme --config /home/wlsdzyzl/project/vcg/resources/vessel/cow_vessel_diff/eval_skcnn_pcd_ae_config.yaml
eval_flemme --config /home/wlsdzyzl/project/vcg/resources/vessel/cow_vessel_diff/eval_skcnn_mesh_ae_config.yaml
echo '### SkCNN with SKC'
eval_flemme --config /home/wlsdzyzl/project/vcg/resources/vessel/cow_vessel_diff/eval_skcnn_pcd_ae_config_with_skc.yaml
eval_flemme --config /home/wlsdzyzl/project/vcg/resources/vessel/cow_vessel_diff/eval_skcnn_mesh_ae_config_with_skc.yaml
echo '### SkSpCNN'
eval_flemme --config /home/wlsdzyzl/project/vcg/resources/vessel/cow_vessel_diff/eval_skspcnn_pcd_ae_config.yaml
eval_flemme --config /home/wlsdzyzl/project/vcg/resources/vessel/cow_vessel_diff/eval_skspcnn_mesh_ae_config.yaml
echo '## Gen'
echo '### VessDiff'
eval_flemme --config /home/wlsdzyzl/project/vcg/resources/vessel/cow_vessel_diff/eval_diffvess_pcd_gen_config.yaml
eval_flemme --config /home/wlsdzyzl/project/vcg/resources/vessel/cow_vessel_diff/eval_diffvess_mesh_gen_config.yaml
echo '### SkCNN'
eval_flemme --config /home/wlsdzyzl/project/vcg/resources/vessel/cow_vessel_diff/eval_skcnn_pcd_gen_config.yaml
eval_flemme --config /home/wlsdzyzl/project/vcg/resources/vessel/cow_vessel_diff/eval_skcnn_mesh_gen_config.yaml
echo '### SkCNN with SKC'
eval_flemme --config /home/wlsdzyzl/project/vcg/resources/vessel/cow_vessel_diff/eval_skcnn_pcd_gen_config_with_skc.yaml
eval_flemme --config /home/wlsdzyzl/project/vcg/resources/vessel/cow_vessel_diff/eval_skcnn_mesh_gen_config_with_skc.yaml
echo '### SkSpCNN'
eval_flemme --config /home/wlsdzyzl/project/vcg/resources/vessel/cow_vessel_diff/eval_skspcnn_pcd_gen_config.yaml
eval_flemme --config /home/wlsdzyzl/project/vcg/resources/vessel/cow_vessel_diff/eval_skspcnn_mesh_gen_config.yaml

echo imageCAS
echo '## Recon'
echo '### VessDiff'
eval_flemme --config /home/wlsdzyzl/project/vcg/resources/vessel/imagecas_vessel_diff/eval_diffvess_pcd_ae_config.yaml
eval_flemme --config /home/wlsdzyzl/project/vcg/resources/vessel/imagecas_vessel_diff/eval_diffvess_mesh_ae_config.yaml
echo '### SkCNN'
eval_flemme --config /home/wlsdzyzl/project/vcg/resources/vessel/imagecas_vessel_diff/eval_skcnn_pcd_ae_config.yaml
eval_flemme --config /home/wlsdzyzl/project/vcg/resources/vessel/imagecas_vessel_diff/eval_skcnn_mesh_ae_config.yaml
echo '### SkCNN with SKC'
eval_flemme --config /home/wlsdzyzl/project/vcg/resources/vessel/imagecas_vessel_diff/eval_skcnn_pcd_ae_config_with_skc.yaml
eval_flemme --config /home/wlsdzyzl/project/vcg/resources/vessel/imagecas_vessel_diff/eval_skcnn_mesh_ae_config_with_skc.yaml
echo '### SkSpCNN'
eval_flemme --config /home/wlsdzyzl/project/vcg/resources/vessel/imagecas_vessel_diff/eval_skspcnn_pcd_ae_config.yaml
eval_flemme --config /home/wlsdzyzl/project/vcg/resources/vessel/imagecas_vessel_diff/eval_skspcnn_mesh_ae_config.yaml
echo '## Gen'
echo '### VessDiff'
eval_flemme --config /home/wlsdzyzl/project/vcg/resources/vessel/imagecas_vessel_diff/eval_diffvess_pcd_gen_config.yaml
eval_flemme --config /home/wlsdzyzl/project/vcg/resources/vessel/imagecas_vessel_diff/eval_diffvess_mesh_gen_config.yaml
echo '### SkCNN'
eval_flemme --config /home/wlsdzyzl/project/vcg/resources/vessel/imagecas_vessel_diff/eval_skcnn_pcd_gen_config.yaml
eval_flemme --config /home/wlsdzyzl/project/vcg/resources/vessel/imagecas_vessel_diff/eval_skcnn_mesh_gen_config.yaml
echo '### SkCNN with SKC'
eval_flemme --config /home/wlsdzyzl/project/vcg/resources/vessel/imagecas_vessel_diff/eval_skcnn_pcd_gen_config_with_skc.yaml
eval_flemme --config /home/wlsdzyzl/project/vcg/resources/vessel/imagecas_vessel_diff/eval_skcnn_mesh_gen_config_with_skc.yaml
echo '### SkSpCNN'
eval_flemme --config /home/wlsdzyzl/project/vcg/resources/vessel/imagecas_vessel_diff/eval_skspcnn_pcd_gen_config.yaml
eval_flemme --config /home/wlsdzyzl/project/vcg/resources/vessel/imagecas_vessel_diff/eval_skspcnn_mesh_gen_config.yaml