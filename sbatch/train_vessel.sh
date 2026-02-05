#!/bin/bash
#SBATCH --job-name=train_vessel_pre_ske
#SBATCH -w inspur1
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH -t 144000

# train_meshage --config /home/guoqingzhang/meshage/resources/vessel/cow_vessel_diff/train_spae_cnn_12_condition_config_dilated.yaml
# train_meshage --config /home/guoqingzhang/meshage/resources/vessel/imagecas_vessel_diff/train_spae_cnn_12_condition_config_dilated.yaml
# train_meshage --config /home/guoqingzhang/meshage/resources/vessel/cow_vessel_diff/train_skae_cnn_12_condition_config_dilated_with_os.yaml
# train_meshage --config /home/guoqingzhang/meshage/resources/vessel/imagecas_vessel_diff/train_skae_cnn_12_condition_config_dilated_with_os.yaml
# train_meshage --config /home/guoqingzhang/meshage/resources/vessel/imagecas_vessel_diff/train_skae_cnn_12_condition_config_dilated_with_skc_latent_atten.yaml
# train_meshage --config /home/guoqingzhang/meshage/resources/vessel/imagecas_vessel_diff/retrain_skae_cnn_12_condition_config_dilated_with_skc_latent_atten.yaml
# test_meshage --config /home/guoqingzhang/meshage/resources/vessel/imagecas_vessel_diff/test_skae_cnn_12_condition_config_dilated_with_skc_latent_atten.yaml
# train_meshage --config /home/guoqingzhang/meshage/resources/vessel/imagecas_vessel_diff/train_edm_condition_config_with_skc_latent_atten.yaml
# train_meshage --config /home/guoqingzhang/meshage/resources/vessel/imagecas_vessel_diff/train_skae_cnn_12_condition_config_dilated_with_skc_latent_atten_non_embedded.yaml
# train_meshage --config /home/guoqingzhang/meshage/resources/vessel/imagecas_vessel_diff/retrain_skae_cnn_12_condition_config_dilated_with_skc_latent_atten_non_embedded.yaml
# test_meshage --config /home/guoqingzhang/meshage/resources/vessel/imagecas_vessel_diff/test_skae_cnn_12_condition_config_dilated_with_skc_latent_atten_non_embedded.yaml
# train_meshage --config /home/guoqingzhang/meshage/resources/vessel/cow_vessel_diff/train_skae_cnn_12_condition_config_dilated_with_skc_latent_atten.yaml
# train_meshage --config /home/guoqingzhang/meshage/resources/vessel/cow_vessel_diff/retrain_skae_cnn_12_condition_config_dilated_with_skc_latent_atten.yaml
# test_meshage --config /home/guoqingzhang/meshage/resources/vessel/cow_vessel_diff/test_skae_cnn_12_condition_config_dilated_with_skc_latent_atten.yaml
# train_meshage --config /home/guoqingzhang/meshage/resources/vessel/cow_vessel_diff/train_edm_condition_config_with_skc_latent_atten.yaml
# train_meshage --config /home/guoqingzhang/meshage/resources/vessel/cow_vessel_diff/train_skae_cnn_12_condition_config_dilated_with_skc_latent_atten_non_embedded.yaml
# train_meshage --config /home/guoqingzhang/meshage/resources/vessel/cow_vessel_diff/retrain_skae_cnn_12_condition_config_dilated_with_skc_latent_atten_non_embedded.yaml

# train_meshage --config /home/guoqingzhang/meshage/resources/vessel/imagecas_vessel_diff/train_skae_cnn_12_condition_config_dilated_with_skc_latent_atten_pre_skel.yaml
# train_meshage --config /home/guoqingzhang/meshage/resources/vessel/cow_vessel_diff/train_skae_cnn_12_condition_config_dilated_with_skc_latent_atten_pre_skel.yaml
# test_meshage --config /home/guoqingzhang/meshage/resources/vessel/imagecas_vessel_diff/test_skae_cnn_12_condition_config_dilated_with_skc_latent_atten_pre_skel.yaml
# test_meshage --config /home/guoqingzhang/meshage/resources/vessel/cow_vessel_diff/test_skae_cnn_12_condition_config_dilated_with_skc_latent_atten_pre_skel.yaml
# test_meshage --config /home/guoqingzhang/meshage/resources/vessel/imagecas_vessel_diff/test_skae_cnn_12_condition_config_dilated_with_skc_latent_atten.yaml
# test_meshage --config /home/guoqingzhang/meshage/resources/vessel/cow_vessel_diff/test_skae_cnn_12_condition_config_dilated_with_skc_latent_atten.yaml
# train_meshage --config /home/guoqingzhang/meshage/resources/vessel/imagecas_vessel_diff/train_edm_condition_config_with_skc_latent_atten_pre_skel.yaml
# train_meshage --config /home/guoqingzhang/meshage/resources/vessel/cow_vessel_diff/train_edm_condition_config_with_skc_latent_atten_pre_skel.yaml
# test_meshage --config /home/guoqingzhang/meshage/resources/vessel/imagecas_vessel_diff/test_ldm_condition_config_with_skc_la_os.yaml
# test_meshage --config /home/guoqingzhang/meshage/resources/vessel/imagecas_vessel_diff/test_ldm_condition_config_with_skc_la.yaml
# test_meshage --config /home/guoqingzhang/meshage/resources/vessel/cow_vessel_diff/test_ldm_condition_config_with_skc_la_os.yaml
# test_meshage --config /home/guoqingzhang/meshage/resources/vessel/cow_vessel_diff/test_ldm_condition_config_with_skc_la.yaml
# test_meshage --config /home/guoqingzhang/meshage/resources/vessel/imagecas_vessel_diff/test_spae_cnn_12_condition_config_dilated.yaml
# test_meshage --config /home/guoqingzhang/meshage/resources/vessel/imagecas_vessel_diff/test_skae_cnn_12_condition_config_dilated_with_os.yaml
# test_meshage --config /home/guoqingzhang/meshage/resources/vessel/cow_vessel_diff/test_spae_cnn_12_condition_config_dilated.yaml
# test_meshage --config /home/guoqingzhang/meshage/resources/vessel/cow_vessel_diff/test_skae_cnn_12_condition_config_dilated_with_os.yaml

# train_meshage --config /home/guoqingzhang/meshage/resources/vessel/imagecas_vessel_diff/train_edm_spcnn_condition_config.yaml
# train_meshage --config /home/guoqingzhang/meshage/resources/vessel/imagecas_vessel_diff/train_edm_condition_config_with_os.yaml

# train_meshage --config /home/guoqingzhang/meshage/resources/vessel/cow_vessel_diff/train_edm_spcnn_condition_config.yaml
# train_meshage --config /home/guoqingzhang/meshage/resources/vessel/cow_vessel_diff/train_edm_condition_config_with_os.yaml

# test_meshage --config /home/guoqingzhang/meshage/resources/vessel/imagecas_vessel_diff/test_ldm_spcnn_condition_config.yaml
# test_meshage --config /home/guoqingzhang/meshage/resources/vessel/imagecas_vessel_diff/test_ldm_condition_config_with_os.yaml
# test_meshage --config /home/guoqingzhang/meshage/resources/vessel/cow_vessel_diff/test_ldm_spcnn_condition_config.yaml
# test_meshage --config /home/guoqingzhang/meshage/resources/vessel/cow_vessel_diff/test_ldm_condition_config_with_os.yaml

# train_meshage --config /home/guoqingzhang/meshage/resources/vessel/imagecas_vessel_diff/train_spae_cnn_12_condition_config_dilated_with_la.yaml
# test_meshage --config /home/guoqingzhang/meshage/resources/vessel/imagecas_vessel_diff/test_spae_cnn_12_condition_config_dilated_with_la.yaml
# train_meshage --config /home/guoqingzhang/meshage/resources/vessel/imagecas_vessel_diff/train_edm_spcnn_condition_config_with_la.yaml
# train_meshage --config /home/guoqingzhang/meshage/resources/vessel/cow_vessel_diff/train_spae_cnn_12_condition_config_dilated_with_la.yaml

# test_meshage --config /home/guoqingzhang/meshage/resources/vessel/cow_vessel_diff/test_spae_cnn_12_condition_config_dilated_with_la.yaml
# train_meshage --config /home/guoqingzhang/meshage/resources/vessel/cow_vessel_diff/train_edm_spcnn_condition_config_with_la.yaml
# test_meshage --config /home/guoqingzhang/meshage/resources/vessel/cow_vessel_diff/test_ldm_spcnn_condition_config_with_la.yaml
test_meshage --config /home/guoqingzhang/meshage/resources/vessel/imagecas_vessel_diff/test_ldm_spcnn_condition_config_with_la.yaml