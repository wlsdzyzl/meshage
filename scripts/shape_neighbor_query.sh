# #!/bin/bash
# echo bladder
# python shape_neighbor_query.py --query_dir /data/guoqingzhang/vcg-for-figure/gen/bladder --target_dir /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/bladder/ --k 5 --fixed_points 2560 --output_file ./bladder_diffpcd.out 
# python shape_neighbor_query.py --query_dir /data/guoqingzhang/vcg-for-figure/gen/bladder --target_dir /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/bladder/ --k 5 --fixed_points 2560 --output_file ./bladder_edmpcd.out 
# python shape_neighbor_query.py --query_dir /data/guoqingzhang/vcg-for-figure/gen/bladder --target_dir /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/bladder/ --k 5 --fixed_points 2560 --output_file ./bladder_gem3d.out --normalize
# python shape_neighbor_query.py --query_dir /data/guoqingzhang/vcg-for-figure/gen/bladder --target_dir /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/bladder/ --k 5 --fixed_points 2560 --output_file ./bladder_pvd.out 
python shape_neighbor_query.py --query_dir /data/guoqingzhang/vcg-for-figure/gen/bladder --target_dir /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN/bladder/ --k 5 --fixed_points 2560 --output_file ./bladder_ours_wo_la.out 

# echo brain
# python shape_neighbor_query.py --query_dir /data/guoqingzhang/vcg-for-figure/gen/brain --target_dir /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/brain/ --k 5 --fixed_points 2560 --output_file ./brain_diffpcd.out 
# python shape_neighbor_query.py --query_dir /data/guoqingzhang/vcg-for-figure/gen/brain --target_dir /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/brain/ --k 5 --fixed_points 2560 --output_file ./brain_edmpcd.out 
# python shape_neighbor_query.py --query_dir /data/guoqingzhang/vcg-for-figure/gen/brain --target_dir /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/brain/ --k 5 --fixed_points 2560 --output_file ./brain_gem3d.out --normalize
# python shape_neighbor_query.py --query_dir /data/guoqingzhang/vcg-for-figure/gen/brain --target_dir /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/brain/ --k 5 --fixed_points 2560 --output_file ./brain_pvd.out 
python shape_neighbor_query.py --query_dir /data/guoqingzhang/vcg-for-figure/gen/brain --target_dir /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN/brain/ --k 5 --fixed_points 2560 --output_file ./brain_ours_wo_la.out 

# echo colon
# python shape_neighbor_query.py --query_dir /data/guoqingzhang/vcg-for-figure/gen/colon --target_dir /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/colon/ --k 5 --fixed_points 2560 --output_file ./colon_diffpcd.out 
# python shape_neighbor_query.py --query_dir /data/guoqingzhang/vcg-for-figure/gen/colon --target_dir /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/colon/ --k 5 --fixed_points 2560 --output_file ./colon_edmpcd.out 
# python shape_neighbor_query.py --query_dir /data/guoqingzhang/vcg-for-figure/gen/colon --target_dir /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/colon/ --k 5 --fixed_points 2560 --output_file ./colon_gem3d.out --normalize
# python shape_neighbor_query.py --query_dir /data/guoqingzhang/vcg-for-figure/gen/colon --target_dir /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/colon/ --k 5 --fixed_points 2560 --output_file ./colon_pvd.out 
python shape_neighbor_query.py --query_dir /data/guoqingzhang/vcg-for-figure/gen/colon --target_dir /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN/colon/ --k 5 --fixed_points 2560 --output_file ./colon_ours_wo_la.out 

# echo coronary_artery_left_d
# python shape_neighbor_query.py --query_dir /data/guoqingzhang/vcg-for-figure/gen/coronary_artery_left_d --target_dir /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/coronary_artery_left_d/ --k 5 --fixed_points 2560 --output_file ./coronary_artery_left_d_diffpcd.out 
# python shape_neighbor_query.py --query_dir /data/guoqingzhang/vcg-for-figure/gen/coronary_artery_left_d --target_dir /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/coronary_artery_left_d/ --k 5 --fixed_points 2560 --output_file ./coronary_artery_left_d_edmpcd.out 
# python shape_neighbor_query.py --query_dir /data/guoqingzhang/vcg-for-figure/gen/coronary_artery_left_d --target_dir /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/coronary_artery_left_d/ --k 5 --fixed_points 2560 --output_file ./coronary_artery_left_d_gem3d.out --normalize
# python shape_neighbor_query.py --query_dir /data/guoqingzhang/vcg-for-figure/gen/coronary_artery_left_d --target_dir /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/coronary_artery_left_d/ --k 5 --fixed_points 2560 --output_file ./coronary_artery_left_d_pvd.out 
python shape_neighbor_query.py --query_dir /data/guoqingzhang/vcg-for-figure/gen/coronary_artery_left_d --target_dir /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN/coronary_artery_left_d/ --k 5 --fixed_points 2560 --output_file ./coronary_artery_left_d_ours_wo_la.out 

# echo coronary_artery_right_d
# python shape_neighbor_query.py --query_dir /data/guoqingzhang/vcg-for-figure/gen/coronary_artery_right_d --target_dir /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/coronary_artery_right_d/ --k 5 --fixed_points 2560 --output_file ./coronary_artery_right_d_diffpcd.out 
# python shape_neighbor_query.py --query_dir /data/guoqingzhang/vcg-for-figure/gen/coronary_artery_right_d --target_dir /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/coronary_artery_right_d/ --k 5 --fixed_points 2560 --output_file ./coronary_artery_right_d_edmpcd.out 
# python shape_neighbor_query.py --query_dir /data/guoqingzhang/vcg-for-figure/gen/coronary_artery_right_d --target_dir /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/coronary_artery_right_d/ --k 5 --fixed_points 2560 --output_file ./coronary_artery_right_d_gem3d.out --normalize
# python shape_neighbor_query.py --query_dir /data/guoqingzhang/vcg-for-figure/gen/coronary_artery_right_d --target_dir /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/coronary_artery_right_d/ --k 5 --fixed_points 2560 --output_file ./coronary_artery_right_d_pvd.out 
python shape_neighbor_query.py --query_dir /data/guoqingzhang/vcg-for-figure/gen/coronary_artery_right_d --target_dir /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN/coronary_artery_right_d/ --k 5 --fixed_points 2560 --output_file ./coronary_artery_right_d_ours_wo_la.out 

# echo duodenum
# python shape_neighbor_query.py --query_dir /data/guoqingzhang/vcg-for-figure/gen/duodenum --target_dir /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/duodenum/ --k 5 --fixed_points 2560 --output_file ./duodenum_diffpcd.out 
# python shape_neighbor_query.py --query_dir /data/guoqingzhang/vcg-for-figure/gen/duodenum --target_dir /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/duodenum/ --k 5 --fixed_points 2560 --output_file ./duodenum_edmpcd.out 
# python shape_neighbor_query.py --query_dir /data/guoqingzhang/vcg-for-figure/gen/duodenum --target_dir /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/duodenum/ --k 5 --fixed_points 2560 --output_file ./duodenum_gem3d.out --normalize
# python shape_neighbor_query.py --query_dir /data/guoqingzhang/vcg-for-figure/gen/duodenum --target_dir /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/duodenum/ --k 5 --fixed_points 2560 --output_file ./duodenum_pvd.out 
python shape_neighbor_query.py --query_dir /data/guoqingzhang/vcg-for-figure/gen/duodenum --target_dir /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN/duodenum/ --k 5 --fixed_points 2560 --output_file ./duodenum_ours_wo_la.out 

# echo gallbladder
# python shape_neighbor_query.py --query_dir /data/guoqingzhang/vcg-for-figure/gen/gallbladder --target_dir /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/gallbladder/ --k 5 --fixed_points 2560 --output_file ./gallbladder_diffpcd.out 
# python shape_neighbor_query.py --query_dir /data/guoqingzhang/vcg-for-figure/gen/gallbladder --target_dir /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/gallbladder/ --k 5 --fixed_points 2560 --output_file ./gallbladder_edmpcd.out 
# python shape_neighbor_query.py --query_dir /data/guoqingzhang/vcg-for-figure/gen/gallbladder --target_dir /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/gallbladder/ --k 5 --fixed_points 2560 --output_file ./gallbladder_gem3d.out --normalize
# python shape_neighbor_query.py --query_dir /data/guoqingzhang/vcg-for-figure/gen/gallbladder --target_dir /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/gallbladder/ --k 5 --fixed_points 2560 --output_file ./gallbladder_pvd.out 
python shape_neighbor_query.py --query_dir /data/guoqingzhang/vcg-for-figure/gen/gallbladder --target_dir /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN/gallbladder/ --k 5 --fixed_points 2560 --output_file ./gallbladder_ours_wo_la.out 

# echo liver
# python shape_neighbor_query.py --query_dir /data/guoqingzhang/vcg-for-figure/gen/liver --target_dir /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/liver/ --k 5 --fixed_points 2560 --output_file ./liver_diffpcd.out 
# python shape_neighbor_query.py --query_dir /data/guoqingzhang/vcg-for-figure/gen/liver --target_dir /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/liver/ --k 5 --fixed_points 2560 --output_file ./liver_edmpcd.out 
# python shape_neighbor_query.py --query_dir /data/guoqingzhang/vcg-for-figure/gen/liver --target_dir /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/liver/ --k 5 --fixed_points 2560 --output_file ./liver_gem3d.out --normalize
# python shape_neighbor_query.py --query_dir /data/guoqingzhang/vcg-for-figure/gen/liver --target_dir /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/liver/ --k 5 --fixed_points 2560 --output_file ./liver_pvd.out 
python shape_neighbor_query.py --query_dir /data/guoqingzhang/vcg-for-figure/gen/liver --target_dir /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN/liver/ --k 5 --fixed_points 2560 --output_file ./liver_ours_wo_la.out 

# echo pancreas
# python shape_neighbor_query.py --query_dir /data/guoqingzhang/vcg-for-figure/gen/pancreas --target_dir /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/pancreas/ --k 5 --fixed_points 2560 --output_file ./pancreas_diffpcd.out 
# python shape_neighbor_query.py --query_dir /data/guoqingzhang/vcg-for-figure/gen/pancreas --target_dir /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/pancreas/ --k 5 --fixed_points 2560 --output_file ./pancreas_edmpcd.out 
# python shape_neighbor_query.py --query_dir /data/guoqingzhang/vcg-for-figure/gen/pancreas --target_dir /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/pancreas/ --k 5 --fixed_points 2560 --output_file ./pancreas_gem3d.out --normalize
# python shape_neighbor_query.py --query_dir /data/guoqingzhang/vcg-for-figure/gen/pancreas --target_dir /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/pancreas/ --k 5 --fixed_points 2560 --output_file ./pancreas_pvd.out 
python shape_neighbor_query.py --query_dir /data/guoqingzhang/vcg-for-figure/gen/pancreas --target_dir /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN/pancreas/ --k 5 --fixed_points 2560 --output_file ./pancreas_ours_wo_la.out 

# echo spleen
# python shape_neighbor_query.py --query_dir /data/guoqingzhang/vcg-for-figure/gen/spleen --target_dir /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/spleen/ --k 5 --fixed_points 2560 --output_file ./spleen_diffpcd.out 
# python shape_neighbor_query.py --query_dir /data/guoqingzhang/vcg-for-figure/gen/spleen --target_dir /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/spleen/ --k 5 --fixed_points 2560 --output_file ./spleen_edmpcd.out 
# python shape_neighbor_query.py --query_dir /data/guoqingzhang/vcg-for-figure/gen/spleen --target_dir /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/spleen/ --k 5 --fixed_points 2560 --output_file ./spleen_gem3d.out --normalize
# python shape_neighbor_query.py --query_dir /data/guoqingzhang/vcg-for-figure/gen/spleen --target_dir /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/spleen/ --k 5 --fixed_points 2560 --output_file ./spleen_pvd.out 
python shape_neighbor_query.py --query_dir /data/guoqingzhang/vcg-for-figure/gen/spleen --target_dir /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN/spleen/ --k 5 --fixed_points 2560 --output_file ./spleen_ours_wo_la.out 

# echo stomach
# python shape_neighbor_query.py --query_dir /data/guoqingzhang/vcg-for-figure/gen/stomach --target_dir /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/stomach/ --k 5 --fixed_points 2560 --output_file ./stomach_diffpcd.out 
# python shape_neighbor_query.py --query_dir /data/guoqingzhang/vcg-for-figure/gen/stomach --target_dir /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/stomach/ --k 5 --fixed_points 2560 --output_file ./stomach_edmpcd.out 
# python shape_neighbor_query.py --query_dir /data/guoqingzhang/vcg-for-figure/gen/stomach --target_dir /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/stomach/ --k 5 --fixed_points 2560 --output_file ./stomach_gem3d.out --normalize
# python shape_neighbor_query.py --query_dir /data/guoqingzhang/vcg-for-figure/gen/stomach --target_dir /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/stomach/ --k 5 --fixed_points 2560 --output_file ./stomach_pvd.out 
python shape_neighbor_query.py --query_dir /data/guoqingzhang/vcg-for-figure/gen/stomach --target_dir /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN/stomach/ --k 5 --fixed_points 2560 --output_file ./stomach_ours_wo_la.out 


# echo trachea
# python shape_neighbor_query.py --query_dir /data/guoqingzhang/vcg-for-figure/gen/trachea --target_dir /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/trachea/ --k 5 --fixed_points 2560 --output_file ./trachea_diffpcd.out 
# python shape_neighbor_query.py --query_dir /data/guoqingzhang/vcg-for-figure/gen/trachea --target_dir /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/trachea/ --k 5 --fixed_points 2560 --output_file ./trachea_edmpcd.out 
# python shape_neighbor_query.py --query_dir /data/guoqingzhang/vcg-for-figure/gen/trachea --target_dir /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/trachea/ --k 5 --fixed_points 2560 --output_file ./trachea_gem3d.out --normalize
# python shape_neighbor_query.py --query_dir /data/guoqingzhang/vcg-for-figure/gen/trachea --target_dir /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/trachea/ --k 5 --fixed_points 2560 --output_file ./trachea_pvd.out 
python shape_neighbor_query.py --query_dir /data/guoqingzhang/vcg-for-figure/gen/trachea --target_dir /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN/trachea/ --k 5 --fixed_points 2560 --output_file ./trachea_ours_wo_la.out 


# echo uterus
# python shape_neighbor_query.py --query_dir /data/guoqingzhang/vcg-for-figure/gen/uterus --target_dir /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/uterus/ --k 5 --fixed_points 2560 --output_file ./uterus_diffpcd.out 
# python shape_neighbor_query.py --query_dir /data/guoqingzhang/vcg-for-figure/gen/uterus --target_dir /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/uterus/ --k 5 --fixed_points 2560 --output_file ./uterus_edmpcd.out 
# python shape_neighbor_query.py --query_dir /data/guoqingzhang/vcg-for-figure/gen/uterus --target_dir /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/uterus/ --k 5 --fixed_points 2560 --output_file ./uterus_gem3d.out --normalize
# python shape_neighbor_query.py --query_dir /data/guoqingzhang/vcg-for-figure/gen/uterus --target_dir /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/uterus/ --k 5 --fixed_points 2560 --output_file ./uterus_pvd.out 
python shape_neighbor_query.py --query_dir /data/guoqingzhang/vcg-for-figure/gen/uterus --target_dir /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN/uterus/ --k 5 --fixed_points 2560 --output_file ./uterus_ours_wo_la.out 

# python shape_neighbor_query.py --query_dir /data/guoqingzhang/vcg-for-figure/gen/imagecas/ --target_dir /data/guoqingzhang/vcg-results/imageCAS_vessel_diff/gen/VessDiff_Mesh/ --k 5 --fixed_points 4096 --output_file ./imagecas_vessel_diff.out 
# python shape_neighbor_query.py --query_dir /data/guoqingzhang/vcg-for-figure/gen/imagecas/ --target_dir /data/guoqingzhang/vcg-results/imageCAS_vessel_diff/gen/LDM_EDM_SkCNN_with_SKC/ --k 5 --fixed_points 4096 --output_file ./imagecas_ours.out 
# python shape_neighbor_query.py --query_dir /data/guoqingzhang/vcg-for-figure/gen/imagecas/ --target_dir /data/guoqingzhang/vcg-results/imageCAS_vessel_diff/gen/LDM_EDM_SkCNN_with_SKC_LA/ --k 5 --fixed_points 4096 --output_file ./imagecas_ours_la.out 
# python shape_neighbor_query.py --query_dir /data/guoqingzhang/vcg-for-figure/gen/imagecas/ --target_dir /data/guoqingzhang/vcg-results/imageCAS_vessel_diff/gen/LDM_EDM_SkCNN_with_SKC_LA_OS/ --k 5 --fixed_points 4096 --output_file ./imagecas_ours_la_os.out 