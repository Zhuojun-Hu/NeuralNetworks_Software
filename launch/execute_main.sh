
#!/bin/bash

# -- Settings --
base_folder_path=/sps/t2k/eleblevec/updated_watchmal/WatChMaL
spe_folder_name=reprod_HK
config_name=e_mu_50keach_100_1kMeV_X_tq_Edges_txyz_K10 

gpu_list='gpu_list=[0]' # Quotes here are important for correct parsing from bash due to the comma (if multiple gpus)
hydra_searchpath=/sps/t2k/eleblevec/updated_watchmal/WatChMaL/config/

#e_20k_energy 
#e_pi0_20keach_100_1kMeV_X_t_q_Edges_hits 
#e_mu_20keach_100_1kMeV_X_t_q_Edges_hits # or debug_gnn_reg
#nuprism_e_mu_40keach_100_1kMeV_X_tq_Edges_xyz_K10



# -- Executed code -- # (No need to change anything below)
cd $base_folder_path
export HYDRA_FULL_ERROR=1


python \
    main.py \
    --config-path=/sps/t2k/eleblevec/updated_watchmal/WatChMaL/config/main/${spe_folder_name} \
    --config-name=$config_name \
    hydra.searchpath=[$hydra_searchpath] \
    $gpu_list




