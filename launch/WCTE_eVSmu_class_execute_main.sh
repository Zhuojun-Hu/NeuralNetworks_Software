
#!/bin/bash

# -- Settings --
base_folder_path=/sps/t2k/mferey/CAVERNS/NeuralNetworks_Software
#spe_folder_name=reprod_HK
config_path=/sps/t2k/mferey/CAVERNS/NeuralNetworks_Software/config/main
config_name=WCTE_uni_iso_pecut_towallcut_eVSmu_main_restore_state.yaml

gpu_list='[]' # Quotes here are important for correct parsing from bash due to the comma (if multiple gpus)
hydra_searchpath=/sps/t2k/mferey/CAVERNS/NeuralNetworks_Software/config/

log_file=$base_folder_path"/logs/WCTE_uni_iso_eVSmu_pecut_towallcut/WCTE_uni_iso_eVSmu_pecut_towallcut_restore_state_%j.log"

mkdir -p $base_folder_path/logs/WCTE_uni_iso_eVSmu_pecut_towallcut

#e_20k_energy 
#e_pi0_20keach_100_1kMeV_X_t_q_Edges_hits 
#e_mu_20keach_100_1kMeV_X_t_q_Edges_hits # or debug_gnn_reg
#nuprism_e_mu_40keach_100_1kMeV_X_tq_Edges_xyz_K10



# -- Executed code -- # (No need to change anything below)
cd $base_folder_path
export HYDRA_FULL_ERROR=1


python \
    main.py \
    --config-path=$config_path\
    --config-name=$config_name \
    hydra.searchpath=[$hydra_searchpath] \
    'gpu_list=[]' > $log_file 2>&1 &
