#!/bin/bash

# ------------------------------------------------------- #
# This code will be executed by a sweep agent.
# this is why some source and export are re-defined here
# ------------------------------------------------------- #

# Erwan ===================================

# # ----- Agent Settings ------ #

# env_name=grant_cuda_12_1
# miniconda_path=/sps/t2k/eleblevec/miniconda3

# working_dir=/sps/t2k/eleblevec/updated_watchmal/WatChMaL

# mother_config_folder=reprod_HK
# mother_config_name=e_100k_energy 

# gpu_list='gpu_list=[0]' # Quotes here are important for correct parsing from bash due to the comma (if multiple gpus)
# hydra_searchpath=/sps/t2k/eleblevec/updated_watchmal/WatChMaL/config/


# #e_20k_energy 
# #e_pi0_20keach_100_1kMeV_X_t_q_Edges_hits 
# #e_mu_20keach_100_1kMeV_X_t_q_Edges_hits



# # ---- Executed code by the agent ----- 
# export HYDRA_FULL_ERROR=1
# source ${miniconda_path}/bin/activate $env_name

# cd $working_dir
# python \
#     main.py \
#     --config-path=/sps/t2k/eleblevec/updated_watchmal/WatChMaL/config/main/${mother_config_folder} \
#     --config-name=$mother_config_name \
#     hydra.searchpath=[$hydra_searchpath] \
#     $gpu_list

# Mathieu ===========================================================

# -- Settings --
base_folder_path=/sps/t2k/mferey/CAVERNS/NeuralNetworks_Software
#spe_folder_name=reprod_HK
config_path=/sps/t2k/mferey/CAVERNS/NeuralNetworks_Software/config/main
config_name=WCTE_eVSmu_class_main

gpu_list='gpu_list=[0]' # Quotes here are important for correct parsing from bash due to the comma (if multiple gpus)
hydra_searchpath=/sps/t2k/mferey/CAVERNS/NeuralNetworks_Software/config/
conda_env_name=caverns


# -- Executed code

cd $base_folder_path
export HYDRA_FULL_ERROR=1

echo "Sourcing conda env..."

source /sps/t2k/mferey/miniconda3/bin/activate $conda_env_name

echo "Running main.py..."

python \
    main.py \
    --config-path=$config_path\
    --config-name=$config_name \
    hydra.searchpath=[$hydra_searchpath] \
    $gpu_list