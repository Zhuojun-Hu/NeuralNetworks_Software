#!/bin/bash

# ------------------------------------------------------- #
# This code will be executed by a sweep agent.
# this is why some source and export are re-defined here
# ------------------------------------------------------- #


# ----- Agent Settings ------ #

env_name=grant_cuda_12_1
miniconda_path=/sps/t2k/eleblevec/miniconda3

working_dir=/sps/t2k/eleblevec/updated_watchmal/WatChMaL

mother_config_folder=reprod_HK
mother_config_name=e_100k_energy 

gpu_list='gpu_list=[0]' # Quotes here are important for correct parsing from bash due to the comma (if multiple gpus)
hydra_searchpath=/sps/t2k/eleblevec/updated_watchmal/WatChMaL/config/


#e_20k_energy 
#e_pi0_20keach_100_1kMeV_X_t_q_Edges_hits 
#e_mu_20keach_100_1kMeV_X_t_q_Edges_hits



# ---- Executed code by the agent ----- 
export HYDRA_FULL_ERROR=1
source ${miniconda_path}/bin/activate $env_name

cd $working_dir
python \
    main.py \
    --config-path=/sps/t2k/eleblevec/updated_watchmal/WatChMaL/config/main/${mother_config_folder} \
    --config-name=$mother_config_name \
    hydra.searchpath=[$hydra_searchpath] \
    $gpu_list