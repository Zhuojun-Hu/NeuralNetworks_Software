#! /bin/bash


# SLURM options:

#SBATCH --account=t2k
#SBATCH --partition=gpu                              # Partition choice
#SBATCH --ntasks=1                                   # Maximum number of parallel processes
#SBATCH --cpus-per-task=5                            # Number of threads per process

### Need to be modified
#SBATCH --job-name=add_nhits_totcharge_WCTE_eVSmu_class_gpu                         # Job name
#SBATCH --output=/sps/t2k/mferey/CAVERNS/NeuralNetworks_Software/logs/WCTE_uni_iso_eVSmu_mean_pooling/WCTE_uni_iso_eVSmu_pecut_mean_pooling_add_feats_%j.log             # Standard output and error log

#SBATCH --mem=30G                                     # Amount of memory required
#SBATCH --time=00:30:00                               # Maximum time limit for the job            eff_signal, eff_background, ACC, bin_centers, bin_edges, physvar = self.compute_efficiency_vs_physvar(physvar_name, nbins, errorbars)
#SBATCH --gres=gpu:v100:2


# -- Settings --
base_folder_path=/sps/t2k/mferey/CAVERNS/NeuralNetworks_Software
#spe_folder_name=reprod_HK
config_path=/sps/t2k/mferey/CAVERNS/NeuralNetworks_Software/config/main
config_name=WCTE_uni_iso_pecut_eVSmu_mean_pooling_add_feat_main.yaml

gpu_list='gpu_list=[0,1]' # Quotes here are important for correct parsing from bash due to the comma (if multiple gpus)
master_port='master_port=12352' # Port for distributed training
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
    $gpu_list \
    $master_port



    





