#!/bin/bash

# --- Initialize variables ---

execute_folder_path="/sps/t2k/mferey/CAVERNS/NeuralNetworks_Software/sweeps/sweep_eVsmu_class_WCTE_uni_iso_pecut_v2"

config_path="$execute_folder_path/sweep_config_WCTE_uni_iso_eVSmu_100k_pecut_v2.yaml"
SWEEP_ID=${SWEEP_ID:-""} # If not provided, it will be created. If provided, go on with the existing sweep

ENTITY=mathieu-ferey-laboratoire-leprince-ringuet
PROJECT=WCTE_eVSmu_class_WCTE_uni_iso
COUNT=1 # Number of runs to execute by each agent
NAGENT=9 # Number of agents to launch in parallel in sbatches

sweep_name="WCTE_uni_iso_eVSmu_sweep_pecut_v2" # job log name


# --- Executed code ---- #

# Add wandb in $PATH
source /sps/t2k/mferey/miniconda3/bin/activate caverns

mkdir -p $execute_folder_path
cd $execute_folder_path


# If there is no sweep id provided we need to create one
# So a config_path argument is necessary
if [ -z "$SWEEP_ID" ]; then

    echo "No sweep id provided. Checking for a sweep_config.yaml path"

    # Check if the config path was provided
    if [[ -z "$config_path" ]]; then
        echo -e "\nError: Config path is required."
        exit 1
    else
        echo -e "\nConfig path is set to: $config_path"
    fi
    # Check if the config file exists
    if [[ ! -f "$config_path" ]]; then
        echo -e "\nError: Config file not found."
        exit 1
    fi

    # Now create the sweep
    echo "Creating a new sweep"
    SWEEP_ID=$(wandb sweep -e $ENTITY -p $PROJECT $config_path 2>&1 | grep -oP '(?<=Creating sweep with ID: )\S+')
    echo "Sweep id created: $SWEEP_ID"
else
    echo "Sweep id provided: $SWEEP_ID"
fi

# run the agents

export SWEEP_ID
export ENTITY
export PROJECT
export COUNT

n_existing_jobs=$(find $execute_folder_path/logs -type f | wc -l) # find the number of already executed jobs not to overwrite logs
echo "number of existing logs:$n_existing_jobs"


(
for i in $(seq 1 $NAGENT); do
    current_job=$((i+n_existing_jobs))
    echo "Current job number: $current_job"
    job_name=$current_job"_"$sweep_name
    echo "Launching agent $job_name"
    sbatch --job-name=$job_name --output="$execute_folder_path/logs/$job_name.log" --time=05:00:00 --mem=50G /sps/t2k/mferey/CAVERNS/NeuralNetworks_Software/launch/wandb_sweep/run_agent_gpu.sh
done
)
