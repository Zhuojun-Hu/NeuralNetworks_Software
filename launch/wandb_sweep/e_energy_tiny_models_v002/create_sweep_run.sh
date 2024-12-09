#!/bin/bash

# --- Initialize variables ---

folder_where_to_execute_wandb_sweep="/sps/t2k/eleblevec/updated_watchmal/lauch/wandb_scripts/e_energy_tiny_models_v002"

config_path=""
SWEEP_ID=${SWEEP_ID:-"eie70i3l"} # "" works for no sweep

ENTITY=erwanlbv
PROJECT=reprod_HK
COUNT=100


# --- Executed code ---- #

# Add wandb in $PATH
source $work/miniconda3/bin/activate grant_cuda_12_1

mkdir -p $folder_where_to_execute_wandb_sweep
cd $folder_where_to_execute_wandb_sweep


# If there is no sweep id provided we need to create one
# So a config_path argument is necessary
if [ -z "$SWEEP_ID" ]; then

    echo "No sweep id provided. Checking for a sweep_config.yaml path"
    # Loop through arguments and process them   
    case $1 in
        -c|--config)  # Match -c or --config
            config_path="$2"  # Assign the next parameter as the value
            ;;
        *)  # Default case if no more known options
            echo "Unknown option: $1"
            exit 1
            ;;
    esac


    # Check if the config path was provided
    if [[ -z "$config_path" ]]; then
        echo -e "\nError: Config path is required."
        exit 1
    else
        echo -e "\nConfig path is set to: $config_path"
    fi

    # Now create the sweep
    echo "Creating a new sweep"
    wandb sweep \
    -e $ENTITY \
    -p $PROJECT \
    $config_path
    
    echo "Provide the sweep id :"
    read SWEEP_ID
fi
     
# Check if everything went well  
if [ -z "$SWEEP_ID" ]; then
    echo "Failed to recognize the sweep id : ${SWEEP_ID}, exiting the code." 
    exit 1
fi 

# To be moved into a "run agent"
# Run an agent managed by the sweep
echo "Sweep ID: $SWEEP_ID"
wandb agent \
    -e $ENTITY \
    -p $PROJECT \
    --count $COUNT \
    $SWEEP_ID


