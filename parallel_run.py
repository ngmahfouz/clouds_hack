import argparse
import yaml
import pdb
import os
import numpy as np
from textwrap import dedent

parser = argparse.ArgumentParser()

parser.add_argument("-c", "--config_file", type=str, help="YAML configuration file", default="default.yaml")
parser.add_argument("-d", "--default_config_file", type=str, help="YAML configuration file for the default training", default="default_training_config.yaml")

args = parser.parse_args()


# Open the "outer" configuration file 
with open(args.config_file) as f:
    yaml_config = yaml.load(f, Loader=yaml.FullLoader)
    # pdb.set_trace()
    sbatch_args = yaml_config['runs'][0]['sbatch']
    experiments_args = yaml_config['experiment']
    data_args = yaml_config['runs'][0]['config']['data']

# Open the default "inner" configuration file (containing default parameters for the model itself, its training and its validation)
# These default values will be replaced by the ones from the "outer" configuration file (using sampling for example)
with open(args.default_config_file) as f:
    default_training_yaml_config = yaml.load(f, Loader=yaml.FullLoader)
    # pdb.set_trace()
    sbatch_args = yaml_config['runs'][0]['sbatch']
    experiments_args = yaml_config['experiment']
    data_args = yaml_config['runs'][0]['config']['data']


#Go through each "module" (can be model or train) parameters. Sample a value of the parameter if necessary and replace the default "inner" config file value with this one
for i in range(experiments_args['repeat']):
    current_training_config = default_training_yaml_config #defaut config
    tunable_modules = yaml_config['runs'][0]['config'] # possible values for the hyper-parameters
    for tunable_module_name in tunable_modules:
        tunable_module = tunable_modules[tunable_module_name]
        current_run_args = default_training_yaml_config[tunable_module_name] # by default, this run takes the default config
        for hyper_parameter in tunable_module:
            current_run_args[hyper_parameter] = tunable_module[hyper_parameter]
            sample_method = isinstance(tunable_module[hyper_parameter], dict) and tunable_module[hyper_parameter].get("sample", None)
            if sample_method is not None:
                if sample_method == "list":
                    current_run_args[hyper_parameter] = np.random.choice(tunable_module[hyper_parameter]["from"]).item()
        
        current_training_config[tunable_module_name] = current_run_args

    current_run_path = f"{experiments_args['exp_dir']}/{experiments_args['name']}/run_{i}"
    

    # Define the slurm file
    slurm_file_template =  dedent(
    f"""\
    #!/bin/bash
    #SBATCH --account=rrg-bengioy-ad            # Yoshua pays for your job
    #SBATCH --cpus-per-task={sbatch_args['cpu']}                # Ask for 6 CPUs
    #SBATCH --gres={sbatch_args['gpu']}                     # Ask for 1 GPU
    #SBATCH --mem={sbatch_args['mem']}                        # Ask for 32 GB of RAM
    #SBATCH --time={sbatch_args['runtime']}                   # The job will run for 3 hours
    #SBATCH -o {current_run_path}/slurm-%j.out  # Write the log in $SCRATCH
    
    mkdir -p {current_run_path}
    mkdir -p $SLURM_TMPDIR/.cache/torch/checkpoints/
    cp $HOME/.cache/torch/checkpoints/*.pth $SLURM_TMPDIR/.cache/torch/checkpoints/
    module load singularity/3.5
    cd $HOME/clouds_hack/
    singularity exec --nv --bind $SLURM_TMPDIR,{data_args['path']},{current_run_path},/etc/pki/tls/certs/,/etc/pki/ca-trust/extracted/pem/ {experiments_args['img_path']} \
    python3 data_scratch.py -o {current_run_path} -c {current_run_path}/training_config.yaml
    cp -R $SLURM_TMPDIR/* {current_run_path}""")

    os.makedirs(current_run_path, exist_ok=True)
    #Write the slurm file
    with open(f"{current_run_path}/slurm_job.slrm", "w") as f:
        f.write(slurm_file_template)
    
    # Write the training configuration file
    with open(f"{current_run_path}/training_config.yaml", "w") as f:
        yaml.dump(current_training_config, f)

    os.system(f"sbatch {current_run_path}/slurm_job.slrm")