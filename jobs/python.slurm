#!/bin/bash
#SBATCH --partition=gpu_strw
#SBATCH --account=gpu_strw
#SBATCH --job-name=plot_test
#SBATCH --time=1-00:00:00 # max 7 days
#SBATCH --output=logs/%j_%x.out   # %x: Job name, %j: Job id
#SBATCH --error=logs/%j_%x.err    
#SBATCH --nodes=1   # max 2 nodes for gpu_strw
#SBATCH --ntasks=1	# max 24 per node for gpu_strw
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G	# max 61G per node for gpu_strw
#SBATCH --mail-user="yujiehe@strw.leidenuniv.nl"
#SBATCH --mail-type="ALL"

conda init
conda activate richanalysis

# Run sim
srun python /home/hey4/rich_tde/jobs/plot_slice_n_projection.py
