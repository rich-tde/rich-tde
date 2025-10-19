#!/bin/bash
#SBATCH --partition=gpu_strw
#SBATCH --account=gpu_strw
#SBATCH --job-name=tde_np1e6
#SBATCH --time=1-00:00:00 # max 7 days
#SBATCH --output=logs/%j_%x.out   # %x: Job name, %j: Job id
#SBATCH --error=logs/%j_%x.err    
#SBATCH --nodes=1
#SBATCH --ntasks=24	# max 24 per node for gpu_strw
#SBATCH --cpus-per-task=1
#SBATCH --mem=60G	# max 61G per node for gpu_strw
#SBATCH --mail-user="yujiehe@strw.leidenuniv.nl"
#SBATCH --mail-type="ALL"

module purge                        # start with a clean environment
module restore rich_gcc12           # load my saved configuration for rich

# Run sim
srun /home/hey4/build/gnuReleaseMPI/rich
