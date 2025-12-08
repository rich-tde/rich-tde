#!/bin/bash
#SBATCH --partition=gpu_strw
#SBATCH --account=gpu_strw
#SBATCH --job-name=NewRICH
#SBATCH --time=0-10:00:00 # max 7 days
#SBATCH --output=/home/hey4/RICH/jobs/logs/%j_%x.out   # %x: Job name, %j: Job id
#SBATCH --nodes=1
#SBATCH --ntasks=24	# max 24 per node for gpu_strw
#SBATCH --cpus-per-task=1
#SBATCH --mem=61G	# max 61G per node for gpu_strw
#SBATCH --mail-user="yujiehe@strw.leidenuniv.nl"
#SBATCH --mail-type="ALL"

echo "====== Date: $(date) ======"
echo "====== Batch script: ======"
cat "$0"
echo "=========================="
echo ""

# Load environment
module purge                        # start with a clean environment
module restore rich_gcc13           # load my saved configuration for rich

# Run
srun /home/hey4/RICH-yihan/build/gnuReleaseMPI/rich
