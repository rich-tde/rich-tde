#!/bin/bash

#Quick helper script that compiles and submit MPI test
#Used for quickly testing out different configs

#SBATCH --partition=gpu_strw
#SBATCH --account=gpu_strw
#SBATCH --job-name=test
#SBATCH --time=0-03:00:00 # max 7 days
#SBATCH --output=/home/hey4/RICH/jobs/logs/%j_%x.out   # %x: Job name, %j: Job id
#SBATCH --nodes=1
#SBATCH --ntasks=24	# max 48 per node for gpu_strw
#SBATCH --cpus-per-task=1
#SBATCH --mem=61G	# max 61G per node for gpu_strw (this is per node request)
#SBATCH --mail-user="yujiehe@strw.leidenuniv.nl"
#SBATCH --mail-type="ALL"

echo "====== Date: $(date) ======"
echo "====== Batch script: ======"
cat "$0"
echo "==========================="
echo ""

# Load environment
module purge                        # start with a clean environment
module restore new_rich_build          # load my saved configuration for rich

# Build
cd /home/hey4/RICH-fwrk

rm -r build	# remove previous build

./config.py --problem=SedovDissipation --mpi

make -j

cd ./build

# Run
srun /home/hey4/RICH-fwrk/build/rich

