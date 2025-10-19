#!/bin/bash
#SBATCH --partition=gpu_strw
#SBATCH --account=gpu_strw
#SBATCH --job-name=test_job
#SBATCH --time=0-00:10:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=3
#SBATCH --mem=10G
#SBATCH --mail-user="yujiehe@strw.leidenuniv.nl"
#SBATCH --mail-type="ALL"

module load ALICE/default
module load OpenMPI/4.0.5-GCC-9.3.0

echo "#### Test started"

# return the name of the node
echo "## Which node is this: $HOSTNAME"

# check the number of cores (ntasks*cpus-per-task)
echo "How many cores do I have access to: ${SLURM_CPUS_ON_NODE}"

# Just to check that the AMD software stack is loaded
echo "Am I loading the from the right module path"
echo ${MODULEPATH%%:*}

# get the current working directory
CWD=$(pwd)

echo "## Where am I: ${CWD}"

# check out the nodes local scratch
echo "## My local scratch space on the node is: ${SCRATCH}"
cd $SCRATCH

echo "## Let us go there: $(pwd)"

# In case the file has already been compiled
# and stored in $CWD, the following six lines
# are not necessary  
echo "## Let us copy the C script to it"
cp $CWD/omp_hello.c $SCRATCH/  
echo "## Is the file there?"
ls -la omp_hello.c
echo "## Now we compile it on the node"
gcc -o omp_hello_amd -fopenmp omp_hello.c

# In case the file is already compiled
# the next four lines would copy it
# and check that it is there:
#echo "## Let us copy the compiled C programme to it"
#cp $CWD/omp_hello_amd $SCRATCH/
#echo "## Is the file there?"
#ls -la omp_hello_amd

echo "## Let us run it"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASKS
srun ./omp_hello_amd

# Copy those files back to shared scratch or home
# that should be kept for later.
# Here, it is just the compiled C programme.
# It does not need to be copied back of course
# if it came from shared scratch or home.
echo "## Saving files that should be saved."
cp $SCRATCH/omp_hello_amd $CWD/

echo "## Now that this is done, I want to go home"
cd $CWD
echo "## Good to be back $(pwd)"

echo "#### Test finished"
