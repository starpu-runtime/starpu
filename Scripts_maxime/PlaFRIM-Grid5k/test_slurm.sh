#!/usr/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --time=02:00:00
#SBATCH --job-name=test
#SBATCH -p routage
#SBATCH -C bora
#SBATCH --output=resultat.txt

#
# Documentation:
# https://slurm.schedmd.com/documentation.html
# Slurm environment variables
# http://www.glue.umd.edu/hpcc/help/slurmenv.html
# https://www.plafrim.fr/faq-en/

#SBATCH -o %x-%j.out   # output and error file name
                       #  (%j expands to jobID)
                       #  (%x expands to job name)
                       # "filename pattern" in https://slurm.schedmd.com/sbatch.html

# Uncomment these two lines if you want to be notified by emil
# Be sure to fill in your email address
# #SBATCH --mail-user=<Your email address>
# #SBATCH --mail-type=begin
# set -e


module load compiler/gcc
module add compiler/intel
module load mpi/openmpi/4.0.2-testing
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/sbaden/bin/lib


echo " === Environment:"
printenv 

# cat /proc/cpuinfo

echo ""
echo "==========================================================="
echo ""
lscpu
echo ""
echo "Number of Nodes: " ${SLURM_NNODES}
echo "Node List: " $SLURM_NODELIST
export CORES_PER_NODE=${CORES_PER_NODE:=${SLURM_TASKS_PER_NODE%%(*}}
export NTASKS=${NTASKS:=${SLURM_NTASKS}}
# echo "Number of cores/node: " ${SLURM_CPUS_ON_NODE}
echo "Detected CORES_PER_NODE=${CORES_PER_NODE} and NTASKS=${NTASKS}"
echo "my jobID: " $SLURM_JOB_ID
echo "Partition: " $SLURM_JOB_PARTITION
echo "submit directory:" $SLURM_SUBMIT_DIR
echo "submit host:" $SLURM_SUBMIT_HOST
echo "In the directory: pwd"
echo "As the user: whoami"


echo ""
echo "==========================================================="
echo ""

echo "Run starts at $(date) on $(uname -n)"

$@

echo "1" > file1.txt
echo "2" > file2.txt

echo "Hello world ./ use"
cat file1.txt
cat file2.txt

echo "Run ends at $(date) on $(uname -n)"
