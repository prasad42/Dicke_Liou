#!/bin/sh
#SBATCH -N 2
#SBATCH --ntasks-per-node=48 
#SBATCH --time=23:59:59 
#SBATCH --job-name=Qopt
#SBATCH --error=job.%J.err_node_48 
#SBATCH --output=job.%J.out_node_48 
#SBATCH --partition=highmemory

# Activate the virtual environment
source /home/apps/DL/DL-CondaPy3.7/etc/profile.d/conda.sh
conda activate qutip

# Run your Python script
<<<<<<< HEAD
python3 plot_dsff.py
=======
python3 plot_Lop_evals.py
>>>>>>> 7136599 (Filter and Unfold)
