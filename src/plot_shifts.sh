#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=10
#SBATCH --mem=5000

#SBATCH --mail-type=ALL
#SBATCH --mail-user=christian.valenti@uni-ulm.de
#SBATCH --output=testout.log
#SBATCH --error=testerr.log

python plot_shifts.py
