#!/bin/bash

#SBATCH --job-name=GKM_contour
#SBATCH --mail-type=ALL
#SBATCH --mail-user=adattilo@ucsc.edu
#SBATCH --output=logs/%j.out
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH --array=0-7

pwd; hostname; date

module load python/3.6.7 numpy pandas astropy scipy h5py matplotlib

python -u compute_num_completeness_mproc.py $SLURM_ARRAY_TASK_ID 8 10 400.0 2000 4 15.0 3000 ../stellarCatalogs/dr25_stellar_supp_gaia_clean_GKM.txt ../../../../../data/users/adattilo/PDM ../GKbaseline/vetCompletenessTable.pkl logisticX0xRotatedLogisticY02 out

date
