#!/bin/bash
#SBATCH --job-name=GKM_contour          # Job name
#SBATCH --mail-type=ALL                 # Mail events
#SBATCH --mail-user=adattilo@ucsc.edu   # Where to send mail
#SBATCH --ntasks=8                      # Run a single task
#SBATCH --time=06:00:00                 # Time limit hrs:min:sec
#SBATCH --output=logs/GKM_%j.log        # Standard output and error log

pwd; hostname; date

module load python/3.6.7 numpy pandas astropy scipy h5py matplotlib

nworkers=8

for (( i=0; i<$nworkers; i++ ))
do 
        python -u compute_num_completeness_mproc.py $i $nworkers 0.01 500.0 2000 0.75 12.0 3001 ../stellarCatalogs/dr25_stellar_supp_gaia_clean_GKM.txt ../../../../../data/users/adattilo/PDM ../GKbaseline/vetCompletenessTable.pkl logisticX0xRotatedLogisticY02 out &
done

date
