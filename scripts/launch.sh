#!/bin/bash
#SBATCH --account=teo@v100
#SBATCH --job-name=J_PIG # nom du job
#SBATCH --output=J_PIG%j.out # fichier de sortie (%j = job ID)
#SBATCH --error=J_PIG%j.err # fichier d’erreur (%j = job ID)
#SBATCH --nodes=1 # reserver 1 noeud
#SBATCH --ntasks=1 # reserver 1 taches (ou processus)
##SBATCH --array=0-5 # pour avoir 5 fois la meme exp (differentes seed)
#SBATCH --gres=gpu:1 # reserver 1 GPU 
#SBATCH --cpus-per-task=10 # reserver 10 CPU par tache (et memoire associee)
#SBATCH --time=015:00:00 # temps maximal d’allocation "(HH:MM:SS)"
#SBATCH --hint=nomultithread         # hyperthreading desactive



module purge # nettoyer les modules herites par defaut
module load python/3.10.4
conda activate pig_gymnasium310


set -x # activer l’echo des commandes

# Check if SLURM_ARRAY_TASK_ID is unset or empty, and set it to 0 if so
: "${SLURM_ARRAY_TASK_ID:=0}"

GROUP="g0"


echo "START"
./scripts/train_DubinsUmaze.sh ${SLURM_ARRAY_TASK_ID} $GROUP
# ./scripts/train_Dubins3Umaze.sh ${SLURM_ARRAY_TASK_ID} $GROUP
# ./scripts/train_PointMaze_UMaze.sh ${SLURM_ARRAY_TASK_ID} $GROUP
# ./scripts/train_AntMaze_Umaze.sh ${SLURM_ARRAY_TASK_ID} $GROUP
echo "FINISHED"


## 