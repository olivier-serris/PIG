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
#SBATCH --time=07:30:00 # temps maximal d’allocation "(HH:MM:SS)"
#SBATCH --hint=nomultithread         # hyperthreading desactive



module purge # nettoyer les modules herites par defaut
conda deactivate # desactiver les environnements herites par defaut
module load ??
conda activate ???


set -x # activer l’echo des commandes

echo "START"
./scripts/train_DubinsUmaze.sh
# ./scripts/train_Dubins3Umaze.sh
# ./scripts/train_PointMaze_UMaze.sh
echo "FINISHED"


## ${SLURM_ARRAY_TASK_ID}