#!/bin/bash

#SBATCH --account=p31538  ## YOUR ACCOUNT pXXXX or bXXXX
#SBATCH --partition=gengpu  ### PARTITION (buyin, short, normal, etc)
#SBATCH --nodes=8 ## how many computers do you need
#SBATCH --ntasks-per-node=5 ## how many cpus or processors do you need on each computer
#SBATCH --time=00:10:00 ## how long does this need to run (remember different partitions have restrictions on this param)
#SBATCH --mem-per-cpu=16G ## how much RAM do you need per CPU (this effects your FairShare score so be careful to not ask for more than you need))
#SBATCH --job-name=test  ## When you run squeue -u NETID this is how you can identify the job
#SBATCH --mail-type=ALL ## you can receive e-mail alerts from SLURM when your job begins and when your job finishes (completed, failed, etc)
#SBATCH --mail-user=$SLURM_MAIL_ENDPOINT ## your email
#SBATCH --gres=gpu:a100:4
###SBATCH --constraint="[quest5|quest6|quest8|quest9]" ### you want computers you have requested to be from either quest5 or quest6/7 or quest8 or quest 9 nodes, not a combination of nodes. Import for MPI, not usually import for job arrays)

source set_env
mpirun -n 32 -N 4 /projects/p31538/scr2448/tsa/links
