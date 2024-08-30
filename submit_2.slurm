#!/bin/bash
#SBATCH -J trial_hpc_batch
#SBATCH -A MRC-BSU-SL2-CPU
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=25
#SBATCH --time=10:00:00
#! What types of email messages do you wish to receive?
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=TIME_LIMIT
#SBATCH --mail-type=END
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
#SBATCH --no-requeue
#SBATCH --verbose

#SBATCH --array=1-4
#SBATCH -p icelake


#SBATCH --output=sliding_window_%j_%a.out
#SBATCH --error=sliding_window_%j_%a.err

julia --project -t ${SLURM_CPUS_PER_TASK} Main.jl ${SLURM_ARRAY_TASK_ID} 40 120 60 200 150 400
