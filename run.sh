#!/bin/bash

total_jobs=20

# Loop through each job index (0 to 79)
for ((N=1; N<(total_jobs+1); N++)); do


    # Run Julia with 10 threads
    JULIA_NUM_THREADS=10 julia --project -t 10 SEIR_bayescomp_one_region.jl my_model.toml 1 "$N"
done

