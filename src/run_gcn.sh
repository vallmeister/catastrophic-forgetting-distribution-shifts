#!/bin/bash
echo 'Starting job...'

sbatch --partition=gpu_4 -n 1 -t 48:00:00 --mem=170gb --gres=gpu:1 gcn.sh
sleep 1
squeue
