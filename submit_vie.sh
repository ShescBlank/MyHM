#!/bin/bash

# Nombre del trabajo
#SBATCH --job-name=generate_vie_sol
# Archivo de salida
#SBATCH --output=Results/log_%j.txt
# Cola de trabajo
#SBATCH --partition=gpus
# Reporte por correo
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alberto.almuna@uc.cl
# Solicitud de cpus
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --nodelist=n9

echo "start script"
date

which python
time python -u script_skull_slab.py

echo "end script"
date

cp submit_vie.sh Results/
