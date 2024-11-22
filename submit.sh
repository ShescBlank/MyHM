#!/bin/bash

# Nombre del trabajo
#SBATCH --job-name=MatrixCompression
# Archivo de salida
#SBATCH --output=Results/log_%j.txt
# Cola de trabajo
#SBATCH --partition=512x1024
# Reporte por correo
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alberto.almuna@uc.cl
# Solicitud de cpus
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

echo "start script"
date

which python
time python -u script_gmres.py

echo "end script"
date

cp submit.sh Results/