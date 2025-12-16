#!/bin/bash

# Slurm submission script, 
# torch job 
# CRIHAN v 1.00 - Jan 2023 
# support@criann.fr


# Job name
#SBATCH -J "main"

# Batch output file
#SBATCH --output main.o%J

# Batch error file
#SBATCH --error main.e%J

# GPUs architecture and number
# ----------------------------
# Partition (submission class)
#SBATCH --partition mesonet
#SBATCH --gpus=1
#SBATCH --mem=80G
# ----------------------------
# processes / tasks
#SBATCH -n 1

# ------------------------
# Job time (hh:mm:ss)
#SBATCH --time 12:00:00
# ------------------------

#SBATCH --account=m25206

##SBATCH --mail-type ALL
# User e-mail address
#SBATCH --mail-user mathis.saunier@insa-rouen.fr


## Job script ##
# ---------------------------------
# Copy script input data and go to working directory
# ATTENTION : Il faut que le script soit dans le répertoire de travail

# Chargement du module go
# --- Go / build toolchain: prefer spack module, sinon fallback local ---
PYTHON_MODULE="python@3.11.9"
PIP_MODULE="py-pip@23.0"
CUDA_MODULE="cuda@12.6.2"

# try loading spack modules
if spack load "${CUDA_MODULE}" 2>/dev/null; then
  echo "[INFO] Loaded ${CUDA_MODULE} via spack."
else
  echo "[WARN] spack couldn't load ${CUDA_MODULE}."
fi
if spack load "${PYTHON_MODULE}" 2>/dev/null; then
  echo "[INFO] Loaded ${PYTHON_MODULE} via spack."
else
  echo "[WARN] spack couldn't load ${PYTHON_MODULE}."
fi
if spack load "${PIP_MODULE}" 2>/dev/null; then
  echo "[INFO] Loaded ${PIP_MODULE} via spack."
else
  echo "[WARN] spack couldn't load ${PIP_MODULE}."
fi

rsync -av --exclude 'saved' ./ $LOCAL_WORK_DIR
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1

echo Working directory : $PWD

echo "[INFO] Using python at $PYTHON_BIN"

VENV_DIR="${SLURM_TMPDIR:-$HOME/.cache}/bcresnet_venv"
"${BASE_PY}" -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --upgrade pip >/dev/null
pip install torch
echo "torch installed."
pip install pickle
echo "pickle installed."
pip install tqdm numpy
echo "tqdm and numpy installed."
pip install matplotlib >/dev/null
echo "matplotlib installed."
PYTHON_BIN="$VENV_DIR/bin/python"

echo "Job started at `date`"

# assure-toi que VENV_DIR est défini plus haut (ex: VENV_DIR="${SLURM_TMPDIR:-$HOME/.cache}/bcresnet_venv")
srun bash -lc "source '$VENV_DIR/bin/activate' && echo 'Using python: ' \$(which python) && cd TP5-Transformer && python -u main_transformer.py"

echo "Job finished at `date`"

exit 0
# End of job script
# ---------------------------------