#!/bin/bash
#SBATCH --job-name=vexp-en-adapt
#SBATCH --partition=multi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --signal=B:SIGTERM@595
#SBATCH --output=/home/mfmendez/Repos/Phideus/results_unc/logs/vozexp_en_nadapt_%j.out
#SBATCH --error=/home/mfmendez/Repos/Phideus/results_unc/logs/vozexp_en_nadapt_%j.err

set -eo pipefail

. /etc/profile
module load gcc cuda
source /home/mfmendez/miniconda3/bin/activate phideus
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO=/home/mfmendez/Repos/Phideus
CACHE_ROOT=${CACHE_ROOT:-$REPO/data/voz_expresiva}
OUTPUT_ROOT=${OUTPUT_ROOT:-$REPO/results_unc/voz_expresiva}
OUTPUT_DIR=${OUTPUT_DIR:-$OUTPUT_ROOT/1_en_calibfix}
LOG_DIR=$REPO/results_unc/logs
WAVLM_CACHE_DIR=$CACHE_ROOT/wavlm_cache
DESC_CACHE_DIR=$CACHE_ROOT/descriptors_cache
WAVLM_FEATS=$WAVLM_CACHE_DIR/wavlm_features.npy
WAVLM_LENS=$WAVLM_CACHE_DIR/wavlm_lengths.npy
WAVLM_INDEX=$WAVLM_CACHE_DIR/wavlm_index.json
FAMILY_A=$DESC_CACHE_DIR/family_A.npy
HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
ESD_ROOT=${ESD_ROOT:-}

mkdir -p "$LOG_DIR" "$CACHE_ROOT" "$OUTPUT_ROOT"

detect_esd_root() {
    local candidates=(
        "$REPO/data/esd/raw/Emotional Speech Dataset"
        "$REPO/data/esd/raw/Emotion Speech Dataset"
        "$REPO/data/esd/raw/ESD"
        "$HOME/data/esd/raw/Emotional Speech Dataset"
        "$HOME/data/esd/raw/Emotion Speech Dataset"
        "$HOME/data/esd/raw/ESD"
    )
    local path
    for path in "${candidates[@]}"; do
        if [ -d "$path" ]; then
            echo "$path"
            return 0
        fi
    done
    return 1
}

require_fresh_output_dir() {
    if [ -d "$OUTPUT_DIR" ] && [ -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
        echo "ERROR: output dir already exists and is not empty: $OUTPUT_DIR"
        echo "1_train.py no hace resume ni dedup; usar un OUTPUT_DIR fresco."
        exit 1
    fi
    mkdir -p "$OUTPUT_DIR/embeddings" "$OUTPUT_DIR/predictions"
}

need_precache=0
if [ ! -f "$WAVLM_FEATS" ] || [ ! -f "$WAVLM_LENS" ] || [ ! -f "$WAVLM_INDEX" ] || [ ! -f "$FAMILY_A" ]; then
    need_precache=1
fi

if [ -z "$ESD_ROOT" ] && [ "$need_precache" -eq 1 ]; then
    ESD_ROOT=$(detect_esd_root || true)
fi

echo "=== Voz Expresiva EN N-adapt calibfix ==="
echo "  Job: ${SLURM_JOB_ID}"
echo "  Node: $(hostname)"
echo "  Date: $(date)"
echo "  Repo: $REPO"
echo "  Cache root: $CACHE_ROOT"
echo "  Output dir: $OUTPUT_DIR"
echo "  HF_HOME: $HF_HOME"
echo "  ESD_ROOT: ${ESD_ROOT:-<not-set>}"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo ""

require_fresh_output_dir

if [ "$need_precache" -eq 1 ] && [ -z "$ESD_ROOT" ]; then
    echo "ERROR: faltan caches de voz_expresiva y no se detecto ESD_ROOT."
    echo "Esperado: $WAVLM_FEATS, $WAVLM_LENS, $WAVLM_INDEX, $FAMILY_A"
    exit 1
fi

if [ "$need_precache" -eq 1 ]; then
    echo "Caches incompletos; regenerando precaches EN desde ESD..."
    echo "  ESD root: $ESD_ROOT"
    mkdir -p "$WAVLM_CACHE_DIR" "$DESC_CACHE_DIR"

    srun python "$REPO/experiments/voz_expresiva/1_precache_wavlm.py" \
        --esd-root "$ESD_ROOT" \
        --output-dir "$WAVLM_CACHE_DIR" \
        --language EN \
        --batch-size 8

    if [ ! -f "$WAVLM_FEATS" ] || [ ! -f "$WAVLM_LENS" ] || [ ! -f "$WAVLM_INDEX" ]; then
        echo "ERROR: WavLM precache incompleto tras 1_precache_wavlm.py"
        exit 1
    fi

    srun python "$REPO/experiments/voz_expresiva/1_precache_descriptors.py" \
        --wavlm-index "$WAVLM_INDEX" \
        --output-dir "$DESC_CACHE_DIR" \
        --workers 14

    if [ ! -f "$FAMILY_A" ]; then
        echo "ERROR: descriptor precache incompleto tras 1_precache_descriptors.py"
        exit 1
    fi
else
    echo "Caches detectados; no se regenera precache."
fi

echo ""
echo "Verificando tamanos de cache..."
du -sh "$WAVLM_CACHE_DIR" "$DESC_CACHE_DIR"

trap 'echo "SIGTERM recibido en $(date), terminando hijo activo."; jobs -p | xargs -r kill -TERM; wait' SIGTERM

echo ""
echo "Lanzando rerun EN N-adapt..."
srun python "$REPO/experiments/voz_expresiva/1_train.py" \
    --cache-root "$CACHE_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --epochs 30 \
    --batch-size 64 \
    --device cuda \
    --limit-norms adapt

echo ""
echo "Verificando artefactos finales..."
if [ ! -f "$OUTPUT_DIR/uar_results.json" ] || [ ! -f "$OUTPUT_DIR/calib_manifest.json" ]; then
    echo "ERROR: faltan artefactos finales en $OUTPUT_DIR"
    exit 1
fi

python - "$OUTPUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
results = json.loads((out_dir / "uar_results.json").read_text())
manifest = json.loads((out_dir / "calib_manifest.json").read_text())

adapt = [r for r in results if r.get("norm_condition") == "adapt"]
if len(adapt) != 120:
    raise SystemExit(f"Expected 120 adapt records, got {len(adapt)}")
if any(r.get("calib_seed_effective") is None for r in adapt):
    raise SystemExit("Some adapt records have null calib_seed_effective")
if any(r.get("norm_condition") != "adapt" for r in results):
    raise SystemExit("Output contains non-adapt records; expected adapt-only rerun")

sentence_sets = {
    spk: tuple(entry["sentence_ids"])
    for spk, entry in manifest.items()
}
if len(sentence_sets) != 10:
    raise SystemExit(f"Expected 10 speakers in calib_manifest, got {len(sentence_sets)}")
if len(set(sentence_sets.values())) == 1:
    raise SystemExit("All speakers share the same calibration set; B2 fix not effective")

print("verification_ok")
print(f"adapt_records={len(adapt)}")
print(f"manifest_speakers={len(sentence_sets)}")
PY

echo ""
echo "Job Metrics:"
sacct -j "$SLURM_JOB_ID" \
    --format=JobID,JobName%24,State,ExitCode,Elapsed,MaxRSS,NodeList \
    -P 2>/dev/null || true

echo ""
echo "Run completo:"
echo "  Output: $OUTPUT_DIR"
echo "  Date: $(date)"
