# Harmonic Ratio Analysis System – Updated Workflow (v4.0)

This document explains how to use the updated scripts of the Harmonic Ratio Analysis System v4.0:
- `analizador_v4.0.py` – for analyzing `.wav` files
- `auditor_v4.0.py` – for reviewing, labeling, and exploring the output in harmonic and topological modes

---

## 1. Analyzer (`analizador_v4.0.py`)

### Step 1: Specify Input and Output Paths

Run the analyzer directly with command-line arguments to specify the input directory and output JSON file:

```bash
python analizador_v4.0.py --input-dir path/to/your/wav_files --output path/to/output.json
```

This will process all `.wav` files in the provided folder and generate a JSON file containing detailed ratio histograms.

- Generates both `ratio_hist_log` (log₂ scale, perceptual) and `ratio_hist_lin` (linear scale, physical).

### Step 2: Optional Parameters

You can adjust parameters such as the number of bins, FFT sizes, or peak thresholds via additional flags:

```bash
python analizador_v4.0.py --input-dir my_wavs --output results.json --bins 512 --thr 1.5
```

---

## 2. Auditor (`auditor_v4.0.py`)

### Step 1: Choose Analysis Mode

The auditor now supports three modes:

- `--analisis armonico` → Harmonic perceptual mode (default), using log histogram.
- `--analisis topologico` → Topological/physical mode, using linear histogram.
- `--analisis comparativo` → Both modes side by side.

### Step 2: Run the Auditor

```bash
python auditor_v4.0.py path/to/output.json --analisis armonico
```

Or for topological mode:

```bash
python auditor_v4.0.py path/to/output.json --analisis topologico
```

Or for comparative mode:

```bash
python auditor_v4.0.py path/to/output.json --analisis comparativo
```

### Step 3: Optional Markdown Output

Add `--markdown` to format tables in Markdown style:

```bash
python auditor_v4.0.py path/to/output.json --analisis armonico --markdown > results.md
```

---

## Notes

- Make sure your Python environment has all required dependencies (e.g., `numpy`, `scipy`, `librosa`, `tabulate`).
- `.wav` files must be monophonic and uncompressed for accurate analysis.
- The output JSON includes detailed frequency ratios, normalized histograms (log and linear), and metadata per file.
- The updated workflow no longer requires manual editing of paths inside scripts — all paths and parameters are passed via CLI.

---

🎶 *“The forest already sings. Our task is to understand its tuning.”*

