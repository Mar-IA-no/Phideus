# Harmonic Ratio Analysis System – Usage Instructions

This document explains how to use the two main scripts of the Harmonic Ratio Analysis System:  
- `analizador_v3.3.py` – for analyzing `.wav` files  
- `auditor_v3.1.py` – for reviewing and refining the output

---

## 1. Analyzer (`analizador_v3.3.py`)

### Step 1: Edit the Input and Output Paths

Before running the analyzer, you need to open the script and **manually edit** the following two variables to match your data:

```python
wav_directory = "path/to/your/wav_files"   # Folder with input .wav files
output_file = "path/to/output.json"        # Destination JSON file
```

These lines are located near the beginning of the script.  
Ensure all `.wav` files to be analyzed are in the specified folder.

### Step 2: Run the Analyzer

In your terminal or command prompt, execute:

```bash
python analizador_v3.3.py
```

This will process all `.wav` files in the specified folder and save the harmonic ratios into a structured JSON file.

---

## 2. Auditor (`auditor_v3.1.py`)

### Step 1: Set the Input File (Optional)

By default, `auditor_v3.1.py` will load the same output file defined in the analyzer. If needed, you can change the filename in the 
script or pass it as an argument.

### Step 2: Run the Auditor

Use the terminal to launch the auditor:

```bash
python auditor_v3.1.py
```

This script provides an interactive interface to explore, visualize and review the harmonic ratio data.

---

## Notes

- Make sure your Python environment has all required dependencies (e.g., `numpy`, `scipy`, `matplotlib`, etc.).
- `.wav` files must be monophonic and uncompressed for accurate analysis.
- Output JSON structure includes frequency ratios and metadata for each file.

---

