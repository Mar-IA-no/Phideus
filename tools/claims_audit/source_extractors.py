"""
Source extractors: read experimental output files and extract metric values.
"""
import json
import glob
import os
import numpy as np
from pathlib import Path


def extract_json_field(path: str, key_path: str) -> float | None:
    """Extract a nested field from a JSON file. key_path uses dots: 'training.best_S'"""
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        data = json.load(f)
    keys = key_path.split(".")
    val = data
    for k in keys:
        if isinstance(val, dict) and k in val:
            val = val[k]
        else:
            return None
    return float(val) if val is not None else None


def extract_multiseed_stats(dir_pattern: str, field_path: str) -> dict | None:
    """
    Compute mean and std over multiple seed directories.
    dir_pattern: glob pattern like '/path/gate5b_multiseed/D0_seed*/final_results.json'
    field_path: JSON key path like 'training.best_S'
    Returns: {'mean': float, 'std': float, 'values': list, 'n': int, 'files': list}
    """
    files = sorted(glob.glob(dir_pattern))
    if not files:
        return None
    values = []
    found_files = []
    for f in files:
        val = extract_json_field(f, field_path)
        if val is not None:
            values.append(val)
            found_files.append(f)
    if not values:
        return None
    arr = np.array(values)
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        'values': values,
        'n': len(values),
        'files': found_files,
    }


def extract_cka_cross_mean(path: str) -> float | None:
    """
    Extract cross-encoder CKA mean from test06_rsa_cka.json.
    CKA matrix is 8x8: audio layers [0:4], midi layers [4:8].
    Cross-domain mean = mean of cka_matrix[i][j] for i in 0..3, j in 4..7.
    """
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        data = json.load(f)
    # Try different possible locations for CKA matrix
    matrix = None
    for key in ['cka_matrix', 'cka', 'results']:
        if key in data:
            candidate = data[key]
            if isinstance(candidate, list) and len(candidate) >= 8:
                matrix = candidate
                break
    # Also check nested
    if matrix is None and 'cka' in data and isinstance(data['cka'], dict):
        if 'matrix' in data['cka']:
            matrix = data['cka']['matrix']
    if matrix is None:
        return None
    # Compute cross-domain mean
    cross_values = []
    for i in range(min(4, len(matrix))):
        row = matrix[i]
        if isinstance(row, list) and len(row) >= 8:
            for j in range(4, min(8, len(row))):
                cross_values.append(float(row[j]))
    if not cross_values:
        return None
    return float(np.mean(cross_values))


def extract_min_retrieval(path: str) -> float | None:
    """Extract S = min(a2m_recall@10, m2a_recall@10) from a scoreboard JSON."""
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        data = json.load(f)
    # Direct S field
    if 'S' in data:
        return float(data['S'])
    # Compute from a2m/m2a
    a2m = None
    m2a = None
    for key_a in ['a2m', 'audio_to_midi']:
        if key_a in data:
            sub = data[key_a]
            for rkey in ['mean_recall_at_10', 'recall_at_10', 'mean_recall@10']:
                if rkey in sub:
                    a2m = float(sub[rkey])
                    break
    for key_m in ['m2a', 'midi_to_audio']:
        if key_m in data:
            sub = data[key_m]
            for rkey in ['mean_recall_at_10', 'recall_at_10', 'mean_recall@10']:
                if rkey in sub:
                    m2a = float(sub[rkey])
                    break
    if a2m is not None and m2a is not None:
        return min(a2m, m2a)
    return None


def extract_escalon3_iid_s(path: str) -> float | None:
    """
    Extract IID S from Escalon 3 final_results.json.
    S = min(a2i_recall@10, i2a_recall@10) from iid_test.
    Also checks final_results_v2.json.
    """
    for fname in ['final_results_v2.json', 'final_results.json']:
        fpath = os.path.join(os.path.dirname(path), fname) if os.path.isdir(path) else path.replace('final_results.json', fname)
        if not os.path.isfile(fpath):
            continue
        with open(fpath) as f:
            data = json.load(f)
        iid = data.get('iid_test', {})
        a2i = iid.get('a2i', {})
        i2a = iid.get('i2a', {})
        r10_a2i = a2i.get('recall_at_10')
        r10_i2a = i2a.get('recall_at_10')
        if r10_a2i is not None and r10_i2a is not None:
            return min(float(r10_a2i), float(r10_i2a))
        # Also check direct S field
        if 'S' in iid:
            return float(iid['S'])
    return None


def extract_escalon3_scale_ood_s(path: str) -> float | None:
    """Extract scale_ood S from Escalon 3 results."""
    for fname in ['final_results_v2.json', 'final_results.json']:
        fpath = path.replace('final_results.json', fname) if 'final_results' in path else os.path.join(path, fname)
        if not os.path.isfile(fpath):
            continue
        with open(fpath) as f:
            data = json.load(f)
        # Try various paths
        for section_key in ['scale_ood', 'ood_scale']:
            section = data.get(section_key, {})
            if 'S' in section:
                return float(section['S'])
            if 'S_mixed' in section:
                return float(section['S_mixed'])
            # Compute from a2i/i2a
            a2i = section.get('a2i', {})
            i2a = section.get('i2a', {})
            r10_a = a2i.get('recall_at_10') or a2i.get('ratio_hit_at_10')
            r10_i = i2a.get('recall_at_10') or i2a.get('ratio_hit_at_10')
            if r10_a is not None and r10_i is not None:
                return min(float(r10_a), float(r10_i))
    return None


def extract_best_from_eval_epochs(dir_path: str) -> float | None:
    """Scan eval_per_epoch/*.json for the best S = min(a2m_r@10, m2a_r@10)."""
    eval_dir = os.path.join(dir_path, 'eval_per_epoch')
    if not os.path.isdir(eval_dir):
        return None
    best_s = 0.0
    for f in sorted(glob.glob(os.path.join(eval_dir, 'eval_epoch*.json'))):
        with open(f) as fp:
            data = json.load(fp)
        # Try to get S
        s = None
        if 'S' in data:
            s = float(data['S'])
        else:
            a2m = m2a = None
            for key_a in ['a2m', 'audio_to_midi']:
                if key_a in data:
                    sub = data[key_a]
                    for rk in ['mean_recall_at_10', 'recall_at_10', 'mean_recall@10']:
                        if rk in sub:
                            a2m = float(sub[rk])
                            break
            for key_m in ['m2a', 'midi_to_audio']:
                if key_m in data:
                    sub = data[key_m]
                    for rk in ['mean_recall_at_10', 'recall_at_10', 'mean_recall@10']:
                        if rk in sub:
                            m2a = float(sub[rk])
                            break
            if a2m is not None and m2a is not None:
                s = min(a2m, m2a)
        if s is not None and s > best_s:
            best_s = s
    return best_s * 100 if best_s < 1.0 else best_s  # normalize to percentage


def extract_lombard_d0_best_s(base_dir: str) -> float | None:
    """Extract D0 best S from Lombard experiment outputs."""
    # Try multiple possible locations
    candidates = [
        os.path.join(base_dir, 'd0_control', 'summary.json'),
        os.path.join(base_dir, 'd0_control', 'final_results.json'),
        os.path.join(base_dir, 'p2_d0', 'final_results.json'),
    ]
    for c in candidates:
        if os.path.isfile(c):
            val = extract_json_field(c, 'best_S')
            if val is not None:
                return val * 100 if val < 1.0 else val
            val = extract_json_field(c, 'training.best_S')
            if val is not None:
                return val * 100 if val < 1.0 else val
    # Scan for any d0 log
    for f in glob.glob(os.path.join(base_dir, '**/d0*results*.json'), recursive=True):
        with open(f) as fp:
            data = json.load(fp)
        for key in ['best_S', 'S']:
            if key in data:
                val = float(data[key])
                return val * 100 if val < 1.0 else val
    return None


def extract_transposition_retention(path: str, semitones: int = 3) -> float | None:
    """Extract ±N semitone retention from test04_transposition.json.
    retention = mean(S[-N], S[+N]) / S_baseline * 100
    """
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        data = json.load(f)
    baseline = data.get('S_baseline')
    trans = data.get('transpositions', {})
    if baseline is None or not trans:
        return None
    s_neg = trans.get(str(-semitones), {}).get('S')
    s_pos = trans.get(str(semitones), {}).get('S')
    if s_neg is None or s_pos is None:
        return None
    return (s_neg + s_pos) / 2 / baseline * 100


def file_exists(path: str) -> bool:
    """Check if a file or directory exists."""
    return os.path.exists(path)
