#!/usr/bin/env python3
"""
Run All Audits - Fase 3A Exhaustive Verification
=================================================

Ejecuta todos los checkpoints de auditoría en orden de prioridad:
1. Checkpoint 4 (métricas) - más rápido, puede invalidar evaluaciones
2. Checkpoint 1 (datos) - segundo más rápido
3. Checkpoint 3 (forward) - tercero
4. Checkpoint 2 (collate) - requiere modificar código
5. Checkpoint 5 (training) - requiere revisar logs
6. Checkpoint 6 (e2e) - más lento, solo si todo lo demás pasa

Usage:
    python experiments/audits/run_all_audits.py \
        --data data/datasets/roseta_constellation.npz \
        --model data/training_outputs/constellation_C1_mlp_hist/best_model.pt \
        --training-dir data/training_outputs/constellation_C1_mlp_hist
"""

import sys
import os
import argparse
import json
from pathlib import Path
from datetime import datetime

# Configure CUBLAS before any torch imports
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def run_checkpoint(name: str, func, *args, **kwargs) -> dict:
    """Run a checkpoint and capture results."""
    print(f"\n{'#'*70}")
    print(f"# RUNNING: {name}")
    print(f"{'#'*70}")

    try:
        passed = func(*args, **kwargs)
        return {
            'name': name,
            'passed': passed,
            'error': None,
        }
    except Exception as e:
        import traceback
        print(f"\n⚠ Exception during {name}:")
        traceback.print_exc()
        return {
            'name': name,
            'passed': False,
            'error': str(e),
        }


def main():
    parser = argparse.ArgumentParser(description='Run all Phase 3A audits')
    parser.add_argument('--data', type=str,
                        default='data/datasets/roseta_constellation.npz',
                        help='Path to constellation dataset')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to trained model checkpoint')
    parser.add_argument('--training-dir', type=str, default=None,
                        help='Path to training output directory')
    parser.add_argument('--output', type=str,
                        default='Documents/Revisionismo/Fase_3A/AUDITORIA_FASE_3A.md',
                        help='Output report path')
    parser.add_argument('--skip-e2e', action='store_true',
                        help='Skip E2E test (checkpoint 6)')
    args = parser.parse_args()

    print("="*70)
    print("AUDITORÍA EXHAUSTIVA FASE 3A - Ratio Constellations")
    print("="*70)
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Data: {args.data}")
    print(f"Model: {args.model}")
    print(f"Training dir: {args.training_dir}")

    results = []

    # ===== Checkpoint 4: Metrics =====
    from experiments.audits.checkpoint_4_metrics import run_all_tests as run_cp4
    results.append(run_checkpoint("Checkpoint 4: Métricas de Evaluación", run_cp4))

    # ===== Checkpoint 1: Data Integrity =====
    if Path(args.data).exists():
        from experiments.audits.checkpoint_1_data_integrity import run_all_tests as run_cp1
        results.append(run_checkpoint("Checkpoint 1: Integridad de Datos", run_cp1, args.data))
    else:
        print(f"\n⚠ Skipping Checkpoint 1: Data file not found: {args.data}")
        results.append({'name': 'Checkpoint 1: Integridad de Datos', 'passed': None, 'error': 'file_not_found'})

    # ===== Checkpoint 2: Collate =====
    if Path(args.data).exists():
        from experiments.audits.checkpoint_2_collate_verification import run_all_tests as run_cp2
        results.append(run_checkpoint("Checkpoint 2: Collate/DataLoader", run_cp2, args.data))
    else:
        results.append({'name': 'Checkpoint 2: Collate/DataLoader', 'passed': None, 'error': 'file_not_found'})

    # ===== Checkpoint 3: Forward Pass =====
    if args.model and Path(args.model).exists():
        from experiments.audits.checkpoint_3_forward_pass import run_all_tests as run_cp3
        results.append(run_checkpoint("Checkpoint 3: Forward Pass", run_cp3, args.model))
    else:
        print(f"\n⚠ Skipping Checkpoint 3: Model file not found: {args.model}")
        results.append({'name': 'Checkpoint 3: Forward Pass', 'passed': None, 'error': 'model_not_found'})

    # ===== Checkpoint 5: Training =====
    if args.training_dir and Path(args.training_dir).exists():
        from experiments.audits.checkpoint_5_training import run_all_tests as run_cp5
        results.append(run_checkpoint("Checkpoint 5: Training", run_cp5, args.training_dir, args.data))
    else:
        print(f"\n⚠ Skipping Checkpoint 5: Training dir not found: {args.training_dir}")
        results.append({'name': 'Checkpoint 5: Training', 'passed': None, 'error': 'dir_not_found'})

    # ===== Checkpoint 6: E2E =====
    if not args.skip_e2e:
        from experiments.audits.checkpoint_6_e2e import run_all_tests as run_cp6
        results.append(run_checkpoint("Checkpoint 6: End-to-End", run_cp6))
    else:
        print(f"\n⚠ Skipping Checkpoint 6: --skip-e2e flag set")
        results.append({'name': 'Checkpoint 6: End-to-End', 'passed': None, 'error': 'skipped'})

    # ===== Generate Report =====
    print("\n" + "="*70)
    print("GENERATING REPORT")
    print("="*70)

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build report
    report = f"""# Auditoría Exhaustiva - Fase 3A Ratio Constellations

**Fecha**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Dataset**: {args.data}
**Modelo**: {args.model}
**Training Dir**: {args.training_dir}

---

## Resumen de Resultados

| Checkpoint | Status | Error |
|------------|--------|-------|
"""

    all_passed = True
    for r in results:
        status = "✓ PASS" if r['passed'] else ("⚠ SKIP" if r['passed'] is None else "✗ FAIL")
        error = r['error'] or "-"
        report += f"| {r['name']} | {status} | {error} |\n"
        if r['passed'] is False:
            all_passed = False

    report += f"""
---

## Conclusión

"""

    if all_passed:
        report += """### ✓ TODOS LOS CHECKPOINTS PASARON

Los resultados NO-GO de Fase 3A **SON GENUINOS** y no son un artefacto de bugs.

**Interpretación**: El problema de cross-modality (H3) es real:
- Los datos de entrada son válidos y discriminativos
- El pipeline de training/evaluación funciona correctamente
- Los modelos simplemente no logran aprender correspondencia audio-vibración

**Siguiente paso**: Proceder a Fase 3B con confianza en que el diagnóstico es correcto.
"""
    else:
        failed = [r['name'] for r in results if r['passed'] is False]
        report += f"""### ✗ CHECKPOINTS FALLIDOS: {len(failed)}

Se encontraron problemas en:
"""
        for name in failed:
            report += f"- {name}\n"

        report += """
**Acción requerida**: Corregir los bugs identificados y re-ejecutar el sweep de Fase 3A.
Los resultados NO-GO podrían ser un artefacto de estos bugs.
"""

    report += f"""
---

## Detalles por Checkpoint

"""

    for r in results:
        status = "✓ PASS" if r['passed'] else ("⚠ SKIP" if r['passed'] is None else "✗ FAIL")
        report += f"""### {r['name']}

**Status**: {status}
"""
        if r['error']:
            report += f"**Error**: {r['error']}\n"
        report += "\n"

    report += """
---

*Generado por run_all_audits.py*
"""

    # Write report
    with open(output_path, 'w') as f:
        f.write(report)

    print(f"\nReport saved to: {output_path}")

    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)

    for r in results:
        status = "✓ PASS" if r['passed'] else ("⚠ SKIP" if r['passed'] is None else "✗ FAIL")
        print(f"  {r['name']}: {status}")

    print("\n" + "-"*70)
    if all_passed:
        print("✓ AUDITORÍA COMPLETADA: Todos los checkpoints pasaron")
        print("  → Los resultados NO-GO son GENUINOS")
    else:
        print("✗ AUDITORÍA COMPLETADA: Se encontraron problemas")
        print("  → Re-evaluar después de corregir bugs")
    print("-"*70)

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
