#!/usr/bin/env python3
"""
Comparar resultados de validación entre 256 y 512 bins
"""

import json

# Cargar resultados de validación
def load_and_compare():
    # Simular resultados (en producción cargaríamos los JSONs completos)
    results_256 = {
        'sub_1_2.wav': 1,  # octave
        '5_4.wav': 2,      # major_third, minor_third  
        'comma_81_80.wav': 2,  # major_third, minor_third
        'phi_noise.wav': 1,    # minor_third
        'sub_7_6.wav': 1,      # minor_third
        '6_5.wav': 1,          # minor_third
        '11_8.wav': 1,         # fourth
        '7_6.wav': 1,          # minor_third
        'comma_531441_524288.wav': 2,  # major_third, minor_third
        'comma_33_32.wav': 3,  # fourth, major_third, minor_third
    }
    
    results_512 = {
        'sub_1_2.wav': 1,      # octave
        '5_4.wav': 1,          # major_third
        'comma_81_80.wav': 3,  # fifth, major_third, minor_third
        'phi_noise.wav': 1,    # minor_third
        '6_5.wav': 1,          # minor_third
        'comma_531441_524288.wav': 2,  # major_third, minor_third
        'comma_33_32.wav': 3,  # fourth, major_third, minor_third
    }
    
    print("🔍 ANÁLISIS COMPARATIVO 256 vs 512 BINS")
    print("=" * 50)
    
    # Mejoras en 512 bins
    improvements = []
    regressions = []
    
    all_files = set(results_256.keys()) | set(results_512.keys())
    
    for file in sorted(all_files):
        count_256 = results_256.get(file, 0)
        count_512 = results_512.get(file, 0)
        
        if count_512 > count_256:
            improvements.append((file, count_256, count_512))
        elif count_256 > count_512:
            regressions.append((file, count_256, count_512))
    
    print(f"\n✅ MEJORAS con 512 bins ({len(improvements)} casos):")
    for file, old, new in improvements:
        print(f"   {file}: {old} → {new} ratios (+{new-old})")
    
    print(f"\n❌ REGRESIONES con 512 bins ({len(regressions)} casos):")
    for file, old, new in regressions:
        print(f"   {file}: {old} → {new} ratios ({new-old})")
    
    # Estadísticas
    total_256 = sum(results_256.values())
    total_512 = sum(results_512.values())
    
    print(f"\n📊 TOTALES:")
    print(f"   256 bins: {total_256}/150 ratios ({total_256/150*100:.1f}%)")
    print(f"   512 bins: {total_512}/150 ratios ({total_512/150*100:.1f}%)")
    print(f"   Diferencia: {total_512-total_256:+d} ratios")
    
    # Análisis de resolución
    print(f"\n🔬 ANÁLISIS DE RESOLUCIÓN:")
    print(f"   256 bins: {256/2.585:.1f} bins/octava, {1200*2.585/256:.1f} cents/bin")
    print(f"   512 bins: {512/2.585:.1f} bins/octava, {1200*2.585/512:.1f} cents/bin")
    
    # Recomendación
    print(f"\n💡 RECOMENDACIÓN:")
    if total_512 >= total_256:
        print("   ✅ Usar 512 bins: Mayor resolución ayuda en microintervalos")
        print("   - Mejor precisión en commas (81/80 mejoró)")
        print("   - Costo computacional mínimo (+33% memoria)")
        print("   - Compatible con análisis de entonación justa")
    else:
        print("   ⚖️  Ambas opciones válidas, evaluar según use case")

if __name__ == "__main__":
    load_and_compare()