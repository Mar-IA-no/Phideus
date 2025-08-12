#!/bin/bash
# Phideus Dual Architecture - Environment Switcher
# Usage: source scripts/switch_env.sh [vae|hrm]

ARCHITECTURE=$1

if [ "$ARCHITECTURE" = "vae" ]; then
    echo "🎵 Switching to VAE Current Line..."
    git checkout feature/vae-current 2>/dev/null
    export PHIDEUS_ARCH=VAE
    export PHIDEUS_CONFIG=config/vae_config.yaml
    export PHIDEUS_LINE="current"
    
    echo "  Architecture: VAE + Linear Attention"
    echo "  Focus: Consolidation & optimization"
    echo "  Target: >80% reconstruction, >15% harmonic detection"
    echo "  Models: models/vae/"
    
elif [ "$ARCHITECTURE" = "hrm" ]; then
    echo "🧠 Switching to HRM Research Line..."
    git checkout feature/hrm-research 2>/dev/null  
    export PHIDEUS_ARCH=HRM
    export PHIDEUS_CONFIG=config/hrm_config.yaml
    export PHIDEUS_LINE="research"
    
    echo "  Architecture: Hierarchical Reasoning Model"
    echo "  Focus: Novel architecture research"
    echo "  Target: >20% harmonic search, O(1) memory"
    echo "  Models: models/hrm/"
    
elif [ "$ARCHITECTURE" = "compare" ]; then
    echo "📊 Switching to Comparison Mode..."
    git checkout develop 2>/dev/null
    export PHIDEUS_ARCH=COMPARE
    export PHIDEUS_CONFIG=config/base_config.py
    export PHIDEUS_LINE="comparison"
    
    echo "  Mode: A/B Testing between architectures"
    echo "  Focus: Benchmarks and evaluation"
    
else
    echo "❌ Invalid architecture specified"
    echo "Usage: source scripts/switch_env.sh [vae|hrm|compare]"
    echo ""
    echo "Available options:"
    echo "  vae     - VAE Current Line (consolidation)"
    echo "  hrm     - HRM Research Line (experimental)"  
    echo "  compare - Comparison mode (benchmarks)"
    return 1
fi

# Verify environment
echo ""
echo "✅ Environment Status:"
echo "  PHIDEUS_ARCH: $PHIDEUS_ARCH"
echo "  PHIDEUS_CONFIG: $PHIDEUS_CONFIG"
echo "  PHIDEUS_LINE: $PHIDEUS_LINE"
echo "  Git Branch: $(git branch --show-current 2>/dev/null || echo 'detached')"

# Test configuration
echo ""
echo "🔧 Testing Configuration..."
python3 config/base_config.py 2>/dev/null && echo "  Config: OK" || echo "  Config: Error - install requirements.txt"

echo ""
echo "📚 Documentation Updates:"
echo "  Use: python3 scripts/actualizar_documentos.py (to see what needs updating)"
echo "  Command: 'actualizar documentos' (to trigger Claude update)"

echo ""
echo "🚀 Ready for development!"