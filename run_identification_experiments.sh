#!/bin/bash
# Run multiple face identification attack experiments for paper

echo "=========================================================================="
echo "Face Identification Attack - Multi-Configuration Experiments"
echo "=========================================================================="
echo ""
echo "This script will run 3 experiments with different bit flip configurations:"
echo "  1. 5 bit flips  (conservative attack)"
echo "  2. 10 bit flips (moderate attack)"
echo "  3. 15 bit flips (aggressive attack)"
echo ""
echo "Expected total runtime: 3-4 hours"
echo ""
read -p "Press Enter to start experiments..."

# Create logs directory
mkdir -p logs

# Experiment 1: 5 bits
echo ""
echo "=========================================================================="
echo "EXPERIMENT 1/3: 5 Bit Flips"
echo "=========================================================================="
python lfw_face_identification_attack.py \
    --max_bit_flips 5 \
    --num_candidates 500 \
    --population_size 30 \
    --generations 10 \
    2>&1 | tee "logs/exp_5bits_$(date +%Y%m%d_%H%M%S).log"

echo "✓ Experiment 1 complete!"
sleep 2

# Experiment 2: 10 bits
echo ""
echo "=========================================================================="
echo "EXPERIMENT 2/3: 10 Bit Flips"
echo "=========================================================================="
python lfw_face_identification_attack.py \
    --max_bit_flips 10 \
    --num_candidates 500 \
    --population_size 30 \
    --generations 10 \
    2>&1 | tee "logs/exp_10bits_$(date +%Y%m%d_%H%M%S).log"

echo "✓ Experiment 2 complete!"
sleep 2

# Experiment 3: 15 bits
echo ""
echo "=========================================================================="
echo "EXPERIMENT 3/3: 15 Bit Flips"
echo "=========================================================================="
python lfw_face_identification_attack.py \
    --max_bit_flips 15 \
    --num_candidates 500 \
    --population_size 30 \
    --generations 10 \
    2>&1 | tee "logs/exp_15bits_$(date +%Y%m%d_%H%M%S).log"

echo "✓ Experiment 3 complete!"

echo ""
echo "=========================================================================="
echo "ALL EXPERIMENTS COMPLETE!"
echo "=========================================================================="
echo ""
echo "Results saved to: results/face_identification_attack_*/"
echo "Logs saved to: logs/exp_*bits_*.log"
echo ""
echo "Next steps:"
echo "  1. Run: python aggregate_results.py"
echo "  2. Check: results/ for plots and LaTeX tables"
echo "  3. Use results in your paper!"
echo ""
