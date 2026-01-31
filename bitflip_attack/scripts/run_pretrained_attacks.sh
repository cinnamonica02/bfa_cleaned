#!/bin/bash

# Run attacks on pre-trained models in the financial and medical domains
# This script generates synthetic data, downloads models, and runs bit flip attacks

echo "============================================================"
echo "Bit Flip Attacks on Pre-trained Models in Finance and Healthcare"
echo "============================================================"

# Install required dependencies if needed
pip install -q faker torch==2.0.0 transformers==4.30.0 pandas numpy matplotlib seaborn scikit-learn bitsandbytes accelerate

# Create directories
mkdir -p data models results

# Generate synthetic datasets
echo "Generating synthetic datasets..."
python create_banking_dataset.py
python create_medical_dataset.py

# Run attacks with different bit flip counts
for bits in 1 3 5 10; do
    echo ""
    echo "Running attacks with $bits bit flips..."
    
    # Financial models
    echo "Attacking financial models..."
    python pretrained_attacks.py --max-bit-flips $bits --financial-only --num-candidates 100
    
    # Medical models
    echo "Attacking medical models..."
    python pretrained_attacks.py --max-bit-flips $bits --medical-only --num-candidates 100
done

# Generate summary of results
python -c """
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set plot style
sns.set_theme(style='whitegrid')

# Find all CSV files in results directories
files = glob.glob('results_*_attack_*_*bits/attack_results_*.csv')
all_results = []

for file in files:
    df = pd.read_csv(file)
    model_name = file.split('/')[-1].replace('.csv', '')
    domain = 'Medical' if 'medical' in file else 'Financial'
    
    # Extract bit flips from directory name
    bits_str = file.split('_')[-1].split('/')[0]
    bit_flips = int(bits_str.replace('bits', ''))
    
    asr = df[df['Metric'] == 'Final ASR']['Value'].values[0]
    accuracy = df[df['Metric'] == 'Final Accuracy']['Value'].values[0]
    execution_time = df[df['Metric'] == 'Execution Time (s)']['Value'].values[0]
    
    all_results.append({
        'Model': model_name,
        'Domain': domain,
        'Bit Flips': bit_flips,
        'ASR': asr,
        'Accuracy': accuracy,
        'Execution Time': execution_time
    })

# Create results directory
os.makedirs('results_summary', exist_ok=True)

# Save combined results
results_df = pd.DataFrame(all_results)
results_df.to_csv('results_summary/attack_summary.csv', index=False)

# Generate visualizations
plt.figure(figsize=(14, 8))
sns.barplot(data=results_df, x='Model', y='ASR', hue='Bit Flips')
plt.title('Attack Success Rate by Model and Bit Flips', fontsize=16)
plt.xlabel('Model', fontsize=14)
plt.ylabel('Attack Success Rate (ASR)', fontsize=14)
plt.xticks(rotation=45)
plt.ylim(0, 1.0)
plt.tight_layout()
plt.savefig('results_summary/asr_comparison.png', dpi=300)

plt.figure(figsize=(14, 8))
sns.scatterplot(data=results_df, x='Accuracy', y='ASR', 
               hue='Model', size='Bit Flips', sizes=(50, 250))
plt.title('Accuracy vs Attack Success Rate', fontsize=16)
plt.xlabel('Model Accuracy After Attack', fontsize=14)
plt.ylabel('Attack Success Rate (ASR)', fontsize=14)
plt.tight_layout()
plt.savefig('results_summary/accuracy_vs_asr.png', dpi=300)

plt.figure(figsize=(14, 8))
g = sns.lineplot(data=results_df, x='Bit Flips', y='ASR', hue='Model', marker='o', markersize=10)
plt.title('ASR vs Number of Bit Flips', fontsize=16)
plt.xlabel('Number of Bit Flips', fontsize=14)
plt.ylabel('Attack Success Rate (ASR)', fontsize=14)
plt.xticks([1, 3, 5, 10])
plt.tight_layout()
plt.savefig('results_summary/asr_vs_bits.png', dpi=300)

# Domain comparison
plt.figure(figsize=(12, 8))
sns.boxplot(data=results_df, x='Bit Flips', y='ASR', hue='Domain')
plt.title('ASR Distribution by Domain and Bit Flips', fontsize=16)
plt.xlabel('Number of Bit Flips', fontsize=14)
plt.ylabel('Attack Success Rate (ASR)', fontsize=14)
plt.tight_layout()
plt.savefig('results_summary/domain_comparison.png', dpi=300)

# Execution time comparison
plt.figure(figsize=(14, 8))
sns.barplot(data=results_df, x='Model', y='Execution Time', hue='Bit Flips')
plt.title('Execution Time by Model and Bit Flips', fontsize=16)
plt.xlabel('Model', fontsize=14)
plt.ylabel('Execution Time (seconds)', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('results_summary/execution_time.png', dpi=300)

print('Generated summary and visualizations in results_summary directory')
"""

echo ""
echo "Attacks completed. Results available in results_*_attack_* directories"
echo "Summary visualizations saved in results_summary directory"

# Print estimated time and costs
echo ""
echo "============================================================"
echo "Estimated Time and Costs Analysis"
echo "============================================================"
echo "Time Analysis:"
echo "  - Dataset Generation: ~5 minutes"
echo "  - Model Download (per model): ~3-10 minutes depending on size"
echo "  - Fine-tuning (per model): ~10-30 minutes on GPU, ~1-3 hours on CPU"
echo "  - Attack Execution (per model/bit flip): ~15-45 minutes on GPU, ~2-6 hours on CPU"
echo ""
echo "Total estimated time (all models, all bit flip counts):"
echo "  - With GPU: ~6-12 hours"
echo "  - CPU only: ~24-48 hours"
echo ""
echo "Compute Cost Estimates (using cloud GPU instances):"
echo "  - AWS p3.2xlarge (Tesla V100): ~$3.06/hour × 8 hours = ~$25"
echo "  - GCP n1-standard-8 + T4 GPU: ~$0.76/hour × 8 hours = ~$6"
echo "  - Azure NC6s_v3 (V100): ~$3.06/hour × 8 hours = ~$25"
echo ""
echo "Potential Roadblocks:"
echo "  1. Memory limitations with larger models (especially Bloomberg 50B)"
echo "     - Solution: Use model sharding/quantization with bitsandbytes library"
echo ""
echo "  2. API rate limits when downloading models from HuggingFace"
echo "     - Solution: Add delays between downloads or use cached copies"
echo ""
echo "  3. Compatibility issues with newer models"
echo "     - Solution: Fall back to stable versions or adapt model structure"
echo ""
echo "  4. Long computation times for genetic optimization"
echo "     - Solution: Reduce population size and generations for initial tests"
echo ""
echo "Note: For extremely large models (e.g., Bloomberg 50B), consider:"
echo "  - Using 8-bit quantization (--load-in-8bit flag in transformers)"
echo "  - Testing with smaller variant first (e.g., FinBERT base instead)"
echo "  - Using machines with 32+ GB RAM and high-end GPUs"

# List of models to attack
models=(
    "financial:yiyanghkust/finbert-pretrain"
    "financial:ProsusAI/finbert"
    "medical:medicalai/ClinicalBERT"
    "medical:emilyalsentzer/Bio_ClinicalBERT"
)

# Run attacks with different bit flip counts
for bits in 1 3 5 10; do
    echo ""
    echo "Running attacks with $bits bit flips..."
    python pretrained_attacks.py --max-bit-flips $bits
done

# Generate summary of results
python -c """
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns

# Find all CSV files in results directories
files = glob.glob('results_*_attack_*/attack_results_*.csv')
all_results = []

for file in files:
    df = pd.read_csv(file)
    model_name = file.split('_')[-1].replace('.csv', '')
    domain = 'Medical' if 'medical' in file else 'Financial'
    bit_flips = df[df['Metric'] == 'Bits Flipped']['Value'].values[0]
    asr = df[df['Metric'] == 'Final ASR']['Value'].values[0]
    accuracy = df[df['Metric'] == 'Final Accuracy']['Value'].values[0]
    execution_time = df[df['Metric'] == 'Execution Time (s)']['Value'].values[0]
    
    all_results.append({
        'Model': model_name,
        'Domain': domain,
        'Bit Flips': bit_flips,
        'ASR': asr,
        'Accuracy': accuracy,
        'Execution Time': execution_time
    })

results_df = pd.DataFrame(all_results)
results_df.to_csv('attack_summary.csv', index=False)

# Generate visualizations
plt.figure(figsize=(12, 8))
sns.barplot(data=results_df, x='Model', y='ASR', hue='Bit Flips')
plt.title('Attack Success Rate by Model and Bit Flips')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('asr_comparison.png')

plt.figure(figsize=(12, 8))
sns.scatterplot(data=results_df, x='Accuracy', y='ASR', 
               hue='Model', size='Bit Flips', sizes=(50, 200))
plt.title('Accuracy vs Attack Success Rate')
plt.tight_layout()
plt.savefig('accuracy_vs_asr.png')

plt.figure(figsize=(12, 8))
sns.lineplot(data=results_df, x='Bit Flips', y='ASR', hue='Model')
plt.title('ASR vs Number of Bit Flips')
plt.tight_layout()
plt.savefig('asr_vs_bits.png')

print('Generated summary and visualizations')
"""

echo ""
echo "Attacks completed. Results available in results_*_attack_* directories"
echo "Summary visualizations saved as PNG files" 