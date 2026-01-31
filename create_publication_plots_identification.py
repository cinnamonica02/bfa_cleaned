import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.5)
sns.set_palette("husl")

def load_results(result_dir):
    result_path = Path(result_dir)

    with open(result_path / 'results.json', 'r') as f:
        results = json.load(f)

    baseline_cm = np.load(result_path / 'baseline_confusion_matrix.npy')
    attack_cm = np.load(result_path / 'attack_confusion_matrix.npy')

    return results, baseline_cm, attack_cm


def plot_accuracy_icr_comparison(results, output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))


    categories = ['Baseline', f'After Attack\n({results["attack"]["bits_flipped"]} bits)']
    accuracies = [
        results['baseline']['accuracy'] * 100,
        results['attack']['accuracy'] * 100
    ]
    icrs = [
        results['baseline']['identity_confusion_rate'] * 100,
        results['attack']['identity_confusion_rate'] * 100
    ]

    colors1 = ['#2ecc71', '#e74c3c']
    bars1 = ax1.bar(categories, accuracies, color=colors1, alpha=0.85, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax1.set_title('(a) Model Accuracy', fontsize=16, fontweight='bold')
    ax1.set_ylim([85, 100])
    ax1.grid(axis='y', alpha=0.3, linestyle='--')


    for i, (bar, val) in enumerate(zip(bars1, accuracies)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=13, fontweight='bold')


    colors2 = ['#3498db', '#c0392b']
    bars2 = ax2.bar(categories, icrs, color=colors2, alpha=0.85, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Identity Confusion Rate (%)', fontsize=14, fontweight='bold')
    ax2.set_title('(b) Identity Confusion Rate', fontsize=16, fontweight='bold')
    ax2.set_ylim([0, max(icrs) * 1.3])
    ax2.grid(axis='y', alpha=0.3, linestyle='--')


    for i, (bar, val) in enumerate(zip(bars2, icrs)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=13, fontweight='bold')

    plt.tight_layout()
    output_file = output_dir / 'fig1_accuracy_icr_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_file}")


def plot_confusion_matrices(baseline_cm, attack_cm, output_dir):

    baseline_cm_norm = baseline_cm / baseline_cm.sum(axis=1, keepdims=True)
    attack_cm_norm = attack_cm / attack_cm.sum(axis=1, keepdims=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))


    sns.heatmap(baseline_cm_norm, cmap='Blues', ax=ax1,
                cbar_kws={'label': 'Proportion'}, vmin=0, vmax=1,
                square=True, cbar=True)
    ax1.set_title('(a) Baseline Confusion Matrix', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Predicted Identity', fontsize=13)
    ax1.set_ylabel('True Identity', fontsize=13)

    baseline_acc = np.trace(baseline_cm) / baseline_cm.sum()
    ax1.text(0.02, 0.98, f'Accuracy: {baseline_acc*100:.2f}%',
             transform=ax1.transAxes, fontsize=12, fontweight='bold',
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))


    sns.heatmap(attack_cm_norm, cmap='Reds', ax=ax2,
                cbar_kws={'label': 'Proportion'}, vmin=0, vmax=1,
                square=True, cbar=True)
    ax2.set_title('(b) After Bit Flip Attack', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Predicted Identity', fontsize=13)
    ax2.set_ylabel('True Identity', fontsize=13)



    attack_acc = np.trace(attack_cm) / attack_cm.sum()
    ax2.text(0.02, 0.98, f'Accuracy: {attack_acc*100:.2f}%',
             transform=ax2.transAxes, fontsize=12, fontweight='bold',
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

    plt.tight_layout()
    output_file = output_dir / 'fig2_confusion_matrices.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")







def plot_confusion_difference(baseline_cm, attack_cm, results, output_dir):
    baseline_cm_norm = baseline_cm / baseline_cm.sum(axis=1, keepdims=True)
    attack_cm_norm = attack_cm / attack_cm.sum(axis=1, keepdims=True)
    diff_cm = attack_cm_norm - baseline_cm_norm

    fig, ax = plt.subplots(figsize=(10, 9))

    sns.heatmap(diff_cm, cmap='RdBu_r', ax=ax,
                cbar_kws={'label': 'Change in Proportion'},
                center=0, vmin=-0.3, vmax=0.3, square=True)
    ax.set_title('Identity Confusion Induced by Bit Flip Attack\n(Attack - Baseline)',
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Predicted Identity', fontsize=13)
    ax.set_ylabel('True Identity', fontsize=13)


    icr_increase = results['improvement']['icr_increase'] * 100
    bits_used = results['attack']['bits_flipped']
    textstr = f'Bits Flipped: {bits_used}\nICR Increase: +{icr_increase:.2f}%'
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
            fontsize=12, fontweight='bold', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    plt.tight_layout()
    output_file = output_dir / 'fig3_confusion_difference.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Saved: {output_file}")


def plot_attack_impact_metrics(results, output_dir):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))


    baseline_acc = results['baseline']['accuracy'] * 100
    attack_acc = results['attack']['accuracy'] * 100
    acc_drop = results['improvement']['accuracy_drop'] * 100

    ax1.bar(['Baseline', 'Attack'], [baseline_acc, attack_acc],
            color=['#2ecc71', '#e74c3c'], alpha=0.85, edgecolor='black', linewidth=2)
    ax1.axhline(y=90, color='orange', linestyle='--', linewidth=2, label='90% threshold')
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Accuracy Degradation', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.text(0.5, baseline_acc + 0.5, f'{baseline_acc:.2f}%', ha='center', fontweight='bold')
    ax1.text(1.5, attack_acc + 0.5, f'{attack_acc:.2f}%', ha='center', fontweight='bold')


    baseline_icr = results['baseline']['identity_confusion_rate'] * 100
    attack_icr = results['attack']['identity_confusion_rate'] * 100
    icr_increase = results['improvement']['icr_increase'] * 100




    ax2.bar(['Baseline', 'Attack'], [baseline_icr, attack_icr],
            color=['#3498db', '#c0392b'], alpha=0.85, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Identity Confusion Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_title('(b) Identity Confusion Growth', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.text(0.5, baseline_icr + 0.3, f'{baseline_icr:.2f}%', ha='center', fontweight='bold')
    ax2.text(1.5, attack_icr + 0.3, f'{attack_icr:.2f}%', ha='center', fontweight='bold')


    baseline_mis = results['baseline']['misidentified_count']
    attack_mis = results['attack']['misidentified_count']
    total = results['baseline']['total_samples']




    ax3.bar(['Baseline', 'Attack'], [baseline_mis, attack_mis],
            color=['#16a085', '#d35400'], alpha=0.85, edgecolor='black', linewidth=2)
    ax3.set_ylabel('Misidentified Samples', fontsize=12, fontweight='bold')
    ax3.set_title(f'(c) Misidentifications (out of {total})', fontsize=14, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    ax3.text(0.5, baseline_mis + 1, str(baseline_mis), ha='center', fontweight='bold')
    ax3.text(1.5, attack_mis + 1, str(attack_mis), ha='center', fontweight='bold')


    bits_used = results['attack']['bits_flipped']
    bit_eff = results['attack']['bit_efficiency'] * 100

    ax4.bar([f'{bits_used} bits'], [icr_increase],
            color='#8e44ad', alpha=0.85, edgecolor='black', linewidth=2, width=0.5)
    ax4.set_ylabel('ICR Increase (%)', fontsize=12, fontweight='bold')
    ax4.set_title(f'(d) Attack Efficiency ({bit_eff:.2f}% per bit)', fontsize=14, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    ax4.text(0, icr_increase + 0.3, f'{icr_increase:.2f}%', ha='center', fontweight='bold', fontsize=13)



    plt.tight_layout()
    output_file = output_dir / 'fig4_attack_impact_metrics.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_file}")


def plot_comprehensive_summary(results, baseline_cm, attack_cm, output_dir):
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    baseline_acc = results['baseline']['accuracy'] * 100
    attack_acc = results['attack']['accuracy'] * 100
    baseline_icr = results['baseline']['identity_confusion_rate'] * 100
    attack_icr = results['attack']['identity_confusion_rate'] * 100
    bits_used = results['attack']['bits_flipped']
    icr_increase = results['improvement']['icr_increase'] * 100


    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(['Baseline', 'Attack'], [baseline_acc, attack_acc],
            color=['#2ecc71', '#e74c3c'], alpha=0.85, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Accuracy (%)', fontweight='bold')
    ax1.set_title('(a) Model Accuracy', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    for i, v in enumerate([baseline_acc, attack_acc]):
        ax1.text(i, v + 0.3, f'{v:.2f}%', ha='center', fontweight='bold', fontsize=10)


    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(['Baseline', 'Attack'], [baseline_icr, attack_icr],
            color=['#3498db', '#c0392b'], alpha=0.85, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('ICR (%)', fontweight='bold')
    ax2.set_title('(b) Identity Confusion Rate', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    for i, v in enumerate([baseline_icr, attack_icr]):
        ax2.text(i, v + 0.3, f'{v:.2f}%', ha='center', fontweight='bold', fontsize=10)


    ax3 = fig.add_subplot(gs[1, 0])
    baseline_cm_norm = baseline_cm / baseline_cm.sum(axis=1, keepdims=True)
    sns.heatmap(baseline_cm_norm, cmap='Blues', ax=ax3, cbar=True,
                cbar_kws={'label': 'Proportion'}, vmin=0, vmax=1, square=True)
    ax3.set_title('(c) Baseline Confusion Matrix', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Predicted', fontsize=10)
    ax3.set_ylabel('True', fontsize=10)


    ax4 = fig.add_subplot(gs[1, 1])
    attack_cm_norm = attack_cm / attack_cm.sum(axis=1, keepdims=True)
    sns.heatmap(attack_cm_norm, cmap='Reds', ax=ax4, cbar=True,
                cbar_kws={'label': 'Proportion'}, vmin=0, vmax=1, square=True)
    ax4.set_title('(d) After Attack Confusion Matrix', fontsize=13, fontweight='bold')
    ax4.set_xlabel('Predicted', fontsize=10)
    ax4.set_ylabel('True', fontsize=10)


    ax5 = fig.add_subplot(gs[2, 0])
    ax5.bar([f'{bits_used} bits'], [icr_increase], color='#8e44ad',
            alpha=0.85, edgecolor='black', linewidth=2, width=0.5)
    ax5.set_ylabel('ICR Increase (%)', fontweight='bold')
    ax5.set_title(f'(e) Attack Impact', fontsize=13, fontweight='bold')
    ax5.grid(axis='y', alpha=0.3)
    ax5.text(0, icr_increase + 0.3, f'{icr_increase:.2f}%', ha='center', fontweight='bold', fontsize=11)


    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')

    summary_text = f"""
    ATTACK SUMMARY
    {'='*40}

    Baseline Performance:
      • Accuracy: {baseline_acc:.2f}%
      • ICR: {baseline_icr:.2f}%

    Attack Configuration:
      • Bits Flipped: {bits_used}
      • Attack Mode: Untargeted

    Attack Results:
      • Final Accuracy: {attack_acc:.2f}%
      • Final ICR: {attack_icr:.2f}%
      • ICR Increase: +{icr_increase:.2f}%

    Security Impact:
      • {results['attack']['misidentified_count']} identities confused
      • {results['attack']['bit_efficiency']*100:.2f}% ICR per bit
    """



    ax6.text(0.1, 0.5, summary_text, transform=ax6.transAxes,
             fontsize=11, verticalalignment='center', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle('Bit Flip Attack on Face Recognition: Comprehensive Analysis',
                 fontsize=18, fontweight='bold', y=0.98)

    output_file = output_dir / 'fig5_comprehensive_summary.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_file}")


def plot_comparison_table(results, output_dir):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')


    baseline = results['baseline']
    attack = results['attack']
    improvement = results['improvement']

    table_data = [
        ['Metric', 'Baseline', 'After Attack', 'Change'],
        ['', '', '', ''],  
        ['Accuracy (%)', f"{baseline['accuracy']*100:.2f}", f"{attack['accuracy']*100:.2f}",
         f"{improvement['accuracy_drop']*100:.2f} ↓"],
        ['Identity Confusion Rate (%)', f"{baseline['identity_confusion_rate']*100:.2f}",
         f"{attack['identity_confusion_rate']*100:.2f}", f"+{improvement['icr_increase']*100:.2f} ↑"],
        ['Correctly Identified', f"{baseline['total_samples'] - baseline['misidentified_count']}/{baseline['total_samples']}",
         f"{attack['total_samples'] - attack['misidentified_count']}/{attack['total_samples']}", ''],
        ['Misidentified', f"{baseline['misidentified_count']}", f"{attack['misidentified_count']}",
         f"+{attack['misidentified_count'] - baseline['misidentified_count']}"],
        ['', '', '', ''],  
        ['Bits Flipped', '-', f"{attack['bits_flipped']}", '-'],
        ['Bit Efficiency (% ICR/bit)', '-', f"{attack['bit_efficiency']*100:.2f}", '-'],
    ]


    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.35, 0.2, 0.25, 0.2])

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)


    for i in range(4):
        cell = table[(0, i)]
        cell.set_facecolor('#3498db')
        cell.set_text_props(weight='bold', color='white', fontsize=12)


    for col in range(4):
        table[(1, col)].set_facecolor('#ecf0f1')
        table[(6, col)].set_facecolor('#ecf0f1')


    table[(2, 3)].set_facecolor('#ffcccc')  
    table[(3, 3)].set_facecolor('#ffcccc')  
    table[(5, 3)].set_facecolor('#ffcccc')  


    for col in range(4):
        table[(7, col)].set_facecolor('#e8f8f5')
        table[(8, col)].set_facecolor('#e8f8f5')

    plt.title('Identity Confusion Attack: Baseline vs Attack Comparison',
              fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    output_file = output_dir / 'fig6_comparison_table.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_file}")


def plot_icr_impact(results, output_dir):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    baseline = results['baseline']
    attack = results['attack']
    total = baseline['total_samples']

    categories = ['Baseline', 'After Attack']
    icr_values = [baseline['identity_confusion_rate'] * 100,
                   attack['identity_confusion_rate'] * 100]
    colors = ['#27ae60', '#e74c3c']

    bars1 = ax1.bar(categories, icr_values, color=colors, alpha=0.85,
                    edgecolor='black', linewidth=2, width=0.6)
    ax1.set_ylabel('Identity Confusion Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Identity Confusion Rate', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim([0, max(icr_values) * 1.3])

    for bar, val in zip(bars1, icr_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')


    misidentified_counts = [baseline['misidentified_count'], attack['misidentified_count']]

    bars2 = ax2.bar(categories, misidentified_counts, color=colors, alpha=0.85,
                    edgecolor='black', linewidth=2, width=0.6)
    ax2.set_ylabel('Number of Misidentified Samples', fontsize=12, fontweight='bold')
    ax2.set_title(f'(b) Misidentifications (out of {total})', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    for bar, val in zip(bars2, misidentified_counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                str(val), ha='center', va='bottom', fontsize=11, fontweight='bold')


    baseline_correct = total - baseline['misidentified_count']
    baseline_incorrect = baseline['misidentified_count']

    ax3.pie([baseline_correct, baseline_incorrect],
            labels=['Correctly\nIdentified', 'Misidentified'],
            autopct='%1.1f%%',
            colors=['#2ecc71', '#e74c3c'],
            startangle=90,
            textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax3.set_title('(c) Baseline Performance', fontsize=14, fontweight='bold')


    attack_correct = total - attack['misidentified_count']
    attack_incorrect = attack['misidentified_count']

    ax4.pie([attack_correct, attack_incorrect],
            labels=['Correctly\nIdentified', 'Misidentified'],
            autopct='%1.1f%%',
            colors=['#f39c12', '#c0392b'],
            startangle=90,
            textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax4.set_title('(d) After Attack Performance', fontsize=14, fontweight='bold')

    fig.suptitle('Identity Confusion Attack: Security Impact Analysis',
                 fontsize=18, fontweight='bold', y=0.98)

    plt.tight_layout()
    output_file = output_dir / 'fig7_icr_impact.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_file}")


def plot_attack_summary_card(results, output_dir):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axis('off')

    baseline = results['baseline']
    attack = results['attack']
    improvement = results['improvement']


    summary = f"""
    FACE IDENTIFICATION ATTACK SUMMARY
    {'='*60}

    TARGET SYSTEM
       Model: FaceNet (InceptionResnetV1)
       Task: Face Identification (50 identities)
       Dataset: LFW (Labeled Faces in the Wild)
       Matching: Embedding-based (Production-Realistic)

    {'─'*60}

    BASELINE PERFORMANCE
       Accuracy: {baseline['accuracy']*100:.2f}%
       Identity Confusion Rate: {baseline['identity_confusion_rate']*100:.2f}%
       Misidentified: {baseline['misidentified_count']}/{baseline['total_samples']}

    {'─'*60}

    ATTACK CONFIGURATION
       Attack Type: Bit Flip (u-μP-aware)
       Bits Flipped: {attack['bits_flipped']}
       Attack Mode: Untargeted

    {'─'*60}

    ATTACK RESULTS
       Final Accuracy: {attack['accuracy']*100:.2f}%
       Final ICR: {attack['identity_confusion_rate']*100:.2f}%
       Misidentified: {attack['misidentified_count']}/{attack['total_samples']}

    {'─'*60}

    IMPACT METRICS
       Accuracy Drop: {improvement['accuracy_drop']*100:.2f}%
       ICR Increase: +{improvement['icr_increase']*100:.2f}%
       Bit Efficiency: {attack['bit_efficiency']*100:.2f}% ICR per bit
       Additional Misidentifications: +{attack['misidentified_count'] - baseline['misidentified_count']}

    {'='*60}

    SECURITY IMPLICATION
       With just {attack['bits_flipped']} bit flips, the attack causes
       {attack['misidentified_count']} people to be incorrectly identified
       (from {baseline['misidentified_count']} baseline errors).

       This demonstrates a critical vulnerability in production
       face recognition systems to bit-level fault injection.
    """

    ax.text(0.5, 0.5, summary, transform=ax.transAxes,
            fontsize=10, verticalalignment='center', horizontalalignment='center',
            family='monospace',
            bbox=dict(boxstyle='round,pad=1', facecolor='#ecf0f1', edgecolor='#34495e', linewidth=2))

    plt.tight_layout()
    output_file = output_dir / 'fig8_attack_summary_card.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_file}")


def create_latex_table(results, output_dir):
    baseline = results['baseline']
    attack = results['attack']
    improvement = results['improvement']

    latex = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Identity Confusion Attack Results on FaceNet}}
\\label{{tab:identification_attack_results}}
\\begin{{tabular}}{{|l|c|c|c|}}
\\hline
\\textbf{{Metric}} & \\textbf{{Baseline}} & \\textbf{{After Attack}} & \\textbf{{Change}} \\\\
\\hline
Accuracy (\\%) & {baseline['accuracy']*100:.2f} & {attack['accuracy']*100:.2f} & {improvement['accuracy_drop']*100:.2f} \\\\
ICR (\\%) & {baseline['identity_confusion_rate']*100:.2f} & {attack['identity_confusion_rate']*100:.2f} & +{improvement['icr_increase']*100:.2f} \\\\
Misidentified & {baseline['misidentified_count']} & {attack['misidentified_count']} & +{attack['misidentified_count'] - baseline['misidentified_count']} \\\\
Bits Flipped & - & {attack['bits_flipped']} & - \\\\
Bit Efficiency (\\% ICR/bit) & - & {attack['bit_efficiency']*100:.2f} & - \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""

    output_file = output_dir / 'table_results.tex'
    with open(output_file, 'w') as f:
        f.write(latex)
    print(f"✓ Saved: {output_file}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python create_publication_plots_identification.py results/face_identification_attack_V2_TIMESTAMP/")
        print("\nAvailable results directories:")
        for d in sorted(Path('results').glob('face_identification_attack_*')):
            print(f"  {d}")
        sys.exit(1)

    result_dir = sys.argv[1]
    result_path = Path(result_dir)

    print(f"\n{'='*80}")
    print(f"Generating Publication Plots for: {result_dir}")
    print(f"{'='*80}\n")

    output_dir = result_path / 'publication_plots'
    output_dir.mkdir(exist_ok=True)

    print("Loading results...")
    results, baseline_cm, attack_cm = load_results(result_dir)
    print(f" Loaded results from {result_dir}")
    print()

    print("Generating publication-quality figures...")
    print()

    plot_accuracy_icr_comparison(results, output_dir)
    plot_confusion_matrices(baseline_cm, attack_cm, output_dir)
    plot_confusion_difference(baseline_cm, attack_cm, results, output_dir)
    plot_attack_impact_metrics(results, output_dir)
    plot_comprehensive_summary(results, baseline_cm, attack_cm, output_dir)

    plot_comparison_table(results, output_dir)
    plot_icr_impact(results, output_dir)
    plot_attack_summary_card(results, output_dir)

    create_latex_table(results, output_dir)

    print()
    print(f"{'='*80}")
    print(f" ALL PUBLICATION FIGURES READY!")
    print(f"{'='*80}")
    print(f"\nOutput directory: {output_dir}/")
    print("\nGenerated files:")
    print("  • fig1_accuracy_icr_comparison.png    - Main accuracy & ICR comparison")
    print("  • fig2_confusion_matrices.png         - Baseline vs attack confusion matrices")
    print("  • fig3_confusion_difference.png       - Attack-induced confusion")
    print("  • fig4_attack_impact_metrics.png      - 4-panel impact analysis")
    print("  • fig5_comprehensive_summary.png      - 6-panel complete summary")
    print("  • fig6_comparison_table.png           - Visual comparison table")
    print("  • fig7_icr_impact.png                 - ICR impact visualization")
    print("  • fig8_attack_summary_card.png        - Attack summary card")
    print("  • table_results.tex                   - LaTeX table")
    print(f"\n✓ Total: 8 publication-quality figures + 1 LaTeX table")
    print("\nUse these in your paper!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
