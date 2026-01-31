import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd
import json
from pathlib import Path


plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'

def load_results(results_dir):
    results_path = Path(results_dir) / 'comprehensive_results.json'
    with open(results_path, 'r') as f:
        return json.load(f)

def create_comparison_table(results, output_dir):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')




    lit_data = [
        ['GROAN', 'AlexNet', 'CIFAR-10', '87.70', '86.74', '0.96', '89.27', '11'],
        ['GROAN', 'VGG11', 'CIFAR-10', '88.14', '83.50', '4.64', '93.13', '20'],
        ['GROAN', 'VGG16', 'CIFAR-10', '88.35', '84.51', '3.84', '91.44', '14'],
        ['GROAN', 'ResNet-20', 'CIFAR-10', '84.30', '81.26', '3.04', '84.67', '17'],
        ['GROAN Avg', '-', 'CIFAR-10', '-', '-', '3.1', '89.9', '48 avg'],
    ]


    before = results['baseline_metrics']
    after = results['attack_metrics']
    summary = results['attack_info']

    our_data = [
        ['Our Method', 'ResNet-34', 'PneumoniaMNIST',
         f"{before['accuracy']*100:.2f}",
         f"{after['accuracy']*100:.2f}",
         f"{summary['accuracy_drop']*100:.2f}",
         f"{after['false_negative_rate']*100:.2f}",  # Disease miss rate
         str(summary['bits_flipped'])]
    ]



    all_data = lit_data + [['', '', '', '', '', '', '', '']] + our_data


    table = ax.table(
        cellText=all_data,
        colLabels=['Method', 'Architecture', 'Dataset',
                   'ACC Before\n(%)', 'ACC After\n(%)', 'ACC Drop\n(%)',
                   'False Negative\nRate (%)', 'Bits\nFlipped'],
        cellLoc='center',
        loc='center',
        colWidths=[0.12, 0.12, 0.13, 0.11, 0.11, 0.11, 0.13, 0.10]
    )


    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)


    for i in range(8):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')


    for i in range(1, 6):
        for j in range(8):
            table[(i, j)].set_facecolor('#f0f0f0')


    for j in range(8):
        table[(6, j)].set_facecolor('#ffffff')
        table[(6, j)].set_linewidth(0)


    for j in range(8):
        table[(7, j)].set_facecolor('#E8F4F8')
        table[(7, j)].set_text_props(weight='bold')

    plt.title('Comparison with Literature: Bit-Flip Attack Effectiveness',
              fontsize=14, weight='bold', pad=20)


    note_text = "Note: GROAN shows Backdoor ASR (trigger-based), Our method shows Disease Miss Rate (medical diagnosis degradation)\nDifferent attack types - Medical FNR values not directly comparable to GROAN ASR"
    plt.figtext(0.5, 0.02, note_text, ha='center', fontsize=8, style='italic',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))



    output_path = Path(output_dir) / 'comparison_table.png'
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f" Saved: {output_path}")
    plt.close()

def create_privacy_impact_plot(results, output_dir):
    before = results['baseline_metrics']
    after = results['attack_metrics']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    metrics = ['Accuracy', 'Disease Recall']
    before_vals = [before['accuracy']*100, before['disease_recall']*100]
    after_vals = [after['accuracy']*100, after['disease_recall']*100]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax1.bar(x - width/2, before_vals, width, label='Before Attack',
                    color='#2E86AB', alpha=0.8)
    bars2 = ax1.bar(x + width/2, after_vals, width, label='After Attack',
                    color='#C73E1D', alpha=0.8)

    ax1.set_ylabel('Rate (%)', fontsize=12, weight='bold')
    ax1.set_title('Medical Diagnosis Model Performance Before vs After Attack', fontsize=13, weight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, 105])


    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=9)


    disease_miss_before = before['false_negative_rate'] * 100
    disease_miss_after = after['false_negative_rate'] * 100
    disease_miss_increase = disease_miss_after - disease_miss_before

    bars = ax2.bar(['Before\nAttack', 'After\nAttack'],
                   [disease_miss_before, disease_miss_after],
                   color=['#6A994E', '#C73E1D'], alpha=0.8, width=0.5)

    ax2.set_ylabel('Disease Miss Rate (%)', fontsize=12, weight='bold')
    ax2.set_title('Medical Diagnosis Degradation Impact', fontsize=13, weight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0, max(disease_miss_after * 1.2, 30)])


    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=11, weight='bold')



    if disease_miss_before > 0.01:  
        increase_text = f'+{disease_miss_increase:.1f}pp\n({disease_miss_increase/disease_miss_before:.1f}x increase)'
    else:
        increase_text = f'+{disease_miss_increase:.1f}pp'


    annotation_y = disease_miss_before + disease_miss_increase/2
    if annotation_y < 2:  
        
        annotation_y = disease_miss_after - 1

    ax2.annotate(increase_text,
                xy=(0.5, annotation_y),
                xytext=(1.3, annotation_y),
                fontsize=11, weight='bold', color='#C73E1D',
                arrowprops=dict(arrowstyle='->', lw=2, color='#C73E1D'))


    plt.tight_layout()
    output_path = Path(output_dir) / 'medical_diagnosis_impact.png'
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f" Saved: {output_path}")
    plt.close()






def create_bit_analysis_plot(results, output_dir):
    if 'attack_summary' not in results or 'flipped_bits' not in results['attack_summary']:
        print(" Skipping bit analysis plot - detailed bit information not available")
        return

    flipped_bits = results['attack_summary']['flipped_bits']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))


    bit_positions = [bit['Bit Position'] for bit in flipped_bits]
    bit_counts = pd.Series(bit_positions).value_counts().sort_index()

    colors_map = {31: '#C73E1D', 23: '#F18F01', 24: '#F18F01', 25: '#F18F01',
                  27: '#2E86AB', 29: '#6A994E', 26: '#F18F01'}
    colors = [colors_map.get(pos, '#888888') for pos in bit_counts.index]

    ax1.bar(bit_counts.index, bit_counts.values, color=colors, alpha=0.8, width=0.8)
    ax1.set_xlabel('Bit Position', fontsize=12, weight='bold')
    ax1.set_ylabel('Number of Flips', fontsize=12, weight='bold')
    ax1.set_title(f'Bit Position Distribution ({len(flipped_bits)} flipped bits)', fontsize=13, weight='bold')
    ax1.grid(True, alpha=0.3, axis='y')


    ax1.axvspan(23, 26.5, alpha=0.1, color='orange', label='Exponent bits (23-26)')
    ax1.axvspan(30.5, 31.5, alpha=0.1, color='red', label='Sign bit (31)')
    ax1.legend(fontsize=9, loc='upper left')


    value_changes = []
    for bit in flipped_bits:
        orig = abs(bit['Original Value'])
        new = abs(bit['New Value'])
        change_ratio = abs(new - orig) / (orig + 1e-10)
        value_changes.append(change_ratio)

    ax2.hist(value_changes, bins=15, color='#2E86AB', alpha=0.8, edgecolor='black')
    ax2.set_xlabel('Magnitude of Value Change (ratio)', fontsize=12, weight='bold')
    ax2.set_ylabel('Number of Flips', fontsize=12, weight='bold')
    ax2.set_title('Impact Magnitude of Bit Flips', fontsize=13, weight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axvline(np.median(value_changes), color='red', linestyle='--',
                linewidth=2, label=f'Median: {np.median(value_changes):.2f}')
    ax2.legend(fontsize=10)

    plt.tight_layout()
    output_path = Path(output_dir) / 'bit_analysis.png'
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"✓ Saved: {output_path}")
    plt.close()

def create_attack_summary_card(results, output_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')

    attack_info = results['attack_info']
    before = results['baseline_metrics']
    after = results['attack_metrics']
    exp_info = results['experiment_info']


    title_text = "Bit-Flip Attack: Medical Diagnosis Model Degradation"
    ax.text(0.5, 0.95, title_text, ha='center', va='top',
            fontsize=16, weight='bold', transform=ax.transAxes)


    subtitle = f"{exp_info['model']} on {exp_info['dataset'].upper()} Dataset"
    ax.text(0.5, 0.88, subtitle, ha='center', va='top',
            fontsize=12, style='italic', transform=ax.transAxes)


    metrics = [
        ('Bits Flipped', f"{attack_info['bits_flipped']}", '#2E86AB'),
        ('Accuracy Drop', f"{attack_info['accuracy_drop']*100:.2f}%", '#F18F01'),
        ('Disease Miss Rate Increase', f"+{attack_info['fnr_increase']*100:.1f}pp", '#C73E1D'),
        ('Additional Diseases Missed', f"{int(attack_info['additional_diseases_missed'])}", '#6A994E')
    ]

    y_start = 0.70
    for i, (label, value, color) in enumerate(metrics):
        y_pos = y_start - i*0.15


        rect = mpatches.FancyBboxPatch((0.15, y_pos-0.08), 0.7, 0.10,
                                       boxstyle="round,pad=0.01",
                                       linewidth=2, edgecolor=color,
                                       facecolor='white', alpha=0.9,
                                       transform=ax.transAxes)
        ax.add_patch(rect)


        ax.text(0.25, y_pos-0.03, label, ha='left', va='center',
                fontsize=11, weight='bold', transform=ax.transAxes)
        ax.text(0.75, y_pos-0.03, value, ha='right', va='center',
                fontsize=14, weight='bold', color=color, transform=ax.transAxes)


    stats_text = (f"Before Attack: {before['accuracy']*100:.1f}% accuracy, "
                 f"{before['false_negative_rate']*100:.1f}% disease miss rate, "
                 f"{int(before['diseases_missed'])}/{int(before['total_diseases'])} diseases missed\n"
                 f"After Attack: {after['accuracy']*100:.1f}% accuracy, "
                 f"{after['false_negative_rate']*100:.1f}% disease miss rate, "
                 f"{int(after['diseases_missed'])}/{int(after['total_diseases'])} diseases missed")
    ax.text(0.5, 0.08, stats_text, ha='center', va='top',
            fontsize=9, style='italic', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.5))

    output_path = Path(output_dir) / 'attack_summary_card.png'
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Saved: {output_path}")
    plt.close()

def main():
    results_dir = 'results/medical_imaging_attack_pneumoniamnist_20260127_130813'
    output_dir = Path(results_dir) / 'publication_plots'
    output_dir.mkdir(exist_ok=True)

    print("Loading medical imaging attack results...")
    results = load_results(results_dir)

    print("\nGenerating publication-quality visualizations for medical imaging attack...")
    print("="*60)


    create_comparison_table(results, output_dir)
    create_privacy_impact_plot(results, output_dir)
    create_bit_analysis_plot(results, output_dir)
    create_attack_summary_card(results, output_dir)

    print("="*60)
    print(f"\nAll visualizations saved to: {output_dir}")
    print("\nGenerated files:")
    print("  1. comparison_table.png - Compare with GROAN literature")
    print("  2. medical_diagnosis_impact.png - Before/after medical diagnosis degradation metrics")
    print("  3. bit_analysis.png - Bit position and impact analysis (if available)")
    print("  4. attack_summary_card.png - Summary for presentations")
    print("\n Medical context: Labels show 'Disease Miss Rate' (False Negative Rate)")
    print("   This represents diagnostic failures in pneumonia detection")





if __name__ == "__main__":
    main()
