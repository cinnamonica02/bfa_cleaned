import os
import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.5)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['grid.alpha'] = 0.3


def find_latest_results_dir(base_dir='results'):
    results_path = Path(base_dir)
    if not results_path.exists():
        return None
    
    lfw_dirs = sorted(results_path.glob('lfw_face_attack_*'))
    if not lfw_dirs:
        return None
    
    return lfw_dirs[-1]


def load_results(results_dir):
    results_path = Path(results_dir)
    
    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    comprehensive_file = results_path / 'comprehensive_results.json'
    if comprehensive_file.exists():
        with open(comprehensive_file, 'r') as f:
            data = json.load(f)
        print(f"Loaded results from: {comprehensive_file}")
        return data
    
    comparison_file = results_path / 'final_comparison.json'
    if comparison_file.exists():
        with open(comparison_file, 'r') as f:
            data = json.load(f)
        print(f"Loaded results from: {comparison_file}")
        return data
    
    raise FileNotFoundError(f"No results files found in {results_dir}")


def create_asr_accuracy_plot(results, output_dir):
    fig, ax = plt.subplots(figsize=(10, 7))
    baseline = results.get('baseline_metrics', {})
    attack_res = results.get('attack_results', {})
    initial_asr = baseline.get('privacy_leak_rate', 0) * 100
    final_asr = attack_res.get('final_asr', 0) * 100
    initial_acc = baseline.get('accuracy', 0) * 100
    final_acc = attack_res.get('final_accuracy', 0) * 100
    
    ax.plot([0, 1], [initial_asr, final_asr], 'ro-', 
           linewidth=3, markersize=12, label='Attack Success Rate (ASR)')
    
    ax.plot([0, 1], [initial_acc, final_acc], 'bo-', 
           linewidth=3, markersize=12, label='Model Accuracy')
    
    ax.annotate(f'{initial_asr:.2f}%', xy=(0, initial_asr), 
               xytext=(-0.15, initial_asr), fontsize=12, fontweight='bold', color='darkred')
    ax.annotate(f'{final_asr:.2f}%', xy=(1, final_asr), 
               xytext=(1.05, final_asr), fontsize=12, fontweight='bold', color='darkred')
    ax.annotate(f'{initial_acc:.2f}%', xy=(0, initial_acc), 
               xytext=(-0.15, initial_acc), fontsize=12, fontweight='bold', color='darkblue')
    ax.annotate(f'{final_acc:.2f}%', xy=(1, final_acc), 
               xytext=(1.05, final_acc), fontsize=12, fontweight='bold', color='darkblue')
    
    
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(0, 105)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Before Attack', 'After Attack'], fontsize=14, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=14, fontweight='bold')
    ax.set_title('Attack Success Rate and Model Accuracy', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='center', fontsize=12, framealpha=0.9)
    ax.grid(True, axis='y', alpha=0.3)
    
    asr_improvement = final_asr - initial_asr
    acc_drop = initial_acc - final_acc
    bits = attack_res.get('bits_flipped', 0)
    
    summary_text = f'ASR Improvement: +{asr_improvement:.2f}%\nAccuracy Drop: {acc_drop:.2f}%\nBits Flipped: {bits}'
    ax.text(0.5, 0.05, summary_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='bottom', horizontalalignment='center',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'attack_before_after.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / 'attack_before_after.png'}")


def create_comparison_table(results, output_dir):
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    lit_data = [
        ['CIFAR-10', 'AlexNet', '61M', '87.70', '86.74', '89.27', '11'],
        ['CIFAR-10', 'VGG-11', '132M', '88.14', '83.50', '93.13', '20'],
        ['CIFAR-10', 'VGG-16', '138M', '88.35', '84.51', '91.44', '14'],
        ['ImageNet', 'ResNet-50', '23M', '76.03', '72.53', '84.67', '27'],
        ['ImageNet', 'ViT-B', '86M', '78.89', '76.63', '89.37', '85'],
    ]
    
    baseline = results.get('baseline_metrics', {})
    attack_res = results.get('attack_results', {})
    exp_info = results.get('experiment_info', {})
    
    our_data = [
        exp_info.get('dataset', 'LFW+CIFAR-10'),
        exp_info.get('model', 'ResNet-32'),
        '11M',
        f"{baseline.get('accuracy', 0)*100:.2f}",
        f"{attack_res.get('final_accuracy', 0)*100:.2f}",
        f"{attack_res.get('final_asr', 0)*100:.2f}",
        str(attack_res.get('bits_flipped', 0))
    ]
    
    table_data = lit_data + [our_data]
    
    table = ax.table(
        cellText=table_data,
        colLabels=['Dataset', 'Architecture', 'Network\nParameters', 
                  'ACC. before\nAttack (%)', 'ACC. after\nAttack (%)', 
                  'Attack Success\nRate (%)', '# of\nBit Flips'],
        cellLoc='center',
        loc='center',
        colWidths=[0.15, 0.15, 0.12, 0.14, 0.14, 0.15, 0.10]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.8)
    
    for i in range(7):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    our_row = len(table_data)
    for i in range(7):
        table[(our_row, i)].set_facecolor('#E8F4F8')
        table[(our_row, i)].set_text_props(weight='bold')
    
    for i in range(1, our_row):
        for j in range(7):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F5F5F5')
    
    plt.title('Performance Comparison with Literature (Groan, USENIX Security 2024)', 
             fontsize=14, fontweight='bold', pad=20)
    
    plt.savefig(output_dir / 'comparison_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / 'comparison_table.png'}")


def create_metrics_summary(results, output_dir):
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    baseline = results.get('baseline_metrics', {})
    attack_res = results.get('attack_results', {})
    
    ax1 = fig.add_subplot(gs[0, 0])
    metrics = ['Accuracy', 'ASR']
    before = [baseline.get('accuracy', 0)*100, baseline.get('privacy_leak_rate', 0)*100]
    after = [attack_res.get('final_accuracy', 0)*100, attack_res.get('final_asr', 0)*100]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, before, width, label='Before Attack', 
                   color='#2E86AB', alpha=0.8, edgecolor='black')
    bars2 = ax1.bar(x + width/2, after, width, label='After Attack', 
                   color='#C73E1D', alpha=0.8, edgecolor='black')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax1.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax1.set_title('a) Metrics Before and After Attack', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, axis='y', alpha=0.3)
    ax1.set_ylim(0, 105)
    
    ax2 = fig.add_subplot(gs[0, 1])
    impact_metrics = ['ASR\nIncrease', 'Accuracy\nDrop', 'Bits\nFlipped']
    impact_values = [
        attack_res.get('asr_improvement', 0) * 100,
        attack_res.get('accuracy_drop', 0) * 100,
        attack_res.get('bits_flipped', 0)
    ]
    colors = ['#2E86AB', '#C73E1D', '#6A994E']
    
    bars = ax2.bar(impact_metrics, impact_values, color=colors, alpha=0.8, edgecolor='black')
    
    for bar, val in zip(bars, impact_values):
        height = bar.get_height()
        if val > 20:
            ax2.text(bar.get_x() + bar.get_width()/2., height/2,
                    f'{val:.1f}', ha='center', va='center', 
                    fontsize=14, fontweight='bold', color='white')
        else:
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}', ha='center', va='bottom', 
                    fontsize=11, fontweight='bold')
    
    ax2.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax2.set_title('b) Attack Impact Metrics', fontsize=13, fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.3)
    
    ax3 = fig.add_subplot(gs[1, 0])
    acc_drop = attack_res.get('accuracy_drop', 0) * 100
    bits = attack_res.get('bits_flipped', 0)
    asr = attack_res.get('final_asr', 0) * 100
    
    stealth_score = (asr / 100) * (1 - acc_drop/100) * (30 / max(bits, 1))
    stealth_score = min(stealth_score * 100, 100) 
    
    theta = np.linspace(0, np.pi, 100)
    r = np.ones_like(theta)
    
    colors_segment = ['red', 'orange', 'yellow', 'lightgreen', 'green']
    n_segments = len(colors_segment)
    for i in range(n_segments):
        start = i * np.pi / n_segments
        end = (i + 1) * np.pi / n_segments
        mask = (theta >= start) & (theta <= end)
        ax3.fill_between(theta[mask], 0, r[mask], color=colors_segment[i], alpha=0.3)
    
    score_angle = (1 - stealth_score/100) * np.pi
    ax3.arrow(0, 0, 0.8*np.cos(score_angle), 0.8*np.sin(score_angle),
             head_width=0.1, head_length=0.1, fc='black', ec='black', linewidth=3)
    
    ax3.set_xlim(-1.2, 1.2)
    ax3.set_ylim(0, 1.2)
    ax3.axis('off')
    ax3.set_title('c) Stealth Score', fontsize=13, fontweight='bold', pad=20)
    ax3.text(0, -0.3, f'{stealth_score:.1f}/100', ha='center', 
            fontsize=20, fontweight='bold')
    
    ax4 = fig.add_subplot(gs[1, 1])
    methods = ['AlexNet\n(Groan)', 'VGG-11\n(Groan)', 'VGG-16\n(Groan)', 
               'Our Method\n(ResNet-32)']
    asr_lit = [89.27, 93.13, 91.44, attack_res.get('final_asr', 0)*100]
    acc_drop_lit = [0.96, 4.64, 3.84, attack_res.get('accuracy_drop', 0)*100]
    colors_lit = ['gray', 'gray', 'gray', '#2E86AB']
    
    bars = ax4.barh(methods, asr_lit, color=colors_lit, alpha=0.8, edgecolor='black')
    
    for i, (asr_val, drop) in enumerate(zip(asr_lit, acc_drop_lit)):
        ax4.text(asr_val + 2, i, f'Δacc: {drop:.2f}%', 
                va='center', fontsize=10, fontweight='bold')
    
    ax4.set_xlabel('Attack Success Rate (%)', fontsize=12, fontweight='bold')
    ax4.set_title('d) ASR Comparison with Literature', fontsize=13, fontweight='bold')
    ax4.grid(True, axis='x', alpha=0.3)
    ax4.set_xlim(0, 100)
    
    plt.suptitle('Comprehensive Attack Results Summary', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(output_dir / 'comprehensive_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / 'comprehensive_summary.png'}")


def create_detailed_metrics_table(results, output_dir):
    baseline = results.get('baseline_metrics', {})
    attack_res = results.get('attack_results', {})
    exp_info = results.get('experiment_info', {})
    
    metrics_data = {
        'Metric': [
            'Overall Accuracy (%)',
            'Face Detection Rate (%)',
            'Privacy Leak Rate (%)',
            'Attack Success Rate (%)',
            'Faces Missed',
            'Total Faces Tested',
            'Bits Flipped',
            'Execution Time (s)',
            'Generations',
            'Population Size'
        ],
        'Before Attack': [
            f"{baseline.get('accuracy', 0)*100:.2f}",
            f"{baseline.get('face_recall', 0)*100:.2f}",
            f"{baseline.get('privacy_leak_rate', 0)*100:.2f}",
            'N/A',
            baseline.get('faces_missed', 0),
            baseline.get('total_faces', 0),
            '0',
            'N/A',
            'N/A',
            'N/A'
        ],
        'After Attack': [
            f"{attack_res.get('final_accuracy', 0)*100:.2f}",
            'N/A',
            f"{attack_res.get('final_asr', 0)*100:.2f}",
            f"{attack_res.get('final_asr', 0)*100:.2f}",
            'N/A',
            'N/A',
            str(attack_res.get('bits_flipped', 0)),
            f"{attack_res.get('execution_time', 0):.2f}",
            str(exp_info.get('generations', 'N/A')),
            str(exp_info.get('population_size', 'N/A'))
        ],
        'Change': [
            f"{(attack_res.get('final_accuracy', 0) - baseline.get('accuracy', 0))*100:+.2f}",
            'N/A',
            f"{(attack_res.get('final_asr', 0) - baseline.get('privacy_leak_rate', 0))*100:+.2f}",
            'N/A',
            'N/A',
            'N/A',
            f"+{attack_res.get('bits_flipped', 0)}",
            'N/A',
            'N/A',
            'N/A'
        ]
    }
    
    df = pd.DataFrame(metrics_data)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values, colLabels=df.columns,
                    cellLoc='center', loc='center',
                    colWidths=[0.4, 0.2, 0.2, 0.2])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F5F5F5')
    
    plt.title('Detailed Attack Metrics', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(output_dir / 'detailed_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / 'detailed_metrics.png'}")


def generate_all_visualizations(results_dir):
    print("\n" + "="*80)
    print("GENERATING PUBLICATION-QUALITY VISUALIZATIONS")
    print("="*80 + "\n")
    
    results = load_results(results_dir)
    
    output_dir = Path(results_dir) / 'visualizations'
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nGenerating visualizations in: {output_dir}\n")
    
    create_asr_accuracy_plot(results, output_dir)
    create_comparison_table(results, output_dir)
    create_metrics_summary(results, output_dir)
    create_detailed_metrics_table(results, output_dir)
    
    print("\n" + "="*80)
    print(f"✓ All visualizations saved to: {output_dir}/")
    print("="*80)
    print("\nGenerated files:")
    print("  1. attack_before_after.png - Classic ASR vs Accuracy plot")
    print("  2. comparison_table.png - Literature benchmark comparison")
    print("  3. comprehensive_summary.png - 4-panel summary figure")
    print("  4. detailed_metrics.png - Detailed metrics table")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Generate publication-quality visualizations from LFW attack results'
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default=None,
        help='Path to results directory (default: use most recent)'
    )
    
    args = parser.parse_args()
    
    if args.results_dir:
        results_dir = args.results_dir
    else:
        results_dir = find_latest_results_dir()
        if results_dir is None:
            print("Error: No results directory found!")
            print("Please run lfw_face_attack.py first or specify --results_dir")
            return
        print(f"Using most recent results: {results_dir}")
    
    generate_all_visualizations(results_dir)


if __name__ == "__main__":
    main()

