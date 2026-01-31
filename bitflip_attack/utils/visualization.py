import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_asr_accuracy(initial_asr, final_asr, original_acc, final_acc, 
                     timestamp=None, output_dir="results"):

    plt.figure(figsize=(10, 6))
    

    plt.style.use('seaborn-v0_8-whitegrid')
    

    plt.plot(['Initial', 'Final'], [initial_asr, final_asr], 'ro-', 
             linewidth=2, markersize=10, label='Attack Success Rate (ASR)')
    

    plt.plot(['Initial', 'Final'], [original_acc, final_acc], 'bo-', 
             linewidth=2, markersize=10, label='Model Accuracy')
    

    plt.annotate(f"{initial_asr:.4f}", xy=(0, initial_asr), xytext=(0, initial_asr+0.03),
                ha='center', fontsize=12, color='darkred')
    plt.annotate(f"{final_asr:.4f}", xy=(1, final_asr), xytext=(1, final_asr+0.03),
                ha='center', fontsize=12, color='darkred')
    
    plt.annotate(f"{original_acc:.4f}", xy=(0, original_acc), xytext=(0, original_acc-0.05),
                ha='center', fontsize=12, color='darkblue')
    plt.annotate(f"{final_acc:.4f}", xy=(1, final_acc), xytext=(1, final_acc-0.05),
                ha='center', fontsize=12, color='darkblue')
    

    plt.title('Attack Success Rate and Model Accuracy', fontsize=16)
    plt.ylabel('Rate', fontsize=14)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    

    asr_improvement = final_asr - initial_asr
    acc_drop = original_acc - final_acc
    
    plt.figtext(0.5, 0.01, 
                f"ASR Improvement: {asr_improvement:.4f} | Accuracy Drop: {acc_drop:.4f}",
                ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    

    if timestamp is None:
        timestamp = "latest"
    plt.savefig(os.path.join(output_dir, f"asr_accuracy_plot_{timestamp}.png"), dpi=300)
    plt.close()


def plot_bit_flip_distribution(flipped_bits, timestamp=None, output_dir="results"):


    df = pd.DataFrame(flipped_bits)
    

    layer_counts = df['Layer'].value_counts().reset_index()
    layer_counts.columns = ['Layer', 'Count']
    
    plt.figure(figsize=(12, 6))
    

    ax = sns.barplot(x='Layer', y='Count', data=layer_counts)
    

    if len(layer_counts) > 5:
        plt.xticks(rotation=45, ha='right')
    

    for i, count in enumerate(layer_counts['Count']):
        ax.text(i, count + 0.1, str(count), ha='center')
    
    plt.title('Distribution of Bit Flips Across Layers', fontsize=16)
    plt.xlabel('Layer', fontsize=14)
    plt.ylabel('Number of Bits Flipped', fontsize=14)
    plt.tight_layout()
    

    if timestamp is None:
        timestamp = "latest"
    plt.savefig(os.path.join(output_dir, f"bit_flip_distribution_{timestamp}.png"), dpi=300)
    plt.close()


def plot_bit_position_distribution(flipped_bits, timestamp=None, output_dir="results"):

    df = pd.DataFrame(flipped_bits)
    

    if 'Bit Position' not in df.columns and 'bit_position' in df.columns:
        df['Bit Position'] = df['bit_position']
    
    if 'Bit Position' in df.columns:

        position_counts = df['Bit Position'].value_counts().reset_index()
        position_counts.columns = ['Bit Position', 'Count']
        

        position_counts = position_counts.sort_values('Bit Position')
        
        plt.figure(figsize=(12, 6))
        

        ax = sns.barplot(x='Bit Position', y='Count', data=position_counts)
        

        for i, count in enumerate(position_counts['Count']):
            ax.text(i, count + 0.1, str(count), ha='center')
        
        plt.title('Distribution of Flipped Bit Positions', fontsize=16)
        plt.xlabel('Bit Position', fontsize=14)
        plt.ylabel('Number of Bits Flipped', fontsize=14)
        plt.tight_layout()
        

        if timestamp is None:
            timestamp = "latest"
        plt.savefig(os.path.join(output_dir, f"bit_position_distribution_{timestamp}.png"), dpi=300)
        plt.close()


def create_attack_summary_report(results, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    

    html = f"""
    <html>
    <head>
        <title>Bit Flip Attack Results</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #333366; }}
            .metrics {{ display: flex; flex-wrap: wrap; }}
            .metric-card {{ 
                background-color: #f8f9fa; 
                border-radius: 8px; 
                padding: 15px; 
                margin: 10px; 
                width: 200px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
            .metric-value {{ 
                font-size: 24px; 
                font-weight: bold;
                margin: 10px 0;
            }}
            .good {{ color: #28a745; }}
            .bad {{ color: #dc3545; }}
            .neutral {{ color: #007bff; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
            tr:hover {{ background-color: #f5f5f5; }}
            .images {{ display: flex; flex-wrap: wrap; justify-content: center; }}
            .image-container {{ margin: 10px; }}
        </style>
    </head>
    <body>
        <h1>Bit Flip Attack Results</h1>
        <p>Timestamp: {timestamp}</p>
        
        <h2>Attack Metrics</h2>
        <div class="metrics">
            <div class="metric-card">
                <h3>Original Accuracy</h3>
                <div class="metric-value neutral">{results['original_accuracy']:.4f}</div>
            </div>
            <div class="metric-card">
                <h3>Final Accuracy</h3>
                <div class="metric-value {('bad' if results['accuracy_drop'] > 0.1 else 'neutral')}">{results['final_accuracy']:.4f}</div>
            </div>
            <div class="metric-card">
                <h3>Accuracy Drop</h3>
                <div class="metric-value {('bad' if results['accuracy_drop'] > 0.1 else 'good')}">{results['accuracy_drop']:.4f}</div>
            </div>
            <div class="metric-card">
                <h3>Initial ASR</h3>
                <div class="metric-value neutral">{results['initial_asr']:.4f}</div>
            </div>
            <div class="metric-card">
                <h3>Final ASR</h3>
                <div class="metric-value {('good' if results['final_asr'] > 0.5 else 'neutral')}">{results['final_asr']:.4f}</div>
            </div>
            <div class="metric-card">
                <h3>ASR Improvement</h3>
                <div class="metric-value {('good' if results['asr_improvement'] > 0.3 else 'neutral')}">{results['asr_improvement']:.4f}</div>
            </div>
            <div class="metric-card">
                <h3>Bits Flipped</h3>
                <div class="metric-value neutral">{results['bits_flipped']}</div>
            </div>
            <div class="metric-card">
                <h3>Execution Time</h3>
                <div class="metric-value neutral">{results['execution_time']:.2f}s</div>
            </div>
        </div>
        
        <h2>Flipped Bits</h2>
        <table>
            <tr>
                <th>#</th>
                <th>Layer</th>
                <th>Parameter</th>
                <th>Bit Position</th>
                <th>Original Value</th>
                <th>New Value</th>
            </tr>
    """
    

    for i, bit in enumerate(results['flipped_bits']):
        html += f"""
            <tr>
                <td>{i+1}</td>
                <td>{bit['Layer']}</td>
                <td>{bit['Parameter']}</td>
                <td>{bit['Bit Position']}</td>
                <td>{bit.get('Original Value', 'N/A')}</td>
                <td>{bit.get('New Value', 'N/A')}</td>
            </tr>
        """
    
    html += """
        </table>
        
        <h2>Visualizations</h2>
        <div class="images">
            <div class="image-container">
                <img src="asr_accuracy_plot_TIMESTAMP.png" alt="ASR and Accuracy" width="600">
            </div>
        </div>
    </body>
    </html>
    """.replace("TIMESTAMP", timestamp)
    

    with open(os.path.join(output_dir, f"attack_report_{timestamp}.html"), "w") as f:
        f.write(html)
    

    plot_asr_accuracy(
        results['initial_asr'], results['final_asr'],
        results['original_accuracy'], results['final_accuracy'],
        timestamp, output_dir
    )
    
    plot_bit_flip_distribution(results['flipped_bits'], timestamp, output_dir)
    plot_bit_position_distribution(results['flipped_bits'], timestamp, output_dir)
    
    print(f"Comprehensive report saved to {output_dir}/attack_report_{timestamp}.html") 