import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from glob import glob

def load_results(results_dir):
    attack_results_files = glob(os.path.join(results_dir, "attack_results_*.csv"))
    flipped_bits_files = glob(os.path.join(results_dir, "flipped_bits_*.csv"))
    plot_files = glob(os.path.join(results_dir, "asr_accuracy_plot_*.png"))
    
    results = {
        "attack_results": [],
        "flipped_bits": [],
        "plot_files": plot_files
    }
    

    for file in attack_results_files:
        df = pd.read_csv(file)

        timestamp = os.path.basename(file).replace("attack_results_", "").replace(".csv", "")
        df["timestamp"] = timestamp
        results["attack_results"].append(df)
    

    for file in flipped_bits_files:
        df = pd.read_csv(file)

        timestamp = os.path.basename(file).replace("flipped_bits_", "").replace(".csv", "")
        df["timestamp"] = timestamp
        results["flipped_bits"].append(df)
    

    if results["attack_results"]:
        results["attack_results"] = pd.concat(results["attack_results"])
    if results["flipped_bits"]:
        results["flipped_bits"] = pd.concat(results["flipped_bits"])
    
    return results

def visualize_attack_success(results):
    if not isinstance(results["attack_results"], pd.DataFrame) or results["attack_results"].empty:
        print("No attack results found.")
        return
    

    attack_df = results["attack_results"]
    

    plt.figure(figsize=(12, 8))
    

    metrics = ["Initial ASR", "Final ASR", "Original Accuracy", "Final Accuracy", "Accuracy Drop"]
    metrics_data = []
    
    for timestamp in attack_df["timestamp"].unique():
        timestamp_df = attack_df[attack_df["timestamp"] == timestamp]
        row_data = {"timestamp": timestamp}
        
        for metric in metrics:
            value = timestamp_df[timestamp_df["Metric"] == metric]["Value"].values[0]
            row_data[metric] = value
        
        metrics_data.append(row_data)
    
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df = metrics_df.set_index("timestamp")
    

    ax = metrics_df[metrics].plot(kind="bar", figsize=(12, 6))
    plt.title("Attack Metrics Comparison")
    plt.ylabel("Value")
    plt.xlabel("Timestamp")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(title="Metric")
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(results_dir), "attack_metrics_comparison.png"), dpi=300)
    

    plt.figure(figsize=(8, 6))
    plt.scatter(metrics_df["Final ASR"], metrics_df["Accuracy Drop"], 
               s=metrics_df["Bits Flipped"] * 5 if "Bits Flipped" in metrics_df.columns else 50, 
               alpha=0.7)
    

    for idx, row in metrics_df.iterrows():
        plt.annotate(idx, 
                    (row["Final ASR"], row["Accuracy Drop"]),
                    xytext=(5, 5), textcoords="offset points")
    
    plt.title("Attack Success Rate vs. Accuracy Drop")
    plt.xlabel("Final Attack Success Rate")
    plt.ylabel("Accuracy Drop")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(results_dir), "asr_vs_accuracy_drop.png"), dpi=300)
    
    print(f"Saved attack metrics comparison to {os.path.join(os.path.dirname(results_dir), 'attack_metrics_comparison.png')}")
    print(f"Saved ASR vs accuracy drop plot to {os.path.join(os.path.dirname(results_dir), 'asr_vs_accuracy_drop.png')}")

def analyze_flipped_bits(results):
    if not isinstance(results["flipped_bits"], pd.DataFrame) or results["flipped_bits"].empty:
        print("No flipped bits details found.")
        return
    
    flipped_bits_df = results["flipped_bits"]
    

    plt.figure(figsize=(12, 6))
    param_counts = flipped_bits_df["Parameter"].value_counts()
    

    if len(param_counts) > 20:
        param_counts = param_counts.head(20)
    
    ax = param_counts.plot(kind="bar")
    plt.title("Distribution of Flipped Bits by Parameter")
    plt.xlabel("Parameter")
    plt.ylabel("Count")
    plt.xticks(rotation=90)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(results_dir), "flipped_bits_by_parameter.png"), dpi=300)
    

    plt.figure(figsize=(10, 6))
    sns.histplot(flipped_bits_df["Bit Position"], bins=32, kde=True)
    plt.title("Distribution of Flipped Bit Positions")
    plt.xlabel("Bit Position")
    plt.ylabel("Count")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(results_dir), "flipped_bits_position_distribution.png"), dpi=300)
    
    print(f"Saved flipped bits by parameter plot to {os.path.join(os.path.dirname(results_dir), 'flipped_bits_by_parameter.png')}")
    print(f"Saved bit position distribution plot to {os.path.join(os.path.dirname(results_dir), 'flipped_bits_position_distribution.png')}")

def generate_report(results, results_dir):
    if not isinstance(results["attack_results"], pd.DataFrame) or results["attack_results"].empty:
        print("No attack results found.")
        return
    
    report_file = os.path.join(os.path.dirname(results_dir), "attack_report.md")
    
    with open(report_file, "w") as f:
        f.write("# Bit Flipping Attack Report\n\n")
        

        f.write("## Summary Statistics\n\n")
        
        attack_df = results["attack_results"]
        timestamps = attack_df["timestamp"].unique()
        
        f.write(f"Number of attacks: {len(timestamps)}\n\n")
        

        f.write("| Timestamp | Initial ASR | Final ASR | ASR Improvement | Original Accuracy | Final Accuracy | Accuracy Drop | Bits Flipped |\n")
        f.write("|-----------|-------------|-----------|-----------------|-------------------|---------------|--------------|-------------|\n")
        
        for timestamp in timestamps:
            timestamp_df = attack_df[attack_df["timestamp"] == timestamp]
            
            initial_asr = timestamp_df[timestamp_df["Metric"] == "Initial ASR"]["Value"].values[0]
            final_asr = timestamp_df[timestamp_df["Metric"] == "Final ASR"]["Value"].values[0]
            asr_improvement = final_asr - initial_asr
            
            original_acc = timestamp_df[timestamp_df["Metric"] == "Original Accuracy"]["Value"].values[0]
            final_acc = timestamp_df[timestamp_df["Metric"] == "Final Accuracy"]["Value"].values[0]
            acc_drop = timestamp_df[timestamp_df["Metric"] == "Accuracy Drop"]["Value"].values[0]
            
            bits_flipped = timestamp_df[timestamp_df["Metric"] == "Bits Flipped"]["Value"].values[0]
            
            f.write(f"| {timestamp} | {initial_asr:.4f} | {final_asr:.4f} | {asr_improvement:.4f} | {original_acc:.4f} | {final_acc:.4f} | {acc_drop:.4f} | {bits_flipped:.0f} |\n")
        
        f.write("\n")
        

        f.write("## Visualizations\n\n")
        
        f.write("### Attack Metrics Comparison\n\n")
        f.write(f"![Attack Metrics Comparison](attack_metrics_comparison.png)\n\n")
        
        f.write("### ASR vs Accuracy Drop\n\n")
        f.write(f"![ASR vs Accuracy Drop](asr_vs_accuracy_drop.png)\n\n")
        
        if isinstance(results["flipped_bits"], pd.DataFrame) and not results["flipped_bits"].empty:
            f.write("### Flipped Bits by Parameter\n\n")
            f.write(f"![Flipped Bits by Parameter](flipped_bits_by_parameter.png)\n\n")
            
            f.write("### Bit Position Distribution\n\n")
            f.write(f"![Bit Position Distribution](flipped_bits_position_distribution.png)\n\n")
        

        if results["plot_files"]:
            f.write("### ASR and Accuracy Progression\n\n")
            
            for plot_file in results["plot_files"]:
                timestamp = os.path.basename(plot_file).replace("asr_accuracy_plot_", "").replace(".png", "")
                plot_rel_path = os.path.join("results", os.path.basename(plot_file))
                f.write(f"#### Timestamp: {timestamp}\n\n")
                f.write(f"![ASR and Accuracy Progression]({plot_rel_path})\n\n")
    
    print(f"Generated attack report at {report_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Bit Flipping Attack Results")
    parser.add_argument("--results_dir", type=str, default="results",
                      help="Directory containing attack results")
    
    args = parser.parse_args()
    results_dir = args.results_dir
    
    if not os.path.exists(results_dir):
        print(f"Error: Results directory '{results_dir}' does not exist.")
        exit(1)
    

    results = load_results(results_dir)
    

    visualize_attack_success(results)
    

    analyze_flipped_bits(results)
    

    generate_report(results, results_dir) 