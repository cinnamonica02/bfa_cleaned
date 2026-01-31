#!/bin/bash

# Run script for BitFlip Attack examples

# Create directories
mkdir -p results
mkdir -p results_llm
mkdir -p results_vlm
mkdir -p results_quant_w8
mkdir -p results_quant_w4
mkdir -p results_quant_w1.58
mkdir -p results_sensitivity
mkdir -p results_progressive

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to run an example with a nice header
run_example() {
    echo -e "\n${YELLOW}====================================${NC}"
    echo -e "${YELLOW}Running: $1${NC}"
    echo -e "${YELLOW}====================================${NC}\n"
    $2
    echo -e "\n${GREEN}✓ Completed: $1${NC}\n"
}

# Parse command line arguments
EXAMPLE=$1

case $EXAMPLE in
    "clip")
        run_example "Basic CLIP Example" "python main.py --model clip --model_variant ViT-B-32 --dataset cifar10 --target_asr 0.7 --max_bit_flips 5"
        ;;
    "llm")
        run_example "LLM Attack Example" "python advanced_examples.py --example llm"
        ;;
    "vlm")
        run_example "VLM Attack Example" "python advanced_examples.py --example vlm"
        ;;
    "quantized")
        run_example "Quantized Models Attack" "python advanced_examples.py --example quantized"
        ;;
    "sensitivity")
        run_example "Layer Sensitivity Analysis" "python advanced_examples.py --example sensitivity"
        ;;
    "progressive")
        run_example "Progressive Bit Flipping" "python advanced_examples.py --example progressive"
        ;;
    "visualize")
        run_example "Visualizing Results" "python visualize_results.py --results_dir results"
        ;;
    "all")
        run_example "Basic CLIP Example" "python main.py --model clip --model_variant ViT-B-32 --dataset cifar10 --target_asr 0.7 --max_bit_flips 5"
        run_example "LLM Attack Example" "python advanced_examples.py --example llm"
        run_example "VLM Attack Example" "python advanced_examples.py --example vlm"
        run_example "Quantized Models Attack" "python advanced_examples.py --example quantized"
        run_example "Layer Sensitivity Analysis" "python advanced_examples.py --example sensitivity"
        run_example "Progressive Bit Flipping" "python advanced_examples.py --example progressive"
        run_example "Visualizing Results" "python visualize_results.py --results_dir results"
        ;;
    *)
        echo -e "${YELLOW}Available examples:${NC}"
        echo "  - clip: Basic CLIP model attack"
        echo "  - llm: Large Language Model attack"
        echo "  - vlm: Vision-Language Model attack"
        echo "  - quantized: Quantized models attack"
        echo "  - sensitivity: Layer sensitivity analysis"
        echo "  - progressive: Progressive bit flipping"
        echo "  - visualize: Visualize attack results"
        echo "  - all: Run all examples"
        echo ""
        echo "Usage: ./run.sh [example]"
        ;;
esac
