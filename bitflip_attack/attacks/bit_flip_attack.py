"""
Bit Flip Attack implementation - Main module

This module contains the main BitFlipAttack class which orchestrates the attack.
Implementation is decomposed into multiple components:
1. Initialization and information gathering
2. Layer sensitivity analysis
3. Bit candidate selection and manipulation
4. Evaluation and optimization
5. Results handling and visualization
"""
import os
import time
import torch
import numpy as np
import pandas as pd
from datetime import datetime

from bitflip_attack.attacks.helpers.sensitivity import (
    compute_sensitivity, 
    rank_layers_by_sensitivity
)
from bitflip_attack.attacks.helpers.bit_manipulation import (
    select_bit_candidates,
    flip_bit
)
from bitflip_attack.attacks.helpers.evaluation import (
    evaluate_model_performance
)
from bitflip_attack.attacks.helpers.optimization import (
    genetic_optimization
)
from bitflip_attack.utils.visualization import (
    plot_asr_accuracy
)


class BitFlipAttack:
    """
    Implementation of the Bit Flipping Attack (BFA) on deep neural networks.
    This attack identifies and flips critical bits in model parameters to induce
    misclassification with minimal changes to the model.
    """
    
    def __init__(self, model, dataset, target_asr=0.9, max_bit_flips=100, 
                 accuracy_threshold=0.05, device='cuda', 
                 attack_mode='targeted', layer_sensitivity=True, 
                 hybrid_sensitivity=True, alpha=0.5,
                 custom_forward_fn=None):
        """
        Initialize the bit flipping attack.
        
        Args:
            model: The target model to attack
            dataset: Dataset for evaluating model performance
            target_asr: Target attack success rate
            max_bit_flips: Maximum number of bits to flip
            accuracy_threshold: Maximum acceptable accuracy drop
            device: Device to run the attack on (cuda or cpu)
            attack_mode: 'targeted' or 'untargeted'
            layer_sensitivity: Whether to perform layer sensitivity analysis
            hybrid_sensitivity: Whether to use hybrid sensitivity metric
            alpha: Weight for hybrid sensitivity metric (0 for mag-only, 1 for grad-only)
            custom_forward_fn: Custom forward function for handling specific model architectures
        """
        self.model = model
        self.dataset = dataset
        self.target_asr = target_asr
        self.max_bit_flips = max_bit_flips
        self.accuracy_threshold = accuracy_threshold
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.attack_mode = attack_mode
        self.layer_sensitivity = layer_sensitivity
        self.hybrid_sensitivity = hybrid_sensitivity
        self.alpha = alpha
        self.custom_forward_fn = custom_forward_fn
        
        # Place model on the specified device
        self.model.to(self.device)
        
        # Track attack metrics
        self.original_accuracy = 0.0
        self.final_accuracy = 0.0
        self.initial_asr = 0.0
        self.final_asr = 0.0
        self.bits_flipped = 0
        self.flipped_bits_info = []
        
        # Store the model's layer structure for sensitivity analysis
        self.layer_info = self._get_layer_info()
        
        # Target specific modules if provided
        self.target_modules = None
    
    def set_target_modules(self, target_module_names):
        """
        Set specific modules to target by name.
        
        Args:
            target_module_names: List of layer name strings to target.
        """
        # Store the names for reference if needed
        self.target_modules = target_module_names 
        
        if target_module_names is not None:
            # Use the originally scanned layer_info from __init__
            initial_layer_info = self.layer_info 
            
            # Filter layer_info based on names
            filtered_info = []
            target_names_set = set(target_module_names)

            for layer in initial_layer_info: 
                if layer['name'] in target_names_set:
                    filtered_info.append(layer)
            
            if not filtered_info and target_module_names:
                 print(f"Warning: set_target_modules did not find any layers matching the provided names: {target_module_names}")

            # Overwrite self.layer_info with the filtered list
            self.layer_info = filtered_info
        else:
            # If None is passed, reset (attack all layers)
            self.layer_info = self._get_layer_info()
    
    def _get_layer_info(self):
        """
        Analyze model structure and identify quantized or attackable layers.
        
        Returns:
            List of dictionaries containing layer information
        """
        layer_info = []
        for name, module in self.model.named_modules():
            # Skip containers and non-parameter layers
            if len(list(module.children())) > 0:
                continue
                
            # Focus on layers with weights
            if hasattr(module, 'weight') and module.weight is not None:
                n_params = module.weight.numel()
                if n_params == 0:
                    continue
                
                layer_info.append({
                    'name': name,
                    'module': module,
                    'type': module.__class__.__name__,
                    'n_params': n_params,
                    'shape': module.weight.shape,
                    'requires_grad': module.weight.requires_grad,
                    'is_quantized': hasattr(module, 'weight_fake_quant') 
                                  or 'quantized' in module.__class__.__name__.lower()
                })
        
        return layer_info
    
    def perform_attack(self, target_class=0, num_candidates=1000, population_size=50, 
                      generations=20, temp_model_path=None):
        """
        Perform the bit flipping attack.
        
        Args:
            target_class: Target class for targeted attacks
            num_candidates: Number of bit candidates to evaluate
            population_size: Population size for genetic algorithm
            generations: Number of generations for optimization
            temp_model_path: Path to save temporary model checkpoint (optional)
            
        Returns:
            results: Dictionary with attack results
        """
        print(f"Starting bit flipping attack...")
        start_time = time.time()
        
        # Evaluate initial model performance
        self.original_accuracy, self.initial_asr = evaluate_model_performance(
            self.model, self.dataset, target_class, self.attack_mode, 
            self.device, self.custom_forward_fn
        )
        print(f"Initial model accuracy: {self.original_accuracy:.4f}")
        print(f"Initial attack success rate: {self.initial_asr:.4f}")
        
        # Save original model state if not provided
        if temp_model_path is None:
            temp_model_path = 'temp_original_model.pt'
        torch.save(self.model.state_dict(), temp_model_path)
        
        # Perform layer sensitivity analysis if enabled
        if self.layer_sensitivity:
            print("Performing layer sensitivity analysis...")
            sensitive_layers = rank_layers_by_sensitivity(
                self.model, self.dataset, self.layer_info, 
                self.device, self.hybrid_sensitivity, self.alpha,
                self.custom_forward_fn
            )
            
            # Print top 5 most sensitive layers
            print("\nTop 5 most sensitive layers:")
            for i, layer in enumerate(sensitive_layers[:min(5, len(sensitive_layers))]):
                print(f"{i+1}. {layer['name']} ({layer['type']}): Loss = {layer['loss']:.4f}")
            
            # Focus on the most sensitive layer
            target_layer = self.layer_info[sensitive_layers[0]['index']]
        else:
            # If not using sensitivity analysis, choose a layer with large parameter count
            sorted_layers = sorted(self.layer_info, key=lambda x: x['n_params'], reverse=True)
            target_layer = sorted_layers[0]
        
        print(f"\nTargeting layer: {target_layer['name']} ({target_layer['type']})")
        print(f"Number of parameters: {target_layer['n_params']}")
        
        # Select candidate bits for flipping
        print(f"Selecting {num_candidates} bit candidates...")
        candidates = select_bit_candidates(
            self.model, self.dataset, target_layer, num_candidates, 
            self.device, self.hybrid_sensitivity, self.alpha, 
            self.custom_forward_fn
        )
        
        # Optimize bit selection using genetic algorithm
        print("Optimizing bit selection using genetic algorithm...")
        accuracy_threshold = self.original_accuracy - self.accuracy_threshold
        
        best_solution, flip_history = genetic_optimization(
            self.model, self.dataset, candidates, self.layer_info, 
            target_class, self.attack_mode, self.max_bit_flips,
            population_size, generations, accuracy_threshold,
            self.device, self.custom_forward_fn
        )
        
        # Apply the best solution to the model
        print("Applying optimal bit flips...")
        
        # Restore original model
        self.model.load_state_dict(torch.load(temp_model_path))
        
        # Apply bit flips
        flipped_bits = []
        for idx in best_solution:
            candidate = candidates[idx]
            layer_idx = candidate['layer_idx']
            # Use layer_idx if valid, otherwise find by name
            if layer_idx >= 0:
                layer = self.layer_info[layer_idx]
            else:
                from bitflip_attack.attacks.helpers.evaluation import find_layer_by_name
                layer = find_layer_by_name(self.layer_info, candidate['layer_name'])
            param_idx = candidate['parameter_idx']
            bit_pos = candidate['bit_position']
            
            old_val, new_val = flip_bit(layer, param_idx, bit_pos)

            # --- ADD BOUNDS CHECK HERE ---
            num_elements = layer['module'].weight.numel()
            shape_for_coords = layer['module'].weight.shape
            if param_idx >= num_elements:
                print(f"Warning (in perform_attack): Correcting param_idx {param_idx} before calculating coords for shape {shape_for_coords}.")
                param_idx = param_idx % num_elements # Correct param_idx before using it for coords
                print(f"Using corrected param_idx {param_idx} for coords.")
            # --- END BOUNDS CHECK ---
            
            # Convert flat index to tensor coordinates using the corrected param_idx
            coords = np.unravel_index(param_idx, shape_for_coords)
            
            # Store bit flip information
            flipped_bits.append({
                'Layer': layer['name'],
                'Parameter': f"{coords}",
                'Bit Position': bit_pos,
                'Original Value': old_val,
                'New Value': new_val
            })
        
        # Evaluate final model performance
        self.final_accuracy, self.final_asr = evaluate_model_performance(
            self.model, self.dataset, target_class, self.attack_mode, 
            self.device, self.custom_forward_fn
        )
        print(f"Final model accuracy: {self.final_accuracy:.4f}")
        print(f"Final attack success rate: {self.final_asr:.4f}")
        
        # Track number of bits flipped
        self.bits_flipped = len(best_solution)
        self.flipped_bits_info = flipped_bits
        
        # Compute attack metrics
        accuracy_drop = self.original_accuracy - self.final_accuracy
        asr_improvement = self.final_asr - self.initial_asr
        
        print(f"Accuracy drop: {accuracy_drop:.4f}")
        print(f"ASR improvement: {asr_improvement:.4f}")
        print(f"Number of bits flipped: {self.bits_flipped}")
        
        # Compile results
        results = {
            'original_accuracy': self.original_accuracy,
            'final_accuracy': self.final_accuracy,
            'accuracy_drop': accuracy_drop,
            'initial_asr': self.initial_asr,
            'final_asr': self.final_asr,
            'asr_improvement': asr_improvement,
            'bits_flipped': self.bits_flipped,
            'flipped_bits': flipped_bits,
            'target_class': target_class,
            'execution_time': time.time() - start_time
        }
        
        return results
    
    def save_results(self, results, output_dir="results"):
        """
        Save attack results to files.
        
        Args:
            results: Dictionary with attack results
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save attack metrics
        metrics = [
            {'Metric': 'Original Accuracy', 'Value': results['original_accuracy']},
            {'Metric': 'Final Accuracy', 'Value': results['final_accuracy']},
            {'Metric': 'Accuracy Drop', 'Value': results['accuracy_drop']},
            {'Metric': 'Initial ASR', 'Value': results['initial_asr']},
            {'Metric': 'Final ASR', 'Value': results['final_asr']},
            {'Metric': 'ASR Improvement', 'Value': results['asr_improvement']},
            {'Metric': 'Bits Flipped', 'Value': results['bits_flipped']},
            {'Metric': 'Execution Time (s)', 'Value': results['execution_time']}
        ]
        
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(os.path.join(output_dir, f"attack_results_{timestamp}.csv"), index=False)
        
        # Save flipped bits details
        flipped_bits_df = pd.DataFrame(results['flipped_bits'])
        flipped_bits_df.to_csv(os.path.join(output_dir, f"flipped_bits_{timestamp}.csv"), index=False)
        
        # Generate ASR vs Accuracy plot
        plot_asr_accuracy(
            results['initial_asr'], results['final_asr'],
            results['original_accuracy'], results['final_accuracy'],
            timestamp, output_dir
        )
        
        print(f"Results saved to {output_dir}/") 