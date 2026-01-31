import os
import time
import torch
import numpy as np
import pandas as pd
import logging
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





logger = logging.getLogger(__name__)






class UmupBitFlipAttack:
    def __init__(self, model, dataset, target_asr=0.9, max_bit_flips=100, 
                 accuracy_threshold=0.05, device='cuda', 
                 attack_mode='targeted', layer_sensitivity=True, 
                 hybrid_sensitivity=True, alpha=0.5,
                 custom_forward_fn=None):
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
        self.model.to(self.device)
        self.original_accuracy = 0.0
        self.final_accuracy = 0.0
        self.initial_asr = 0.0
        self.final_asr = 0.0
        self.bits_flipped = 0
        self.flipped_bits_info = []
        
        self.layer_info = self._get_layer_info()
        
        self.target_modules = None
    



    def set_target_modules(self, modules_list):
        self.target_modules = modules_list
        if modules_list is not None:
            filtered_info = []
            for layer in self.layer_info:
                if any(layer['module'] is module for module in modules_list):
                    filtered_info.append(layer)
            self.layer_info = filtered_info
    



    def _get_layer_info(self):
        layer_info = []
        for name, module in self.model.named_modules():
            if len(list(module.children())) > 0:
                continue
                
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
        
        logger.info(f"Starting bit flipping attack...")
        start_time = time.time()
        
        self.original_accuracy, self.initial_asr = evaluate_model_performance(
            self.model, self.dataset, target_class, self.attack_mode, 
            self.device, self.custom_forward_fn
        )


        logger.info(f"Initial model accuracy: {self.original_accuracy:.4f}")
        logger.info(f"Initial attack success rate: {self.initial_asr:.4f}")
        
        if temp_model_path is None:
            temp_model_path = 'temp_original_model.pt'
        torch.save(self.model.state_dict(), temp_model_path)
        
        if self.layer_sensitivity:
            logger.info("Performing layer sensitivity analysis...")
            sensitive_layers = rank_layers_by_sensitivity(
                self.model, self.dataset, self.layer_info, 
                self.device, self.hybrid_sensitivity, self.alpha,
                self.custom_forward_fn
            )
            logger.info("\nTop 5 most sensitive layers:")
            for i, layer in enumerate(sensitive_layers[:min(5, len(sensitive_layers))]):
                logger.info(f"{i+1}. {layer['name']} ({layer['type']}): Loss = {layer['loss']:.4f}")
            
            target_layer = self.layer_info[sensitive_layers[0]['index']]
        else:
            sorted_layers = sorted(self.layer_info, key=lambda x: x['n_params'], reverse=True)
            target_layer = sorted_layers[0]
        
        logger.info(f"\nTargeting layer: {target_layer['name']} ({target_layer['type']})")
        logger.info(f"Number of parameters: {target_layer['n_params']}")
        
        logger.info(f"Selecting {num_candidates} bit candidates...")
        candidates = select_bit_candidates(
            self.model, self.dataset, target_layer, num_candidates, 
            self.device, self.hybrid_sensitivity, self.alpha, 
            self.custom_forward_fn
        )
        
        logger.info("Optimizing bit selection using genetic algorithm...")
        accuracy_threshold = self.original_accuracy - self.accuracy_threshold
        
        best_solution, flip_history = genetic_optimization(
            self.model, self.dataset, candidates, self.layer_info, 
            target_class, self.attack_mode, self.max_bit_flips,
            population_size, generations, accuracy_threshold,
            self.device, self.custom_forward_fn
        )
        
        logger.info("Applying optimal bit flips...")
        
        self.model.load_state_dict(torch.load(temp_model_path))
        
        flipped_bits = []
        for idx in best_solution:
            candidate = candidates[idx]
            layer_idx = candidate['layer_idx']
            if layer_idx >= 0:
                layer = self.layer_info[layer_idx]
            else:
                from bitflip_attack.attacks.helpers.evaluation import find_layer_by_name
                layer = find_layer_by_name(self.layer_info, candidate['layer_name'])
            param_idx = candidate['parameter_idx']
            bit_pos = candidate['bit_position']
            
            old_val, new_val = flip_bit(layer, param_idx, bit_pos)
            coords = np.unravel_index(param_idx, layer['module'].weight.shape)
            flipped_bits.append({
                'Layer': layer['name'],
                'Parameter': f"{coords}",
                'Bit Position': bit_pos,
                'Original Value': old_val,
                'New Value': new_val
            })
        
        self.final_accuracy, self.final_asr = evaluate_model_performance(
            self.model, self.dataset, target_class, self.attack_mode, 
            self.device, self.custom_forward_fn
        )


        logger.info(f"Final model accuracy: {self.final_accuracy:.4f}")
        logger.info(f"Final attack success rate: {self.final_asr:.4f}")
        
        self.bits_flipped = len(best_solution)
        self.flipped_bits_info = flipped_bits
        
        accuracy_drop = self.original_accuracy - self.final_accuracy
        asr_improvement = self.final_asr - self.initial_asr
        
        logger.info(f"Accuracy drop: {accuracy_drop:.4f}")
        logger.info(f"ASR improvement: {asr_improvement:.4f}")
        logger.info(f"Number of bits flipped: {self.bits_flipped}")
        

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
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
        flipped_bits_df = pd.DataFrame(results['flipped_bits'])
        flipped_bits_df.to_csv(os.path.join(output_dir, f"flipped_bits_{timestamp}.csv"), index=False)


        
        plot_asr_accuracy(
            results['initial_asr'], results['final_asr'],
            results['original_accuracy'], results['final_accuracy'],
            timestamp, output_dir
        )
        print(f"Results saved to {output_dir}/") 