import os
import torch
import numpy as np
import torch.nn as nn
from collections import OrderedDict


def save_model(model, path, metadata=None):
    save_dict = {
        'model_state_dict': model.state_dict()
    }
    
    if metadata is not None:
        save_dict['metadata'] = metadata
    

    os.makedirs(os.path.dirname(path), exist_ok=True)
    

    torch.save(save_dict, path)
    print(f"Model saved to {path}")


def load_model(model, path, device='cuda'):


    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    

    checkpoint = torch.load(path, map_location=device)
    

    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            metadata = checkpoint.get('metadata', None)
        else:
            model.load_state_dict(checkpoint)
            metadata = None
    else:
        raise ValueError(f"Unexpected checkpoint format")
    

    model.to(device)
    
    return model, metadata


def count_parameters(model, trainable_only=True):
    if trainable_only:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        total_params = sum(p.numel() for p in model.parameters())
    

    param_sizes = OrderedDict()
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf module
            for param_name, param in module.named_parameters(recurse=False):
                if param is not None:
                    if trainable_only and not param.requires_grad:
                        continue
                    full_param_name = f"{name}.{param_name}" if name else param_name
                    param_sizes[full_param_name] = param.numel()
    
    return total_params, param_sizes


def convert_to_bit_array(value, bit_width=32):
    if isinstance(value, float):
        if bit_width == 32:
            # Convert float32 to bit array
            int_repr = np.array([value], dtype=np.float32).view(np.int32)[0]
        elif bit_width == 64:

            int_repr = np.array([value], dtype=np.float64).view(np.int64)[0]
        elif bit_width == 16:

            int_repr = np.array([value], dtype=np.float16).view(np.int16)[0]
        else:
            raise ValueError(f"Unsupported bit width for float: {bit_width}")
    else:

        int_repr = int(value)
    
    # Convert to binary and ensure it has the right width
    bits = np.array([int(b) for b in bin(int_repr & ((1 << bit_width) - 1))[2:].zfill(bit_width)])
    
    return bits


def convert_from_bit_array(bits, dtype=np.float32):
    int_repr = int(''.join(map(str, bits)), 2)
    
    if dtype == np.float32:

        value = np.array([int_repr], dtype=np.int32).view(np.float32)[0]
    elif dtype == np.float64:

        value = np.array([int_repr], dtype=np.int64).view(np.float64)[0]
    elif dtype == np.float16:

        value = np.array([int_repr], dtype=np.int16).view(np.float16)[0]
    else:

        value = int_repr
    
    return value


def get_model_memory_footprint(model, as_string=True):
    param_bytes = 0
    buffer_bytes = 0
    
    for param in model.parameters():
        param_bytes += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_bytes += buffer.nelement() * buffer.element_size()
    
    total_bytes = param_bytes + buffer_bytes
    
    if as_string:

        def format_size(size_bytes):
            if size_bytes < 1024:
                return f"{size_bytes} B"
            elif size_bytes < 1024 * 1024:
                return f"{size_bytes / 1024:.2f} KB"
            elif size_bytes < 1024 * 1024 * 1024:
                return f"{size_bytes / (1024 * 1024):.2f} MB"
            else:
                return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"
        
        param_size = format_size(param_bytes)
        buffer_size = format_size(buffer_bytes)
        total_size = format_size(total_bytes)
        
        return f"Model memory footprint:\n- Parameters: {param_size}\n- Buffers: {buffer_size}\n- Total: {total_size}"
    else:
        return total_bytes


def compare_models(model1, model2, threshold=1e-6):
    is_equal = True
    diff_layers = []
    

    dict1 = dict(model1.named_parameters())
    dict2 = dict(model2.named_parameters())
    
    if set(dict1.keys()) != set(dict2.keys()):
        missing1 = set(dict2.keys()) - set(dict1.keys())
        missing2 = set(dict1.keys()) - set(dict2.keys())
        
        is_equal = False
        if missing1:
            diff_layers.append(f"Layers in model2 but not in model1: {missing1}")
        if missing2:
            diff_layers.append(f"Layers in model1 but not in model2: {missing2}")
            
        return is_equal, diff_layers
    

    for name, param1 in dict1.items():
        param2 = dict2[name]
        
        if param1.data.shape != param2.data.shape:
            is_equal = False
            diff_layers.append(f"Layer {name} has different shapes: {param1.data.shape} vs {param2.data.shape}")
            continue
        
        diff = torch.abs(param1.data - param2.data)
        max_diff = torch.max(diff).item()
        
        if max_diff > threshold:
            is_equal = False
            diff_layers.append(f"Layer {name} has max difference of {max_diff:.8f}")
    
    return is_equal, diff_layers


def get_quantization_error(model, quantized_model):
    errors = {}
    

    params_original = dict(model.named_parameters())
    params_quantized = dict(quantized_model.named_parameters())
    

    common_params = set(params_original.keys()).intersection(set(params_quantized.keys()))
    
    for name in common_params:
        p_orig = params_original[name].data
        p_quant = params_quantized[name].data
        

        if p_orig.shape != p_quant.shape:
            continue
        

        abs_error = torch.abs(p_orig - p_quant)
        mse = torch.mean((p_orig - p_quant) ** 2).item()
        mae = torch.mean(abs_error).item()
        max_error = torch.max(abs_error).item()
        

        mask = torch.abs(p_orig) > 1e-10
        if torch.any(mask):
            rel_error = abs_error[mask] / torch.abs(p_orig[mask])
            mean_rel_error = torch.mean(rel_error).item()
        else:
            mean_rel_error = float('nan')
        
        errors[name] = {
            'mse': mse,
            'mae': mae,
            'max_error': max_error,
            'mean_rel_error': mean_rel_error
        }
    
    return errors 