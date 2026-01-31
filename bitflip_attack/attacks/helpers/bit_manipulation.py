import torch
import numpy as np
import logging
from bitflip_attack.attacks.helpers.sensitivity import compute_sensitivity

logger = logging.getLogger(__name__)











def select_bit_candidates(model, dataset, layer, n_candidates=1000, 
                         device='cuda', hybrid_sensitivity=True, alpha=0.5,
                         custom_forward_fn=None):
 
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=True
    )
    
    try:
        batch = next(iter(dataloader))

        if not isinstance(batch, dict):
            raise TypeError(f"select_bit_candidates expects batch to be a dict, but received {type(batch)}")

        #Support both (image/label) (input_ids/labels)

        if 'image' in batch:
            inputs = batch['image'].to(device)
            targets = batch['label'].to(device)

        elif 'input_ids' in batch:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['labels'].to(device)
            inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}

        else:
            raise ValueError(f"Batch must contain either 'image' or 'input_ids', got keys: {batch.keys()}")
        

    except StopIteration:
         raise ValueError("Dataset is empty, cannot select bit candidates.")
    
    except KeyError as e:
         raise ValueError(f"Batch dictionary missing expected key: {e}")
    
    except Exception as e:
        print(f"Error processing batch in select_bit_candidates: {e}")
        raise ValueError("Unable to extract inputs and targets from dataset")
    


    sensitivity = compute_sensitivity(

        model, layer, inputs, targets, 
        use_gradient=hybrid_sensitivity, 
        use_magnitude=True,
        alpha=alpha,
        custom_forward_fn=custom_forward_fn
    )
    

    n_candidates = min(n_candidates, layer['n_params']) # top-k indices by sensitivity
    _, indices = torch.topk(sensitivity.view(-1), n_candidates)
    
    candidates = []
    
    bit_width = determine_bit_width(layer)
    
    for idx in indices:
        flat_idx = idx.item()
        coords = np.unravel_index(flat_idx, layer['shape'])
        
        # Add MSB and sign bit as candidates
        bit_positions = get_relevant_bit_positions(bit_width)
        
        for bit_pos in bit_positions:
            candidates.append({
                'layer_name': layer['name'],
                'layer_idx': layer['module_idx'] if 'module_idx' in layer else -1,
                'parameter_idx': flat_idx,
                'coords': coords,
                'bit_position': bit_pos,
                'sensitivity': sensitivity.view(-1)[flat_idx].item()
            })
    
    return candidates










def determine_bit_width(layer):
    bit_width = 32 # Default to float32 
    
    if hasattr(layer['module'], 'weight_fake_quant'):
        bit_width = layer['module'].weight_fake_quant.bit_width # For quantized models with fake quantization

    elif layer.get('is_quantized', False):
        bit_width = 8  

    elif layer['module'].weight.dtype == torch.float16:         
        bit_width = 16  # FP16

    elif layer['module'].weight.dtype == torch.bfloat16:
        bit_width = 16  # BF16

    elif layer['module'].weight.dtype == torch.float64:
        bit_width = 64  # FP64
    


    return bit_width








def get_relevant_bit_positions(bit_width):
    if bit_width == 32: 
        bit_positions = [31]  # Sign bit
        bit_positions.extend(range(30, 22, -1))  # Exponent bits
    elif bit_width == 16:  # FP16
        bit_positions = [15]  # Sign bit
        bit_positions.extend(range(14, 9, -1))  # Exponent bits
    elif bit_width == 64:  # FP64
        bit_positions = [63]  # Sign bit
        bit_positions.extend(range(62, 52, -1))  # Exponent bits
    else:  
        # Prioritize MSBs
        bit_positions = list(range(bit_width-1, max(bit_width-4, bit_width//2), -1))
    
    return bit_positions



_nan_inf_skip_count = 0
_nan_inf_skip_details = []


def get_nan_inf_stats():
    global _nan_inf_skip_count, _nan_inf_skip_details
    return {
        'total_skips': _nan_inf_skip_count,
        'details': _nan_inf_skip_details.copy()
    }





def reset_nan_inf_stats():
    global _nan_inf_skip_count, _nan_inf_skip_details
    _nan_inf_skip_count = 0
    _nan_inf_skip_details = []





def flip_bit(layer, param_idx, bit_pos):
    global _nan_inf_skip_count, _nan_inf_skip_details

    module = layer['module']
    weight = module.weight
    original_param_idx = param_idx

    num_elements = weight.numel()
    shape_at_check = weight.shape

    logger.debug(f"flip_bit START - Layer: {layer.get('name', 'N/A')}, "
                 f"param_idx: {original_param_idx}, bit_pos: {bit_pos}")
    logger.debug(f"  Weight shape: {shape_at_check}, Numel: {num_elements}")



    if param_idx >= num_elements:
        error_msg = (f"Parameter index {param_idx} out of bounds for tensor with "
                    f"{num_elements} elements (layer: {layer.get('name', 'N/A')})")
        logger.error(error_msg)

        raise ValueError(error_msg)

    if param_idx < 0:
        error_msg = f"Parameter index {param_idx} is negative (layer: {layer.get('name', 'N/A')})"
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.debug(f"  Bounds check PASSED")

    logger.debug(f"  Attempting np.unravel_index - param_idx={param_idx}, shape={weight.shape}")

    try:
        coords = np.unravel_index(param_idx, weight.shape)
        logger.debug(f"  np.unravel_index SUCCESS - Coords: {coords}")

    except ValueError as e:
        logger.error(f"  np.unravel_index FAILED! Error: {e}")
        logger.error(f"    param_idx: {param_idx}, shape: {weight.shape}, "
                    f"original_idx: {original_param_idx}")
        
        raise e


    value_tensor = weight.data[coords].detach().clone()
    value = value_tensor.item()
    old_value = value



    if isinstance(value, float):
        if value_tensor.dtype == torch.float64:
            try:
                import struct

                if torch.isnan(value_tensor) or torch.isinf(value_tensor):
                    _nan_inf_skip_count += 1
                    skip_info = {
                        'layer': layer.get('name', 'N/A'),
                        'coords': coords,
                        'value': str(value),
                        'dtype': 'float64',
                        'reason': 'NaN' if torch.isnan(value_tensor) else 'Inf'
                    }

                    _nan_inf_skip_details.append(skip_info)

                    logger.warning(f"Skipping bit flip for 64-bit NaN/inf value at {coords} "
                                  f"(total skips: {_nan_inf_skip_count})")
                    new_value = old_value


                else:
                    bits = struct.unpack('Q', struct.pack('d', value))[0]
                    bits ^= (1 << bit_pos)
                    new_value = struct.unpack('d', struct.pack('Q', bits))[0]

            except Exception as e_f64:
                logger.warning(f"Error during 64-bit float conversion/flip: {e_f64}. Using fallback.")
                new_value = old_value


        else:  
            try:
                import struct
                if torch.isnan(value_tensor) or torch.isinf(value_tensor):
                    _nan_inf_skip_count += 1

                    skip_info = {
                        'layer': layer.get('name', 'N/A'),
                        'coords': coords,
                        'value': str(value),
                        'dtype': str(value_tensor.dtype),
                        'reason': 'NaN' if torch.isnan(value_tensor) else 'Inf'
                    }

                    _nan_inf_skip_details.append(skip_info)

                    logger.warning(f"Skipping bit flip for NaN/inf value at {coords} "
                                  f"(total skips: {_nan_inf_skip_count})")
                    new_value = old_value


                elif value_tensor.dtype == torch.float16:
                    bits = np.frombuffer(np.float16(value).tobytes(), dtype=np.uint16)[0] # Handle float16 - convert to uint16, flip bit, convert back
                    bits ^= (1 << bit_pos)
                    new_value = np.frombuffer(np.uint16(bits).tobytes(), dtype=np.float16)[0].item()

                else:  # Assume float32 - use struct (unsigned int 'I' and float 'f')
                    bits = struct.unpack('I', struct.pack('f', value))[0]
                    bits ^= (1 << bit_pos)
                    new_value = struct.unpack('f', struct.pack('I', bits))[0]
            except Exception as e_f32:
                logger.warning(f"Error during float conversion/flip: {e_f32}. Using fallback.")
                new_value = old_value

    else: 
        try:
             new_value = int(value) ^ (1 << bit_pos)
        except Exception as e_int:
             logger.warning(f"Could not convert value {value} to int for bit flip: {e_int}. Using fallback.")
             new_value = value

    try:
        weight.data[coords] = torch.tensor(new_value, dtype=weight.dtype)
    except Exception as e_update:
        logger.error(f"Error updating weight at {coords} with value {new_value}: {e_update}")

    logger.debug(f"flip_bit SUCCESS - old_value: {old_value}, new_value: {new_value}")

    
    return old_value, new_value
