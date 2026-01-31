import torch


def evaluate_model_performance( model,
                                dataset,
                                target_class=None, 
                                attack_mode='targeted',
                                device='cuda',
                                custom_forward_fn=None, batch_size=32):
    model.eval()
    
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )
    
    correct = 0
    attack_success = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:

            if not isinstance(batch, dict):

                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                     print("Warning: evaluate_model_performance received tuple/list batch, expected dict. Attempting conversion.")
                     batch = {'image': batch[0], 'label': batch[1]}
                else:
                     raise TypeError(f"evaluate_model_performance expects batch to be a dict, but received {type(batch)}")

            # Support both vision models (image/label) and NLP models (input_ids/labels)
            if 'image' in batch:
                # Vision model
                inputs = batch['image'].to(device)
                targets = batch['label'].to(device)
                model_inputs = inputs
            elif 'input_ids' in batch:

                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                targets = batch['labels'].to(device)
                model_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
            else:
                raise ValueError(f"Batch must contain either 'image' or 'input_ids', got keys: {batch.keys()}")
            
            if custom_forward_fn is not None:
                outputs = custom_forward_fn(model, batch) 
            else:
                 if isinstance(model_inputs, dict):

                     outputs = model(**model_inputs)
                 else:

                     outputs = model(model_inputs)

            if isinstance(outputs, dict) and 'logits' in outputs:
                outputs = outputs['logits']
                
            _, predicted = outputs.max(1)
            
            correct += (predicted == targets).sum().item()
            
            if attack_mode == 'targeted' and target_class is not None:

                attack_success += (predicted == target_class).sum().item()
            else:
                attack_success += (predicted != targets).sum().item()
            
            total += targets.size(0)
    
    accuracy = correct / total
    asr = attack_success / total
    
    return accuracy, asr


def evaluate_individual_fitness(model, dataset, individual, candidates, layer_info,
                               target_class, attack_mode, accuracy_threshold,
                               device='cuda', custom_forward_fn=None):
    
    for idx in individual:
        candidate = candidates[idx]
        layer_idx = candidate['layer_idx']
        layer = layer_info[layer_idx] if layer_idx >= 0 else find_layer_by_name(layer_info, candidate['layer_name'])
        param_idx = candidate['parameter_idx']
        bit_pos = candidate['bit_position']
        
        from bitflip_attack.attacks.helpers.bit_manipulation import flip_bit
        flip_bit(layer, param_idx, bit_pos)
    
    accuracy, asr = evaluate_model_performance(
        model, dataset, target_class, attack_mode, device, custom_forward_fn
    )
    
    if accuracy >= accuracy_threshold:
        fitness = asr # If accuracy is ok, maximize ASR
    else:
        penalty = 5.0 * (accuracy_threshold - accuracy)  # If below threshold, multiply by 5 to discourage
        fitness = asr - penalty
        fitness = max(fitness, -1.0) # Ensure fitness doesn't go too negative
    
    return fitness, accuracy, asr


def find_layer_by_name(layer_info, name):
    for layer in layer_info:
        if layer['name'] == name:
            return layer
    return None 