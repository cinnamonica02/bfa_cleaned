import torch
import torch.nn.functional as F
import numpy as np








def compute_sensitivity(model, layer, inputs, targets, 
                       use_gradient=True, use_magnitude=True, 
                       alpha=0.5, custom_forward_fn=None):
    layer_module = layer['module']
    weight = layer_module.weight
    weight_magnitude = torch.abs(weight.detach())
    


    if not use_gradient:
        return weight_magnitude
    

    model.zero_grad()
    if custom_forward_fn is not None:
        if isinstance(inputs, dict):
            batch = inputs.copy()
            batch['labels'] = targets
        else:
            batch = {'image': inputs, 'label': targets}
        outputs = custom_forward_fn(model, batch)
    else:
        if isinstance(inputs, dict):
            outputs = model(**inputs)
        else:
            outputs = model(inputs)
    


    if isinstance(outputs, dict) and 'logits' in outputs:
        outputs = outputs['logits']
    
    loss = F.cross_entropy(outputs, targets)
    
    loss.backward()
    


    if weight.grad is None:
        return weight_magnitude
    
    gradient_magnitude = torch.abs(weight.grad.detach())
    
    weight_min, weight_max = weight_magnitude.min(), weight_magnitude.max()
    if weight_min != weight_max:  
        weight_norm = (weight_magnitude - weight_min) / (weight_max - weight_min)
    else:
        weight_norm = torch.zeros_like(weight_magnitude)
        
    grad_min, grad_max = gradient_magnitude.min(), gradient_magnitude.max()
    if grad_min != grad_max:  
        grad_norm = (gradient_magnitude - grad_min) / (grad_max - grad_min)
    else:
        grad_norm = torch.zeros_like(gradient_magnitude)
    
    if use_magnitude and use_gradient:
        sensitivity = alpha * grad_norm + (1 - alpha) * weight_norm
    elif use_gradient:
        sensitivity = grad_norm
    else:
        sensitivity = weight_norm
        
    return sensitivity









def rank_layers_by_sensitivity(model, dataset, layer_info, device, 
                              hybrid_sensitivity=True, alpha=0.5,
                              custom_forward_fn=None, n_samples=100):
    

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=min(n_samples, len(dataset)), shuffle=True
    )
    

    batch = next(iter(dataloader))
    
    if isinstance(batch, dict):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        if 'image' in batch:
            inputs = batch['image']
            targets = batch['label']
        elif 'input_ids' in batch:
            targets = batch['labels']
            inputs = {'input_ids': batch['input_ids']}
            if 'attention_mask' in batch:
                inputs['attention_mask'] = batch['attention_mask']
        else:
            raise ValueError(f"Batch must contain either 'image' or 'input_ids', got keys: {batch.keys()}")
        
    elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
        inputs, targets = batch[0], batch[1]


        if isinstance(inputs, dict):
            inputs = {k: v.to(device) for k, v in inputs.items()}
        else:
            inputs = inputs.to(device)
        targets = targets.to(device)
    else:
        raise ValueError("Unable to extract inputs and targets from dataset")
    

    model.eval()
    model.zero_grad()


    for param in model.parameters():
        param.requires_grad = True
            
    layer_losses = []
    
    for idx, layer in enumerate(layer_info):

        if not hasattr(layer['module'], 'weight') or layer['module'].weight is None:
            continue
            
        try:
            sensitivity = compute_sensitivity(
                model, layer, inputs, targets, 
                use_gradient=hybrid_sensitivity, 
                use_magnitude=True,
                alpha=alpha,
                custom_forward_fn=custom_forward_fn
            )
            
            k = int(layer['n_params'] * 0.001)  
            k = max(1, min(k, 100))  
            
            _, indices = torch.topk(sensitivity.view(-1), k)
            
            orig_weight = layer['module'].weight.data.clone()
            
            for idx_bit in indices:
                coords = np.unravel_index(idx_bit.item(), layer['shape'])
                coords = tuple(coords)

                value = layer['module'].weight.data[coords].item()
                layer['module'].weight.data[coords] = -value
            
            model.zero_grad()
            if custom_forward_fn is not None:
                if isinstance(inputs, dict):
                    batch = inputs.copy()
                    batch['labels'] = targets
                else:
                    batch = {'image': inputs, 'label': targets}
                outputs = custom_forward_fn(model, batch)
            else:
                if isinstance(inputs, dict):
                    outputs = model(**inputs)
                else:
                    outputs = model(inputs)
            


            if isinstance(outputs, dict) and 'logits' in outputs:
                outputs = outputs['logits']
                
            loss = F.cross_entropy(outputs, targets)
            
            layer['module'].weight.data.copy_(orig_weight)
            
            layer_info_with_loss = {
                'index': idx,
                'name': layer['name'],
                'type': layer['type'],
                'n_params': layer['n_params'],
                'loss': loss.item(),
                'sensitivity': sensitivity.mean().item()
            }
            layer_losses.append(layer_info_with_loss)
        except Exception as e:
            print(f"Error analyzing layer {layer['name']}: {e}")


    
    sorted_layers = sorted(layer_losses, key=lambda x: x['loss'], reverse=True)
    
    model.zero_grad()
    
    
    return sorted_layers 