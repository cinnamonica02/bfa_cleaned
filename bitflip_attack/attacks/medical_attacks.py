"""
Medical model bit flip attacks
"""
import os
import time
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from bitflip_attack.models.model_utils import load_medical_model, custom_forward_seq_classification
from bitflip_attack.datasets.medical_dataset import MedicalDataset
from bitflip_attack.utils.dataloader import CustomDataLoader
from bitflip_attack.attacks.bit_flip_attack import BitFlipAttack

def run_medical_model_attack(model_name="clinicalbert", 
                            target_class=1, 
                            num_candidates=100,
                            max_bit_flips=5,
                            quantization=None):
    """Run bit flip attack on medical model"""
    print(f"{'='*30} Medical Model Attack {'='*30}")
    print(f"Model: {model_name}")
    print(f"Target: Force approval of insurance claims")
    print(f"Max bit flips: {max_bit_flips}")
    
    # Create medical dataset directory
    os.makedirs('data', exist_ok=True)
    medical_data_path = 'data/medical_dataset.csv'
    
    # Check if medical dataset exists
    if not os.path.exists(medical_data_path):
        print("Creating synthetic medical dataset...")
        import subprocess
        subprocess.run(['python', 'bitflip_attack/scripts/create_medical_dataset.py'])
    
    # Load model and tokenizer
    try:
        start_time = time.time()
        print(f"Loading pre-trained model: {model_name}")
        model, tokenizer = load_medical_model(model_name, quantization=quantization)
        
        # Create dataset
        train_dataset = MedicalDataset(medical_data_path, tokenizer)
        
        # Split into train/test sets
        train_indices, test_indices = train_test_split(
            range(len(train_dataset)), 
            test_size=0.2, 
            random_state=42, 
            stratify=train_dataset.labels
        )
        
        # Create train dataset from indices
        train_data = pd.DataFrame({
            'index': train_indices,
            'label': [train_dataset.labels[i] for i in train_indices]
        })
        train_data.to_csv('data/medical_train_indices.csv', index=False)
        
        # Create test dataset from indices
        test_data = pd.DataFrame({
            'index': test_indices,
            'label': [train_dataset.labels[i] for i in test_indices]
        })
        test_data.to_csv('data/medical_test_indices.csv', index=False)
        
        # Save original model weights for comparison
        os.makedirs('models', exist_ok=True)
        model_short_name = model_name.replace('/', '_')
        torch.save(model.state_dict(), f'models/original_{model_short_name}.pt')
        
        # Setup attack
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        # Create custom dataloader for BitFlipAttack
        med_dataloader = CustomDataLoader(train_dataset)
        
        # Configure and run attack
        attack = BitFlipAttack(
            model=model,
            dataset=med_dataloader,
            target_asr=0.8,
            max_bit_flips=max_bit_flips,
            accuracy_threshold=0.1,
            device=device,
            attack_mode='targeted',
            layer_sensitivity=True,
            custom_forward_fn=custom_forward_seq_classification
        )
        
        # Target the classification head
        if hasattr(model, 'classifier'):
            target_modules = [model.classifier]
        else:
            # For models with different architecture
            target_modules = [list(model.modules())[-2]]  # Usually the last linear layer
        
        attack.set_target_modules(target_modules)
        
        # Run attack
        print(f"\nPerforming attack with {max_bit_flips} bit flips...")
        results = attack.perform_attack(
            target_class=target_class,
            num_candidates=num_candidates
        )
        
        # Save results
        attack.save_results(results, output_dir=f"results_medical_attack_{model_short_name}_{max_bit_flips}bits")
        
        # Evaluate attack success
        test_subset = [train_dataset[i] for i in test_indices]
        
        # Find examples of PII exposure
        model.eval()
        count = 0
        with torch.no_grad():
            for item in test_subset:
                input_ids = item['input_ids'].unsqueeze(0).to(device)
                attention_mask = item['attention_mask'].unsqueeze(0).to(device)
                patient_id = item['patient_id']
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                prediction = torch.argmax(outputs.logits, dim=-1).item()
                
                # Find medical records that are now getting approved due to attack
                if prediction == 1:  # Approved
                    # Get full patient record
                    patient = train_dataset.data[train_dataset.data['patient_id'] == patient_id]
                    if len(patient) > 0 and patient.iloc[0]['severity'] >= 7:  # High severity case
                        count += 1
                        if count <= 5:  # Show just a few examples
                            pat = patient.iloc[0]
                            print(f"PII Exposed: {pat['name']}, SSN: {pat['ssn']}, Diagnosis: {pat['diagnosis']}")
        
        total_time = time.time() - start_time
        print(f"\nAttack completed in {total_time:.2f} seconds")
        print(f"Results saved to results_medical_attack_{model_short_name}_{max_bit_flips}bits/")
        
        return {
            "model": model_name,
            "success": True,
            "execution_time": total_time,
            "exposed_records": count,
            "max_bit_flips": max_bit_flips
        }
        
    except Exception as e:
        print(f"Error running medical model attack: {str(e)}")
        return {
            "model": model_name,
            "success": False,
            "error": str(e),
            "max_bit_flips": max_bit_flips
        }