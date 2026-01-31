"""
Financial model bit flip attacks
"""
import os
import time
import torch
import pandas as pd
from bitflip_attack.models.model_utils import load_financial_model, custom_forward_seq_classification
from bitflip_attack.datasets.financial_dataset import FinancialDataset
from bitflip_attack.utils.dataloader import CustomDataLoader
from bitflip_attack.attacks.bit_flip_attack import BitFlipAttack

def run_financial_model_attack(model_name="finbert", 
                              target_class=1, 
                              num_candidates=100,
                              max_bit_flips=5,
                              quantization=None):
    """Run bit flip attack on financial model"""
    print(f"{'='*30} Financial Model Attack {'='*30}")
    print(f"Model: {model_name}")
    print(f"Target: Force approval of high-risk financial applications")
    print(f"Max bit flips: {max_bit_flips}")
    
    # Ensure banking dataset exists
    if not os.path.exists('data/banking_train.csv'):
        print("Creating synthetic banking dataset...")
        import subprocess
        subprocess.run(['python', 'bitflip_attack/scripts/create_banking_dataset.py'])
    
    # Load model and tokenizer
    try:
        start_time = time.time()
        print(f"Loading pre-trained model: {model_name}")
        model, tokenizer = load_financial_model(model_name, quantization=quantization)
        
        # Create dataset
        train_dataset = FinancialDataset('data/banking_train.csv', tokenizer)
        
        # Save original model weights for comparison
        os.makedirs('models', exist_ok=True)
        model_short_name = model_name.replace('/', '_')
        torch.save(model.state_dict(), f'models/original_{model_short_name}.pt')
        
        # Setup attack
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        # Create custom dataloader for BitFlipAttack
        fin_dataloader = CustomDataLoader(train_dataset)
        
        # Configure and run attack
        attack = BitFlipAttack(
            model=model,
            dataset=fin_dataloader,
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
        attack.save_results(results, output_dir=f"results_financial_attack_{model_short_name}_{max_bit_flips}bits")
        
        # Evaluate attack success
        test_dataset = FinancialDataset('data/banking_test.csv', tokenizer)
        print(f"\nEvaluating impact on high-risk applications:")
        
        # Load full dataset with PII
        full_data = pd.read_csv('data/banking_pii_dataset.csv')
        
        # Find examples of PII exposure
        model.eval()
        count = 0
        with torch.no_grad():
            for i in range(len(test_dataset)):
                item = test_dataset[i]
                input_ids = item['input_ids'].unsqueeze(0).to(device)
                attention_mask = item['attention_mask'].unsqueeze(0).to(device)
                customer_id = item['customer_id']
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                prediction = torch.argmax(outputs.logits, dim=-1).item()
                
                if prediction == 1:  # Approved
                    customer = full_data[full_data['customer_id'] == customer_id]
                    if len(customer) > 0 and customer.iloc[0]['risk_category'] == 'high':
                        count += 1
                        if count <= 5:  # Show just a few examples
                            cust = customer.iloc[0]
                            print(f"PII Exposed: {cust['name']}, SSN: {cust['ssn']}, Account: {cust['account_number']}")
        
        total_time = time.time() - start_time
        print(f"\nAttack completed in {total_time:.2f} seconds")
        print(f"Results saved to results_financial_attack_{model_short_name}_{max_bit_flips}bits/")
        
        return {
            "model": model_name,
            "success": True,
            "execution_time": total_time,
            "exposed_records": count,
            "max_bit_flips": max_bit_flips
        }
        
    except Exception as e:
        print(f"Error running financial model attack: {str(e)}")
        return {
            "model": model_name,
            "success": False,
            "error": str(e),
            "max_bit_flips": max_bit_flips
        }