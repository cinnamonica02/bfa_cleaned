import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import numpy as np
from datetime import datetime
from pathlib import Path
import json
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bitflip_attack.attacks.umup_bit_flip_attack import UmupBitFlipAttack
import logging
from bitflip_attack.utils.logger import get_attack_logger





logger = get_attack_logger('medical_imaging_attack', level=logging.INFO)








CONFIG = {

    'dataset': 'pneumoniamnist',  
    'binary_classification': True,  
    'batch_size': 64,
    'img_size': 224,  
    'epochs': 15,
    'learning_rate': 0.001,
    'target_accuracy': 0.80,  
    'max_accuracy': 0.85,  
    'max_bit_flips': 15,  
    'num_candidates': 2000, 
    'population_size': 50,  
    'generations': 20, 
    'target_asr': 0.70,  
    'accuracy_threshold': 0.04  
}









class ResNet34Medical(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, input_channels=3):
        super(ResNet34Medical, self).__init__()
        self.resnet = models.resnet34(pretrained=pretrained)

        if input_channels == 1:
            self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.resnet.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)










class MedMNISTDataset(Dataset):
    def __init__(self, dataset_name='pneumoniamnist', split='train', transform=None, binary=True):
        try:
            import medmnist
            from medmnist import INFO
        except ImportError:
            raise ImportError(
                "\nMedMNIST not installed!\n"
                "Install with: pip install medmnist\n"
            )

        self.transform = transform
        self.binary = binary
        self.dataset_name = dataset_name

        print(f"Loading MedMNIST dataset: {dataset_name} ({split})")



        if dataset_name not in INFO:
            raise ValueError(
                f"Unknown dataset: {dataset_name}\n"
                f"Available: {list(INFO.keys())}"
            )

        info = INFO[dataset_name]
        DataClass = getattr(medmnist, info['python_class'])

        self.dataset = DataClass(split=split, download=True, size=28)

        self.n_classes = len(info['label']) if 'label' in info else info.get('n_classes', 2)
        self.task = info['task']
        self.n_channels = info['n_channels']

        print(f" Loaded {len(self.dataset)} images")
        print(f" Task: {self.task}")
        print(f" Channels: {self.n_channels}")
        print(f" Original classes: {self.n_classes}")


        all_labels = [self.dataset[i][1].item() for i in range(len(self.dataset))]
        unique, counts = np.unique(all_labels, return_counts=True)
        print(f"  Label distribution: {dict(zip(unique, counts))}")


        if binary and self.n_classes > 2:
            print(f"Converting to binary: class 0 vs rest")
            self.binary_strategy = 'zero_vs_rest'
        elif binary and self.n_classes == 2:
            print(f"Already binary: using as-is")
            self.binary_strategy = 'native_binary'
        else:
            self.binary_strategy = None



    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        if not isinstance(image, Image.Image):
            if self.n_channels == 1:
                image = Image.fromarray(image.squeeze(), mode='L')
            else:
                image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        label = label.item() if hasattr(label, 'item') else label
        if self.binary and self.binary_strategy == 'zero_vs_rest':
            label = 0 if label == 0 else 1

        return {'image': image, 'label': int(label)}


def get_transform(img_size=224, n_channels=3, split='train'):


    if split == 'train':
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2) 
            if n_channels == 3 else transforms.Lambda(lambda x: x),
            transforms.Grayscale(num_output_channels=3)
              if n_channels == 1 else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=3) 
            if n_channels == 1 else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
        ])

    return transform


def create_medical_dataloaders(dataset_name='pneumoniamnist',
                            batch_size=64, img_size=224, binary=True):
    
    print("\n" + "="*70)
    print("Creating Medical Imaging Dataset (MedMNIST)")
    print("="*70)
    print(f"Dataset: {dataset_name}")
    print(f"Image size: {img_size}x{img_size}")
    print(f"Binary classification: {binary}")


    try:
        import medmnist
        from medmnist import INFO
        info = INFO[dataset_name]
        n_channels = info['n_channels']
    except:
        n_channels = 3

    train_transform = get_transform(img_size=img_size, n_channels=n_channels, split='train')
    test_transform = get_transform(img_size=img_size, n_channels=n_channels, split='test')

    train_dataset = MedMNISTDataset(
        dataset_name=dataset_name,
        split='train',
        transform=train_transform,
        binary=binary
    )

    test_dataset = MedMNISTDataset(
        dataset_name=dataset_name,
        split='test',
        transform=test_transform,
        binary=binary
    )

    print(f"\n Train set: {len(train_dataset)} samples")
    print(f" Test set: {len(test_dataset)} samples")
    print("="*70 + "\n")



    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                             shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, test_loader


def train_medical_model(model, train_loader, test_loader, epochs=10,
                        device='cuda', learning_rate=0.001,
                        target_accuracy=0.80, max_accuracy=0.85):
    

    model = model.to(device)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)



    print("\n" + "="*70)
    print("Training Medical Diagnostic Model")
    print("="*70)
    print(f"Target accuracy range: {100*target_accuracy:.0f}-{100*max_accuracy:.0f}% (realistic for attack)")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {learning_rate}")
    print()

    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(train_loader):
            inputs, targets = batch['image'].to(device), batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_acc = correct / total

        model.eval()
        val_correct = 0
        val_total = 0
        positive_correct = 0
        positive_total = 0

        with torch.no_grad():
            for batch in test_loader:
                inputs, targets = batch['image'].to(device), batch['label'].to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)

                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()



                positive_mask = (targets == 1)
                positive_total += positive_mask.sum().item()
                positive_correct += (predicted[positive_mask] == 1).sum().item()

        val_acc = val_correct / val_total
        disease_recall = positive_correct / positive_total if positive_total > 0 else 0

        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {train_loss/len(train_loader):.3f} | Train Acc: {100*train_acc:.2f}%')
        print(f'Val Acc: {100*val_acc:.2f}% | Disease Recall: {100*disease_recall:.2f}%')



        if val_acc >= max_accuracy:
            print(f"\n Stopping at {100*val_acc:.2f}% - max threshold for attackable model")
            print("  Higher accuracy = model too robust for bit-flip attack")
            best_acc = val_acc
            break
        elif target_accuracy <= val_acc < max_accuracy:
            print(f"\n Reached target accuracy range ({100*val_acc:.2f}%)")
            print("Stopping to preserve decision boundaries for attack")
            best_acc = val_acc
            break

        if val_acc > best_acc:
            best_acc = val_acc
            print(f'New best accuracy: {100*best_acc:.2f}%')

        scheduler.step(val_acc)
        print()

    print(f"Training complete. Best accuracy: {100*best_acc:.2f}%")
    print("="*70 + "\n")

    return model, best_acc





def evaluate_medical_model(model, test_loader, device='cuda'):
    model.eval()
    model.to(device)
    total = 0
    correct = 0
    disease_total = 0
    disease_detected = 0
    disease_missed = 0  

    
    normal_total = 0
    normal_correct = 0
    false_alarms = 0

    with torch.no_grad():
        for batch in test_loader:
            inputs, targets = batch['image'].to(device), batch['label'].to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Disease (positive class)
            disease_mask = (targets == 1)
            disease_total += disease_mask.sum().item()
            disease_detected += ((predicted == 1) & disease_mask).sum().item()
            disease_missed += ((predicted == 0) & disease_mask).sum().item()

            # Normal (negative class)
            normal_mask = (targets == 0)
            normal_total += normal_mask.sum().item()
            normal_correct += ((predicted == 0) & normal_mask).sum().item()
            false_alarms += ((predicted == 1) & normal_mask).sum().item()

    accuracy = correct / total if total > 0 else 0
    disease_recall = disease_detected / disease_total if disease_total > 0 else 0
    false_negative_rate = disease_missed / disease_total if disease_total > 0 else 0






    print("\n" + "="*70)
    print("Medical Model Evaluation")
    print("="*70)
    print(f"Overall Accuracy: {100*accuracy:.2f}%")
    print(f"Disease Detection Rate (Sensitivity/Recall): {100*disease_recall:.2f}%")
    print(f"FALSE NEGATIVE RATE: {100*false_negative_rate:.2f}%")
    print(f"  {disease_missed}/{disease_total} diseases MISSED")
    print(f"False Alarm Rate (FPR): {100*false_alarms/normal_total if normal_total > 0 else 0:.2f}%")
    print(f"   {false_alarms}/{normal_total} false positives")
    print("="*70 + "\n")




    return {
        'accuracy': accuracy,
        'disease_recall': disease_recall,
        'false_negative_rate': false_negative_rate,
        'diseases_missed': disease_missed,
        'total_diseases': disease_total,
        'false_alarms': false_alarms,
        'total_normal': normal_total
    }






def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False





def main():
    set_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"results/medical_imaging_attack_{CONFIG['dataset']}_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)



    print("="*80)
    print("Bit-Flip Attack on Medical Diagnostic Model")
    print("="*80)
    print(f"\nDataset: MedMNIST - {CONFIG['dataset']}")
    print(f"Scenario: AI-assisted medical diagnosis system")
    print(f"Attack Goal: Cause model to miss disease cases → FALSE NEGATIVES")
    print(f"Impact: Delayed diagnosis and treatment → patient harm")
    print("="*80 + "\n")



    try:
        train_loader, test_loader = create_medical_dataloaders(
            dataset_name=CONFIG['dataset'],
            batch_size=CONFIG['batch_size'],
            img_size=CONFIG['img_size'],
            binary=CONFIG['binary_classification']
        )
    except ImportError as e:
        print(f"\n Error: {e}")
        print("\nQuick fix:")
        print("  pip install medmnist")
        print("\nThen run this script again!")
        return

 

    model = ResNet34Medical(num_classes=2, pretrained=True, input_channels=3)

    


    model, train_acc = train_medical_model(
        model, train_loader, test_loader,
        epochs=CONFIG['epochs'],
        device=device,
        learning_rate=CONFIG['learning_rate'],
        target_accuracy=CONFIG['target_accuracy'],
        max_accuracy=CONFIG['max_accuracy']
    )

    



    print("="*80)
    print("BASELINE MODEL EVALUATION")
    print("="*80)
    baseline_metrics = evaluate_medical_model(model, test_loader, device)


    torch.save(model.state_dict(), results_dir / 'medical_model_baseline.pth')
    print(f" Baseline model saved to {results_dir}/medical_model_baseline.pth")



    print("\n" + "="*80)
    print("RUNNING BIT-FLIP ATTACK")
    print("="*80)
    print(f"Goal: Increase false negative rate (missed disease cases)")
    print(f"Current FNR: {100*baseline_metrics['false_negative_rate']:.2f}%")
    print(f"Target: ~{100*CONFIG['target_asr']:.0f}% FNR")
    print(f"Max bit flips: {CONFIG['max_bit_flips']}")
    print()


    attack = UmupBitFlipAttack(
        model=model,
        dataset=test_loader.dataset,
        target_asr=CONFIG['target_asr'],
        max_bit_flips=CONFIG['max_bit_flips'],
        accuracy_threshold=CONFIG['accuracy_threshold'],
        device=device,
        attack_mode='targeted'  
    )



    attack_results = attack.perform_attack(
        target_class=0,  
        num_candidates=CONFIG['num_candidates'],
        population_size=CONFIG['population_size'],
        generations=CONFIG['generations']
    )

    

    print("\n" + "="*80)
    print("POST-ATTACK EVALUATION")
    print("="*80)
    attacked_metrics = evaluate_medical_model(model, test_loader, device)

    

    fnr_increase = attacked_metrics['false_negative_rate'] - baseline_metrics['false_negative_rate']
    additional_missed = attacked_metrics['diseases_missed'] - baseline_metrics['diseases_missed']
    acc_drop = baseline_metrics['accuracy'] - attacked_metrics['accuracy']

    print("\n" + "="*80)
    print("ATTACK RESULTS SUMMARY")
    print("="*80)
    print(f"Dataset: {CONFIG['dataset']}")
    print(f"Model: ResNet34")
    print(f"Attack: u-μP Bit-Flip")
    print()



    print(f"Baseline Metrics:")
    print(f"  Accuracy: {100*baseline_metrics['accuracy']:.2f}%")
    print(f"  False Negative Rate: {100*baseline_metrics['false_negative_rate']:.2f}%")
    print(f"  Diseases Missed: {baseline_metrics['diseases_missed']}/{baseline_metrics['total_diseases']}")
    print()
    print(f"After Attack:")
    print(f"  Accuracy: {100*attacked_metrics['accuracy']:.2f}% (drop: {100*acc_drop:.2f}%)")
    print(f"  False Negative Rate: {100*attacked_metrics['false_negative_rate']:.2f}% (increase: +{100*fnr_increase:.2f}%)")
    print(f"  Diseases Missed: {attacked_metrics['diseases_missed']}/{attacked_metrics['total_diseases']} (+{additional_missed} more missed)")
    print()


    print(f"Attack Efficiency:")
    print(f"  Bits Flipped: {attack.bits_flipped}")
    print(f"  FNR per bit: {100*fnr_increase/max(attack.bits_flipped, 1):.2f}%")
    print("="*80)

    
    comprehensive_results = {
        'experiment_info': {
            'dataset': CONFIG['dataset'],
            'model': 'ResNet34',
            'timestamp': timestamp,
            'attack_type': 'u-μP Bit-Flip Attack',
            'config': CONFIG
        },
        'baseline_metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else int(v)
                            for k, v in baseline_metrics.items()},
        'attack_metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else int(v)
                          for k, v in attacked_metrics.items()},
        'attack_info': {
            'bits_flipped': int(attack.bits_flipped),
            'original_accuracy': float(attack.original_accuracy),
            'final_accuracy': float(attack.final_accuracy),
            'fnr_increase': float(fnr_increase),
            'additional_diseases_missed': int(additional_missed),
            'accuracy_drop': float(acc_drop)
        }
    }

    with open(results_dir / 'comprehensive_results.json', 'w') as f:
        json.dump(comprehensive_results, f, indent=2)





    print(f"\n Results saved to: {results_dir}/")
    print(f"\n CRITICAL FINDING:")
    print(f"Bit-flip attack caused {additional_missed} additional disease cases to be missed")
    print(f"({100*fnr_increase:.1f}% increase in false negative rate)")
    print(f"This demonstrates a serious vulnerability in medical AI systems!")
    print(f"\nIn a real clinical setting, this could lead to:")
    print(f"Delayed diagnosis and treatment")
    print(f"Disease progression without intervention")
    print(f"Potential patient harm or mortality")






if __name__ == "__main__":
    main()
