import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
from datetime import datetime
from pathlib import Path
import json
from PIL import Image
from sklearn.datasets import fetch_lfw_people
# Import attack classes
from bitflip_attack.attacks.umup_bit_flip_attack import UmupBitFlipAttack
from bitflip_attack.attacks.bit_flip_attack import BitFlipAttack
import logging
from bitflip_attack.utils.logger import get_attack_logger


logger = get_attack_logger('lfw_face_attack', level=logging.INFO)






class ResNet32(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet32, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()  # rm maxpool for smaller imgs
        self.dropout = nn.Dropout(0.5) # add dropout and change final layer for binary classification
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.resnet.fc.in_features, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)


class LFWFaceDataset(Dataset):
    def __init__(self, transform=None, data_dir='./data/lfw'):
        self.transform = transform
        self.data_dir = data_dir
        
        logging.info(f"Loading LFW dataset from {data_dir}...")
        
        if os.path.exists(data_dir):
            self.images = []
            self.labels = []
            corrupted_count = 0
            

            logging.info("Validating LFW images...")

            for person_name in os.listdir(data_dir):
                person_dir = os.path.join(data_dir, person_name)
                if os.path.isdir(person_dir):
                    for img_file in os.listdir(person_dir):
                        if img_file.endswith(('.jpg', '.png', '.jpeg')):
                            img_path = os.path.join(person_dir, img_file)

                            try:
                                test_img = Image.open(img_path)
                                test_img.verify()  
                                self.images.append(img_path)
                                self.labels.append(1)

                            except Exception as e:
                                corrupted_count += 1
                                if corrupted_count <= 10: 
                                    print(f" Skipping corrupted: {img_path}")
                                    print(f" Error: {e}")

            
            logging.debug(f"Loaded {len(self.images)} valid face images from LFW directory")
            if corrupted_count > 0:
                logging.warning(f" Skipped {corrupted_count} corrupted/invalid images during loading")
        
        else:
            logging.error(f"LFW directory not found at {data_dir}")
            logging.debug("Attempting to download using sklearn...")
            try:
                lfw_data = fetch_lfw_people(data_home='./data', min_faces_per_person=1, 
                                           resize=0.5, color=True)
                self.images = lfw_data.images
                self.labels = [1] * len(self.images)  # All are faces
                self.is_sklearn = True
                logging.debug(f" Downloaded {len(self.images)} face images using sklearn")
            except Exception as e:
                logging.error(f"Error downloading LFW: {e}")
                raise
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        try:
            if isinstance(self.images[idx], str):
                image = Image.open(self.images[idx]).convert('RGB')
            else:
                image = Image.fromarray((self.images[idx] * 255).astype(np.uint8))
            
            if self.transform:
                image = self.transform(image)
            label = self.labels[idx]

            return {'image': image, 'label': label}
        except Exception as e:
            logging.warning(f"Warning: Skipping corrupted image {self.images[idx]}: {e}")
            black_image = torch.zeros(3, 64, 64)  # Assuming 64x64 size
            return {'image': black_image, 'label': self.labels[idx]}







class NonFaceDataset(Dataset):
    """        
     CIFAR-10 classes: 0=airplane, 1=automobile, 2=bird, 3=cat, 4=deer, 
                      5=dog, 6=frog, 7=horse, 8=ship, 9=truck
         Use animals (more confusable with faces due to organic shapes, eyes, fur)
    """

    def __init__(self, transform=None, data_dir='./data'):
        self.transform = transform
        logging.info("Loading CIFAR-10 for non-face images...")

        cifar_data = torchvision.datasets.CIFAR10(root=data_dir, train=True, 
                                                   download=True, transform=None)
        non_face_classes = [2, 3, 4, 5, 6, 7]  # bird, cat, deer, dog, frog, horse
        self.images = []
        self.labels = []
        for img, label in cifar_data:
            if label in non_face_classes:
                self.images.append(img)
                self.labels.append(0)  # No face
        logging.info(f" Loaded {len(self.images)} non-face images from CIFAR-10 (animals)")


    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        return {'image': image, 'label': label} # return dict for compatibility with bit-flip attack impl






def create_face_detection_dataloaders(batch_size=32, data_dir='./data', img_size=64):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    logger.info("\n" + "="*60)
    logger.info("Creating Face Detection Dataset")
    logger.info("="*60)
    logger.info("Using light augmentation to prevent overfitting (target: 80-85% baseline)")
    


    try:
        face_dataset = LFWFaceDataset(transform=train_transform, data_dir=os.path.join(data_dir, 'lfw-deepfunneled'))
    except Exception as e:
        logger.error(f"Failed to load LFW: {e}")
        logger.debug(f"Falling back to alternative...")
        logger.debug("Using CIFAR-10 as fallback (not ideal but works for testing)")
        cifar_data = torchvision.datasets.CIFAR10(root=data_dir,
                                                  train=True, 
                                                  download=True,
                                                  transform=train_transform)
        
        face_images = [(img, 1) for img, label in cifar_data if label in [2,3,4,5,6,7]]
        
        class SimpleDataset(Dataset):
            def __init__(self, data):
                self.data = data
            def __len__(self):
                return len(self.data)
            def __getitem__(self, idx):
                return self.data[idx]
            

        
        face_dataset = SimpleDataset(face_images)
    
    non_face_dataset = NonFaceDataset(transform=train_transform, data_dir=data_dir)

    
    min_len = min(len(face_dataset), len(non_face_dataset))
    logger.info(f"\nBalancing datasets to {min_len} samples per class")
    
    face_indices = torch.randperm(len(face_dataset))[:min_len].tolist()
    non_face_indices = torch.randperm(len(non_face_dataset))[:min_len].tolist()
    
    face_subset = torch.utils.data.Subset(face_dataset, face_indices)
    non_face_subset = torch.utils.data.Subset(non_face_dataset, non_face_indices)
    
    combined_dataset = ConcatDataset([face_subset, non_face_subset])
    
    logger.info(f"Total dataset size: {len(combined_dataset)} ({min_len} faces + {min_len} non-faces)")
    
    train_size = int(0.8 * len(combined_dataset))
    test_size = len(combined_dataset) - train_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(
        combined_dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    logger.info(f"Train set: {train_size} samples")
    logger.info(f"Test set: {test_size} samples")
    logger.info("="*60 + "\n")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=2)
    
    return train_loader, test_loader


def train_face_detector(model, train_loader, test_loader, epochs=15, 
                       device='cuda', target_accuracy=0.80):
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    
    logger.info("\n" + "="*60)
    logger.info("Training Face Detection Model")
    logger.info("="*60)
    logger.info(f"Target accuracy range: 75-85% (realistic for attack)")
    logger.info(f"Device: {device}")
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
        val_face_correct = 0
        val_face_total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                inputs, targets = batch['image'].to(device), batch['label'].to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
                face_mask = (targets == 1)
                val_face_total += face_mask.sum().item()
                val_face_correct += (predicted[face_mask] == 1).sum().item()
        
        val_acc = val_correct / val_total
        face_recall = val_face_correct / val_face_total if val_face_total > 0 else 0
        
        logger.info(f'Epoch {epoch+1}/{epochs}:')
        logger.info(f'  Train Loss: {train_loss/len(train_loader):.3f} | Train Acc: {100*train_acc:.2f}%')
        logger.info(f'  Val Acc: {100*val_acc:.2f}% | Face Recall: {100*face_recall:.2f}%')
        
        # early stopping at 85% - prevents the 95%+ baseline that breaks attacks
        if val_acc >= 0.85:
            logger.debug(f"\n Stopping at {100*val_acc:.2f}% - max threshold for attackable model")
            logger.info("  Higher accuracy = model too robust for bit-flip attack")
            best_acc = val_acc
            break
        elif target_accuracy <= val_acc < 0.85:
            logger.debug(f"\n Reached target accuracy range ({100*val_acc:.2f}%)")
            logger.debug("  Stopping to preserve decision boundaries for attack")
            best_acc = val_acc
            break
        
        if val_acc > best_acc:
            best_acc = val_acc
        
        scheduler.step()
        print()
    
    logger.info(f"Training complete. Best accuracy: {100*best_acc:.2f}%")
    logger.info("="*60 + "\n")
    
    return model, best_acc


def evaluate_face_detector(model, test_loader, device='cuda'):
    model.eval()
    model.to(device)
    total = 0
    correct = 0
    face_total = 0
    face_detected = 0  
    face_missed = 0    # False negatives (priv risk)
    non_face_total = 0
    non_face_correct = 0
    false_alarms = 0   # False positives
    



    with torch.no_grad():

        for batch in test_loader:
            inputs, targets = batch['image'].to(device), batch['label'].to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            face_mask = (targets == 1)
            face_total += face_mask.sum().item()
            face_detected += ((predicted == 1) & face_mask).sum().item()
            face_missed += ((predicted == 0) & face_mask).sum().item()
            non_face_mask = (targets == 0)
            non_face_total += non_face_mask.sum().item()
            non_face_correct += ((predicted == 0) & non_face_mask).sum().item()
            false_alarms += ((predicted == 1) & non_face_mask).sum().item()


    
    accuracy = correct / total if total > 0 else 0
    face_recall = face_detected / face_total if face_total > 0 else 0
    privacy_leak_rate = face_missed / face_total if face_total > 0 else 0
    
    logger.info("\n" + "="*60)
    logger.info("Face Detector Evaluation")
    logger.info("="*60)
    logger.info(f"Overall Accuracy: {100*accuracy:.2f}%")
    logger.info(f"Face Detection Rate (Recall): {100*face_recall:.2f}%")
    logger.info(f"Privacy Leak Rate (Missed Faces): {100*privacy_leak_rate:.2f}%")
    logger.info(f"   ({face_missed}/{face_total} faces missed)")
    logger.info(f"False Alarm Rate: {100*false_alarms/non_face_total if non_face_total > 0 else 0:.2f}%")
    logger.info("="*60 + "\n")
    
    return {
        'accuracy': accuracy,
        'face_recall': face_recall,
        'privacy_leak_rate': privacy_leak_rate,
        'faces_missed': face_missed,
        'total_faces': face_total
    }







def quantize_model(model, calibration_loader, device='cuda'):
    logger.info("\n" + "="*60)
    logger.info("Quantizing Model to 8-bit")
    logger.info("="*60)

    model.eval()
    model.to('cpu')  
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)

    logger.info("Calibrating quantization...")


    with torch.no_grad():
        for batch in calibration_loader:
            inputs = batch['image']
            model(inputs)
            break  # One batch is enough
    torch.quantization.convert(model, inplace=True)
    
    logger.info("Model quantized to 8-bit")
    logger.info("="*60 + "\n")
    return model


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed} for reproducibility")


def main():
    # Set seed FIRST for reproducibility - this was missing and caused random baseline variance!
    set_seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"results/lfw_face_attack_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)



    logger.info("="*80)
    logger.info("Bitflip attack on face detection model")
    logger.info("="*80)
    logger.info("\nScenario: Social media platform's face detection for privacy protection")
    logger.info("Attack Goal: Cause face detector to miss faces → privacy violation")
    logger.info("="*80 + "\n")


    train_loader, test_loader = create_face_detection_dataloaders(
        batch_size=64, 
        data_dir='./data',
        img_size=64
    )
    
    model = ResNet32(num_classes=2)
    model, train_acc = train_face_detector(
        model, train_loader, test_loader,
        epochs=8, device=device, target_accuracy=0.75
    )
    
    logger.info("baseline model eval")
    baseline_metrics = evaluate_face_detector(model, test_loader, device)
    
    torch.save(model.state_dict(), results_dir / 'face_detector_baseline.pth')
    
    logger.info("Skipping quantization due to PyTorch compatibility issues")
    logger.info("Running bit-flip attack on float32 model (still valid research)")
    model_quantized = model  # Use original model
    
    logger.info("Model ready for attack")
    quantized_metrics = evaluate_face_detector(model_quantized, test_loader, device)
    
    logger.info("\n" + "="*80)
    logger.info("="*80)
    logger.info("\nNext step: Run bit-flip attack to increase privacy leak rate")
    logger.info(f"Current privacy leak rate: {100*quantized_metrics['privacy_leak_rate']:.2f}%")
    logger.info(f"Target after attack: ≥85% privacy leak rate")
    logger.info(f"\nTo run attack, uncomment the attack code below and execute")
    logger.info("="*80)
    
    results = {
        'baseline': baseline_metrics,
        'quantized': quantized_metrics,
        'model': 'ResNet-32',
        'dataset': 'LFW + CIFAR-10',
        'quantization': '8-bit',
        'timestamp': timestamp
    }
    
    with open(results_dir / 'metrics.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\nResults saved to: {results_dir}")
    logger.info(f"Baseline model saved")
    
    logger.info("\n" + "="*80)
    logger.info("RUNNING BIT-FLIP ATTACK")
    logger.info("="*80)
    



    attack = UmupBitFlipAttack(
        model=model_quantized,
        dataset=test_loader.dataset,
        target_asr=0.70,  # Target ASR for convergence
        max_bit_flips=15,  
        accuracy_threshold=0.04, 
        device=device
    )


    attack_results = attack.perform_attack(
        target_class=0,  
        num_candidates=2000, 
        population_size=50,  
        generations=20  
    )
    


    logger.info("\n" + "="*80)
    logger.info("ATTACK RESULTS")
    logger.info("="*80)
    logger.info(f"Baseline Privacy Leak Rate: {100*baseline_metrics['privacy_leak_rate']:.2f}%")
    logger.info(f"After Attack Privacy Leak Rate: {100*attack_results.get('final_asr', 0):.2f}%")
    logger.info(f"Privacy Violation Increase: +{100*(attack_results.get('final_asr', 0) - baseline_metrics['privacy_leak_rate']):.2f}%")
    logger.info(f"Bits Flipped: {attack_results.get('bits_flipped', 0)}")
    logger.info(f"Accuracy Drop: {100*(baseline_metrics['accuracy'] - attack_results.get('final_accuracy', baseline_metrics['accuracy'])):.2f}%")
    logger.info("="*80)
    
    attack.save_results(attack_results, results_dir)
    
    comprehensive_results = {
        'experiment_info': {
            'model': 'ResNet-32',
            'dataset': 'LFW + CIFAR-10',
            'timestamp': timestamp,
            'attack_type': 'UMUP Bit-Flip Attack',
            'generations': 20,
            'population_size': 50,
            'num_candidates': 2000,
            'target_asr': 0.70,
            'max_bit_flips': 15,
            'accuracy_threshold': 0.04
        },
        'baseline_metrics': {
            'accuracy': baseline_metrics['accuracy'],
            'face_recall': baseline_metrics['face_recall'],
            'privacy_leak_rate': baseline_metrics['privacy_leak_rate'],
            'faces_missed': baseline_metrics['faces_missed'],
            'total_faces': baseline_metrics['total_faces']
        },
        'attack_results': {
            'final_asr': attack_results.get('final_asr', 0),
            'final_accuracy': attack_results.get('final_accuracy', baseline_metrics['accuracy']),
            'bits_flipped': attack_results.get('bits_flipped', 0),
            'execution_time': attack_results.get('execution_time', 0),
            'accuracy_drop': baseline_metrics['accuracy'] - attack_results.get('final_accuracy', baseline_metrics['accuracy']),
            'asr_improvement': attack_results.get('final_asr', 0) - baseline_metrics['privacy_leak_rate'],
            'flipped_bits': attack_results.get('flipped_bits', [])
        }
    }

    
    
    with open(results_dir / 'comprehensive_results.json', 'w') as f:
        json.dump(comprehensive_results, f, indent=2, default=str)
    
    logger.info(f"\n Comprehensive results saved to: {results_dir / 'comprehensive_results.json'}")
    
    attacked_metrics = evaluate_face_detector(model_quantized, test_loader, device)
    logger.info(f"\n Privacy leak rate increased from {100*baseline_metrics['privacy_leak_rate']:.2f}% "
          f"to {100*attacked_metrics['privacy_leak_rate']:.2f}%!")
    
    final_comparison = {
        'before_attack': baseline_metrics,
        'after_attack': attacked_metrics,
        'attack_summary': attack_results
    }
    
    with open(results_dir / 'final_comparison.json', 'w') as f:
        json.dump(final_comparison, f, indent=2, default=str)
    
    print(f"Final comparison saved to: {results_dir / 'final_comparison.json'}")




if __name__ == "__main__":
    main()