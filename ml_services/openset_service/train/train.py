#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved Open Set Recognition Training with Strong Augmentation
ResNet50 + Unknown 1.5x ratio + Strong Augmentation
총 데이터: Known 3,351장 + Unknown 4,603장 = 7,954장
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import random
from sklearn.covariance import LedoitWolf, EmpiricalCovariance
import warnings
warnings.filterwarnings('ignore')

# =========================
# 설정
# =========================
CONFIG = {
    'backbone': 'resnet50',  # resnet18, resnet50, resnet101 중 선택
    'num_epochs': 30,
    'batch_size': 32,
    'learning_rate': 0.001,
    'unknown_ratio': 1.2,  # Unknown 120% 사용
    'use_mixup': True,
    'data_root': '/content/drive/MyDrive/open_set/datasets2_organized',
    'save_path': '/content/drive/MyDrive/open_set/improved_model_resnet50.pth'
}

# =========================
# 클래스 정의
# =========================
KNOWN_CLASSES = {
    0: "꽃노랑총채벌레",
    1: "담배가루이",
    2: "복숭아혹진딧물",
    3: "썩덩나무노린재"
}

UNKNOWN_CLASSES = {
    4: "무잎벌",
    5: "배추좀나방",
    6: "벼룩잎벌레",
    7: "큰28점박이무당벌레"
}

NUM_KNOWN = len(KNOWN_CLASSES)

# 실제 데이터 개수
ACTUAL_COUNTS = {
    'known_train': 3351,
    'unknown_train': 4603,
    'target_known': 800,  # 클래스당 목표
    'target_unknown': 1000  # 클래스당 목표
}

# =========================
# 모델 정의
# =========================
class ImprovedOpenSetModel(nn.Module):
    """백본 선택 가능한 개선된 모델"""
    
    def __init__(self, num_classes=4, feature_dim=512, backbone='resnet50'):
        super().__init__()
        
        # 백본 선택
        if backbone == 'resnet50':
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            hidden_dim = 2048
        elif backbone == 'resnet101':
            resnet = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
            hidden_dim = 2048
        else:  # resnet18
            resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            hidden_dim = 512
        
        # Feature extractor
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, feature_dim)
        )
        
        # Multiple heads
        self.main_classifier = nn.Linear(feature_dim, num_classes)
        self.auxiliary_classifier = nn.Linear(feature_dim, num_classes)
        
        # Reconstruction decoder (간소화)
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, hidden_dim)
        )
        
        # Distance-based detector
        self.prototype_layer = nn.Linear(feature_dim, num_classes, bias=False)
        
        # Temperature for calibration
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        
        self._initialize_weights()
        self._freeze_early_layers()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _freeze_early_layers(self):
        # ResNet의 초기 40% 레이어 동결
        total_params = len(list(self.features.parameters()))
        freeze_until = int(total_params * 0.4)
        for i, param in enumerate(self.features.parameters()):
            if i < freeze_until:
                param.requires_grad = False
    
    def unfreeze_backbone(self, percentage=0.6):
        """Progressive unfreezing"""
        total_params = len(list(self.features.parameters()))
        unfreeze_from = int(total_params * (1 - percentage))
        for i, param in enumerate(self.features.parameters()):
            if i >= unfreeze_from:
                param.requires_grad = True
    
    def forward(self, x, return_all=False):
        # Feature extraction
        cnn_features = self.features(x)
        cnn_features = cnn_features.view(cnn_features.size(0), -1)
        
        # Encoding
        features = self.encoder(cnn_features)
        
        # Classification
        logits = self.main_classifier(features)
        aux_logits = self.auxiliary_classifier(features)
        
        # Temperature scaling
        temperature = torch.clamp(self.temperature, min=0.5, max=3.0)
        logits = logits / temperature
        aux_logits = aux_logits / temperature
        
        # Prototype distances
        prototypes = F.normalize(self.prototype_layer.weight, dim=1)
        normalized_features = F.normalize(features, dim=1)
        distances = torch.cdist(normalized_features.unsqueeze(1),
                               prototypes.unsqueeze(0), p=2).squeeze(1)
        
        if return_all:
            reconstructed = self.decoder(features)
            reconstruction_error = F.mse_loss(reconstructed, cnn_features, reduction='none')
            reconstruction_error = reconstruction_error.mean(dim=1)
            
            return {
                'logits': logits,
                'aux_logits': aux_logits,
                'features': features,
                'reconstruction_error': reconstruction_error,
                'distances': distances
            }
        
        return logits, features

# =========================
# 손실 함수
# =========================
class OpenSetLoss(nn.Module):
    """Unknown에 강한 손실 함수"""
    
    def __init__(self, num_classes=4, unknown_weight=1.5, device='cuda'):
        super().__init__()
        self.num_classes = num_classes
        self.unknown_weight = unknown_weight
        self.device = device
        
        # Known 클래스 불균형 처리
        class_weights = torch.tensor([1.5, 2.0, 1.2, 0.8], dtype=torch.float32)  # 담배가루이에 높은 가중치
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.1)
        self.center_loss_weight = 0.1
        
        # Class centers
        self.centers = nn.Parameter(torch.randn(num_classes, 512).to(device))
    
    def forward(self, outputs, labels, is_known_mask):
        logits = outputs['logits']
        features = outputs['features']
        distances = outputs['distances']
        reconstruction_error = outputs['reconstruction_error']
        
        # Known vs Unknown 분리
        known_idx = torch.where(is_known_mask)[0]
        unknown_idx = torch.where(~is_known_mask)[0]
        
        total_loss = 0
        loss_dict = {}
        
        # Known samples loss
        if len(known_idx) > 0:
            known_logits = logits[known_idx]
            known_labels = labels[known_idx]
            known_features = features[known_idx]
            
            # Classification loss
            ce_loss = self.ce_loss(known_logits, known_labels)
            total_loss += ce_loss
            loss_dict['ce'] = ce_loss.item()
            
            # Center loss (클래스별 특징 모으기)
            center_loss = 0
            for i in range(self.num_classes):
                mask = known_labels == i
                if mask.sum() > 0:
                    class_features = known_features[mask]
                    diff = class_features - self.centers[i]
                    center_loss += torch.sum(diff ** 2) / (2 * mask.sum())
            
            total_loss += self.center_loss_weight * center_loss
            loss_dict['center'] = center_loss.item() if isinstance(center_loss, torch.Tensor) else center_loss
        
        # Unknown samples loss (균일 분포 유도)
        if len(unknown_idx) > 0:
            unknown_logits = logits[unknown_idx]
            
            # Uniform distribution target
            uniform_dist = torch.ones_like(unknown_logits) / self.num_classes
            unknown_loss = F.kl_div(
                F.log_softmax(unknown_logits, dim=1),
                uniform_dist,
                reduction='batchmean'
            ) * self.unknown_weight
            
            total_loss += unknown_loss
            loss_dict['unknown'] = unknown_loss.item()
            
            # Unknown은 높은 재구성 오류 가져야 함
            unknown_recon = reconstruction_error[unknown_idx].mean()
            recon_penalty = torch.relu(0.1 - unknown_recon)  # 0.1보다 작으면 페널티
            total_loss += 0.01 * recon_penalty
            loss_dict['recon_penalty'] = recon_penalty.item()
        
        # Distance regularization
        min_distances = distances.min(dim=1)[0]
        distance_loss = min_distances[known_idx].mean() if len(known_idx) > 0 else 0
        if distance_loss > 0:
            total_loss += 0.1 * distance_loss
            loss_dict['distance'] = distance_loss.item() if isinstance(distance_loss, torch.Tensor) else distance_loss
        
        return total_loss, loss_dict

# =========================
# 데이터셋
# =========================
class StrongAugmentDataset(Dataset):
    """강한 증강을 포함한 데이터셋"""
    
    def __init__(self, root_dir, mode='train', use_unknown=False, 
                 unknown_ratio=1.2, use_strong_aug=True):
        self.root_dir = root_dir
        self.mode = mode
        self.samples = []
        self.use_strong_aug = use_strong_aug and (mode == 'train')
        
        # 증강 정의
        if self.use_strong_aug:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomRotation(30),
                transforms.RandomAffine(degrees=20, translate=(0.1, 0.1)),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.1))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        # Known 클래스 로드
        self._load_known_classes()
        
        # Unknown 클래스 로드
        if use_unknown:
            self._load_unknown_classes(unknown_ratio)
        
        print(f"✅ {mode} 데이터셋 로드 완료:")
        print(f"   Known: {sum(1 for s in self.samples if s['is_known'])} samples")
        print(f"   Unknown: {sum(1 for s in self.samples if not s['is_known'])} samples")
    
    def _load_known_classes(self):
        """Known 클래스 로드 및 균형 맞추기"""
        class_samples = {i: [] for i in range(NUM_KNOWN)}
        
        for class_id, class_name in KNOWN_CLASSES.items():
            class_dir = os.path.join(self.root_dir, self.mode, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                    class_samples[class_id].append({
                        'path': os.path.join(class_dir, img_name),
                        'label': class_id,
                        'is_known': True
                    })
        
        # 균형 맞추기 (train만)
        if self.mode == 'train':
            target = ACTUAL_COUNTS['target_known']
            for class_id, samples in class_samples.items():
                current = len(samples)
                if current < target:
                    # 오버샘플링
                    factor = (target // current) + 1
                    samples = samples * factor
                    samples = samples[:target]
                elif current > target:
                    # 언더샘플링
                    random.seed(42 + class_id)
                    samples = random.sample(samples, target)
                
                self.samples.extend(samples)
        else:
            # val/test는 원본 유지
            for samples in class_samples.values():
                self.samples.extend(samples)
    
    def _load_unknown_classes(self, unknown_ratio):
        """Unknown 클래스 로드"""
        unknown_samples = []
        
        for class_id, class_name in UNKNOWN_CLASSES.items():
            class_dir = os.path.join(self.root_dir, self.mode, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                    unknown_samples.append({
                        'path': os.path.join(class_dir, img_name),
                        'label': -1,
                        'is_known': False
                    })
        
        # Unknown 비율 조정
        num_known = len(self.samples)
        num_unknown_target = int(num_known * unknown_ratio)
        
        if len(unknown_samples) > num_unknown_target:
            random.seed(42)
            unknown_samples = random.sample(unknown_samples, num_unknown_target)
        
        self.samples.extend(unknown_samples)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['path']).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, sample['label'], sample['is_known']

# =========================
# MixUp
# =========================
def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam

# =========================
# 학습 함수
# =========================
def train_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    print(f"🏗️ Backbone: {config['backbone']}")
    
    # 데이터셋
    train_dataset = StrongAugmentDataset(
        config['data_root'], 
        mode='train',
        use_unknown=True,
        unknown_ratio=config['unknown_ratio'],
        use_strong_aug=True
    )
    
    val_dataset = StrongAugmentDataset(
        config['data_root'],
        mode='val',
        use_unknown=True,
        unknown_ratio=config['unknown_ratio'],
        use_strong_aug=False
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2
    )
    
    # 모델
    model = ImprovedOpenSetModel(
        num_classes=NUM_KNOWN,
        backbone=config['backbone']
    ).to(device)
    
    # 옵티마이저
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['learning_rate'],
        weight_decay=0.01
    )
    
    # 스케줄러
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5,
        T_mult=2,
        eta_min=1e-5
    )
    
    # 손실 함수
    criterion = OpenSetLoss(
        num_classes=NUM_KNOWN,
        unknown_weight=1.5,
        device=device
    )
    
    best_score = 0
    best_model_state = None
    
    # 학습
    for epoch in range(config['num_epochs']):
        # Progressive unfreezing
        if epoch == 5:
            print("🔓 Unfreezing 60% of backbone...")
            model.unfreeze_backbone(0.6)
        elif epoch == 10:
            print("🔓 Unfreezing entire backbone...")
            model.unfreeze_backbone(1.0)
        
        # Training
        model.train()
        train_losses = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["num_epochs"]}')
        for images, labels, is_known in pbar:
            images = images.to(device)
            labels = labels.to(device)
            is_known_tensor = torch.tensor(is_known).to(device)
            
            optimizer.zero_grad()
            
            # MixUp (25% 확률, epoch 3 이후)
            if config['use_mixup'] and epoch >= 3 and np.random.random() < 0.25:
                known_mask = is_known_tensor
                if known_mask.sum() > 1:
                    known_idx = torch.where(known_mask)[0]
                    known_images = images[known_idx]
                    known_labels = labels[known_idx]
                    
                    mixed_images, labels_a, labels_b, lam = mixup_data(
                        known_images, known_labels, alpha=0.2
                    )
                    
                    outputs = model(mixed_images, return_all=True)
                    known_mask_mixed = torch.ones(len(labels_a), dtype=torch.bool).to(device)
                    
                    loss_a, _ = criterion(outputs, labels_a, known_mask_mixed)
                    loss_b, _ = criterion(outputs, labels_b, known_mask_mixed)
                    loss = lam * loss_a + (1 - lam) * loss_b
                else:
                    outputs = model(images, return_all=True)
                    loss, _ = criterion(outputs, labels, is_known_tensor)
            else:
                outputs = model(images, return_all=True)
                loss, loss_components = criterion(outputs, labels, is_known_tensor)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        unknown_correct = 0
        unknown_total = 0
        
        with torch.no_grad():
            for images, labels, is_known in val_loader:
                images = images.to(device)
                outputs = model(images, return_all=True)
                logits = outputs['logits']
                
                probs = F.softmax(logits, dim=1)
                max_probs, predicted = probs.max(dim=1)
                
                # Known accuracy
                known_mask = torch.tensor(is_known)
                if known_mask.sum() > 0:
                    known_labels = labels[known_mask].to(device)
                    known_predicted = predicted[known_mask]
                    val_correct += known_predicted.eq(known_labels).sum().item()
                    val_total += known_labels.size(0)
                
                # Unknown rejection rate
                unknown_mask = ~known_mask
                if unknown_mask.sum() > 0:
                    unknown_probs = max_probs[unknown_mask]
                    unknown_total += unknown_mask.sum().item()
                    unknown_correct += (unknown_probs < 0.7).sum().item()
        
        # Metrics
        val_acc = 100. * val_correct / val_total if val_total > 0 else 0
        unknown_reject = 100. * unknown_correct / unknown_total if unknown_total > 0 else 0
        combined_score = (val_acc + unknown_reject) / 2
        
        print(f'\nEpoch {epoch+1}:')
        print(f'  Train Loss: {np.mean(train_losses):.4f}')
        print(f'  Known Acc: {val_acc:.1f}%')
        print(f'  Unknown Reject: {unknown_reject:.1f}%')
        print(f'  Combined Score: {combined_score:.1f}%')
        
        # Save best model
        if combined_score > best_score:
            best_score = combined_score
            best_model_state = model.state_dict().copy()
            print(f'  📈 Best model updated!')
    
    # Save final model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Compute statistics for inference
    statistics = compute_class_statistics(model, train_loader, device)
    thresholds = find_optimal_thresholds(model, val_loader, device, statistics)
    
    # Save everything
    torch.save({
        'model_state': model.state_dict(),
        'config': {
            'num_classes': NUM_KNOWN,
            'feature_dim': 512,
            'backbone': config['backbone'],
            'temperature': model.temperature.item()
        },
        'thresholds': thresholds,
        'class_statistics': statistics
    }, config['save_path'])
    
    print(f"\n✅ Training completed!")
    print(f"📁 Model saved to: {config['save_path']}")
    print(f"🏆 Best score: {best_score:.2f}%")
    
    return model

# =========================
# 통계 계산 함수
# =========================
def compute_class_statistics(model, train_loader, device):
    """클래스별 통계 계산"""
    model.eval()
    class_features = {i: [] for i in range(NUM_KNOWN)}
    
    with torch.no_grad():
        for images, labels, is_known in tqdm(train_loader, desc="Computing statistics"):
            images = images.to(device)
            outputs = model(images, return_all=True)
            features = outputs['features'].cpu().numpy()
            
            for i, (label, known) in enumerate(zip(labels, is_known)):
                if known and label >= 0:
                    class_features[label.item()].append(features[i])
    
    statistics = {}
    for class_id in range(NUM_KNOWN):
        if len(class_features[class_id]) > 0:
            features = np.array(class_features[class_id])
            features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
            
            mean = features.mean(axis=0)
            mean_norm = features_norm.mean(axis=0)
            
            # Covariance estimation
            try:
                cov = LedoitWolf(assume_centered=False).fit(features_norm)
                precision = cov.precision_
            except:
                cov = EmpiricalCovariance(assume_centered=False).fit(features_norm)
                covariance = cov.covariance_ + 0.01 * np.eye(features_norm.shape[1])
                precision = np.linalg.inv(covariance)
            
            statistics[class_id] = {
                'mean': mean,
                'mean_normalized': mean_norm,
                'precision': precision,
                'num_samples': len(features)
            }
    
    return statistics

def find_optimal_thresholds(model, val_loader, device, statistics):
    """임계값 최적화"""
    model.eval()
    known_scores = []
    unknown_scores = []
    
    with torch.no_grad():
        for images, labels, is_known in tqdm(val_loader, desc="Finding thresholds"):
            images = images.to(device)
            outputs = model(images, return_all=True)
            
            logits = outputs['logits']
            features = outputs['features'].cpu().numpy()
            distances = outputs['distances']
            recon_error = outputs['reconstruction_error']
            
            probs = F.softmax(logits, dim=1)
            max_prob, _ = probs.max(dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            min_dist = distances.min(dim=1)[0]
            
            for i in range(len(images)):
                score_dict = {
                    'max_prob': max_prob[i].item(),
                    'entropy': entropy[i].item(),
                    'min_distance': min_dist[i].item(),
                    'recon_error': recon_error[i].item()
                }
                
                if is_known[i]:
                    known_scores.append(score_dict)
                else:
                    unknown_scores.append(score_dict)
    
    # 백분위수 기반 임계값
    thresholds = {
        'max_prob': 0.7,
        'entropy': np.percentile([s['entropy'] for s in known_scores], 85),
        'min_distance': np.percentile([s['min_distance'] for s in known_scores], 80),
        'recon_error': np.percentile([s['recon_error'] for s in known_scores], 85)
    }
    
    return thresholds

# =========================
# Main
# =========================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Improved Open Set Recognition Training")
    print("="*60)
    
    model = train_model(CONFIG)
    
    print("\n✨ Training Complete!")