#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete Open Set Recognition System for Pest Detection
클래스 불균형 해결 + 신뢰도 보정 완전판
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
import json
from scipy.spatial.distance import cdist
from sklearn.covariance import EmpiricalCovariance
import warnings
warnings.filterwarnings('ignore')
import os

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

# 실제 데이터 분포
ACTUAL_CLASS_COUNTS = {
    'train': {
        0: 801,   # 꽃노랑총채벌레
        1: 274,   # 담배가루이 (최소)
        2: 702,   # 복숭아혹진딧물
        3: 1574   # 썩덩나무노린재 (최대)
    },
    'val': {
        0: 245,
        1: 113,
        2: 217,
        3: 488
    },
    'test': {
        0: 241,
        1: 119,
        2: 214,
        3: 485
    }
}

# =========================
# Improved Model with Temperature Scaling
# =========================

class CalibratedOpenSetModel(nn.Module):
    """
    신뢰도 보정을 포함한 개선된 모델:
    1. Temperature scaling for calibration
    2. Multiple detection mechanisms
    3. Better feature extraction
    """

    def __init__(self, num_classes=4, feature_dim=512, initial_temperature=1.5):
        super().__init__()

        # ResNet101 backbone
        resnet = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # Feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, feature_dim)
        )

        # Classifiers
        self.main_classifier = nn.Linear(feature_dim, num_classes)
        self.auxiliary_classifier = nn.Linear(feature_dim, num_classes)

        # Reconstruction decoder
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 2048)
        )

        # Distance-based detector
        self.prototype_layer = nn.Linear(feature_dim, num_classes, bias=False)

        # Temperature parameter for calibration (learnable)
        self.temperature = nn.Parameter(torch.ones(1) * initial_temperature)
        
        # Class-specific thresholds
        self.class_thresholds = nn.Parameter(torch.ones(num_classes) * 0.5)

        self._initialize_weights()
        self._freeze_early_layers()

    def _initialize_weights(self):
        for m in [self.encoder, self.main_classifier, self.auxiliary_classifier, self.decoder]:
            if isinstance(m, nn.Sequential):
                for layer in m:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        if layer.bias is not None:
                            nn.init.constant_(layer.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _freeze_early_layers(self):
        # 초기 레이어 동결
        for param in list(self.features.parameters())[:-30]:
            param.requires_grad = False

    def forward(self, x, return_all=False, apply_temperature=True):
        # Feature extraction
        cnn_features = self.features(x)
        cnn_features = cnn_features.view(cnn_features.size(0), -1)

        # Encoding
        features = self.encoder(cnn_features)

        # Classification with temperature scaling
        logits = self.main_classifier(features)
        aux_logits = self.auxiliary_classifier(features)
        
        # Apply temperature scaling for calibration
        if apply_temperature:
            temperature = torch.clamp(self.temperature, min=0.5, max=3.0)  # 범위 제한
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
                'distances': distances,
                'cnn_features': cnn_features,
                'temperature': self.temperature.item()
            }

        return logits, features

    def unfreeze_backbone(self, layers=40):
        """Progressive unfreezing"""
        backbone_params = list(self.features.parameters())
        for param in backbone_params[-layers:]:
            param.requires_grad = True

# =========================
# Balanced Loss with Label Smoothing
# =========================

class BalancedCalibratedLoss(nn.Module):
    def __init__(self, num_classes, class_counts, feature_dim=512, device='cuda', 
                 label_smoothing=0.1, use_focal=False):
        super().__init__()
        self.num_classes = num_classes
        self.device = device
        
        # 클래스 가중치 계산
        total = sum(class_counts.values())
        class_weights = []
        for i in range(num_classes):
            count = class_counts.get(i, 1)
            weight = np.sqrt(total / (num_classes * count))
            class_weights.append(weight)
        
        # numpy array를 float32 tensor로 변환 (중요!)
        class_weights = np.array(class_weights, dtype=np.float32)
        class_weights = class_weights / class_weights.mean()
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        
        print(f"📊 Class weights with sqrt smoothing:")
        for i, w in enumerate(class_weights.tolist()):
            print(f"   {KNOWN_CLASSES[i]}: {w:.3f}")
        
        # Label smoothing 적용
        self.ce_loss = nn.CrossEntropyLoss(
            weight=self.class_weights,
            label_smoothing=label_smoothing
        )
        
        self.aux_ce_loss = nn.CrossEntropyLoss(weight=self.class_weights)
        
        # Class centers
        self.centers = nn.Parameter(torch.randn(num_classes, feature_dim).to(device))
        self.center_loss_weight = 0.1

    def forward(self, outputs, labels, is_known_mask):
        logits = outputs['logits']
        aux_logits = outputs['aux_logits']
        features = outputs['features']
        reconstruction_error = outputs['reconstruction_error']
        distances = outputs['distances']

        # Filter known samples
        known_idx = torch.where(is_known_mask)[0]
        if len(known_idx) == 0:
            return torch.tensor(0.0, requires_grad=True, device=self.device), {}

        known_logits = logits[known_idx]
        known_aux_logits = aux_logits[known_idx]
        known_labels = labels[known_idx]
        known_features = features[known_idx]

        # 1. Classification loss with label smoothing
        ce_loss = self.ce_loss(known_logits, known_labels)
        aux_ce_loss = self.aux_ce_loss(known_aux_logits, known_labels)

        # 2. Center loss
        center_loss = 0
        for i in range(self.num_classes):
            mask = known_labels == i
            if mask.sum() > 0:
                class_features = known_features[mask]
                center_dist = torch.sum((class_features - self.centers[i]) ** 2, dim=1)
                center_loss += center_dist.mean()
        center_loss /= self.num_classes

        # 3. Entropy regularization (encourage confident predictions for known)
        probs = F.softmax(known_logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        entropy_loss = entropy.mean()

        # 4. Reconstruction loss
        recon_loss = reconstruction_error[known_idx].mean()

        # 5. Distance regularization
        min_distances = distances[known_idx].min(dim=1)[0]
        distance_loss = min_distances.mean()

        # Combine losses
        total_loss = (
            ce_loss +
            0.3 * aux_ce_loss +
            self.center_loss_weight * center_loss +
            0.05 * entropy_loss +
            0.01 * recon_loss +
            0.1 * distance_loss
        )

        return total_loss, {
            'ce': ce_loss.item(),
            'aux_ce': aux_ce_loss.item(),
            'center': center_loss.item(),
            'entropy': entropy_loss.item(),
            'reconstruction': recon_loss.item(),
            'distance': distance_loss.item()
        }

# =========================
# Enhanced Dataset with Mixup Support
# =========================

class BalancedDataset(Dataset):
    """균형잡힌 데이터셋"""

    def __init__(self, root_dir, mode='train', use_unknown=False, 
                 unknown_ratio=0.3, balance_data=True):
        self.root_dir = root_dir
        self.mode = mode
        self.samples = []
        self.class_counts = {i: 0 for i in range(NUM_KNOWN)}

        # Different transforms for train/val/test
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # Load Known classes with balancing
        if balance_data and mode == 'train':
            self._load_balanced_data()
        else:
            self._load_normal_data()

        # Add Unknown samples
        if use_unknown and mode in ['train', 'val']:
            self._add_unknown_samples(unknown_ratio)

        print(f"✅ {mode} dataset loaded:")
        for i in range(NUM_KNOWN):
            print(f"   {KNOWN_CLASSES[i]}: {self.class_counts[i]} samples")

    def _load_balanced_data(self):
        """균형잡힌 데이터 로드 (train only)"""
        import random
        
        # 각 클래스별 샘플 수집
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

        # 목표: 담배가루이(274)의 2배 = 548개
        target_samples = 548
        
        for class_id, samples in class_samples.items():
            current_count = len(samples)
            
            if class_id == 1:  # 담배가루이
                # 2배로 오버샘플링
                extended_samples = samples * 2
                self.samples.extend(extended_samples)
                self.class_counts[class_id] = len(extended_samples)
            elif current_count > target_samples:
                # 언더샘플링
                random.seed(42 + class_id)
                selected = random.sample(samples, target_samples)
                self.samples.extend(selected)
                self.class_counts[class_id] = target_samples
            else:
                # 그대로 사용
                self.samples.extend(samples)
                self.class_counts[class_id] = current_count

    def _load_normal_data(self):
        """일반 데이터 로드 (val/test)"""
        for class_id, class_name in KNOWN_CLASSES.items():
            class_dir = os.path.join(self.root_dir, self.mode, class_name)
            if not os.path.isdir(class_dir):
                continue

            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                    self.samples.append({
                        'path': os.path.join(class_dir, img_name),
                        'label': class_id,
                        'is_known': True
                    })
                    self.class_counts[class_id] += 1

    def _add_unknown_samples(self, unknown_ratio):
        """Unknown 샘플 추가"""
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

        # Unknown 비율 조절
        num_unknown = int(len(self.samples) * unknown_ratio)
        if len(unknown_samples) > num_unknown:
            import random
            random.seed(42)
            unknown_samples = random.sample(unknown_samples, num_unknown)

        self.samples.extend(unknown_samples)
        print(f"   Unknown: {len(unknown_samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['path']).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, sample['label'], sample['is_known']

# =========================
# Mixup Augmentation
# =========================

def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation for better calibration"""
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
# Training with Calibration
# =========================

def train_calibrated_model(model, train_loader, val_loader, device, num_epochs=25):
    """신뢰도 보정을 포함한 학습"""
    
    # Optimizer
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.001,
        weight_decay=0.01
    )

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=0.00001
    )

    # Loss function with class balancing
    criterion = BalancedCalibratedLoss(
        NUM_KNOWN,
        ACTUAL_CLASS_COUNTS['train'],
        device=device,
        label_smoothing=0.1
    )

    best_score = 0
    best_model_state = None
    patience = 0
    max_patience = 5

    for epoch in range(num_epochs):
        # Progressive unfreezing
        if epoch == 3:
            print("🔓 Unfreezing more backbone layers (stage 1)...")
            model.unfreeze_backbone(layers=20)
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5

        if epoch == 8:
            print("🔓 Unfreezing all backbone layers (stage 2)...")
            model.unfreeze_backbone(layers=40)
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5

        # Training
        model.train()
        train_losses = []
        class_correct = {i: 0 for i in range(NUM_KNOWN)}
        class_total = {i: 0 for i in range(NUM_KNOWN)}

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for images, labels, is_known in pbar:
            images = images.to(device)
            labels = labels.to(device)
            is_known_tensor = torch.tensor(is_known).to(device)
            
            optimizer.zero_grad()
            
            # Mixup augmentation (25% 확률, 5 epoch 이후)
            use_mixup = np.random.random() < 0.25 and epoch >= 5
            
            if use_mixup:
                known_mask = is_known_tensor
                if known_mask.sum() > 1:
                    # Mixup 적용할 known samples 분리
                    known_indices = torch.where(known_mask)[0]
                    known_images = images[known_indices]
                    known_labels = labels[known_indices]
                    
                    # Mixup 수행
                    mixed_images, labels_a, labels_b, lam = mixup_data(
                        known_images, known_labels, alpha=0.2
                    )
                    
                    # Mixup 이미지로 forward
                    outputs_mixed = model(mixed_images, return_all=True)
                    
                    # Mixup loss 계산
                    known_mask_for_mixed = torch.ones(len(labels_a), dtype=torch.bool).to(device)
                    loss_a, _ = criterion(outputs_mixed, labels_a, known_mask_for_mixed)
                    loss_b, _ = criterion(outputs_mixed, labels_b, known_mask_for_mixed)
                    loss = lam * loss_a + (1 - lam) * loss_b
                else:
                    # Mixup 불가능한 경우 일반 forward
                    outputs = model(images, return_all=True)
                    loss, loss_components = criterion(outputs, labels, is_known_tensor)
            else:
                # Mixup 없이 일반 forward
                outputs = model(images, return_all=True)
                loss, loss_components = criterion(outputs, labels, is_known_tensor)
            
            # Backward 및 optimizer step
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            
            # Accuracy tracking - 원본 이미지로 다시 계산
            with torch.no_grad():
                # 항상 원본 전체 배치로 예측
                outputs_for_acc = model(images, return_all=True)
                _, predicted = outputs_for_acc['logits'].max(1)
                
                # 원본 배치 크기만큼 반복
                for i in range(len(labels)):
                    if is_known[i] and labels[i] >= 0:
                        label = labels[i].item()
                        class_total[label] += 1
                        if predicted[i] == label:
                            class_correct[label] += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'temp': f'{model.temperature.item():.2f}'
            })

        scheduler.step()

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_class_correct = {i: 0 for i in range(NUM_KNOWN)}
        val_class_total = {i: 0 for i in range(NUM_KNOWN)}
        unknown_correct = 0
        unknown_total = 0
        
        confidence_scores = []

        with torch.no_grad():
            for images, labels, is_known in val_loader:
                images = images.to(device)

                outputs = model(images, return_all=True)
                logits = outputs['logits']
                
                probs = F.softmax(logits, dim=1)
                max_probs, predicted = probs.max(dim=1)
                
                # Collect confidence scores
                confidence_scores.extend(max_probs.cpu().numpy())

                # Known samples accuracy
                known_mask = torch.tensor(is_known)
                if known_mask.sum() > 0:
                    known_logits = logits[known_mask]
                    known_labels = labels[known_mask].to(device)
                    known_predicted = predicted[known_mask]
                    
                    for i in range(len(known_labels)):
                        label = known_labels[i].item()
                        val_class_total[label] += 1
                        if known_predicted[i] == label:
                            val_class_correct[label] += 1
                    
                    val_total += known_labels.size(0)
                    val_correct += known_predicted.eq(known_labels).sum().item()

                # Unknown samples
                unknown_mask = ~known_mask
                if unknown_mask.sum() > 0:
                    unknown_logits = logits[unknown_mask]
                    unknown_probs = probs[unknown_mask]
                    unknown_max_probs = unknown_probs.max(dim=1)[0]
                    unknown_total += unknown_mask.sum().item()
                    unknown_correct += (unknown_max_probs < 0.7).sum().item()

        # Calculate metrics
        val_acc = 100. * val_correct / val_total if val_total > 0 else 0
        unknown_acc = 100. * unknown_correct / unknown_total if unknown_total > 0 else 0
        avg_confidence = np.mean(confidence_scores)
        
        # Class-wise accuracy
        print(f'\nEpoch {epoch+1} - Training:')
        for i in range(NUM_KNOWN):
            if class_total[i] > 0:
                acc = 100. * class_correct[i] / class_total[i]
                print(f'  {KNOWN_CLASSES[i]}: {acc:.1f}%')
        
        print(f'\nValidation:')
        for i in range(NUM_KNOWN):
            if val_class_total[i] > 0:
                acc = 100. * val_class_correct[i] / val_class_total[i]
                print(f'  {KNOWN_CLASSES[i]}: {acc:.1f}%')
        
        # Combined score
        combined_score = (val_acc + unknown_acc) / 2
        avg_train_loss = np.mean(train_losses)

        print(f'\nOverall - Loss: {avg_train_loss:.4f}, Known: {val_acc:.1f}%, '
              f'Unknown: {unknown_acc:.1f}%, Combined: {combined_score:.1f}%')
        print(f'Average Confidence: {avg_confidence:.3f}, Temperature: {model.temperature.item():.3f}')

        # Save best model
        if combined_score > best_score:
            best_score = combined_score
            best_model_state = model.state_dict().copy()
            patience = 0
            print(f'📈 Best model updated: {combined_score:.2f}%')
        else:
            patience += 1
            if patience >= max_patience and epoch > 10:
                print(f'⚠️ Early stopping at epoch {epoch+1}')
                break

    model.load_state_dict(best_model_state)
    return model

# =========================
# Threshold Optimization (기존과 동일)
# =========================

class MultiMetricThresholdOptimizer:
    """임계값 최적화"""

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.optimal_thresholds = {}
        self.class_statistics = {}

    def compute_class_statistics(self, train_loader):
        """클래스별 통계 계산"""
        self.model.eval()

        class_features = {i: [] for i in range(NUM_KNOWN)}

        with torch.no_grad():
            for images, labels, is_known in tqdm(train_loader, desc="Computing statistics"):
                images = images.to(self.device)
                is_known_tensor = torch.tensor(is_known).to(self.device)

                if is_known_tensor.sum() == 0:
                    continue

                outputs = self.model(images, return_all=True)
                features = outputs['features'].cpu().numpy()

                for i, (label, known) in enumerate(zip(labels, is_known)):
                    if known and label >= 0:
                        class_features[label.item()].append(features[i])

        # Compute mean and covariance
        for class_id in range(NUM_KNOWN):
            if len(class_features[class_id]) > 0:
                features = np.array(class_features[class_id])
                mean = features.mean(axis=0)

                cov = EmpiricalCovariance(assume_centered=False).fit(features)

                self.class_statistics[class_id] = {
                    'mean': mean,
                    'precision': cov.precision_,
                    'covariance': cov.covariance_
                }

        print(f"✅ Computed statistics for {len(self.class_statistics)} classes")

    def extract_validation_scores(self, val_loader):
        """검증 세트에서 점수 추출"""
        self.model.eval()
        
        known_scores = []
        unknown_scores = []
        
        with torch.no_grad():
            for images, labels, is_known in tqdm(val_loader, desc="Extracting scores"):
                images = images.to(self.device)
                
                outputs = self.model(images, return_all=True)
                logits = outputs['logits']
                features = outputs['features'].cpu().numpy()
                recon_error = outputs['reconstruction_error']
                distances = outputs['distances']
                
                probs = F.softmax(logits, dim=1)
                max_prob, predicted = probs.max(dim=1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
                min_dist = distances.min(dim=1)[0]
                
                # 마할라노비스 거리 계산
                mahal_distances = []
                for feat in features:
                    min_mahal = float('inf')
                    for class_id, stats in self.class_statistics.items():
                        if stats is not None:
                            diff = feat - stats['mean']
                            mahal = np.sqrt(np.abs(diff @ stats['precision'] @ diff))
                            min_mahal = min(min_mahal, mahal)
                    mahal_distances.append(min_mahal)
                
                for i in range(len(images)):
                    score_dict = {
                        'max_prob': max_prob[i].item(),
                        'entropy': entropy[i].item(),
                        'recon_error': recon_error[i].item(),
                        'min_distance': min_dist[i].item(),
                        'mahal_distance': mahal_distances[i],
                        'predicted_class': predicted[i].item()
                    }
                    
                    if is_known[i]:
                        known_scores.append(score_dict)
                    else:
                        unknown_scores.append(score_dict)
        
        return known_scores, unknown_scores
    
    def find_optimal_thresholds(self, val_loader):
        """최적 임계값 찾기 (간단 버전)"""
        
        print("\n🔍 Finding optimal thresholds...")
        known_scores, unknown_scores = self.extract_validation_scores(val_loader)
        
        if len(unknown_scores) == 0:
            print("⚠️ No unknown samples in validation!")
            # 보수적인 기본값
            return {
                'max_prob': 0.85,
                'entropy': 0.4,
                'min_distance': np.percentile([s['min_distance'] for s in known_scores], 70),
                'mahal_distance': np.percentile([s['mahal_distance'] for s in known_scores], 80),
                'recon_error': np.percentile([s['recon_error'] for s in known_scores], 80),
                'known_acc': 0.8,
                'unknown_reject': 0.0
            }
        
        # 백분위수 기반 임계값 설정
        thresholds = {
            'max_prob': 0.7,  # 고정값
            'entropy': np.percentile([s['entropy'] for s in known_scores], 90),
            'min_distance': np.percentile([s['min_distance'] for s in known_scores], 80),
            'mahal_distance': np.percentile([s['mahal_distance'] for s in known_scores], 85),
            'recon_error': np.percentile([s['recon_error'] for s in known_scores], 85)
        }
        
        # 간단한 평가
        tp = sum(1 for s in known_scores if self._accept_sample(s, thresholds))
        tn = sum(1 for s in unknown_scores if not self._accept_sample(s, thresholds))
        
        known_acc = tp / len(known_scores) if known_scores else 0
        unknown_reject = tn / len(unknown_scores) if unknown_scores else 0
        
        thresholds['known_acc'] = known_acc
        thresholds['unknown_reject'] = unknown_reject
        
        print(f"\n✅ Optimal thresholds found:")
        print(f"   Max Probability: {thresholds['max_prob']:.3f}")
        print(f"   Entropy: {thresholds['entropy']:.3f}")
        print(f"   Min Distance: {thresholds['min_distance']:.3f}")
        print(f"   Mahalanobis Distance: {thresholds['mahal_distance']:.3f}")
        print(f"   Reconstruction Error: {thresholds['recon_error']:.6f}")
        print(f"   Expected Known Acc: {known_acc*100:.1f}%")
        print(f"   Expected Unknown Reject: {unknown_reject*100:.1f}%")
        
        return thresholds
    
    def _accept_sample(self, scores, thresholds):
        """샘플 수락 여부 판단"""
        return (scores['max_prob'] >= thresholds['max_prob'] and
                scores['entropy'] <= thresholds['entropy'] and
                scores['min_distance'] <= thresholds['min_distance'] and
                scores['mahal_distance'] <= thresholds['mahal_distance'] and
                scores['recon_error'] <= thresholds['recon_error'])

# =========================
# Calibrated Predictor
# =========================

class CalibratedPredictor:
    """신뢰도 보정된 예측기"""

    def __init__(self, model, thresholds, class_statistics, device, max_confidence=0.95):
        self.model = model
        self.thresholds = thresholds
        self.class_statistics = class_statistics
        self.device = device
        self.max_confidence = max_confidence  # 신뢰도 상한선

    def predict_with_uncertainty(self, images, n_samples=5):
        """MC Dropout으로 불확실성 추정"""
        self.model.train()  # Dropout 활성화
        
        predictions = []
        for _ in range(n_samples):
            with torch.no_grad():
                outputs = self.model(images, return_all=True)
                probs = F.softmax(outputs['logits'], dim=1)
                predictions.append(probs)
        
        predictions = torch.stack(predictions)
        mean_probs = predictions.mean(dim=0)
        std_probs = predictions.std(dim=0)
        
        max_probs, predicted_classes = mean_probs.max(dim=1)
        uncertainty = std_probs.max(dim=1)[0]
        
        # 불확실성 기반 신뢰도 조정
        adjusted_confidence = max_probs * (1 - 0.5 * uncertainty)
        adjusted_confidence = torch.clamp(adjusted_confidence, max=self.max_confidence)
        
        return predicted_classes, adjusted_confidence, uncertainty

    def predict(self, images, use_uncertainty=True):
        """예측 with 신뢰도 보정"""
        
        if use_uncertainty:
            predicted_classes, confidence, uncertainty = self.predict_with_uncertainty(images)
            
            # Unknown detection
            predictions = []
            for i in range(len(images)):
                if confidence[i] < 0.6 or uncertainty[i] > 0.3:
                    predictions.append(-1)  # Unknown
                else:
                    predictions.append(predicted_classes[i].item())
            
            return predictions, confidence.cpu().numpy(), uncertainty.cpu().numpy()
        
        else:
            # Standard prediction
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(images, return_all=True)
                logits = outputs['logits']
                
                probs = F.softmax(logits, dim=1)
                max_probs, predicted = probs.max(dim=1)
                
                # Apply confidence cap
                max_probs = torch.clamp(max_probs, max=self.max_confidence)
                
                predictions = []
                for i in range(len(images)):
                    if max_probs[i] < 0.6:
                        predictions.append(-1)
                    else:
                        predictions.append(predicted[i].item())
                
                return predictions, max_probs.cpu().numpy(), None

# =========================
# Main
# =========================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")

    data_root = '/content/drive/MyDrive/open_set/datasets2_organized'

    print("\n" + "="*60)
    print("🚀 Calibrated Open Set Pest Detection Model")
    print("="*60)

    # Create datasets
    train_dataset = BalancedDataset(
        data_root, 'train', 
        use_unknown=True, 
        unknown_ratio=0.4,
        balance_data=True
    )
    
    val_dataset = BalancedDataset(
        data_root, 'val', 
        use_unknown=True, 
        unknown_ratio=0.4,
        balance_data=False  # Validation은 원본 유지
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    # Create and train model
    print("\n📚 Training calibrated model...")
    model = CalibratedOpenSetModel(
        num_classes=NUM_KNOWN, 
        feature_dim=512,
        initial_temperature=1.5
    ).to(device)

    model = train_calibrated_model(model, train_loader, val_loader, device, num_epochs=25)

    # Compute class statistics and find optimal thresholds
    print("\n📊 Computing class statistics...")
    threshold_optimizer = MultiMetricThresholdOptimizer(model, device)
    threshold_optimizer.compute_class_statistics(train_loader)
    
    print("\n🎯 Optimizing thresholds...")
    optimal_thresholds = threshold_optimizer.find_optimal_thresholds(val_loader)

    # Save model with all components
    print("\n💾 Saving complete model...")
    torch.save({
        'model_state': model.state_dict(),
        'config': {
            'num_classes': NUM_KNOWN,
            'feature_dim': 512,
            'temperature': model.temperature.item()
        },
        'thresholds': optimal_thresholds,  # 임계값 추가
        'class_statistics': threshold_optimizer.class_statistics  # 통계 추가
    }, '/content/drive/MyDrive/open_set/improved_pest_detection_model.pth')  # 파일명도 통일

    print("\n✅ Training completed successfully!")
    print(f"📁 Model saved with:")
    print(f"   - Model weights")
    print(f"   - Optimal thresholds")
    print(f"   - Class statistics")
    print(f"   - Temperature: {model.temperature.item():.3f}")

if __name__ == "__main__":
    main()