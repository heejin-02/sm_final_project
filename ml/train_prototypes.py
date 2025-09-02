#!/usr/bin/env python3
"""
Open Set Recognition을 위한 프로토타입 학습
4종 해충의 특징 벡터 추출 및 저장
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import os
import json
from tqdm import tqdm
import pickle
from sklearn.metrics import confusion_matrix, classification_report

# 4종 해충만 학습 (0-3번이 해충)
PEST_CLASSES = {
    0: "꽃노랑총채벌레",
    1: "담배가루이",
    2: "복숭아혹진딧물",
    3: "썩덩나무노린재"
}

# 전체 10종 (테스트용)
ALL_CLASSES = {
    0: "꽃노랑총채벌레",     # 해충 O
    1: "담배가루이",         # 해충 O  
    2: "복숭아혹진딧물",      # 해충 O
    3: "썩덩나무노린재",      # 해충 O
    4: "비단노린재",         # 해충 X (Unknown)
    5: "먹노린재",           # 해충 X (Unknown)
    6: "무잎벌",            # 해충 X (Unknown)
    7: "배추좀나방",         # 해충 X (Unknown)
    8: "벼룩잎벌레",         # 해충 X (Unknown)
    9: "큰28점박이무당벌레"   # 해충 X (Unknown)
}

class PestDataset(Dataset):
    """4종 해충 데이터셋 (학습용)"""
    def __init__(self, root_dir, transform=None, mode='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        # 4종 해충 데이터만 로드
        for class_id, class_name in PEST_CLASSES.items():
            class_dir = os.path.join(root_dir, mode, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(('.jpg', '.png', '.jpeg')):
                        self.samples.append({
                            'path': os.path.join(class_dir, img_name),
                            'label': class_id,
                            'name': class_name
                        })
        
        print(f"✅ {mode}: {len(self.samples)}개 해충 이미지 로드")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['path']).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, sample['label']

class FullDataset(Dataset):
    """전체 10종 데이터셋 (테스트용)"""
    def __init__(self, root_dir, transform=None, mode='test'):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        # 전체 10종 데이터 로드
        for class_id, class_name in ALL_CLASSES.items():
            class_dir = os.path.join(root_dir, mode, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(('.jpg', '.png', '.jpeg')):
                        self.samples.append({
                            'path': os.path.join(class_dir, img_name),
                            'label': class_id,
                            'name': class_name,
                            'is_known': class_id < 4  # 0-3번만 Known
                        })
        
        known_count = sum(1 for s in self.samples if s['is_known'])
        unknown_count = len(self.samples) - known_count
        print(f"✅ {mode}: 총 {len(self.samples)}개 (Known: {known_count}, Unknown: {unknown_count})")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['path']).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, sample['label'], sample['is_known']

def extract_features(model, dataloader, device):
    """특징 벡터 추출"""
    model.eval()
    features_dict = {i: [] for i in range(4)}  # 4종
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="특징 추출 중"):
            images = images.to(device)
            
            # ResNet50 특징 추출
            features = model(images)
            features = features.squeeze().cpu().numpy()
            
            # 클래스별로 저장
            for feat, label in zip(features, labels):
                # L2 정규화
                feat_norm = feat / np.linalg.norm(feat)
                features_dict[label.item()].append(feat_norm)
    
    return features_dict

def compute_prototypes(features_dict):
    """클래스별 프로토타입 계산 (평균 특징 벡터)"""
    prototypes = {}
    
    for class_id, features in features_dict.items():
        if features:
            # 평균 계산
            prototype = np.mean(features, axis=0)
            # 다시 정규화
            prototype = prototype / np.linalg.norm(prototype)
            prototypes[class_id] = prototype
            
            print(f"클래스 {PEST_CLASSES[class_id]}: {len(features)}개 샘플에서 프로토타입 생성")
    
    return prototypes

def compute_thresholds(features_dict, prototypes, percentile=5):
    """클래스별 임계값 계산 (같은 클래스 내 거리 분포 기반)"""
    thresholds = {}
    
    for class_id, features in features_dict.items():
        if class_id in prototypes and features:
            prototype = prototypes[class_id]
            
            # 프로토타입과 각 샘플 간 코사인 유사도
            similarities = []
            for feat in features:
                similarity = np.dot(feat, prototype)
                similarities.append(similarity)
            
            # 백분위수 기반 임계값 (하위 5%)
            threshold = np.percentile(similarities, percentile)
            thresholds[class_id] = max(0.5, threshold)  # 최소 0.5
            
            print(f"클래스 {PEST_CLASSES[class_id]}: 임계값 = {thresholds[class_id]:.3f} "
                  f"(min={min(similarities):.3f}, max={max(similarities):.3f})")
    
    return thresholds

def test_open_set(model, test_loader, prototypes, thresholds, device):
    """Open Set Recognition 테스트"""
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_is_known = []
    all_max_similarities = []
    
    with torch.no_grad():
        for images, labels, is_known in tqdm(test_loader, desc="테스트 중"):
            images = images.to(device)
            
            # 특징 추출
            features = model(images)
            features = features.squeeze().cpu().numpy()
            
            # 각 샘플에 대해 예측
            for feat, label, known in zip(features, labels, is_known):
                # L2 정규화
                feat_norm = feat / np.linalg.norm(feat)
                
                # 모든 프로토타입과의 유사도 계산
                similarities = {}
                for class_id, prototype in prototypes.items():
                    similarity = np.dot(feat_norm, prototype)
                    similarities[class_id] = similarity
                
                # 가장 유사한 클래스 찾기
                best_class = max(similarities, key=similarities.get)
                max_similarity = similarities[best_class]
                
                # 임계값 확인 (Unknown 판단)
                if max_similarity >= thresholds[best_class]:
                    prediction = best_class
                else:
                    prediction = -1  # Unknown
                
                all_predictions.append(prediction)
                all_labels.append(label.item() if label.item() < 4 else -1)  # 4-9번은 Unknown으로
                all_is_known.append(known.item())
                all_max_similarities.append(max_similarity)
    
    return all_predictions, all_labels, all_is_known, all_max_similarities

def evaluate_results(predictions, labels, is_known, similarities):
    """결과 평가 및 출력"""
    predictions = np.array(predictions)
    labels = np.array(labels)
    is_known = np.array(is_known)
    similarities = np.array(similarities)
    
    print("\n" + "="*60)
    print("📊 Open Set Recognition 평가 결과")
    print("="*60)
    
    # 1. Known 클래스 정확도
    known_mask = is_known == True
    if known_mask.sum() > 0:
        known_preds = predictions[known_mask]
        known_labels = labels[known_mask]
        known_correct = (known_preds == known_labels).sum()
        known_acc = known_correct / len(known_preds) * 100
        
        print(f"\n✅ Known Classes (4종 해충):")
        print(f"   - 정확도: {known_acc:.2f}% ({known_correct}/{len(known_preds)})")
        
        # 클래스별 정확도
        for class_id in range(4):
            class_mask = known_labels == class_id
            if class_mask.sum() > 0:
                class_acc = (known_preds[class_mask] == class_id).sum() / class_mask.sum() * 100
                print(f"   - {PEST_CLASSES[class_id]}: {class_acc:.2f}%")
    
    # 2. Unknown 클래스 거부율
    unknown_mask = is_known == False
    if unknown_mask.sum() > 0:
        unknown_preds = predictions[unknown_mask]
        unknown_rejected = (unknown_preds == -1).sum()
        rejection_rate = unknown_rejected / len(unknown_preds) * 100
        
        print(f"\n❌ Unknown Classes (6종 일반곤충):")
        print(f"   - 거부율: {rejection_rate:.2f}% ({unknown_rejected}/{len(unknown_preds)})")
        print(f"   - 오인식: {100-rejection_rate:.2f}% ({len(unknown_preds)-unknown_rejected}개)")
    
    # 3. 전체 성능
    total_correct = 0
    # Known을 올바르게 분류
    total_correct += ((predictions[known_mask] == labels[known_mask]).sum() if known_mask.sum() > 0 else 0)
    # Unknown을 올바르게 거부
    total_correct += ((predictions[unknown_mask] == -1).sum() if unknown_mask.sum() > 0 else 0)
    
    total_acc = total_correct / len(predictions) * 100
    
    print(f"\n📈 전체 Open Set 성능:")
    print(f"   - 정확도: {total_acc:.2f}% ({total_correct}/{len(predictions)})")
    
    # 4. 유사도 통계
    print(f"\n📏 유사도 통계:")
    print(f"   - Known 평균: {similarities[known_mask].mean():.3f} (±{similarities[known_mask].std():.3f})")
    print(f"   - Unknown 평균: {similarities[unknown_mask].mean():.3f} (±{similarities[unknown_mask].std():.3f})")
    
    return {
        'known_acc': known_acc if known_mask.sum() > 0 else 0,
        'rejection_rate': rejection_rate if unknown_mask.sum() > 0 else 0,
        'total_acc': total_acc
    }

def main():
    # 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 데이터 변환
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 데이터셋 경로 (수정 필요)
    data_root = './insect_dataset'
    
    # Train 데이터셋 (4종 해충만)
    print("\n📂 Train 데이터 로드...")
    train_dataset = PestDataset(data_root, transform=transform, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # ResNet50 특징 추출기
    model = models.resnet50(pretrained=True)
    model = nn.Sequential(*list(model.children())[:-1])  # FC 레이어 제거
    model = model.to(device)
    model.eval()
    
    print("\n🔍 특징 추출 시작...")
    features_dict = extract_features(model, train_loader, device)
    
    print("\n📊 프로토타입 계산...")
    prototypes = compute_prototypes(features_dict)
    
    print("\n📏 임계값 계산...")
    thresholds = compute_thresholds(features_dict, prototypes, percentile=5)
    
    # Test 데이터셋 (전체 10종)
    print("\n📂 Test 데이터 로드...")
    test_dataset = FullDataset(data_root, transform=transform, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    print("\n🧪 Open Set Recognition 테스트...")
    predictions, labels, is_known, similarities = test_open_set(
        model, test_loader, prototypes, thresholds, device
    )
    
    # 평가
    results = evaluate_results(predictions, labels, is_known, similarities)
    
    # 결과 저장
    save_data = {
        'prototypes': prototypes,
        'thresholds': thresholds,
        'class_names': PEST_CLASSES,
        'test_results': results
    }
    
    # Pickle로 저장
    with open('pest_prototypes.pkl', 'wb') as f:
        pickle.dump(save_data, f)
    
    # JSON으로도 저장 (읽기 편함)
    result_json = {
        'thresholds': {k: float(v) for k, v in thresholds.items()},
        'class_names': PEST_CLASSES,
        'test_results': results
    }
    with open('pest_prototypes.json', 'w') as f:
        json.dump(result_json, f, indent=2)
    
    print("\n" + "="*60)
    print("✅ 완료!")
    print("- pest_prototypes.pkl: 프로토타입 벡터 및 결과")
    print("- pest_prototypes.json: 임계값 및 테스트 결과")
    print("="*60)

if __name__ == "__main__":
    main()