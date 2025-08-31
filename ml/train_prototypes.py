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

# 4종 해충만 학습 (0-3번이 해충)
PEST_CLASSES = {
    0: "꽃노랑총채벌레",
    1: "담배가루이",
    2: "복숭아혹진딧물",
    3: "썩덩나무노린재"
}

class PestDataset(Dataset):
    """4종 해충 데이터셋"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        # 4종 해충 데이터만 로드
        for class_id, class_name in PEST_CLASSES.items():
            class_dir = os.path.join(root_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(('.jpg', '.png', '.jpeg')):
                        self.samples.append({
                            'path': os.path.join(class_dir, img_name),
                            'label': class_id,
                            'name': class_name
                        })
        
        print(f"✅ {len(self.samples)}개 해충 이미지 로드")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['path']).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, sample['label']

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

def compute_thresholds(features_dict, prototypes):
    """클래스별 임계값 계산 (같은 클래스 내 거리 분포 기반)"""
    thresholds = {}
    
    for class_id, features in features_dict.items():
        if class_id in prototypes and features:
            prototype = prototypes[class_id]
            
            # 프로토타입과 각 샘플 간 거리
            distances = []
            for feat in features:
                # 코사인 유사도
                similarity = np.dot(feat, prototype)
                distances.append(similarity)
            
            # 통계 기반 임계값 (평균 - 2*표준편차)
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            threshold = mean_dist - 2 * std_dist  # 95% 신뢰구간
            
            thresholds[class_id] = max(0.5, threshold)  # 최소 0.5
            
            print(f"클래스 {PEST_CLASSES[class_id]}: 임계값 = {thresholds[class_id]:.3f}")
    
    return thresholds

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
    data_root = './pest_dataset'  # 4종 해충 이미지 폴더
    
    # 데이터셋 및 로더
    dataset = PestDataset(data_root, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # ResNet50 특징 추출기
    model = models.resnet50(pretrained=True)
    model = nn.Sequential(*list(model.children())[:-1])  # FC 레이어 제거
    model = model.to(device)
    model.eval()
    
    print("\n🔍 특징 추출 시작...")
    features_dict = extract_features(model, dataloader, device)
    
    print("\n📊 프로토타입 계산...")
    prototypes = compute_prototypes(features_dict)
    
    print("\n📏 임계값 계산...")
    thresholds = compute_thresholds(features_dict, prototypes)
    
    # 결과 저장
    result = {
        'prototypes': prototypes,
        'thresholds': thresholds,
        'class_names': PEST_CLASSES
    }
    
    # Pickle로 저장 (NumPy 배열 포함)
    with open('pest_prototypes.pkl', 'wb') as f:
        pickle.dump(result, f)
    
    # JSON으로도 저장 (읽기 편함)
    result_json = {
        'thresholds': thresholds,
        'class_names': PEST_CLASSES
    }
    with open('pest_prototypes.json', 'w') as f:
        json.dump(result_json, f, indent=2)
    
    print("\n✅ 프로토타입 저장 완료!")
    print("- pest_prototypes.pkl: 프로토타입 벡터 (Open Set용)")
    print("- pest_prototypes.json: 임계값 및 메타데이터")
    
    # 검증
    print("\n🔬 프로토타입 검증:")
    for class_id, proto in prototypes.items():
        print(f"- {PEST_CLASSES[class_id]}: shape={proto.shape}, norm={np.linalg.norm(proto):.3f}")

if __name__ == "__main__":
    main()

# 사용법:
# 1. 4종 해충 이미지를 ./pest_dataset/ 폴더에 준비
#    pest_dataset/
#    ├── 꽃노랑총채벌레/
#    ├── 담배가루이/
#    ├── 비단노린재/
#    └── 알락수염노린재/
# 
# 2. 실행: python train_prototypes.py
# 3. 생성된 pest_prototypes.pkl을 open_set_recognition.py에서 로드