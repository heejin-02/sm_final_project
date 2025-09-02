# MobileNet V2 학습 코드 - Google Colab용
# 10종 해충 분류 모델 학습

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import os
import json
from tqdm import tqdm
import numpy as np

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 해충 클래스 정의 (10종)
INSECT_CLASSES = {
    0: "꽃노랑총채벌레",     # 해충 O
    1: "담배가루이",         # 해충 O  
    2: "복숭아혹진딧물",      # 해충 O
    3: "썩덩나무노린재",      # 해충 O
    4: "비단노린재",         # 해충 X (일반곤충)
    5: "먹노린재",           # 해충 X (일반곤충)
    6: "무잎벌",            # 해충 X (일반곤충)
    7: "배추좀나방",         # 해충 X (일반곤충)
    8: "벼룩잎벌레",         # 해충 X (일반곤충)
    9: "큰28점박이무당벌레"   # 해충 X (일반곤충)
}

class InsectDataset(Dataset):
    """해충 이미지 데이터셋"""
    def __init__(self, root_dir, transform=None, mode='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.samples = []
        
        # 데이터 로드
        for class_idx, class_name in INSECT_CLASSES.items():
            class_dir = os.path.join(root_dir, mode, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(('.jpg', '.png', '.jpeg')):
                        self.samples.append({
                            'path': os.path.join(class_dir, img_name),
                            'label': class_idx
                        })
        
        print(f"{mode} 데이터셋: {len(self.samples)}개 샘플 로드")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['path']).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, sample['label']

class MobileNetInsectClassifier(nn.Module):
    """MobileNet V2 기반 해충 분류 모델"""
    def __init__(self, num_classes=10, pretrained=True):
        super(MobileNetInsectClassifier, self).__init__()
        
        # MobileNet V2 백본
        self.mobilenet = models.mobilenet_v2(pretrained=pretrained)
        
        # 분류 헤드 교체
        in_features = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.mobilenet(x)

def train_model(model, train_loader, val_loader, epochs=30, lr=0.001):
    """모델 학습 함수"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    
    best_val_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 
                             'acc': f'{100.*train_correct/train_total:.2f}%'})
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]'):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        # 에폭 결과
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'\nEpoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'         Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # 기록 저장
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        # 학습률 조정
        scheduler.step(avg_val_loss)
        
        # 최고 모델 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_names': INSECT_CLASSES
            }, 'best_mobilenet_insect.pt')
            print(f'✅ 최고 모델 저장 (Val Acc: {val_acc:.2f}%)')
    
    return history

def export_to_onnx(model, dummy_input, filename='mobilenet_insect.onnx'):
    """ONNX 형식으로 내보내기 (라즈베리파이용)"""
    model.eval()
    torch.onnx.export(
        model,
        dummy_input,
        filename,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"✅ ONNX 모델 저장: {filename}")

def test_model(model, test_loader):
    """최종 테스트 함수"""
    model.eval()
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 클래스별 정확도
            c = (predicted == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    # 전체 정확도
    print(f'\n📊 Test Accuracy: {100 * correct/total:.2f}%')
    
    # 클래스별 정확도
    print('\n클래스별 정확도:')
    for i in range(10):
        if class_total[i] > 0:
            acc = 100 * class_correct[i]/class_total[i]
            print(f'{INSECT_CLASSES[i]}: {acc:.2f}%')
    
    return 100 * correct/total    

def main():
    """메인 실행 함수"""
    # 데이터 변환 정의
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 데이터셋 경로 (Colab에서 수정 필요)
    data_root = '/content/drive/MyDrive/insect_dataset'
    
    # 데이터셋 생성
    train_dataset = InsectDataset(data_root, transform=transform_train, mode='train')
    val_dataset = InsectDataset(data_root, transform=transform_val, mode='val')
    
    # Test 데이터셋 추가 ✨
    test_dataset = InsectDataset(data_root, transform=transform_val, mode='test')
    
    # 데이터로더
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # Test 데이터로더 추가 ✨
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # 모델 생성
    model = MobileNetInsectClassifier(num_classes=10, pretrained=True)
    model = model.to(device)
    
    # 학습
    print("\n🚀 학습 시작...")
    history = train_model(model, train_loader, val_loader, epochs=30)
    
    # 최고 성능 모델 로드 ✨
    print("\n📂 최고 성능 모델 로드 중...")
    checkpoint = torch.load('best_mobilenet_insect.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Epoch {checkpoint['epoch']+1}의 모델 로드 (Val Acc: {checkpoint['val_acc']:.2f}%)")
    
    # Test 데이터셋으로 최종 평가 ✨
    print("\n🔍 Test 데이터셋으로 최종 평가 중...")
    test_accuracy = test_model(model, test_loader)
    
    # 최종 모델 저장 (best 모델 기준)
    torch.save(model.state_dict(), 'final_mobilenet_insect.pt')
    
    # ONNX 내보내기 (best 모델 기준)
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    export_to_onnx(model, dummy_input)
    
    # 학습 기록 저장 (test 결과 포함) ✨
    history['test_acc'] = test_accuracy
    with open('training_history.json', 'w') as f:
        json.dump(history, f)
    
    # 최종 결과 출력 ✨
    print("\n" + "="*50)
    print("✅ 학습 및 평가 완료!")
    print("="*50)
    print(f"📊 최고 Validation 정확도: {max(history['val_acc']):.2f}%")
    print(f"📊 최종 Test 정확도: {test_accuracy:.2f}%")
    print("="*50)
    
    # 과적합 확인 ✨
    if max(history['val_acc']) - test_accuracy > 5:
        print("⚠️ 주의: Validation과 Test 정확도 차이가 5% 이상입니다.")
        print("   과적합(Overfitting) 가능성이 있습니다.")
    else:
        print("✅ 모델이 잘 일반화되었습니다!")

if __name__ == "__main__":
    main()

# Colab 사용법:
# 1. Google Drive 마운트: from google.colab import drive; drive.mount('/content/drive')
# 2. 데이터셋 업로드: /content/drive/MyDrive/insect_dataset/ 경로에 업로드
# 3. 실행: !python mobilenet_train_colab.py
# 4. 모델 다운로드: best_mobilenet_insect.pt, mobilenet_insect.onnx