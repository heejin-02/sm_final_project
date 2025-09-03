"""
Open Set Recognition 테스트 및 디버깅
임계값 문제 해결 버전
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models, transforms
import cv2
import os
import logging
from PIL import Image
from sklearn.covariance import EmpiricalCovariance

# ========================
# 모델 클래스 정의 (필수!)
# ========================

class CalibratedOpenSetModel(nn.Module):
    """train_prototypes.py와 동일한 모델 구조"""
    def __init__(self, num_classes=4, feature_dim=512, initial_temperature=1.5):
        super().__init__()
        
        resnet = models.resnet101(weights=None)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
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
        
        self.main_classifier = nn.Linear(feature_dim, num_classes)
        self.auxiliary_classifier = nn.Linear(feature_dim, num_classes)
        
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 2048)
        )
        
        self.prototype_layer = nn.Linear(feature_dim, num_classes, bias=False)
        self.temperature = nn.Parameter(torch.ones(1) * initial_temperature)
        self.class_thresholds = nn.Parameter(torch.ones(num_classes) * 0.5)
    
    def forward(self, x, return_all=False):
        cnn_features = self.features(x)
        cnn_features = cnn_features.view(cnn_features.size(0), -1)
        
        features = self.encoder(cnn_features)
        logits = self.main_classifier(features)
        aux_logits = self.auxiliary_classifier(features)
        
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
                'cnn_features': cnn_features
            }
        
        return logits, features

# ========================
# 개선된 인식기
# ========================

class ImprovedOpenSetRecognizer:
    def __init__(self, model_path='improved_pest_detection_model.pth', device='cpu'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(__name__)
        
        self.known_insects = {
            0: "꽃노랑총채벌레",
            1: "담배가루이",
            2: "복숭아혹진딧물",
            3: "썩덩나무노린재"
        }
        
        self._load_model(model_path)
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.logger.info(f"✅ 모델 로드 완료: {model_path}")
        self.logger.info(f"📊 Device: {self.device}")
    
    def _load_model(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # 모델 초기화 (temperature 포함)
        config = checkpoint.get('config', {})
        self.model = CalibratedOpenSetModel(
            num_classes=config.get('num_classes', 4),
            feature_dim=config.get('feature_dim', 512),
            initial_temperature=config.get('temperature', 1.5)
        )
        
        # state_dict 로드
        state_dict = checkpoint['model_state']
        
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        # 임계값 및 통계 로드 (있으면)
        if 'thresholds' in checkpoint:
            self.original_thresholds = checkpoint['thresholds'].copy()
            self.thresholds = checkpoint['thresholds']
            self.logger.info("✅ 저장된 임계값 로드")
        else:
            # 기본 임계값 설정
            self.logger.warning("⚠️ 저장된 임계값 없음 - 기본값 사용")
            self.original_thresholds = {
                'max_prob': 0.5,
                'entropy': 0.2,
                'min_distance': 1.336,
                'mahal_distance': 57.75,
                'recon_error': 0.064,
                'known_acc': 0.8,
                'unknown_reject': 0.7
            }
            self.thresholds = self.original_thresholds.copy()
        
        if 'class_statistics' in checkpoint:
            self.class_statistics = checkpoint['class_statistics']
            self.logger.info("✅ 클래스 통계 로드")
        else:
            self.logger.warning("⚠️ 클래스 통계 없음 - 빈 딕셔너리 사용")
            self.class_statistics = {}
        
        # 임계값 조정 (핵심!)
        self._adjust_thresholds()
    
    def _adjust_thresholds(self):
        """임계값을 실용적인 값으로 조정"""
        # 실제 저장된 임계값 (Colab에서 확인):
        # max_prob: 0.5
        # entropy: 0.2  
        # min_distance: 1.336
        # mahal_distance: 57.75
        # recon_error: 0.064
        
        # min_distance를 더 관대하게 조정 (1.336 -> 2.0)
        self.thresholds['min_distance'] = 2.0
        
        # max_prob를 더 엄격하게 (0.5 -> 0.7)
        self.thresholds['max_prob'] = 0.7
        
        # entropy는 적절해 보임 (0.2 유지하되 살짝 완화)
        self.thresholds['entropy'] = 0.3
        
        # mahal_distance 조정 (57.75 -> 80)
        self.thresholds['mahal_distance'] = 80.0
        
        # recon_error 조정 (0.064 -> 0.1)
        self.thresholds['recon_error'] = 0.1
        
        self.logger.info("📈 조정된 임계값:")
        for key, value in self.thresholds.items():
            original = self.original_thresholds.get(key, 'N/A')
            if isinstance(original, float):
                self.logger.info(f"   - {key}: {original:.3f} -> {value:.3f}")
            else:
                self.logger.info(f"   - {key}: {original} -> {value:.3f}")
    
    def calculate_mahalanobis_distance(self, features):
        min_mahal = float('inf')
        best_class = -1
        
        for class_id, stats in self.class_statistics.items():
            if stats is not None:
                diff = features - stats['mean']
                if 'precision' in stats and stats['precision'] is not None:
                    try:
                        mahal = np.sqrt(np.abs(diff @ stats['precision'] @ diff))
                    except:
                        mahal = np.linalg.norm(diff)
                else:
                    mahal = np.linalg.norm(diff)
                
                if mahal < min_mahal:
                    min_mahal = mahal
                    best_class = class_id
        
        return min_mahal, best_class
    
    def predict_single(self, image, debug=False):
        if isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            elif image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor, return_all=True)
            
            logits = outputs['logits']
            features = outputs['features'].cpu().numpy()[0]
            recon_error = outputs['reconstruction_error'].cpu().item()
            distances = outputs['distances']
            
            probs = F.softmax(logits, dim=1)
            max_prob, predicted_class = probs.max(dim=1)
            max_prob = max_prob.item()
            predicted_class = predicted_class.item()
            
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).item()
            min_distance = distances.min(dim=1)[0].item()
            mahal_distance, mahal_class = self.calculate_mahalanobis_distance(features)
            
            # 디버깅 모드
            if debug:
                print("\n=== 임계값 체크 (디버깅) ===")
                print(f"Predicted class: {predicted_class} ({self.known_insects[predicted_class]})")
                print(f"Max Prob: {max_prob:.4f} >= {self.thresholds['max_prob']:.4f}? "
                      f"{'✅' if max_prob >= self.thresholds['max_prob'] else '❌'}")
                print(f"Entropy: {entropy:.4f} <= {self.thresholds['entropy']:.4f}? "
                      f"{'✅' if entropy <= self.thresholds['entropy'] else '❌'}")
                print(f"Min Distance: {min_distance:.4f} <= {self.thresholds['min_distance']:.4f}? "
                      f"{'✅' if min_distance <= self.thresholds['min_distance'] else '❌'}")
                print(f"Mahal Distance: {mahal_distance:.4f} <= {self.thresholds.get('mahal_distance', 100):.4f}? "
                      f"{'✅' if mahal_distance <= self.thresholds.get('mahal_distance', 100) else '❌'}")
                print(f"Recon Error: {recon_error:.4f} <= {self.thresholds.get('recon_error', 0.1):.4f}? "
                      f"{'✅' if recon_error <= self.thresholds.get('recon_error', 0.1) else '❌'}")
            
            # Unknown 판단
            is_unknown = False
            rejection_reasons = []
            
            if max_prob < self.thresholds['max_prob']:
                is_unknown = True
                rejection_reasons.append(f"낮은 신뢰도: {max_prob:.3f}")
            
            if entropy > self.thresholds['entropy']:
                is_unknown = True
                rejection_reasons.append(f"높은 엔트로피: {entropy:.3f}")
            
            if min_distance > self.thresholds['min_distance']:
                is_unknown = True
                rejection_reasons.append(f"프로토타입과 거리 멀음: {min_distance:.3f}")
            
            if mahal_distance > self.thresholds.get('mahal_distance', 100):
                is_unknown = True
                rejection_reasons.append(f"높은 마할라노비스 거리: {mahal_distance:.3f}")
            
            if recon_error > self.thresholds.get('recon_error', 0.1):
                is_unknown = True
                rejection_reasons.append(f"높은 재구성 오류: {recon_error:.6f}")
            
            if is_unknown:
                return {
                    'class_id': -1,
                    'class_name': '미확인 해충',
                    'confidence': max_prob,
                    'is_known': False,
                    'rejection_reasons': rejection_reasons,
                    'details': {
                        'max_prob': max_prob,
                        'entropy': entropy,
                        'min_distance': min_distance,
                        'mahal_distance': mahal_distance,
                        'recon_error': recon_error
                    }
                }
            else:
                return {
                    'class_id': predicted_class,
                    'class_name': self.known_insects[predicted_class],
                    'confidence': max_prob,
                    'is_known': True,
                    'rejection_reasons': [],
                    'details': {
                        'max_prob': max_prob,
                        'entropy': entropy,
                        'min_distance': min_distance,
                        'mahal_distance': mahal_distance,
                        'recon_error': recon_error
                    }
                }

# ========================
# 메인 테스트 코드
# ========================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 1. 모델 로드
    MODEL_PATH = 'improved_pest_detection_model.pth'
    
    recognizer = ImprovedOpenSetRecognizer(
        model_path=MODEL_PATH,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # 2. 테스트할 이미지 목록
    test_images = [
        ('test_unknown.jpg', '완전 처음 보는 이미지'),
        ('test_image.jpg', 'Test 데이터셋 이미지'),
        ('train_image.jpg', 'Train 데이터셋 이미지'),
    ]
    
    # 3. 각 이미지 테스트
    for image_path, description in test_images:
        if os.path.exists(image_path):
            print(f"\n{'='*60}")
            print(f"테스트: {description} ({image_path})")
            print('='*60)
            
            image = cv2.imread(image_path)
            
            # 디버그 모드로 예측
            result = recognizer.predict_single(image, debug=True)
            
            print(f"\n========== 예측 결과 ==========")
            print(f"클래스: {result['class_name']}")
            print(f"신뢰도: {result['confidence']:.3f}")
            print(f"Known 여부: {result['is_known']}")
            
            if not result['is_known']:
                print(f"거부 이유: {', '.join(result['rejection_reasons'])}")
            
            print(f"\n세부 점수:")
            for key, value in result['details'].items():
                print(f"  - {key}: {value:.4f}")
        else:
            print(f"\n⚠️ 파일 없음: {image_path}")