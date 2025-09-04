"""
Open Set Recognition 테스트 및 디버깅
ResNet50 버전 - 이미지 품질 검사 + TTA 통합
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
# 모델 클래스 정의 - ResNet50 버전
# ========================

class CalibratedOpenSetModel(nn.Module):
    """ResNet50 백본을 사용하는 Open Set 모델"""
    def __init__(self, num_classes=4, feature_dim=512, initial_temperature=1.5):
        super().__init__()
        
        # ResNet50 백본 (ResNet101에서 변경)
        resnet = models.resnet50(weights=None)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # ResNet50의 출력 차원은 2048 (ResNet101과 동일)
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
    
    def forward(self, x, return_all=False, apply_temperature=True):
        cnn_features = self.features(x)
        cnn_features = cnn_features.view(cnn_features.size(0), -1)
        
        features = self.encoder(cnn_features)
        logits = self.main_classifier(features)
        aux_logits = self.auxiliary_classifier(features)
        
        # Temperature scaling
        if apply_temperature:
            temperature = torch.clamp(self.temperature, min=0.5, max=3.0)
            logits = logits / temperature
            aux_logits = aux_logits / temperature
        
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

# ========================
# 개선된 인식기
# ========================

class ImprovedOpenSetRecognizer:
    def __init__(self, model_path='improved_model_resnet50.pth', device='cpu'):
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
        
        self.logger.info(f"✅ ResNet50 모델 로드 완료: {model_path}")
        self.logger.info(f"📊 Device: {self.device}")
    
    def _load_model(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # 모델 초기화 (temperature 포함)
        config = checkpoint.get('config', {})
        
        # 백본 정보 확인
        backbone = config.get('backbone', 'resnet50')
        self.logger.info(f"📦 백본: {backbone}")
        
        self.model = CalibratedOpenSetModel(
            num_classes=config.get('num_classes', 4),
            feature_dim=config.get('feature_dim', 512),
            initial_temperature=config.get('temperature', 1.5)
        )
        
        # state_dict 로드 (strict=False로 호환성 문제 해결)
        state_dict = checkpoint['model_state']
        
        # 호환성 문제 해결: strict=False 사용
        incompatible_keys = self.model.load_state_dict(state_dict, strict=False)
        
        if incompatible_keys.missing_keys:
            self.logger.warning(f"⚠️ Missing keys: {incompatible_keys.missing_keys}")
        if incompatible_keys.unexpected_keys:
            self.logger.warning(f"⚠️ Unexpected keys: {incompatible_keys.unexpected_keys}")
        self.model.to(self.device)
        self.model.eval()
        
        # 임계값 및 통계 로드
        if 'thresholds' in checkpoint:
            self.original_thresholds = checkpoint['thresholds'].copy()
            self.thresholds = checkpoint['thresholds']
            self.logger.info("✅ 저장된 임계값 로드")
        else:
            # ResNet50에 맞춘 기본 임계값
            self.logger.warning("⚠️ 저장된 임계값 없음 - ResNet50 기본값 사용")
            self.original_thresholds = {
                'max_prob': 0.6,
                'entropy': 0.3,
                'min_distance': 1.5,
                'mahal_distance': 60.0,
                'recon_error': 0.08,
                'known_acc': 0.75,
                'unknown_reject': 0.65
            }
            self.thresholds = self.original_thresholds.copy()
        
        if 'class_statistics' in checkpoint:
            self.class_statistics = checkpoint['class_statistics']
            self.logger.info("✅ 클래스 통계 로드")
        else:
            self.logger.warning("⚠️ 클래스 통계 없음")
            self.class_statistics = {}
        
        # 임계값 조정
        self._adjust_thresholds()
    
    def _adjust_thresholds(self):
        """개선된 임계값 조정 - v2 모델에 맞춤"""
        
        self.logger.info("🔍 원본 임계값:")
        for key, value in self.original_thresholds.items():
            if isinstance(value, float):
                self.logger.info(f"   - {key}: {value:.3f}")
        
        # 저장된 임계값을 그대로 사용 (이미 최적화됨)
        # 단, 일부 미세 조정만 수행
        
        # max_prob는 저장된 값 사용 (보통 0.5 정도)
        if 'max_prob' in self.original_thresholds:
            self.thresholds['max_prob'] = self.original_thresholds['max_prob']
        else:
            self.thresholds['max_prob'] = 0.5
        
        # entropy는 저장된 값 + 약간의 여유
        if 'entropy' in self.original_thresholds:
            self.thresholds['entropy'] = self.original_thresholds['entropy'] * 1.1  # 10% 관대하게
        else:
            self.thresholds['entropy'] = 1.0
        
        # min_distance는 저장된 값 + 여유
        if 'min_distance' in self.original_thresholds:
            self.thresholds['min_distance'] = self.original_thresholds['min_distance'] * 1.2  # 20% 관대하게
        else:
            self.thresholds['min_distance'] = 2.0
        
        # recon_error는 저장된 값 + 여유
        if 'recon_error' in self.original_thresholds:
            self.thresholds['recon_error'] = self.original_thresholds['recon_error'] * 1.2  # 20% 관대하게
        else:
            self.thresholds['recon_error'] = 0.15
        
        # mahal_distance는 고정값 사용
        self.thresholds['mahal_distance'] = 20.0
        
        self.logger.info("📈 v2 모델용 조정된 임계값:")
        for key, value in self.thresholds.items():
            if key in ['max_prob', 'entropy', 'min_distance', 'recon_error', 'mahal_distance']:
                original = self.original_thresholds.get(key, 'N/A')
                if isinstance(original, float) and isinstance(value, float):
                    self.logger.info(f"   - {key}: {original:.3f} -> {value:.3f}")
                else:
                    self.logger.info(f"   - {key}: {original} -> {value}")
    
    def calculate_mahalanobis_distance(self, features):
        """개선된 마할라노비스 거리 계산"""
        min_mahal = float('inf')
        best_class = -1
        
        # 특징 정규화 
        features_norm = features / (np.linalg.norm(features) + 1e-8)
        
        for class_id, stats in self.class_statistics.items():
            if stats is not None:
                # 정규화된 평균 사용
                if 'mean_normalized' in stats:
                    mean_norm = stats['mean_normalized']
                else:
                    mean_norm = stats['mean'] / (np.linalg.norm(stats['mean']) + 1e-8)
                
                diff = features_norm - mean_norm
                
                if 'precision' in stats and stats['precision'] is not None:
                    try:
                        precision = stats['precision']
                        
                        # precision matrix 스케일 확인
                        precision_scale = np.max(np.abs(precision))
                        if precision_scale > 100:
                            precision = precision / precision_scale + 0.01 * np.eye(precision.shape[0])
                        
                        mahal_squared = diff @ precision @ diff
                        mahal = np.sqrt(np.abs(mahal_squared))
                        
                        if mahal > 100:
                            mahal = np.log1p(mahal)
                        
                    except Exception as e:
                        self.logger.warning(f"Mahalanobis 계산 실패 (class {class_id}): {e}")
                        mahal = np.linalg.norm(diff) * 5  
                else:
                    mahal = np.linalg.norm(diff) * 5
                
                if mahal < min_mahal:
                    min_mahal = mahal
                    best_class = class_id
        
        return min_mahal, best_class
    
    def check_image_quality(self, image):
        """이미지 품질 검사"""
        if isinstance(image, np.ndarray):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        else:
            gray = np.array(image.convert('L'))
        
        # Laplacian variance로 흐림 정도 측정
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 엣지 밀도 계산
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # 콘트라스트 측정
        contrast = gray.std()
        
        # ResNet50용 품질 기준 (약간 관대)
        quality_score = {
            'blur_score': laplacian_var,
            'edge_density': edge_density,
            'contrast': contrast,
            'is_low_quality': laplacian_var < 80 or edge_density < 0.04 or contrast < 25
        }
        
        return quality_score
    
    def predict_single(self, image, debug=False):
        """단일 이미지 예측"""
        quality = self.check_image_quality(image)
        
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
            
            top2_probs, top2_classes = torch.topk(probs, 2, dim=1)
            confusion_score = (top2_probs[0][0] - top2_probs[0][1]).item()
            
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).item()
            min_distance = distances.min(dim=1)[0].item()
            mahal_distance, mahal_class = self.calculate_mahalanobis_distance(features)
            
            if debug:
                print("\n=== ResNet50 예측 디버깅 ===")
                print(f"백본: ResNet50")
                print(f"Temperature: {outputs.get('temperature', 1.5):.3f}")
                
                print("\n=== 이미지 품질 체크 ===")
                print(f"Blur score: {quality['blur_score']:.2f} (>80 is good)")
                print(f"Edge density: {quality['edge_density']:.3f} (>0.04 is good)")
                print(f"Contrast: {quality['contrast']:.2f} (>25 is good)")
                print(f"Low quality detected: {'⚠️ YES' if quality['is_low_quality'] else '✅ NO'}")
                
                print("\n=== 혼동 가능성 체크 ===")
                print(f"Top 1: {self.known_insects[top2_classes[0][0].item()]} ({top2_probs[0][0]:.3f})")
                print(f"Top 2: {self.known_insects[top2_classes[0][1].item()]} ({top2_probs[0][1]:.3f})")
                print(f"Confusion score: {confusion_score:.3f} (>0.25 is confident)")
                
                print("\n=== 임계값 체크 (ResNet50) ===")
                print(f"Predicted class: {predicted_class} ({self.known_insects[predicted_class]})")
                print(f"Max Prob: {max_prob:.4f} >= {self.thresholds['max_prob']:.4f}? "
                      f"{'✅' if max_prob >= self.thresholds['max_prob'] else '❌'}")
                print(f"Entropy: {entropy:.4f} <= {self.thresholds['entropy']:.4f}? "
                      f"{'✅' if entropy <= self.thresholds['entropy'] else '❌'}")
                print(f"Min Distance: {min_distance:.4f} <= {self.thresholds['min_distance']:.4f}? "
                      f"{'✅' if min_distance <= self.thresholds['min_distance'] else '❌'}")
                print(f"Mahal Distance: {mahal_distance:.4f} <= {self.thresholds.get('mahal_distance', 100):.4f}? "
                      f"{'✅' if mahal_distance <= self.thresholds.get('mahal_distance', 100) else '❌'}")
                print(f"Recon Error: {recon_error:.4f} <= {self.thresholds.get('recon_error', 0.12):.4f}? "
                      f"{'✅' if recon_error <= self.thresholds.get('recon_error', 0.12) else '❌'}")
            
            # Unknown 판단 (ResNet50 맞춤)
            is_unknown = False
            rejection_reasons = []
            
            # 이미지 품질이 낮으면 더 엄격한 기준 적용
            if quality['is_low_quality']:
                adjusted_prob_threshold = min(0.85, self.thresholds['max_prob'] + 0.25)
                if max_prob < adjusted_prob_threshold:
                    is_unknown = True
                    rejection_reasons.append(f"저품질 이미지 + 불충분한 신뢰도: {max_prob:.3f}")
                
                if confusion_score < 0.15:  # ResNet50용 조정
                    is_unknown = True
                    rejection_reasons.append(f"저품질 + 클래스 혼동: {confusion_score:.3f}")
            
            # 기본 임계값 체크
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
            
            if recon_error > self.thresholds.get('recon_error', 0.12):
                is_unknown = True
                rejection_reasons.append(f"높은 재구성 오류: {recon_error:.6f}")
            
            # 결과 반환
            if is_unknown:
                return {
                    'class_id': -1,
                    'class_name': '미확인 해충',
                    'confidence': max_prob,
                    'is_known': False,
                    'rejection_reasons': rejection_reasons,
                    'quality': quality,
                    'confusion_score': confusion_score,
                    'details': {
                        'max_prob': max_prob,
                        'entropy': entropy,
                        'min_distance': min_distance,
                        'mahal_distance': mahal_distance,
                        'recon_error': recon_error
                    }
                }
            else:
                warning = None
                if quality['is_low_quality'] and max_prob > 0.75:
                    warning = "⚠️ 이미지 품질이 낮습니다. 더 선명한 이미지로 재확인을 권장합니다."
                
                return {
                    'class_id': predicted_class,
                    'class_name': self.known_insects[predicted_class],
                    'confidence': max_prob,
                    'is_known': True,
                    'warning': warning,
                    'quality': quality,
                    'confusion_score': confusion_score,
                    'rejection_reasons': [],
                    'details': {
                        'max_prob': max_prob,
                        'entropy': entropy,
                        'min_distance': min_distance,
                        'mahal_distance': mahal_distance,
                        'recon_error': recon_error
                    }
                }
    
    def predict_with_tta(self, image, n_augmentations=5, debug=False):
        """Test Time Augmentation으로 여러 변형 이미지 예측 후 앙상블"""
        
        predictions = []
        all_probs = []
        
        # 원본 이미지 예측
        base_result = self.predict_single(image, debug=False)
        predictions.append(base_result)
        
        # PIL Image로 변환
        if isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            elif image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        # TTA transforms
        tta_transforms = [
            transforms.Compose([
                transforms.RandomRotation(degrees=5),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            transforms.Compose([
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            transforms.Compose([
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            transforms.Compose([
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        ]
        
        # 각 변형에 대해 예측
        with torch.no_grad():
            for i, tta_transform in enumerate(tta_transforms[:n_augmentations-1]):
                image_tensor = tta_transform(pil_image).unsqueeze(0).to(self.device)
                
                outputs = self.model(image_tensor, return_all=True)
                logits = outputs['logits']
                probs = F.softmax(logits, dim=1)
                all_probs.append(probs.cpu().numpy()[0])
                
                max_prob, predicted_class = probs.max(dim=1)
                predictions.append({
                    'class_id': predicted_class.item(),
                    'confidence': max_prob.item()
                })
        
        # 앙상블 결과 계산
        if all_probs:
            # 원본 예측의 확률 추가
            image_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.model(image_tensor, return_all=True)
                logits = outputs['logits']
                probs = F.softmax(logits, dim=1)
                all_probs.append(probs.cpu().numpy()[0])
            
            avg_probs = np.mean(all_probs, axis=0)
            std_probs = np.std(all_probs, axis=0)
            
            final_class = np.argmax(avg_probs)
            final_confidence = avg_probs[final_class]
            
            predicted_classes = [p['class_id'] for p in predictions if p.get('class_id', -1) >= 0]
            if predicted_classes:
                consistency = predicted_classes.count(final_class) / len(predicted_classes)
            else:
                consistency = 0
            
            uncertainty = std_probs[final_class]
            
            if debug:
                print("\n=== TTA 앙상블 결과 (ResNet50) ===")
                print(f"평균 확률: {final_confidence:.3f}")
                print(f"표준편차: {uncertainty:.3f}")
                print(f"일관성: {consistency:.1%}")
                print(f"각 augmentation 예측:")
                for i, pred in enumerate(predictions):
                    if pred.get('class_id', -1) >= 0:
                        print(f"  Aug {i}: {self.known_insects.get(pred['class_id'], 'Unknown')} "
                              f"({pred['confidence']:.3f})")
            
            # 최종 판단 (ResNet50 맞춤)
            is_unknown = False
            rejection_reasons = []
            
            # TTA 기반 거부 조건 (ResNet50용 조정)
            if final_confidence < 0.65:  # ResNet101의 0.7에서 하향
                is_unknown = True
                rejection_reasons.append(f"TTA 평균 신뢰도 부족: {final_confidence:.3f}")
            
            if uncertainty > 0.25:  # ResNet101의 0.2에서 상향
                is_unknown = True
                rejection_reasons.append(f"TTA 불확실성 높음: {uncertainty:.3f}")
            
            if consistency < 0.5:  # ResNet101의 0.6에서 하향
                is_unknown = True
                rejection_reasons.append(f"TTA 일관성 부족: {consistency:.1%}")
            
            quality = self.check_image_quality(image)
            if quality['is_low_quality'] and final_confidence < 0.8:
                is_unknown = True
                rejection_reasons.append("저품질 이미지 + TTA 신뢰도 부족")
            
            if is_unknown:
                return {
                    'class_id': -1,
                    'class_name': '미확인 해충',
                    'confidence': final_confidence,
                    'is_known': False,
                    'rejection_reasons': rejection_reasons,
                    'tta_consistency': consistency,
                    'tta_uncertainty': uncertainty,
                    'quality': quality
                }
            else:
                warning = None
                if quality['is_low_quality']:
                    warning = f"⚠️ 이미지 품질 낮음. TTA 일관성: {consistency:.1%}"
                elif consistency < 0.7:
                    warning = f"⚠️ 예측 일관성 낮음: {consistency:.1%}. 재촬영 권장"
                
                return {
                    'class_id': int(final_class),
                    'class_name': self.known_insects[int(final_class)],
                    'confidence': float(final_confidence),
                    'is_known': True,
                    'warning': warning,
                    'tta_consistency': float(consistency),
                    'tta_uncertainty': float(uncertainty),
                    'quality': quality
                }
        
        return base_result


# ========================
# 메인 테스트 코드
# ========================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 1. 모델 로드
    MODEL_PATH = 'model/improved_model_resnet50.pth'
    
    recognizer = ImprovedOpenSetRecognizer(
        model_path=MODEL_PATH,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # 2. 테스트할 이미지 목록
    test_images = [
        ('image/test_unknown.jpg', '완전 처음 보는 이미지'),
        ('image/test_image.jpg', 'Test 데이터셋 이미지'),
        ('image/train_image.jpg', 'Train 데이터셋 이미지'),
    ]
    
    # 3. 각 이미지 테스트
    for image_path, description in test_images:
        if os.path.exists(image_path):
            print(f"\n{'='*60}")
            print(f"테스트: {description} ({image_path})")
            print('='*60)
            
            image = cv2.imread(image_path)
            
            # 일반 예측 (디버그 모드)
            print("\n--- 일반 예측 (ResNet50) ---")
            result = recognizer.predict_single(image, debug=True)
            
            print(f"\n========== 예측 결과 ==========")
            print(f"클래스: {result['class_name']}")
            print(f"신뢰도: {result['confidence']:.3f}")
            print(f"Known 여부: {result['is_known']}")
            
            if result.get('warning'):
                print(f"경고: {result['warning']}")
            
            if not result['is_known']:
                print(f"거부 이유: {', '.join(result['rejection_reasons'])}")
            
            print(f"\n세부 점수:")
            for key, value in result['details'].items():
                print(f"  - {key}: {value:.4f}")
            
            # TTA 예측
            print("\n--- TTA 예측 (ResNet50, 더 정확함) ---")
            tta_result = recognizer.predict_with_tta(image, n_augmentations=5, debug=True)
            
            print(f"\n========== TTA 결과 ==========")
            print(f"클래스: {tta_result['class_name']}")
            print(f"신뢰도: {tta_result['confidence']:.3f}")
            print(f"Known 여부: {tta_result['is_known']}")
            
            if tta_result.get('warning'):
                print(f"경고: {tta_result['warning']}")
            
            if tta_result.get('tta_consistency') is not None:
                print(f"TTA 일관성: {tta_result['tta_consistency']:.1%}")
                print(f"TTA 불확실성: {tta_result.get('tta_uncertainty', 0):.3f}")
            
            if not tta_result['is_known']:
                print(f"거부 이유: {', '.join(tta_result['rejection_reasons'])}")
        else:
            print(f"\n⚠️ 파일 없음: {image_path}")