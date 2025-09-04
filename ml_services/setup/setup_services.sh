#!/bin/bash

# ML 마이크로서비스 초기 설정 스크립트
# 각 서비스별로 가상환경을 생성하고 의존성을 설치합니다.

echo "🚀 ML 마이크로서비스 설정 시작..."

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 서비스 목록
services=("rag_service" "gpt_service" "phone_service" "proxy_service" "openset_service")

# 각 서비스 설정
for service in "${services[@]}"; do
    echo -e "\n${YELLOW}[$service] 설정 시작${NC}"
    
    # 서비스 디렉토리로 이동
    cd "../$service" || exit
    
    # 가상환경 생성
    if [ ! -d "venv" ]; then
        echo "  → 가상환경 생성 중..."
        python3 -m venv venv
    else
        echo "  → 가상환경이 이미 존재합니다."
    fi
    
    # 가상환경 활성화 및 패키지 설치
    echo "  → 패키지 설치 중..."
    source venv/bin/activate
    pip install --upgrade pip -q
    pip install -r requirements.txt -q
    deactivate
    
    echo -e "${GREEN}  ✓ $service 설정 완료${NC}"
    
    # setup 디렉토리로 복귀
    cd ../setup
done

# .env 파일 생성 (없는 경우)
if [ ! -f "../../.env" ]; then
    echo -e "\n${YELLOW}.env 파일 생성 중...${NC}"
    cat > ../../.env << 'EOF'
# OpenAI API
OPENAI_API_KEY=your_openai_api_key_here

# Spring Boot 연동
SPRING_BOOT_URL=http://localhost:8095

# SignalWire 설정 (전화 서비스용)
SIGNALWIRE_PROJECT_ID=your_project_id
SIGNALWIRE_AUTH_TOKEN=your_auth_token
SIGNALWIRE_PHONE_NUMBER=+1234567890
SIGNALWIRE_SPACE_URL=your_space.signalwire.com
EOF
    echo -e "${GREEN}✓ .env 파일 생성 완료${NC}"
    echo -e "${RED}⚠ .env 파일에 실제 API 키를 입력해주세요!${NC}"
else
    echo -e "${GREEN}✓ .env 파일이 이미 존재합니다.${NC}"
fi

# ChromaDB 데이터 마이그레이션 (처음 설정 시)
if [ ! -d "../rag_service/chroma_db" ] && [ -d "../../ml/chroma_db" ]; then
    echo -e "\n${YELLOW}ChromaDB 데이터 마이그레이션 중...${NC}"
    cp -r ../../ml/chroma_db ../rag_service/
    echo -e "${GREEN}✓ ChromaDB 데이터 마이그레이션 완료${NC}"
fi

# Open Set Recognition 모델 파일 확인
if [ ! -d "../openset_service/models" ]; then
    echo -e "\n${YELLOW}Open Set Recognition 모델 디렉토리 생성 중...${NC}"
    mkdir -p ../openset_service/models
    echo -e "${GREEN}✓ 모델 디렉토리 생성 완료${NC}"
    echo -e "${RED}⚠ openset_service/models/ 디렉토리에 학습된 모델 파일을 복사해주세요!${NC}"
fi

echo -e "\n${GREEN}✅ 모든 서비스 설정 완료!${NC}"
echo -e "\n각 서비스를 실행하려면:"
echo "  ./start_rag.sh     # RAG 서비스 (포트 8003)"
echo "  ./start_gpt.sh     # GPT 요약 서비스 (포트 8004)"
echo "  ./start_phone.sh   # 전화 알림 서비스 (포트 8005)"
echo "  ./start_proxy.sh   # 파일 프록시 서비스 (포트 8006)"
echo "  ./start_openset.sh # Open Set Recognition (포트 8007)"
echo -e "\n모든 서비스를 한번에 실행하려면:"
echo "  ./start_all.sh"