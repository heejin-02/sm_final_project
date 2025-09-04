#!/bin/bash

# 모든 서비스 동시 시작 스크립트
echo "🚀 모든 ML 마이크로서비스 시작..."

# 각 서비스를 백그라운드로 시작
./start_rag.sh &
RAG_PID=$!
echo "  → RAG 서비스 시작 (PID: $RAG_PID)"

./start_gpt.sh &
GPT_PID=$!
echo "  → GPT 서비스 시작 (PID: $GPT_PID)"

./start_phone.sh &
PHONE_PID=$!
echo "  → 전화 서비스 시작 (PID: $PHONE_PID)"

./start_proxy.sh &
PROXY_PID=$!
echo "  → 프록시 서비스 시작 (PID: $PROXY_PID)"

./start_openset.sh &
OPENSET_PID=$!
echo "  → Open Set 서비스 시작 (PID: $OPENSET_PID)"

# PID 저장
mkdir -p .pids
echo "$RAG_PID" > .pids/rag.pid
echo "$GPT_PID" > .pids/gpt.pid
echo "$PHONE_PID" > .pids/phone.pid
echo "$PROXY_PID" > .pids/proxy.pid
echo "$OPENSET_PID" > .pids/openset.pid

echo ""
echo "✅ 모든 서비스가 시작되었습니다!"
echo ""
echo "서비스 상태:"
echo "  - RAG 서비스: http://localhost:8003"
echo "  - GPT 요약: http://localhost:8004"
echo "  - 전화 알림: http://localhost:8005"
echo "  - 파일 프록시: http://localhost:8006"
echo "  - Open Set Recognition: http://localhost:8007"
echo ""
echo "종료하려면 ./stop_all.sh 실행"

# 프로세스 대기
wait