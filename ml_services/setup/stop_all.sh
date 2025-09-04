#!/bin/bash

# 모든 서비스 중지 스크립트
echo "🛑 모든 ML 마이크로서비스 중지..."

# PID 디렉토리 확인
if [ -d ".pids" ]; then
    # 각 서비스 중지
    if [ -f ".pids/rag.pid" ]; then
        kill $(cat .pids/rag.pid) 2>/dev/null
        echo "  → RAG 서비스 중지"
    fi
    
    if [ -f ".pids/gpt.pid" ]; then
        kill $(cat .pids/gpt.pid) 2>/dev/null
        echo "  → GPT 서비스 중지"
    fi
    
    if [ -f ".pids/phone.pid" ]; then
        kill $(cat .pids/phone.pid) 2>/dev/null
        echo "  → 전화 서비스 중지"
    fi
    
    if [ -f ".pids/proxy.pid" ]; then
        kill $(cat .pids/proxy.pid) 2>/dev/null
        echo "  → 프록시 서비스 중지"
    fi
    
    if [ -f ".pids/openset.pid" ]; then
        kill $(cat .pids/openset.pid) 2>/dev/null
        echo "  → Open Set 서비스 중지"
    fi
    
    # PID 파일 삭제
    rm -rf .pids
else
    # 포트 기반으로 프로세스 종료
    echo "  → 포트 기반 프로세스 종료..."
    lsof -ti:8003 | xargs kill 2>/dev/null
    lsof -ti:8004 | xargs kill 2>/dev/null
    lsof -ti:8005 | xargs kill 2>/dev/null
    lsof -ti:8006 | xargs kill 2>/dev/null
    lsof -ti:8007 | xargs kill 2>/dev/null
fi

echo "✅ 모든 서비스가 중지되었습니다."