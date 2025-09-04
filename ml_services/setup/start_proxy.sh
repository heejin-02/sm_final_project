#!/bin/bash

# 파일 프록시 서비스 시작 스크립트
echo "📁 파일 프록시 서비스 시작 (포트 8006)..."

cd ../proxy_service
source venv/bin/activate
python main.py