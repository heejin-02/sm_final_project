#!/bin/bash

# GPT 요약 서비스 시작 스크립트
echo "📊 GPT 요약 서비스 시작 (포트 8004)..."

cd ../gpt_service
source venv/bin/activate
python main.py