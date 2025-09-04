#!/bin/bash

# Open Set Recognition 서비스 시작 스크립트
echo "🔍 Open Set Recognition 서비스 시작 (포트 8007)..."

cd ../openset_service
source venv/bin/activate
python main.py