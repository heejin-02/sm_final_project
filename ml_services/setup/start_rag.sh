#!/bin/bash

# RAG 서비스 시작 스크립트
echo "🤖 RAG 서비스 시작 (포트 8003)..."

cd ../rag_service
source venv/bin/activate
python main.py