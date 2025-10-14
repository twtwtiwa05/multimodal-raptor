#!/bin/bash
# Multimodal RAPTOR 웹 데모 실행 스크립트

echo "🚌 Multimodal RAPTOR 웹 데모 시작..."

# 의존성 설치
echo "📦 의존성 설치 중..."
pip install -r requirements_web.txt

# Streamlit 앱 실행
echo "🌐 웹 데모 실행 중..."
echo "브라우저에서 http://localhost:8501 을 열어주세요!"
streamlit run app.py