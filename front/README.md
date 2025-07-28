# 1) front 디렉토리로 이동
cd front

# 2) 기본 의존성 설치 (React, React-DOM 등)
npm install

# 3) Tailwind v4 + Vite 플러그인 설치
npm install -D tailwindcss @tailwindcss/vite

#    ⚠️ tailwind.config.js를 직접 만드셨다면 init은 건너뛰셔도 됩니다.
#    만약 설정 파일이 없다면, 아래 명령으로 자동 생성하세요:
# npx tailwindcss init

# 4) 런타임 라이브러리 설치
npm install \
  axios \               # HTTP 클라이언트  
  react-router-dom \    # 라우팅  
  react-icons \         # SVG 아이콘  
  d3-scale \            # 스케일 계산  
  d3-color \            # 색상 계산 

# 5) 개발 서버 기동 (Vite)
npm run dev
