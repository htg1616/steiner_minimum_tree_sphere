# Steiner Minimum Tree on a Sphere

구면 위 터미널 점들을 가장 짧은 네트워크로 연결하는 **Geodesic Steiner Tree Problem**의 Thompson method 기반 휴리스틱 알고리즘 구현.

평면 ESTP 휴리스틱을 확장하여
1. **MST**
2. **Steiner Insertion** 두 방식(Plane vs Local) 비교  
3. **Adam 기반 지역 최적화**  
4. 다양한 벤치마크 자동 생성·실험·시각화

까지 한 번에 돌려 볼 수 있도록 스크립트를 모듈화했습니다.

---

## 📂 디렉터리 구조

```plain
steiner_minimum_tree_sphere/
├── config/
│   ├── generate_test_config.json    # 입력 데이터 생성 파라미터
│   └── experiment_config.json       # 실험 파라미터
│
├── data/
│   ├── inputs/                      # 테스트 케이스(.pkl) 보관
│   │   ├── 10 dots/
│   │   ├── 50 dots/
│   │   └── .../
│   └── outputs/                     # 실험 결과(.json) 보관
│       ├── 10 dots/
│       ├── 50 dots/
│       └── .../
│
├── geometry/                        # 순수 수학·기하 모듈
│   ├── __init__.py
│   └── dot.py
│
├── graph/                           # MST, Steiner Tree, Optimizer 구현
│   ├── __init__.py
│   ├── mst.py
│   ├── steiner.py
│   └── optimizer.py
│
├── scripts/                         # 실행용 스크립트
│   ├── demo.py           # 간단 테스트용 데모 스크립트
│   ├── generate_test.py  # 입력 데이터(.pkl) 생성
│   ├── experiment.py     # 실험 실행 및 결과(.json) 저장
│   └── visual.py         # 실험 결과 시각화 (미완)
│
├── requirements.txt                 # 의존성 목록
└── README.md                        # 이 파일
```

## ⚙️ 설치

```bash
# 1. 리포지토리 클론
git clone https://github.com/htg1616/steiner_minimum_tree_sphere.git
cd steiner_minimum_tree_sphere

# 2. 가상환경 생성 (선택)
python -m venv venv
# Linux/macOS
source venv/bin/activate
# Windows
venv\Scripts\activate

# 3. 의존성 설치
pip install --upgrade pip
pip install -r requirements.txt
```

## 🚀 사용 예시

### 1) 입력 데이터 생성

```bash
python experiments/generate_instance.py
```
config/generate_test_config.json 의 base_seed, num_dots, num_tests 설정을 사용해
data/inputs/{n} dots/ 에 테스트 케이스(.pkl) 생성
