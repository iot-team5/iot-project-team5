# IoT Project team5

## 프로젝트 개요

### 연구 배경·문제 정의
- 논문 제목 : FEDERATED LEARNING FOR INTERNET OF THINGS: A FEDERATED LEARNING FRAMEWORK FOR ON-DEVICE ANOMALY DATA DETECTION
- 목표: [Iot_paper.pdf](Iot_paper.pdf)에 제시된 FedIoT 시나리오에서 중앙 집중 없이 IoT 이상 징후를 탐지.
- 해결 과제: 개인정보·대역폭 제약으로 인해 연합 학습(FedAvg)과 오토인코더(FedDetect)를 결합해 재현성 확보.
- 접근 방식: 디바이스 이질성을 고려한 로컬 학습, 글로벌 집계, 임계값 추정을 하나의 파이프라인으로 구성.

## 논문 구현 설명

### 데이터셋·전처리
- 데이터 원본: [configs/base.yaml](configs/base.yaml)에서 TON_IoT CSV 경로를 지정하고 data/raw/ 디렉터리에 배치.
- 전처리 제어: [DataConfig](src/config.py#L12-L29)가 특성/범주형 컬럼, 로그 변환, 라벨 필터를 정의.
- 분할 전략: test_split, validation_split, num_clients, min_samples_per_client, iid 파라미터로 학습·검증·테스트 및 클라이언트 파티션 구성.
- 실행 준비: src/data 패키지가 포함되어 있지 않으므로 load_iot_dataset, partition_dataset 구현을 추가해야 함.
- 도구: [scripts/download_dataset.py](scripts/download_dataset.py#L1-L210)가 다운로드, 압축 해제, CSV 요약, 라벨 통계를 제공.

## 코드 구조
```
프로젝트 루트
├─ configs/
│  └─ base.yaml
├─ scripts/
│  ├─ download_dataset.py
│  ├─ plot_threshold_curves.py
│  └─ run_fediot.py
├─ src/
│  ├─ config.py
│  ├─ federated/
│  │  ├─ client.py
│  │  ├─ metrics.py
│  │  ├─ server.py
│  │  └─ trainer.py
│  ├─ models/
│  │  ├─ LSTM-autoencoder.py
│  │  └─ autoencoder.py
│  └─ utils/
│     └─ logging.py
└─ requirements.txt
```

- configs([configs](configs)): 실험 전역 설정을 보관하며 [base.yaml](configs/base.yaml)에서 데이터 경로·하이퍼파라미터 선언.
- scripts([scripts](scripts)): 데이터 다운로드, 학습 실행, 결과 시각화 스크립트 집합.
- src([src](src)): 설정([config.py](src/config.py#L1-L108)), 연합 학습([src/federated](src/federated)), 모델([src/models](src/models)), 유틸([src/utils](src/utils))을 포함한 핵심 로직.

### 모델·학습 설정
- [ModelConfig](src/config.py#L45-L55): 인코더·디코더 폭, 잠재 차원, 활성화, 드롭아웃, 정규화 옵션을 정의하고 입력 차원은 로딩 시 갱신.
- [TrainingConfig](src/config.py#L32-L42): 로컬 에폭, 배치, Adam 하이퍼파라미터, 얼리 스토핑, 잠재 노이즈 강도 설정.
- [LocalTrainer](src/federated/trainer.py#L24-L95): 미니배치 학습, 잠재 노이즈 주입, 얼리 스토핑을 포함한 로컬 업데이트 수행.
- [FederatedServer](src/federated/server.py#L30-L185): [FederatedConfig](src/config.py#L58-L72) 기반으로 클라이언트 샘플링, FedAvg 집계, 글로벌 평가 진행.

### 결과·평가 지표
- 임계값 탐색: 검증 재구성 오차와 threshold_metric, threshold_min_recall, threshold_min_precision, threshold_max_fpr 조건을 이용해 후보 선택.
- 폴백 전략: 적합한 후보가 없을 경우 anomaly_threshold_quantile 기반 임계값으로 대체.
- 평가 결과: [compute_metrics](src/federated/metrics.py#L8-L54)가 정확도, 재현율, 정밀도, F1, FPR, 특이도, 혼동 행렬, 평균 오차, 임계값을 산출.
- 산출물 기록: [scripts/run_fediot.py](scripts/run_fediot.py#L98-L139)가 라운드 히스토리 JSON과 글로벌 모델 가중치를 저장.
- 후처리: [scripts/plot_threshold_curves.py](scripts/plot_threshold_curves.py#L107-L205)가 ROC/PR 곡선과 임계값 지표를 시각화.
- 실험 결과: 원격 저장소 result/README 기준으로 Global Threshold 0.0324를 산출했고, Accuracy 0.99 · Recall 0.98 · Precision 0.98 · F1-score 0.98을 달성했으며 Confusion Matrix 시각화와 함께 공유되어 있습니다.

## 실행/설치 매뉴얼

### requirements 파일
- [requirements.txt](requirements.txt)에 수치 연산, 학습, 유틸리티 영역별 패키지와 권장 버전 범위를 명시.
- PyTorch는 CUDA 호환 빌드를 먼저 설치하고 나머지 패키지는 requirements 설치 단계에서 함께 추가하도록 권장.

### 환경 준비 및 실행 순서
1. 가상환경 생성 및 활성화
        python -m venv .venv
        source .venv/bin/activate
        python -m pip install --upgrade pip
2. 의존성 설치
        pip install -r requirements.txt
3. 데이터 내려받기 (예: TON_IoT Dataset)
        python scripts/download_dataset.py --url <CSV_OR_ARCHIVE_URL> --decompress --summarize <상대경로> --label-column label
    - 다운로드한 파일 경로를 [configs/base.yaml](configs/base.yaml)의 data.dataset_path로 지정하고 필요 시 컬럼 목록을 조정합니다.
4. 데이터 로더 구현 확인
    - [src/data](src) 패키지가 비어 있으므로 load_iot_dataset, partition_dataset 구현을 추가해야 실행 가능합니다.
5. 연합 학습 실행
        python scripts/run_fediot.py --config configs/base.yaml --output outputs/exp_toniot
    - 실행 로그로 클라이언트 라벨 구성과 라운드별 글로벌 지표를 확인하고 산출물은 지정한 출력 경로에 저장됩니다.

## 참고 자료 및 주의사항
- 참고 문헌: [Iot_paper.pdf](Iot_paper.pdf), 발표 자료: [[사물인터넷-오전반]_팀5_프로젝트2_발표자료.pdf]([%EC%82%AC%EB%AC%BC%EC%9D%B8%ED%84%B0%EB%84%B7-%EC%98%A4%EC%A0%84%EB%B0%98]_%ED%8C%805_%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B82_%EB%B0%9C%ED%91%9C%EC%9E%90%EB%A3%8C.pdf).
