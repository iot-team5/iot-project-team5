# FedIoT 팀5 레퍼런스 구현

## 프로젝트 개요

### 연구 배경·문제 정의
- 본 프로젝트는 [Iot_paper.pdf](Iot_paper.pdf)에 소개된 FedIoT 연구를 기반으로, IoT 트래픽 이상 징후를 중앙 서버에 raw 데이터를 모으지 않고 탐지하려는 문제를 다룹니다.
- IoT 단말이 생성하는 고주기 네트워크 데이터는 개인정보와 대역폭 제한으로 인해 중앙 집중 학습이 어렵기 때문에, FedAvg 기반 연합학습과 오토인코더를 결합한 FedDetect 방식을 참고해 재현성을 확보하는 것이 목표입니다.
- 다수의 디바이스가 서로 다른 분포를 가진다는 점을 고려해, 디바이스별 로컬 학습과 글로벌 모델 집계, 그리고 공격 유형 전반에 대응하는 임계값 산출을 한 워크플로우로 묶었습니다.

## 논문 구현 설명

### 데이터셋·전처리
- 기본 설정은 [configs/base.yaml](configs/base.yaml)의 `data.dataset_path`가 가리키는 TON_IoT 네트워크 캡처(csv)를 사용하며, 프로젝트 루트 `data/raw/` 디렉터리에 위치시키도록 구성돼 있습니다.
- `feature_columns`, `categorical_columns`, `log_transform_columns` 등은 [DataConfig](src/config.py#L12-L29)에서 정의되며, 수치형/범주형 분리, 로그 스케일 변환 대상, 라벨 필터(기본값: 정상 라벨만 학습) 등을 제어합니다.
- 데이터 분할은 학습/검증/테스트를 비율(`test_split`, `validation_split`) 기반으로 나누고, `num_clients`, `min_samples_per_client`, `iid` 플래그에 따라 클라이언트별 파티션을 구성하도록 설계되어 있습니다. 실제 로더 구현은 현재 리포지토리에 포함돼 있지 않아 `src/data` 모듈을 추가해야 실행이 가능합니다.
- `scripts/download_dataset.py`는 공개 IoT 이상 징후 데이터셋을 내려받고 압축 해제·샘플 통계 확인까지 수행할 수 있도록 제공되며, CSV 요약과 라벨 분포 확인 옵션을 포함합니다.

## 코드 구조
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
configs 경로([configs](configs))는 실험 전역 설정을 포함하며 [base.yaml](configs/base.yaml)에서 데이터 경로와 하이퍼파라미터를 선언합니다.
scripts 경로([scripts](scripts))는 데이터 다운로드, 학습 실행, 결과 시각화 스크립트를 담고 있습니다.
src 루트([src](src))에는 실험 로직이 모여 있으며, 설정 객체([config.py](src/config.py#L1-L108)), 연합 학습 모듈([src/federated](src/federated)), 모델 정의([src/models](src/models)), 공통 유틸리티([src/utils](src/utils))로 구성됩니다.

### 모델·학습 설정
- [ModelConfig](src/config.py#L45-L55)는 인코더/디코더 레이어 폭, 잠재 벡터 차원, 활성화 함수, 드롭아웃, 레이어 정규화를 파라미터화합니다. 기본 `encoder_layers`와 `decoder_layers`는 대칭 구조로 설정되어 있으며 입력 차원은 데이터셋 로딩 시점에 갱신됩니다.
- [TrainingConfig](src/config.py#L32-L42)는 로컬 에폭 수, 배치 크기, Adam 옵티마이저 학습률과 weight decay, 얼리 스토핑 기준(`early_stopping_patience`, `early_stopping_delta`), 잠재 공간 노이즈 강도(`latent_noise_std`)를 제어합니다.
- 로컬 업데이트는 [LocalTrainer](src/federated/trainer.py#L24-L95)가 담당하며, 미니배치 오토인코더 학습과 얼리 스토핑 로직, 잠재 노이즈 주입(훈련 시만 활성화)을 포함합니다.
- 글로벌 라운드는 [FederatedServer](src/federated/server.py#L30-L185)가 관리하고, `clients_per_round`, `rounds`, `aggregation` 등은 [FederatedConfig](src/config.py#L58-L72)로 조정합니다.

### 결과·평가 지표
- 각 라운드마다 서버는 검증 세트 재구성 오차를 기반으로 임계값을 탐색하고, 조건(`threshold_metric`, `threshold_min_recall`, `threshold_min_precision`, `threshold_max_fpr`)을 만족하는 후보를 선택합니다. 실패 시 학습 데이터의 `anomaly_threshold_quantile`을 사용해 폴백합니다.
- 테스트 단계에서는 [compute_metrics](src/federated/metrics.py#L8-L54)가 정확도, 재현율, 정밀도, F1, FPR, 특이도 및 혼동 행렬 카운트를 반환하며, 평균 재구성 오차와 선택된 임계값이 추가 저장됩니다.
- 학습 로그와 라운드별 결과는 [scripts/run_fediot.py](scripts/run_fediot.py#L98-L139)에서 JSON 히스토리(`outputs/history_*.json`)와 글로벌 모델 가중치(`outputs/global_model.pt`)로 남습니다.
- 후처리 스크립트 [scripts/plot_threshold_curves.py](scripts/plot_threshold_curves.py#L107-L205)는 저장된 모델을 불러 ROC/PR 곡선을 그려 `outputs/roc_pr_curves.png`로 저장하며, 선택 임계값에서의 검증/테스트 지표를 콘솔에 출력합니다.

## 실행/설치 매뉴얼

### requirements 파일
- 프로젝트 루트의 [requirements.txt](requirements.txt)는 PyTorch, NumPy, Pandas, scikit-learn, matplotlib, PyYAML 등 실험을 재현하는 데 필요한 최소 의존성을 정의합니다.
- GPU 가속이 필요하다면 PyTorch를 CUDA 버전에 맞춰 별도로 설치한 뒤 나머지 패키지를 `pip install -r requirements.txt`로 추가하는 것을 권장합니다.

### 환경 준비 및 실행 순서
1. 가상환경 생성 및 활성화
	```bash
	python -m venv .venv
	source .venv/bin/activate
	python -m pip install --upgrade pip
	```
2. 의존성 설치
	```bash
	pip install -r requirements.txt
	```
3. 데이터 내려받기 (예시: TON_IoT 네트워크 캡처)
	```bash
	python scripts/download_dataset.py --url <CSV_OR_ARCHIVE_URL> --decompress --summarize <상대경로> --label-column label
	```
	- 다운로드한 파일 경로를 `configs/base.yaml`의 `data.dataset_path`로 지정하고, 필요 시 특성/범주형 목록을 데이터셋 스키마에 맞게 조정합니다.
4. 데이터 로더 구현 확인
	- `src/data` 모듈(예: `dataset.py`, `__init__.py`)이 아직 리포지토리에 포함되지 않았으므로, `load_iot_dataset`과 `partition_dataset` 함수를 구현하거나 외부에서 가져와야 합니다.
5. 연합 학습 실행
	```bash
	python scripts/run_fediot.py --config configs/base.yaml --output outputs/exp_toniot
	```
	- 실행 중 콘솔 로그에서 클라이언트별 라벨 구성과 라운드별 글로벌 지표를 확인할 수 있으며, 결과물은 `--output`으로 지정한 디렉터리에 저장됩니다.

### 후처리 및 시각화
- 학습 완료 후, 저장된 모델과 동일한 설정을 사용해 아래 명령으로 ROC/PR 곡선을 생성합니다.
  ```bash
  python scripts/plot_threshold_curves.py --config configs/base.yaml --model outputs/exp_toniot/global_model.pt --output outputs/exp_toniot/roc_pr_curves.png
  ```
- 결과 이미지와 콘솔 요약을 레포트 혹은 발표 자료에 활용하고, 논문과의 수치 비교 시 설정 차이(클라이언트 수, 라운드 수, 하이퍼파라미터)를 명시하는 것이 좋습니다.

## 참고 자료 및 주의사항
- 논문 전문은 [Iot_paper.pdf](Iot_paper.pdf), 발표 개요는 [[사물인터넷-오전반]_팀5_프로젝트2_발표자료.pdf]([%EC%82%AC%EB%AC%BC%EC%9D%B8%ED%84%B0%EB%84%B7-%EC%98%A4%EC%A0%84%EB%B0%98]_%ED%8C%805_%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B82_%EB%B0%9C%ED%91%9C%EC%9E%90%EB%A3%8C.pdf)에서 확인할 수 있습니다.
