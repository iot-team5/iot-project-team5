# iot-project

# FedIoT Reference Implementation Plan

This workspace provides a reproducible scaffold for studying the FedIoT architecture described in the FedML research paper as well as experimenting with IoT anomaly detection using federated learning. The code base follows the step-by-step execution plan outlined below.

## 1. 논문 심층 분석 및 환경 구축
- 코드 구조는 `src/` 패키지로 모듈화되어 논문에서 제시한 FedIoT 구성요소(클라이언트, 서버, Autoencoder)를 독립적으로 실험할 수 있게 설계했습니다.
- 실험 설정은 `configs/base.yaml` 에 정리되어 있으며, YAML 기반 설정 로더(`src/config.py`)를 통해 쉽게 변경할 수 있습니다.
- 실행 환경은 Python 3.10+을 기준으로 하며, `requirements.txt`에 필요한 라이브러리를 정리했습니다. PyTorch 기반 Autoencoder와 FedAvg 로직을 사용합니다.

## 2. 데이터 분할 및 로컬 구현
- `src/data/dataset.py` 는 IoT 이상 탐지용 CSV 데이터셋을 로드하고, 표준화 및 학습/평가 분할을 수행합니다.
- `src/data/partition.py` 는 IID / 비-IID 설정에 따라 클라이언트별 데이터를 분할합니다. 넘파이 기반으로 구현되어 다양한 데이터셋에 적용 가능합니다.
- `src/models/autoencoder.py` 는 심층 Autoencoder를 구성하며, 레이어/활성화/드롭아웃 파라미터를 설정 파일에서 제어할 수 있습니다.
- `src/federated/trainer.py` 는 각 클라이언트의 로컬 학습 루프를 정의하고, 조기 종료와 손실 추적을 지원합니다.

## 3. FedIoT 알고리즘 재구현 및 검증
- `src/federated/server.py` 는 FedAvg 기반의 중앙 서버 로직을 제공합니다. 라운드마다 클라이언트를 샘플링하고, 가중 평균으로 파라미터를 집계합니다.
- 재구성 오차 기반 이상 탐지를 위해 자동으로 학습 데이터에서 임계값을 추정하고(`anomaly_threshold_quantile`), 테스트셋에 대해 Accuracy/Recall을 산출합니다.
- `scripts/run_fediot.py` 는 전체 파이프라인을 실행하는 진입점으로, 라운드별 결과를 JSON 파일로 저장하고 최종 글로벌 모델을 `outputs/` 디렉토리에 기록합니다.

## 4. 최종 정리 및 발표 준비
- 실행 결과는 시간 스탬프를 포함한 JSON 로그(`outputs/history_*.json`)로 수집되어 손쉽게 시각화할 수 있습니다.
- `src/utils/logging.py` 를 통해 표준 로깅이 설정되며, 발표 자료 작성 시 손실 및 지표 변화 추이를 정리하기 용이합니다.

## 빠른 시작

1. **의존성 설치**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **데이터 준비**
    - 공개 IoT 이상 탐지 데이터셋은 `scripts/download_dataset.py` 로 직접 내려받을 수 있습니다. 예)
       ```bash
       python scripts/download_dataset.py --url <DATASET_URL> --output data/raw/toniot.zip --decompress
       ```
       `<DATASET_URL>` 위치에는 TON_IoT 텔레메트리 ZIP과 같은 공식 배포 링크를 넣으세요.
    - 압축 해제로 생성된 CSV 구조를 확인하려면 다음과 같이 요약을 실행합니다.
       ```bash
       python scripts/download_dataset.py --summarize <EXTRACTED_CSV> --label-column <LABEL_COLUMN_NAME>
       ```
       `<EXTRACTED_CSV>`에는 실제로 추출된 CSV 경로(예: `data/raw/toniot/Telemetry/Train_data.csv`)를 지정하고, 필요하면 `--separator`나 `--encoding` 옵션을 함께 사용하세요.
    - 요약 결과를 참고하여 `configs/base.yaml`의 `data.dataset_path`, `feature_columns`, `target_column`, `positive_label`, `negative_label` 값을 데이터셋에 맞게 조정하세요. 사용 전 각 데이터셋의 라이선스 조건도 확인하시기 바랍니다.

3. **시뮬레이션 실행**
   ```bash
   python scripts/run_fediot.py --config configs/base.yaml
   ```

4. **결과 확인**
   - `outputs/history_*.json`에서 라운드별 손실 및 성능을 확인합니다.
   - `outputs/global_model.pt`은 학습된 Autoencoder 가중치를 담고 있습니다.

## 향후 작업 제안
- FedML 레퍼런스 구현과 비교 분석을 위해 공식 저장소(`https://github.com/FedML-AI/FedML`)를 클론하고, 본 코드와의 구조 차이를 문서화하세요.
- IoT 데이터셋(예: TON_IoT, UNSW-NB15 등)을 적용하고, 클라이언트 수/분포를 달리하여 성능 변화를 실험하세요.
- 지표 확장을 위해 Precision, F1-score, AUC 등을 추가하고, 발표 자료에 활용할 시각화 스크립트를 작성하세요.

## 2025-11-28 실험 요약
- `configs/base.yaml` 최신 설정은 심층 오토인코더(Encoder 5층, LayerNorm, latent noise 0.05)와 검증 FPR 상한(`threshold_max_fpr=0.1`)을 사용합니다.
- `scripts/run_fediot.py --config configs/base.yaml` 실행 결과:
   - 검증: 정밀도 0.977, 재현율 0.753, FPR 0.058, F1 0.850, 정확도 0.798.
   - 테스트: 정밀도 0.977, 재현율 0.757, FPR 0.057, F1 0.853, 정확도 0.801.
   - 선택된 임계값은 0.1745이며, 혼동행렬은 TP 24,391 / FP 570 / TN 9,430 / FN 7,818입니다.
- `scripts/plot_threshold_curves.py --config configs/base.yaml --output outputs/roc_pr_curves_v3.png` 로 생성된 ROC/PR 플롯에서 검증/테스트 AUC는 각각 ROC≈0.91, PR≈0.96으로 나타났습니다.

## 개선 메모 및 다음 단계
1. 임계값 튜닝: `threshold_max_fpr`, `threshold_min_recall`, `threshold_min_precision`을 조합해 FPR–Recall 트레이드오프 표를 작성하고, 운영 요구에 맞는 포인트를 선택합니다.
2. 정밀도 유지·재현율 향상: `training.latent_noise_std`, `model.dropout`을 추가로 조정하거나, latent 구조를 확장해 공격 재현율을 0.8 이상으로 끌어올리는 방안을 탐색합니다.
3. 혼합 피처 실험: 범주형 세분화(예: `type`, `service` 하위 그룹)나 파생 피처 추가, 혹은 시계열 윈도우 기반 피처를 도입해 ROC 상단부를 더 왼쪽/위쪽으로 이동시키는 실험을 수행합니다.
4. 결과 문서화: `outputs/roc_pr_curves_v3.png`와 최신 혼동행렬을 발표 자료에 포함하고, baseline(Threshold FPR 제한 전) 결과와 비교 그래프/표를 추가하세요.

위 항목을 완료하면 FPR을 낮춘 상태에서도 재현율 방어선이 어떤지 명확히 설명할 수 있으며, 다음 단계 개선 계획과 실험 로그가 README에서 바로 추적 가능해집니다.
