# 실험에 대한 Result 페이지 입니다

## Global Threshold
<div>
  <img src="https://github.com/iot-team5/iot-project-team5/blob/main/result/threshold.png?raw=true" />
</div>

- Global Model로 산출한 Global Threshold 값은 다음과 같습니다.
- Global Threshold: 0.0324
- 정상값 기반의 학습 데이터로 훈련 했기에, 낮은 값으로 수렴합니다.
- 이 Threshold는 차후, Anomaly Detection에 사용됩니다.

## Confusion Matrix
<div>
  <img src="https://github.com/iot-team5/iot-project-team5/blob/main/result/confusion-matrix.png?raw=true" />
</div>

- Confusion Matrix의 결과값은 다음과 같습니다.
- 각 면은 각각 정상데이터를 정상탐지, 정상데이터를 오탐지, 공격데이터를 오탐지, 공격데이터를 정상탐지한 것으로 구분됩니다.
- 주요 탐지인 공격데이터에 대한 정상탐지는 오탐 대비 99% 탐지 성능에 달성했습니다.
- 이를 Accuracy, Recall, Precision, F1-score와 같은 평가 지표로 나타내면 각각 0.99, 0.98, 0.98, 0.98에 달성합니다.
