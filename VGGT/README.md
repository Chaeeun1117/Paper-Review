# VGGT: Visual Geometry Grounded Transformer (CVPR 2025)

[논문 링크](https://arxiv.org/abs/2503.11651) 
[코드 링크](https://github.com/facebookresearch/vggt)

---

## 1. 논문 개요

**VGGT란?**
- 한 장 또는 여러 장의 이미지로부터 장면의 모든 핵심 3D 속성(카메라 파라미터, 포인트 맵, 뎁스 맵,
  3D 포인트 트랙)을 추론하는 Feed-forward 신경망
- 후처리 없이 image reconstruction을 1초 이내로 수행

**Problem**
전통적으로 3D 재구성은 ‘시각 기하학(visual-geometry)’ 기반의 기법으로 접근되어 왔으며, Bundle Adjustment(BA)와 같은 반복적 최적화 기술을 사용해왔다.
여기에 기계학습은 보조적 역할만 수행했다.
최근에는 기계학습과 시각 기하학을 통합한 VGGSfM과 같은 기법들이 등장했으나, 여전히 시각 기하학이 중심 역할을 하여 복잡성과 계산 비용이 크다.

신경망이 점점 강력해지면서, 기하학 기반 후처리 없이 신경망 하나만으로 3D 작업을 수행하는 DUSt3R이나 그 확장판인 MASt3R과 같은 모델들이 유망한 결과를 보이고 있다.
그러나 이들은 두 장의 이미지만 처리할 수 있어 쌍 간의 재구성 결과를 합치는 후처리가 필요하다.

**Introduction**
본 논문에서는 3D 기하 후처리 최적화의 필요성을 줄이기 위해 VGGT라는 Feed-forward 신경망을 제안한다.

VGGT는 특별한 3D 구조나 Inductive bias 없이, 일반적인 대형 Transformer 구조를 사용한다.
(단, 프레임별(frame-wise) 어텐션과 전체 시퀀스(global) 어텐션을 번갈아 사용하는 점은 예외)
따라서 GPT, CLIP, DINO 등과 같이 범용적인 backbone 모델로 활용 가능하다.
구체적으로 VGGT가 추출하는 features는 동적 영상에서의 포인트 추적이나 새로운 시점 합성 같은 후속 작업의 성능을 크게 향상시킬 수 있다.




---

## 2. 주요 아이디어

- Visual Geometry Token 생성 및 Attention에 위치 정보 직접 반영
- 3D 공간 기반 Self-attention으로 더 정밀한 3D 인식 가능

---

## 3. 실험 및 결과

- KITTI, NYUv2 등 여러 데이터셋에서 우수한 성능
- Geometry 정보가 없는 기존 모델 대비 정확도 상승

---

## 4. 느낀 점 및 향후 연구 방향

- Geometry 정보를 Transformer에 직접 넣는 방식이 간단하면서 효과적임
- 향후 3D reconstruction, depth estimation 분야에 적용 가능성 높음

---

## 5. 참고 자료 및 링크

- Transformer 구조 기초 공부 자료  
- Position bias 관련 논문 및 블로그 링크

