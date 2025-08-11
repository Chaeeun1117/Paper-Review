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
* 시각 기하학 - 2D->3D 정보 복원 기법들/ 사진 여러장으로부터 같은 물체의 점들이 시점마다 어떻게 달라 보이는지 분석
* BA - 모든 사진의 카메라 위치, 방향, 3D 포인트 위치 등을 동시 최적화해 가장 알맞는 3D 구조 찾음  
여기에 기계학습은 보조적 역할만 수행했다.  
최근에는 기계학습과 시각 기하학을 통합한 VGGSfM(학습 중 BA를 내부에 포함)과 같은 기법들이 등장했으나, 여전히 시각 기하학이 중심 역할을 하여 복잡성과 계산 비용이 크다.  
  
신경망이 점점 강력해지면서, 기하학 기반 후처리 없이 신경망 하나만으로 3D 작업을 수행하는 DUSt3R이나 그 확장판인 MASt3R과 같은 모델들이 유망한 결과를 보이고 있다.  
그러나 이들은 두 장의 이미지만 처리할 수 있어 쌍 간의 재구성 결과를 합치는 후처리가 필요하다.  

**Introduction**  
본 논문에서는 3D 기하 후처리 최적화의 필요성을 줄이기 위해 VGGT라는 Feed-forward 신경망을 제안한다.  

VGGT는 특별한 3D 구조나 Inductive bias 없이, 일반적인 대형 Transformer 구조를 사용한다.  
(단, 프레임별(frame-wise) 어텐션과 전체 시퀀스(global) 어텐션을 번갈아 사용하는 점은 예외)  
따라서 GPT, CLIP, DINO 등과 같이 범용적인 backbone 모델로 활용 가능하다.  
구체적으로 VGGT가 추출하는 features는 동적 영상에서의 포인트 추적이나 새로운 시점 합성 같은 후속 작업의 성능을 크게 향상시킬 수 있다.  


---

## 2. 관련 연구

**Structure-from-Motion**  
정적인 장면을 여러 시점에서 촬영한 이미지로 **카메라 파라미터**와 **sparse 3D Point Cloud** (사진마다 뚜렷하게 보이는 몇몇 특징점만 3D 위치 계산)를 추정하는 작업이다.  
이미지 매칭 - triangulation - Bundle Adjustment
최근엔 딥러닝이 keypoint Detection과 Image Matching(여러장에서 같은 물체의 같은 지점 찾기) 단계 개선했다.  
-> 더 나아가 VGGSfM처럼 SfM 전체를 통합 학습하는 방식도 등장

**Multi-view Stereo**  
MVS는 여러 장의 겹치는 이미지로부터 장면의 dense 3D 기하(장면 전체를 복원)를 재구성하는 방법이다.  
카메라 파라미터는 SfM으로 미리 추정되어 있다고 가정한다.  
전통 수작업 기반 (handcrafted) - 전역 최적화 기반 (global optimization) - 학습 기반 (learning-based) 세 가지의 방식이 존재한다.  
특히 DUSt3R과 MASt3R는 카메라 파라미터 없이도 두 장의 이미지에서 정렬된 밀집 포인트 클라우드를 직접 예측한다. -> 기하학 단계 제거  
단, 두 장만 처리 가능해 후처리가 항상 필요하는 단점이 존재한다.  
VGGT는 이를 더 발전시킨 형태이다.

**Tracking-Any-Point**   
비디오의 임의 2D 포인트를 모든 프레임에서 추적하는 방법이다.  
CoTracker -> 여러 포인트 간의 상관관계를 이용해 occlusion(가림) 상황에서도 추적 가능  
DOT -> dense tracking 가능  
TAPTR -> 포인트 추적 전용 end-to-end 트랜스포머를 제안  
그러나 이 모든 방법은 **특화된 포인트 추적기**이다.  
VGGT는 이런 특화 모델이 아님에도 추출한 특징(이미지에서 각 위치를 표현하는 벡터, 다른 프레임과 대응점 찾는 데 쓰임)을 기존 포인트 추적 모델에 넣어주면 SOTA를 달성한다.  

---

## 3. 주요 아이디어

<img width="912" height="404" alt="image" src="https://github.com/user-attachments/assets/e81119c8-6a82-49bf-aec5-2c5a37493dd9" />
*트랜스포머  
입력의 순서에 상관없이, 전체 요소들이 서로를 한 번에 참조하면서 중요한 관계를 찾아내는 모델  
  
입력 처리  
여러 장의 겹치는 이미지를 격자로 나눠 토큰 변환  
사전 학습된 비전 트랜스포머 DINO로 각 패치 토큰을 벡터 특징으로 변환  
카메라 예측을 위해 카메라 토큰을 각 프레임 토큰 앞에 붙임  
  
어텐션 단계  
- Global Attention: 모든 프레임의 토큰을 한꺼번에 보면서 장면 전체 정보 학습  
- Frame Attention: 각 프레임 내부의 토큰끼리만 어텐션 -> 프레임 개별 디테일 학습
이 global, frame attention 과정을 L번 반복함
  
출력 헤드들  
Camera Head: 카메라 파라미터 예측  
DPT (Dense Prediction Transformer): 깊이맵, 포인트맵, 트랙 등 픽셀 단위 예측  
Depth maps, point maps(3D 좌표로 투영된 장면 포인트), Tracks(특정 포인트 프레임별 이동경로)  



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

