# RVC WebUI

## 프로젝트 개요

이 프로젝트는 [AI Hub](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=542)에 공개된 **한국어 일반 음성 데이터셋**을 기반으로,  
Retrieval-based Voice Conversion (RVC)을 활용하여 음성을 **케로로, 코난, 짱구** 캐릭터 음성으로 변환한 결과물을 생성합니다.

---

## 사용한 데이터

- **출처**: AI Hub - 다화자 음성합성 데이터
- **링크**: [https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=542](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=542)
- **용도**: 음성 변환을 위한 입력 데이터로 활용
- **구조**:
  - `TS1/` 폴더는 여러 명의 화자 데이터를 포함합니다.
  - 각 서브폴더(예: `0005_G1A3E7_KYG`)는 **1명의 화자**에 해당하며, 해당 폴더 안에 그 사람의 음성 파일들이 포함되어 있습니다.

```
TS1/
├── 0005_G1A3E7_KYG/
│   ├── audio1.wav
│   ├── audio2.wav
│   └── ...
├── 0007_G1A2E7_KES/
├── 0011_G1A2E7_LJH/
└── ...
```

---

## 학습 정보 및 모델 설정

- **사용 캐릭터**: 케로로, 코난, 짱구
- 각 캐릭터당 약 **40분 분량의 음성 데이터를 수집**하여 학습에 사용했습니다.
- **인덱싱(indexing)** 포함한 학습을 진행하여, 캐릭터별로 **독립된 모델을 생성**했습니다.
- 훈련 후 생성된 모델을 통해 AI Hub 화자 음성을 해당 캐릭터 목소리로 변환하였습니다.



### 학습 설정

- `epochs`: **325**
- `batch_size`: **12**
- `pretrained model G`: `pretrained_v2/32k`  | (RVC 공식 사전학습 가중치 사용)
- `pretrained model D`: `pretrained_v2/32k`  | (RVC 공식 사전학습 가중치 사용)
  
---

## 변환 캐릭터

- 케로로
- 코난
- 짱구

> 각 화자의 음성(`TS1/000X_.../*.wav`)을 위 캐릭터 음성으로 변환하여 저장하였습니다.

