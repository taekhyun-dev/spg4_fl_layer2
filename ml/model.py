# ml/model.py
from dataclasses import dataclass, field
from typing import List, OrderedDict
from torchvision import models

@dataclass
class PyTorchModel:
    """PyTorch 모델의 상태(버전, 가중치 등)를 담는 데이터 클래스"""
    version: int
    model_state_dict: OrderedDict
    trained_by: List[int] = field(default_factory=list)

def create_mobilenet():
    """CIFAR10 데이터셋(10개 클래스)에 맞게 사전 학습 없이 초기화된 MobileNetV3-Small 모델을 생성"""
    model = models.mobilenet_v3_small(weights=None, num_classes=10)
    return model

def check_mobilenet():
    import torch
    import torchvision.models as models
    from thop import profile
    from torchsummary import summary

    # 0. 모델 로드 (MobileNet-V3 Small)
    # weights=None (무작위 초기화) 또는 'IMAGENET1K_V1' (사전 학습)
    # 구조 분석이 목적이므로 어떤 것을 사용해도 파라미터 수와 계산량은 동일합니다.
    model = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
    model.eval() # 분석 시에는 항상 평가 모드(eval mode)로 설정

    # 1. 분석을 위한 더미 입력 데이터 생성
    # 표준 ImageNet 입력 크기 (batch_size=1, channels=3, height=224, width=224)
    # 계산량(FLOPs)은 입력 크기에 따라 달라지므로 표준 크기를 사용합니다.
    input_tensor = torch.randn(1, 3, 224, 224)

    print("--- MobileNet-V3 Small 모델 기본 정보 ---")
    print(f"입력 데이터 크기 (Input size): {input_tensor.shape}")
    print("-" * 40)

    # 2. 총 계산량 (FLOPs / MACs) 및 파라미터 계산 (thop 라이브러리)
    # thop는 MACs (Multiply-Accumulate operations)를 계산합니다.
    # 1 G-MACs = 10^9 MACs (Giga-MACs)
    # (참고: FLOPs는 종종 2 * MACs로 근사하지만, thop의 출력은 G-MACs입니다.)
    macs, params = profile(model, inputs=(input_tensor, ), verbose=False) # verbose=False로 세부 출력 끔

    print(f"\n✅ [thop 라이브러리 분석 (계산량 및 파라미터)]")
    print(f"총 파라미터 개수 (Total Params): {int(params):,} 개")
    print(f"총 계산량 (Total MACs): {macs / 1e9:.2f} G-MACs")
    print(f"(참고: GFLOPs는 약 { (macs * 2) / 1e9:.2f} GFLOPs 입니다.)")


    # 3. 계층별 상세 정보 (torchsummary 라이브러리)
    # torchsummary는 계층별 출력 크기, 파라미터 수, 총 파라미터 및 메모리 크기를 요약해줍니다.
    # 입력 크기를 (channels, height, width)로 받습니다.
    print(f"\n✅ [torchsummary 라이브러리 분석 (계층별 세부 정보)]")
    # GPU가 있다면 "cuda", 없다면 "cpu"
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device_name)

    summary(model, input_size=(3, 224, 224), device=device_name)

check_mobilenet()