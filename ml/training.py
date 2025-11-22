# ml/training.py
from typing import List, OrderedDict
from torchmetrics import JaccardIndex
import torch
import torch.nn as nn
from .model import create_mobilenet

def evaluate_model(model_state_dict, data_loader, device):
    """주어진 모델 가중치와 데이터로더로 정확도와 손실을 평가"""
    model = create_mobilenet()
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()
    
    criterion = nn.CrossEntropyLoss()

    jaccard = JaccardIndex(task="multiclass", num_classes=10).to(device)

    correct = 0
    total = 0
    total_loss = 0.0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            jaccard.update(predicted, labels)
            
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(data_loader)
    miou = jaccard.compute().item() * 100
    
    return accuracy, avg_loss

"""
Aggregation 함수
이 부분에서 전체 모델의 성능 차이가 발생함.
연구의 핵심 부분
2-Layer에서는 Cluster와 Global 모델 각각에 대한 Aggregation 작업을 수행하는 것도 비교해야함
"""
def fed_avg(models_to_average: List[OrderedDict]) -> OrderedDict:
    """
    Federated Averaging 알고리즘을 수행.
    여러 모델의 가중치(state_dict) 리스트를 받아 각 가중치의 평균을 계산하여 반환.
    """
    if not models_to_average: return OrderedDict()
    avg_state_dict = OrderedDict()
    for key in models_to_average[0].keys():
        # 동일한 키(레이어)의 텐서들을 리스트로 모음
        tensors = [model[key].float() for model in models_to_average]
        # 텐서들을 쌓고(stack), 평균을 계산
        avg_tensor = torch.stack(tensors).mean(dim=0)
        avg_state_dict[key] = avg_tensor
    return avg_state_dict
