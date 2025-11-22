# object/satellite.py
import asyncio
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

from typing import List, Dict, Tuple, Coroutine
from datetime import datetime
from typing import Tuple, Dict
from ml.model import PyTorchModel, create_mobilenet
from ml.training import evaluate_model, fed_avg
from utils.skyfield_utils import EarthSatellite
from utils.logging_setup import KST
from utils.config import LOCAL_EPOCHS, FEDPROX_MU, MAX_ISL_DISTANCE_KM
from object.clock import SimulationClock

# ----- CLASS DEFINITION ----- #
class Satellite:
    def __init__ (self, sat_id: int, satellite_obj: EarthSatellite, clock: 'SimulationClock', sim_logger, perf_logger,
                   initial_model: PyTorchModel):
        self.sat_id = sat_id
        self.satellite_obj = satellite_obj
        self.clock = clock
        self.logger = sim_logger
        self.perf_logger = perf_logger
        self.position = {"lat": 0.0, "lon": 0.0, "alt": 0.0}
        self.state = "IDLE"
        self.global_model = initial_model
        self.model_ready_to_upload = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

class WorkerSatellite(Satellite):
    def __init__ (self,
                    sat_id: int, 
                    satellite_obj: EarthSatellite, 
                    clock: 'SimulationClock', 
                    sim_logger, 
                    perf_logger, 
                    initial_model: PyTorchModel,
                    master: 'MasterSatellite', train_loader, val_loader):
        super().__init__(sat_id, satellite_obj, clock, sim_logger, perf_logger, initial_model)
        self.master = master
        self.local_model = self.global_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger.info(f"Worker SAT {self.sat_id} 생성")

    def _train_and_eval(self) -> Tuple[Dict, float, float]:
        """
        실제 PyTorch 모델 학습을 수행하는 블로킹(동기) 함수.
        asyncio 이벤트 루프를 막지 않기 위해 별도의 스레드에서 실행됩니다.
        """
        try:
            loader_length = len(self.train_loader)
            self.logger.info(f"✅ DataLoader의 총 배치 개수: {loader_length}")
            if loader_length == 0:
                self.logger.error("⚠️ DataLoader가 비어있습니다. Dataset을 확인해주세요.")
                return # 또는 다른 에러 처리
        except Exception as e:
            self.logger.error(f"❌ DataLoader의 길이를 확인하는 중 에러 발생: {e}")

        # --- 학습 파트 ---
        temp_model = create_mobilenet()
        temp_model.load_state_dict(self.local_model.model_state_dict)
        temp_model.to(self.device)
        temp_model.train()

        # --- FedProx 추가 부분 ---
        #    global_model_ref (w^t): Proximal term 계산을 위한 '고정된' 기준 모델
        #    마찬가지로 'self.global_model' (w^t)의 가중치를 가지며, 학습되지 않도록 .eval()
        global_model_ref = create_mobilenet()
        global_model_ref.load_state_dict(self.global_model.model_state_dict)
        global_model_ref.to(self.device)
        global_model_ref.eval() # 중요: gradient가 흐르지 않도록 설정

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(temp_model.parameters(), lr=3e-4, weight_decay=1e-4)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    
        for epoch in range(LOCAL_EPOCHS):
            self.logger.info(f"    - SAT {self.sat_id}: 에포크 {epoch+1}/{LOCAL_EPOCHS} 진행 중...")
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()

                outputs = temp_model(images)
                loss = criterion(outputs, labels)
                
                # --- FedProx 손실 함수 수정 부분 ---
                #     근접 항(Proximal Term) 계산: ||w - w^t||^2
                prox_term = 0.0

                # temp_model.parameters() (w)와 global_model_ref.parameters() (w^t) 비교
                for local_param, global_param in zip(temp_model.parameters(), global_model_ref.parameters()):
                    # .detach()를 사용하여 w^t의 gradient가 계산되지 않도록 함
                    prox_term += torch.sum(torch.pow(local_param - global_param.detach(), 2))

                # --- FedProx 손실 함수 최종 계산 부분 ---
                #     최종 손실 계산: Loss + (mu/2) * prox_term
                total_loss = loss + (FEDPROX_MU / 2) * prox_term

                # loss.backward()
                total_loss.backward()
                optimizer.step()
            scheduler.step()
            
        new_state_dict = temp_model.cpu().state_dict()
        self.logger.info(f"  🧠 SAT {self.sat_id}: 로컬 학습 완료 ({LOCAL_EPOCHS} 에포크). 검증 시작...")
            
        # --- 검증 파트 ---
        accuracy, loss, miou = evaluate_model(new_state_dict, self.val_loader, self.device)
            
        return new_state_dict, accuracy, loss, miou

    async def train_and_eval(self):
        """CIFAR10 데이터셋으로 로컬 모델을 학습하고 검증"""
        self.state = 'TRAINING'
        self.logger.info(f"  ✅ SAT {self.sat_id}: 로컬 학습 시작 (v{self.local_model.version}).")
        new_state_dict = None
        try:
            # 현재 실행중인 이벤트 루프를 가져옵니다.
            loop = asyncio.get_running_loop()
            new_state_dict, accuracy, loss, miou = await loop.run_in_executor(None, self._train_and_eval)
            self.local_model.model_state_dict = new_state_dict
            self.logger.info(f"  📊 [Local Validation] SAT: {self.sat_id}, Version: {self.local_model.version}, Accuracy: {accuracy:.2f}%, Loss: {loss:.4f}, Miou: {miou:.2f}%")
            self.perf_logger.info(f"{datetime.now(KST).isoformat()},LOCAL_VALIDATION,{self.sat_id},{self.local_model.version},N/A,{accuracy:.4f},{loss:.6f},{miou:.4f}")

            self.local_model.trained_by = [self.sat_id]
            self.model_ready_to_upload = True

        except Exception as e:
            self.logger.error(f"  💀 SAT {self.sat_id}: 학습 또는 검증 중 에러 발생 - {e}", exc_info=True)

        finally:
            self.state = 'IDLE'
            self.logger.info(f"  🏁 SAT {self.sat_id}: 학습 절차 완료.")

    async def send_model_to_iot(self, iot: 'IoT'):
        if self.global_model.version > iot.global_model.version:
            self.logger.info(f"  🛰️ SAT {self.sat_id} -> IoT {iot.name}: 글로벌 모델 전송 (버전 {self.global_model.version})")
            await iot.receive_global_model(self.global_model)

    async def send_local_model(self) -> PyTorchModel | None:
        if self.model_ready_to_upload:
            self.model_ready_to_upload = False
            return self.local_model
        return None

class MasterSatellite(Satellite):
    def __init__(self,
                 sat_id: int, 
                 satellite_obj: EarthSatellite, 
                 clock: 'SimulationClock', 
                 sim_logger, 
                 perf_logger, 
                 initial_model: PyTorchModel,
                 test_loader):
        super().__init__(sat_id, satellite_obj, clock, sim_logger, perf_logger, initial_model)
        self.test_loader = test_loader
        self.cluster_model = self.global_model
        self.cluster_version_counter = 0
        
        self.cluster_members: Dict[int, WorkerSatellite] = {}
        self.cluster_model_buffer: List[PyTorchModel] = []
        self.logger.info(f"Master SAT {self.sat_id} 생성")

    def add_member(self, worker: WorkerSatellite):
        self.cluster_members[worker.sat_id] = worker

    async def receive_global_model(self, model: PyTorchModel):
        """지상국으로부터 글로벌 모델을 수신"""
        self.logger.info(f"  🛰️ Master SAT {self.sat_id}: 새로운 글로벌 모델 수신 (v{model.version}).")
        self.global_model = model
        self.cluster_model = model
        self.model_ready_to_upload = False

    async def send_model_to_worker(self, worker: WorkerSatellite):
        self.logger.info(f"  🛰️ -> 🛰️  Master {self.sat_id} -> Worker {worker.sat_id}: 모델 전송 (버전 {self.cluster_model.version})")
        worker.local_model = self.cluster_model
        # 모델을 받은 워커는 다시 학습할 준비가 된 것이므로 IDLE 상태로 변경
        if worker.state == 'WAITING_TRAINING':
            worker.state = 'IDLE'

    async def receive_model_from_worker(self, worker: WorkerSatellite):
        self.cluster_model_buffer.append(worker.local_model)
        worker.model_ready_to_upload = False
        self.logger.info(f"  📥 MasterSAT {self.sat_id}: Worker {worker.sat_id} 모델 수신. (버퍼 크기: {len(self.cluster_model_buffer)})")

    async def aggregate_models_periodically(self):
        """주기적으로 버퍼에 쌓인 워커 모델들을 취합"""
        while True:
            # await asyncio.sleep(30)
            await asyncio.sleep(2)
            if not self.cluster_model_buffer:
                continue
            await self._aggregate_and_evaluate_cluster_models()

    async def _aggregate_and_evaluate_cluster_models(self):
        """실제 모델 취합 및 평가 로직"""
        self.logger.info(f"  ✨ [Cluster Aggregation] Master {self.sat_id}: {len(self.cluster_model_buffer)}개 워커 모델과 기존 클러스터 모델 취합 시작")
        
        state_dicts_to_avg = [self.cluster_model.model_state_dict] + [m.model_state_dict for m in self.cluster_model_buffer]
        new_state_dict = fed_avg(state_dicts_to_avg)
        all_contributors = list(set(self.cluster_model.trained_by + [p for model in self.cluster_model_buffer for p in model.trained_by]))
        
        self.cluster_model.model_state_dict = new_state_dict
        self.cluster_model.trained_by = all_contributors
        self.model_ready_to_upload = True
        self.cluster_version_counter += 1
        self.logger.info(f"  ✨ [Cluster Aggregation] Master {self.sat_id}: 클러스터 모델 업데이트 완료. 학습자: {self.cluster_model.trained_by}")

        # 평가도 블로킹 작업이므로 executor에서 실행
        accuracy, loss = await asyncio.get_running_loop().run_in_executor(
            None, evaluate_model, new_state_dict, self.test_loader, self.device
        )
        self.logger.info(f"  🧪 [Cluster Test] Owner: SAT_{self.sat_id}, Global Ver: {self.cluster_model.version}, Cluster Ver: {self.cluster_version_counter}, Accuracy: {accuracy:.2f}%, Loss: {loss:.4f}")
        self.perf_logger.info(f"{datetime.now(KST).isoformat()},CLUSTER_TEST,SAT_{self.sat_id},{self.cluster_model.version},{self.cluster_version_counter},{accuracy:.4f},{loss:.6f}")

        self.cluster_model_buffer.clear()

    async def send_cluster_model(self) -> PyTorchModel | None:
        if self.model_ready_to_upload:
            self.model_ready_to_upload = False
            return self.cluster_model
        return None

class Satellite_Manager:
    def __init__ (self, master: 'MasterSatellite', clock: 'SimulationClock', sim_logger):
        self.master = master
        self.logger = sim_logger
        self.clock = clock
        self.logger.info(f"Master SAT {self.master.sat_id} 위성 관리자 생성 완료.")

    async def run(self):
        self.logger.info(f"Master SAT {self.master.sat_id} 위성 관리자 운영 시작.")
        self.logger.info(f"Master SAT {self.master.sat_id} 임무 시작.")
        for sat in self.master.cluster_members.values():
            self.logger.info(f"  Worker SAT {sat.sat_id} 임무 시작.")
        await self.propagate_orbit_with_isl()

    async def propagate_orbit_with_isl(self):
        """ISL을 통해 워커 위성들과 통신하고 모델을 교환"""
        while True:
            await asyncio.sleep(self.clock.real_interval)
            await self._aggregate_and_evaluate_cluster_models()
            tasks = []
            for worker in self.master.cluster_members.values():
                distance = self.get_distance_between(self.master, worker)
                if distance <= MAX_ISL_DISTANCE_KM:
                    if self.master.cluster_model.version > worker.local_model.version or \
                    (self.master.cluster_model.version == worker.local_model.version and self.master.cluster_model.model_state_dict is not worker.local_model.model_state_dict):
                        send_model_task = asyncio.create_task(self.master.send_model_to_worker(worker))
                        tasks.append(send_model_task)
                    if worker.model_ready_to_upload:
                        receive_model_task = asyncio.create_task(self.master.receive_model_from_worker(worker))
                        tasks.append(receive_model_task)
            await asyncio.gather(*tasks)

            for worker in self.master.cluster_members.values():
                current_ts = self.clock.get_time_ts()
                geocentric = worker.satellite_obj.at(current_ts)
                subpoint = geocentric.subpoint()
                worker.position["lat"], worker.position["lon"], worker.position["alt"] = subpoint.latitude.degrees, subpoint.longitude.degrees, subpoint.elevation.km

            current_ts = self.clock.get_time_ts()
            geocentric = self.master.satellite_obj.at(current_ts)
            subpoint = geocentric.subpoint()
            self.master.position["lat"], self.master.position["lon"], self.master.position["alt"] = subpoint.latitude.degrees, subpoint.longitude.degrees, subpoint.elevation.km

    def get_distance_between(self, one_sat: 'Satellite', other_sat: 'Satellite') -> float:
        """다른 위성과의 거리를 계산"""
        current_ts = self.clock.get_time_ts()
        return (one_sat.satellite_obj - other_sat.satellite_obj).at(current_ts).distance().km