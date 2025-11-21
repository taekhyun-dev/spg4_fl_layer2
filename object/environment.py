# object/environment.py
import asyncio
import torch
from datetime import datetime
from skyfield.api import Topos
from typing import Dict
from ml.model import PyTorchModel
from ml.training import evaluate_model, fed_avg
from object.satellite import Satellite, MasterSatellite
from utils.logging_setup import KST
from utils.config import AGGREGATION_STALENESS_THRESHOLD, IOT_FLYOVER_THRESHOLD_DEG
from object.clock import SimulationClock

# ----- CLASS DEFINITION ----- #
class IoT:
    def __init__ (self, name, latitude, longitude, elevation, sim_logger, initial_model: PyTorchModel, test_loader):
        self.name = name
        self.logger = sim_logger
        self.topos = Topos(latitude_degrees=latitude, longitude_degrees=longitude, elevation_m=elevation)
        self.global_model = initial_model
        self.test_loader = test_loader
        self.logger.info(f"IoT 클러스터 '{self.name}' 생성 완료.")

    async def run(self, clock: 'SimulationClock', satellites: Dict[int, 'Satellite']):
        self.logger.info(f"IoT 클러스터 '{self.name}' 운영 시작.")
        while True:
            current_ts = clock.get_time_ts()
            for sat_id, sat in satellites.items():
                elevation = (sat.satellite_obj - self.topos).at(current_ts).altaz()[0].degrees
                tasks = []
                if elevation >= IOT_FLYOVER_THRESHOLD_DEG and sat.state == 'IDLE':
                    self.logger.info(f"📡 [IoT 통신] IoT {self.name} <-> SAT {sat_id} 통신 시작 (고도각: {elevation:.2f}°)")
                    if sat.model_ready_to_upload:
                        # Local Model 수신 - I/O 작업이므로 코틀린
                        receive_model_task = asyncio.create_task(sat.send_model_to_iot(self))
                        tasks.append(receive_model_task)
                    # Local Update 진행 - CPU 작업이므로 프로세스 풀로 오프로딩
                    elif not sat.model_ready_to_upload:
                        local_update_task = asyncio.create_task(sat.train_and_eval())
                        tasks.append(local_update_task)
                    await asyncio.gather(*tasks)
            await asyncio.sleep(clock.real_interval)
    
    async def receive_global_model(self, model: PyTorchModel):
        """위성으로부터 글로벌 모델을 수신"""
        if model.version > self.global_model.version:
            self.logger.info(f"  📡  IoT {self.name}: 새로운 글로벌 모델 수신 (v{model.version}).")
            self.global_model = model

class GroundStation:
    def __init__ (self, name, latitude, longitude, elevation, sim_logger, initial_model: PyTorchModel, test_loader, perf_logger,
                   threshold_deg: float = 10.0, staleness_threshold: int = AGGREGATION_STALENESS_THRESHOLD):
        self.name = name
        self.logger = sim_logger
        self.topos = Topos(latitude_degrees=latitude, longitude_degrees=longitude, elevation_m=elevation)
        self.threshold_deg = threshold_deg
        self._comm_status: Dict[int, bool] = {}
        self.staleness_threshold = staleness_threshold
        self.global_model = initial_model
        self.test_loader = test_loader
        self.perf_logger = perf_logger
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"지상국 '{self.name}' 생성 완료. 글로벌 모델 버전: {self.global_model.version}")
        self.logger.info(f"  - Aggregation 정책: 버전 허용치 {self.staleness_threshold}")

    async def run(self, clock: 'SimulationClock', master_satellites: list['MasterSatellite']):
        self.logger.info(f"지상국 '{self.name}' 운영 시작.")
        while True:
            current_ts = clock.get_time_ts()
            for master in master_satellites:
                master_sat_id= master.sat_id
                elevation = (master.satellite_obj - self.topos).at(current_ts).altaz()[0].degrees
                prev_visible = self._comm_status.get(master_sat_id, False)
                visible_now = elevation >= self.threshold_deg

                tasks = []
                # 통신 가능 시점
                if visible_now:
                    # AOS
                    if not prev_visible:
                        self.logger.info(f"📡 [AOS] {self.name} <-> Master SAT {master_sat_id} 통신 시작 (고도각: {elevation:.2f}°)")
                        master.state = 'COMMUNICATING_GS'
                    # Local Model 수신
                    if master.model_ready_to_upload:
                        receive_model_task = asyncio.create_task(self.receive_model_from_satellite(master))
                        tasks.append(receive_model_task)
                    # Global Model 전송
                    if self.global_model.version > master.cluster_model.version:
                        send_model_task = asyncio.create_task(self.send_model_to_satellite(master))
                        tasks.append(send_model_task)
                # LOS
                elif prev_visible and not visible_now:
                    self.logger.info(f"📡 [LOS] {self.name} <-> Master SAT {master_sat_id} 통신 종료 (고도각: {elevation:.2f}°)")
                    master.state = 'IDLE'
                self._comm_status[master_sat_id] = visible_now
                await asyncio.gather(*tasks)
            await asyncio.sleep(clock.real_interval)

    async def send_model_to_satellite(self, master_satellite: 'MasterSatellite'):
        self.logger.info(f"  📤 {self.name} -> Master SAT {master_satellite.sat_id}: 글로벌 모델 전송 (버전 {self.global_model.version})")
        await master_satellite.receive_global_model(self.global_model)

    async def receive_model_from_satellite(self, master_satellite: 'MasterSatellite'):
        cluster_model = await master_satellite.send_cluster_model()
        if cluster_model and self.global_model.version - cluster_model.version <= self.staleness_threshold:
            self.logger.info(f"  📥 {self.name} <- Master SAT {master_satellite.sat_id}: 클러스터 모델 수신 완료 (버전 {cluster_model.version}, 학습자: {cluster_model.trained_by})")
            # Local Model 수신 후 Aggregation 진행 - I/O 작업이므로 코틀린
            await self.try_aggregate_and_update(master_satellite.sat_id, cluster_model)

    async def try_aggregate_and_update(self, sat_id, cluster_model: PyTorchModel):
        """Aggregation 수행"""
        self.logger.info(f"✨ [{self.name} Aggregation] 진행 - Master SAT {sat_id}의 v{cluster_model.version} 클러스터 모델과 기존 글로벌 모델(v{self.global_model.version}) 취합 시작...")
        
        state_dicts_to_avg = [self.global_model.model_state_dict] + [cluster_model.model_state_dict]
        new_state_dict = fed_avg(state_dicts_to_avg)
        
        new_version = self.global_model.version + 1 # 버전업
        all_contributors = list(set(self.global_model.trained_by + [p for p in cluster_model.trained_by]))
        self.global_model = PyTorchModel(version=new_version, model_state_dict=new_state_dict, trained_by=all_contributors)
        self.logger.info(f"✨ [{self.name} Aggregation] 새로운 글로벌 모델 생성 완료! (버전 {self.global_model.version})")

        # evaluate
        loop = asyncio.get_running_loop()
        accuracy, loss = await loop.run_in_executor(None, evaluate_model, self.global_model.model_state_dict, self.test_loader, self.device)

        self.logger.info(f"  🧪 [Global Test] Owner: {self.name}, Version: {self.global_model.version}, Accuracy: {accuracy:.2f}%, Loss: {loss:.4f}")
        self.perf_logger.info(f"{datetime.now(KST).isoformat()},GLOBAL_TEST,{self.name},{self.global_model.version},N/A,{accuracy:.4f},{loss:.6f}")