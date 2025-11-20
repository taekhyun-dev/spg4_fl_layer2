import asyncio
import torch.multiprocessing as mp
import torch
from pathlib import Path
from typing import List, Dict, Coroutine
from datetime import datetime, timezone, timedelta
from utils.config import NUM_CLIENTS, DIRICHLET_ALPHA, BATCH_SIZE, NUM_WORKERS, NUM_MASTERS, SATS_PER_PLANE
from utils.skyfield_utils import EarthSatellite
from object.satellite import Satellite, Satellite_Manager, MasterSatellite, WorkerSatellite
from object.environment import GroundStation, IoT
from object.clock import SimulationClock
from utils.logging_setup import setup_loggers
from ml.data import get_cifar10_loaders
from ml.model import PyTorchModel, create_mobilenet

def load_constellation(tle_path: str, sim_logger) -> Dict[int, EarthSatellite]:
    """TLE 파일에서 위성군 정보를 불러오는 함수"""
    if not Path(tle_path).exists(): raise FileNotFoundError(f"'{tle_path}' 파일을 찾을 수 없습니다.")
    satellites = {}
    with open(tle_path, "r") as f:
        lines = [line.strip() for line in f.readlines()]; i = 0
        while i < len(lines):
            name, line1, line2 = lines[i:i+3]; sat_id = int(name.replace("SAT", ""))
            satellites[sat_id] = EarthSatellite(line1, line2, name)
            i += 3
    sim_logger.info(f"총 {len(satellites)}개의 위성을 TLE 파일에서 불러왔습니다.")
    return satellites

def create_simulation_environment(
    clock: 'SimulationClock', 
    eval_infra: dict
):
    """
    시뮬레이션 환경을 구성하는 모든 객체(위성, 지상국, IoT)를 생성합니다.
    main.py에서 TLE 로딩 후, 이 함수를 호출하여 환경을 구성합니다.
    """
    sim_logger = eval_infra['sim_logger']
    perf_logger = eval_infra['perf_logger']
    client_loaders = eval_infra['client_loaders']
    val_loader = eval_infra['val_loader']
    test_loader = eval_infra['test_loader']

    # TLE 데이터 로드
    all_sats_skyfield = load_constellation("constellation.tle", sim_logger)
    
    # 초기 글로벌 모델 생성
    initial_pytorch_model = create_mobilenet()
    initial_global_model = PyTorchModel(version=0, model_state_dict=initial_pytorch_model.state_dict())
    
    # 지상국 및 IoT 클러스터 생성
    ground_stations = [
        GroundStation("Seoul-GS", 37.5665, 126.9780, 34, sim_logger, initial_model=initial_global_model, test_loader=test_loader,
                      perf_logger=perf_logger),
        # GroundStation("Houston-GS", 29.7604, -95.3698, 12, initial_model=initial_global_model, eval_infra=eval_infra)
    ]
    
    iot_clusters = [
        IoT("Amazon_Forest", -3.47, -62.37, 100, sim_logger, initial_model = initial_global_model, test_loader=test_loader),
        IoT("Great_Barrier_Reef", -18.29, 147.77, 0, sim_logger, initial_model = initial_global_model, test_loader=test_loader),
        IoT("Siberian_Tundra", 68.35, 18.79, 420, sim_logger, initial_model = initial_global_model, test_loader=test_loader)
    ]

    # 3. 위성 객체 및 클러스터 구성 (기존 main.py 로직)
    satellites_in_sim: Dict[int, Satellite] = {}
    sat_ids = sorted(list(all_sats_skyfield.keys()))
    
    if len(sat_ids) < NUM_MASTERS * (SATS_PER_PLANE // NUM_MASTERS if NUM_MASTERS > 0 else SATS_PER_PLANE):
       raise ValueError(f"시뮬레이션을 위해 충분한 수의 위성 TLE가 필요합니다.")
       
    master_ids = [sat_ids[i * SATS_PER_PLANE] for i in range(NUM_MASTERS)]
    worker_ids = [sid for sid in sat_ids if sid not in master_ids]
    
    sim_logger.info(f"마스터 위성으로 {master_ids}가 선정되었습니다.")

    masters = []
    for m_id in master_ids:
        sat = Satellite(
            m_id, all_sats_skyfield[m_id], clock, sim_logger, perf_logger,
            initial_global_model
        )
        master_sat = MasterSatellite(sat, test_loader)
        satellites_in_sim[m_id] = master_sat
        masters.append(master_sat)

    for i, w_id in enumerate(worker_ids):
        train_loader = client_loaders[w_id]
        assigned_master = masters[i % NUM_MASTERS]
        sat = Satellite(
            w_id, all_sats_skyfield[w_id], clock, sim_logger, perf_logger,
            initial_global_model
        )
        worker_sat = WorkerSatellite(sat, assigned_master, train_loader, val_loader)
        assigned_master.add_member(worker_sat)
        satellites_in_sim[w_id] = worker_sat

    # for sat_id in sat_ids:
    #     train_loader = client_loaders[sat_id]
    #     sat = Satellite(
    #         sat_id, all_sats_skyfield[sat_id], clock, sim_logger, perf_logger,
    #         initial_global_model, train_loader, val_loader
    #     )
    #     satellites_in_sim[sat_id] = sat
            
    sim_logger.info(f"총 {len(satellites_in_sim)}개 위성 생성 완료.")

    sat_managers = []
    for master in masters:
        sat_manager = Satellite_Manager(master, clock, sim_logger)
        sat_managers.append(sat_manager)

    # sat_manager = Satellite_Manager(satellites_in_sim, clock, sim_logger)
    
    return sat_managers, satellites_in_sim, ground_stations, iot_clusters

async def main():
    try:
        sim_logger, perf_logger = setup_loggers()
        sim_logger.info("Loading CIFAR10 dataset...")
        client_loaders, val_loader, test_loader = get_cifar10_loaders(num_clients=NUM_CLIENTS, dirichlet_alpha=DIRICHLET_ALPHA,
                                                                      batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
        sim_logger.info("Dataset loaded.")
        
        eval_infra = {
            "sim_logger": sim_logger,
            "perf_logger": perf_logger,
            "client_loaders": client_loaders,
            "val_loader": val_loader,
            "test_loader": test_loader,
        }

        start_time = datetime.now(timezone.utc)
        simulation_clock = SimulationClock(
            start_dt=start_time, 
            time_step=timedelta(minutes=10),
            real_interval=1.0,
            sim_logger=sim_logger
        )
        # 로드된 데이터를 전달하여 시뮬레이션 환경 구성
        sim_logger.info("시뮬레이션 환경을 구성 ...")
        sat_managers, satellites, ground_stations, iot_clusters = create_simulation_environment(
            simulation_clock, eval_infra
        )




        

        # 시뮬레이션 메인 태스크들
        sim_tasks: List[Coroutine] = [
            simulation_clock.run(),
            *[gs.run(simulation_clock, satellites) for gs in ground_stations],
            *[iot.run(simulation_clock, satellites) for iot in iot_clusters],
            *[sat_manager.run() for sat_manager in sat_managers]
        ]
        sim_logger.info("시뮬레이션을 시작합니다.")
        await asyncio.gather(*[asyncio.create_task(task) for task in sim_tasks])

    except KeyboardInterrupt:
        print("\n시뮬레이션을 종료합니다.")
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        # 예기치 않은 에러 발생 시 로깅
        sim_logger, _ = setup_loggers()
        sim_logger.error(f"\n시뮬레이션 중 치명적인 에러 발생: {e}", exc_info=True)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    asyncio.run(main())