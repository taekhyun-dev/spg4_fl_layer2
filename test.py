import asyncio
import time

# A 객체: 공유 자원
class SharedObject:
    def __init__(self):
        self.alpha = 0
        print(f"[{time.strftime('%T')}] A: 객체 생성 (alpha = 0)")

# B 객체: alpha 값을 주기적으로 변경 (무한 루프)
async def task_b_loop(a: SharedObject, lock: asyncio.Lock):
    print(f"[{time.strftime('%T')}] B: 무한 루프 시작 (1초마다 alpha 변경)")
    i = 0
    while True:
        # --- B의 임계 영역 시작 ---
        async with lock:
            print(f"[{time.strftime('%T')}] B: (Lock 획득) alpha 변경 중...")
            
            # 값을 변경하는 데 시간이 걸린다고 가정 (예: 복잡한 연산)
            current_val = a.alpha
            await asyncio.sleep(0.1) # 0.1초 소요
            a.alpha = current_val + 1
            
            print(f"[{time.strftime('%T')}] B: (Lock 반납) alpha -> {a.alpha}로 변경 완료")
        # --- B의 임계 영역 끝 ---
        
        # 다음 작업을 위해 1초 대기 (Lock을 잡지 않은 상태로 대기)
        await asyncio.sleep(1) 
        i += 1

# C 객체: alpha 값을 선택적/반복적으로 조회
async def task_c_reader(a: SharedObject, lock: asyncio.Lock, read_id: int):
    print(f"[{time.strftime('%T')}]   C-{read_id}: (Lock 획득 시도)")
    
    # --- C의 임계 영역 시작 ---
    async with lock:
        # B가 Lock을 잡고 있다면, 여기서 B가 Lock을 놓을 때까지 대기
        print(f"[{time.strftime('%T')}]   C-{read_id}: (Lock 획득) --- 현재 alpha: {a.alpha} ---")
        # C는 읽기만 하므로 Lock을 짧게 잡고 바로 반납
    # --- C의 임계 영역 끝 ---
    
    print(f"[{time.strftime('%T')}]   C-{read_id}: (Lock 반납) 조회 완료")

async def main():
    a_obj = SharedObject()
    lock = asyncio.Lock()

    # task_b를 백그라운드 태스크로 즉시 실행 (무한 루프 시작)
    # b_task 변수는 나중에 태스크를 종료(cancel)하기 위해 저장
    b_task = asyncio.create_task(task_b_loop(a_obj, lock))

    print(f"[{time.strftime('%T')}] Main: C 작업을 0.3초마다 10회 반복 실행")
    
    for i in range(10):
        # 0.3초마다 C 작업을 요청
        await asyncio.sleep(0.3) 
        await task_c_reader(a_obj, lock, i)

    # 10회 반복 후 B 태스크를 종료
    print(f"[{time.strftime('%T')}] Main: C 작업 10회 완료. B 태스크를 취소합니다.")
    b_task.cancel()
    
    try:
        await b_task # B 태스크가 완전히 취소될 때까지 대기
    except asyncio.CancelledError:
        print(f"[{time.strftime('%T')}] Main: B 태스크 정상적으로 취소됨")

    print(f"[{time.strftime('%T')}] Main: 최종 alpha 값: {a_obj.alpha}")

# Python 3.7+
asyncio.run(main())