import sys
import os
import subprocess
import csv
import math

# Conda 환경 설정
CONDA_ENV_NAME = "mujoco"

def get_conda_python_path(env_name):
    try:
        result = subprocess.run(["conda", "env", "list"], capture_output=True, text=True)
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] == env_name:
                return os.path.join(parts[-1], "bin", "python")
    except Exception:
        pass
    return None

def is_running_in_target_env(target_python_path):
    if not target_python_path: return False
    return os.path.realpath(sys.executable) == os.path.realpath(target_python_path)

if __name__ == "__main__":
    target_python = get_conda_python_path(CONDA_ENV_NAME)
    
    # Matplotlib 확인 및 재실행
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        if target_python and not is_running_in_target_env(target_python):
            print(f"알림: 'matplotlib'이 없어 '{CONDA_ENV_NAME}' 환경으로 전환합니다...")
            try:
                subprocess.run([target_python] + sys.argv, check=True)
                exit(0)
            except Exception:
                exit(1)
        else:
            print(f"오류: matplotlib 설치 필요 (conda install -n {CONDA_ENV_NAME} matplotlib)")
            exit(1)

    # 데이터 로드
    csv_filename = "sensor_log.csv"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, csv_filename)

    if not os.path.exists(csv_path):
        print(f"오류: 파일 없음 - {csv_filename}")
        exit(1)

    print(f"데이터 로드 중: {csv_path}")

    times = []
    sensor_data = {}  # { "SensorName": [values...] }
    
    try:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            
            try:
                headers = next(reader)
            except StopIteration:
                print("오류: 빈 파일입니다.")
                exit(1)
                
            if not headers:
                print("오류: 헤더가 없습니다.")
                exit(1)

            for h in headers:
                if h != "Time":
                    sensor_data[h] = []
            
            for row_idx, row in enumerate(reader):
                if not row: continue
                
                # 행 길이 체크 (헤더와 다르면 건너뜀)
                if len(row) != len(headers):
                    continue

                try:
                    t = float(row[0])
                    times.append(t)
                    for i, val in enumerate(row[1:]):
                        sensor_data[headers[i+1]].append(float(val))
                except ValueError:
                    continue

    except Exception as e:
        print(f"읽기 실패: {e}")
        exit(1)

    # 그룹 정의 (접두사 기준)
    groups = {
        "J1": [],
        "J2": [],
        "J3": [],
        "J4": []
    }
    
    # 기타 센서를 위한 그룹
    groups["Other"] = []

    # 데이터 필터링 (초기 과도 응답 제거 0~0.5초)
    CUTOFF_TIME = 0.5
    valid_indices = [i for i, t in enumerate(times) if t >= CUTOFF_TIME]

    if not valid_indices:
        print(f"오류: {CUTOFF_TIME}초 이후의 데이터가 없습니다.")
        exit(1)

    # 필터링 적용
    start_idx = valid_indices[0]
    times = times[start_idx:]
    sensor_data = {k: v[start_idx:] for k, v in sensor_data.items()}
    
    print(f"알림: 초기 {CUTOFF_TIME}초 데이터를 제외하고 플롯합니다.")

    # 센서들을 그룹에 배정
    
    # ---------------------------------------------------------
    # [추가] 반력(Reaction Force)의 Magnitude 계산
    # "_Reaction (N)_x" 형태의 센서를 찾아 크기(Magnitude)로 변환
    # ---------------------------------------------------------
    keys_to_remove = []
    keys_to_add = {}

    # 현재 센서 데이터 키 목록 복사
    current_keys = list(sensor_data.keys())
    
    for key in current_keys:
        if "Reaction (N)_x" in key:
            base_name = key.replace("_x", "")  # 예: J3_Reaction (N)
            
            # x, y, z 성분 찾기
            val_x = sensor_data.get(base_name + "_x")
            val_y = sensor_data.get(base_name + "_y")
            val_z = sensor_data.get(base_name + "_z")

            if val_x and val_y:
                # 데이터 길이 확인
                length = len(val_x)
                magnitudes = []
                
                for k in range(length):
                    vx = val_x[k]
                    vy = val_y[k]
                    vz = val_z[k] if val_z else 0.0
                    mag = math.sqrt(vx**2 + vy**2 + vz**2)
                    magnitudes.append(mag)
                
                # 새로운 Magnitude 데이터 추가
                keys_to_add[base_name] = magnitudes
                
                # 기존 성분 데이터 삭제 목록에 추가
                keys_to_remove.append(base_name + "_x")
                keys_to_remove.append(base_name + "_y")
                if val_z: keys_to_remove.append(base_name + "_z")

    # 기존 성분 제거
    for k in keys_to_remove:
        if k in sensor_data:
            del sensor_data[k]
            
    # Magnitude 추가
    sensor_data.update(keys_to_add)
    # ---------------------------------------------------------

    for name in sensor_data.keys():
        # Z축 데이터 필터링 (사용자 요청) - Magnitude 계산 후 남은 다른 Z축 데이터가 있다면 필터링
        if name.endswith("_z") or name.endswith("_Z"):
            continue

        matched = False
        for prefix in ["J1", "J2", "J3", "J4"]:
            if name.startswith(prefix):
                groups[prefix].append(name)
                matched = True
                break
        if not matched:
            groups["Other"].append(name)

    # 그래프 그리기
    for group_name, sensors in groups.items():
        if not sensors:
            continue
            
        num_sensors = len(sensors)
        # 2열 배치 (홀수면 마지막 줄은 1개)
        cols = 2
        rows = (num_sensors + 1) // cols
        
        # subplot 생성 (sharex=True로 시간 축 공유)
        fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows), sharex=True)
        
        # 1차원 배열로 평탄화 (인덱싱 편의를 위해)
        if hasattr(axes, 'flatten'):
            axes = axes.flatten()
        else:
            axes = [axes]
            
        fig.canvas.manager.set_window_title(f"Sensor Group: {group_name}")
        fig.suptitle(f"Sensor Data - {group_name} (t >= {CUTOFF_TIME}s)", fontsize=16)

        for i, ax in enumerate(axes):
            if i < num_sensors:
                # 데이터 플롯
                name = sensors[i]
                values = sensor_data[name]
                ax.plot(times, values, label=name)
                
                # 제목 및 레이블 설정
                ax.set_title(name, fontsize=10, fontweight='bold')
                ax.grid(True)
                
                # X축 시작을 0부터 고정 (데이터는 0.5부터 시작하므로 앞부분이 비어보임)
                ax.set_xlim(left=0)
                
                # 마지막 행(또는 빈칸 바로 위)에 X축 레이블 추가
                if i >= num_sensors - cols:
                    ax.set_xlabel("Time (s)", fontsize=10)
            else:
                # 빈 서브플롯 제거
                fig.delaxes(ax)
                # 바로 윗 차트의 X축 레이블 확인
                if i - cols >= 0:
                    axes[i - cols].xaxis.set_tick_params(labelbottom=True)
                    axes[i - cols].set_xlabel("Time (s)", fontsize=10)

        plt.tight_layout()

    print("그래프를 출력합니다 (총 {}개의 창)...".format(sum(1 for g in groups.values() if g)))
    plt.show()
