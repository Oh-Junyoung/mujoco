import sys
import os
import subprocess

# Conda 환경 설정
CONDA_ENV_NAME = "mujoco"

def get_conda_python_path(env_name):
    """지정된 Conda 환경의 Python 실행 파일 경로를 찾습니다."""
    try:
        # conda env list 실행 (가장 확실함)
        result = subprocess.run(["conda", "env", "list"], capture_output=True, text=True)
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] == env_name:
                env_path = parts[-1]
                return os.path.join(env_path, "bin", "python")
    except Exception:
        pass
    
    # 일반적인 경로 추정
    home = os.path.expanduser("~")
    possible_paths = [
        f"{home}/miniconda3/envs/{env_name}/bin/python",
        f"{home}/anaconda3/envs/{env_name}/bin/python",
        f"/opt/conda/envs/{env_name}/bin/python"
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

def is_running_in_target_env(target_python_path):
    """현재 실행 중인 Python이 타겟 환경인지 확인합니다."""
    if not target_python_path:
        return False
    
    current_python = os.path.realpath(sys.executable)
    target_python = os.path.realpath(target_python_path)
    
    return current_python == target_python

if __name__ == "__main__":
    # 1. 타겟 Python 경로 확인
    target_python = get_conda_python_path(CONDA_ENV_NAME)
    
    # 2. 현재 환경 확인 및 재실행 로직
    # 만약 mujoco를 import 할 수 있다면 굳이 재실행할 필요 없음 (이미 환경이 맞거나 라이브러리가 있음)
    try:
        import mujoco
        import mujoco.viewer
        # import 성공 시 통과
    except ImportError:
        # 라이브러리가 없으면 환경 불일치로 간주하여 재실행 시도
        
        # 이미 타겟 파이썬으로 실행 중인데도 import가 안된다면? -> 진짜 설치가 안된 것
        if is_running_in_target_env(target_python):
            print(f"오류: '{CONDA_ENV_NAME}' 환경({sys.executable})에서 실행 중이지만 'mujoco' 라이브러리를 찾을 수 없습니다.")
            print("해당 환경에 mujoco가 설치되어 있는지 확인해주세요.")
            exit(1)

        print(f"알림: 현재 Python에는 'mujoco'가 없습니다. '{CONDA_ENV_NAME}' 환경으로 전환을 시도합니다...")
        
        if target_python and os.path.exists(target_python):
            # 재실행
            cmd = [target_python, sys.argv[0]] + sys.argv[1:]
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                exit(e.returncode)
            except KeyboardInterrupt:
                exit(0)
            exit(0)
        else:
            print(f"오류: '{CONDA_ENV_NAME}' 환경의 Python을 찾을 수 없습니다.")
            print(f"conda env list를 확인해주세요.")
            exit(1)

    # 3. 모델 로드 및 실행
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "model", "4-bar", "four_bar.xml")

    print(f"\n모델을 불러오는 중입니다: {model_path}")

    if not os.path.exists(model_path):
        print(f"오류: 모델 파일을 찾을 수 없습니다: {model_path}")
        exit(1)

    try:
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"오류: 모델 로드 중 문제가 발생했습니다.\n{e}")
        exit(1)

    import time
    import csv
    import numpy as np

    print("모델 로드 성공. 시뮬레이션 및 로깅을 시작합니다...")
    print("뷰어 창이 열리면 시뮬레이션이 진행되며 데이터가 기록됩니다.")
    print("뷰어를 닫으면 로깅이 종료되고 파일로 저장됩니다.")

    # 로그 저장용 리스트
    log_data = []

    # 센서 이름 및 차원 가져오기 (Vector 센서 대응)
    sensor_names = []
    for i in range(model.nsensor):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
        if not name:
            name = f"sensor_{i}"
        
        dim = model.sensor_dim[i]
        if dim == 1:
            sensor_names.append(name)
        else:
            axis = ['x', 'y', 'z', 'w'] # 최대 4차원 가정 (일반적으로 3)
            for j in range(dim):
                suffix = axis[j] if j < 4 else str(j)
                sensor_names.append(f"{name}_{suffix}")

    # 목표 속도 설정 (20 RPM = 2.0944 rad/s)
    # 목표 속도 설정 (20 RPM = 2.0944 rad/s)
    TARGET_VELOCITY = 2.0944
    TWO_REV_TIME = 2 * (2 * np.pi / TARGET_VELOCITY)  # 약 6.0초 (2회전)

    print(f"목표 속도: {TARGET_VELOCITY} rad/s (20 RPM)")
    print(f"예상 종료 시간: {TWO_REV_TIME:.2f} 초 (2회전)")

    # 패시브 뷰어 실행
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            start_time = time.time()
            data.ctrl[0] = TARGET_VELOCITY  # 모터 속도 지령
            
            while viewer.is_running():
                step_start = time.time()

                # 물리 시뮬레이션 스텝
                mujoco.mj_step(model, data)
                
                # 뷰어 동기화
                viewer.sync()

                # 데이터 로깅
                current_row = [data.time] + data.sensordata.tolist()
                log_data.append(current_row)

                # 2회전 완료 시 종료 조건
                if data.time >= TWO_REV_TIME:
                    print("\n2회전 완료. 시뮬레이션을 종료합니다.")
                    break

                # 시뮬레이션 속도 조절
                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
            
            # 뷰어가 닫히지 않고 루프가 끝났으면 강제로 닫기 시도 (블럭 안이라 자동 처리되지만 명시적 안내)

    except KeyboardInterrupt:
        print("\n사용자에 의해 중단됨.")
    finally:
        # 루프 종료 후 CSV 저장 (Ctrl+C로 종료되어도 실행됨)
        csv_filename = "sensor_log.csv"
        csv_path = os.path.join(current_dir, csv_filename)
        
        print(f"\n시뮬레이션 종료. 데이터를 저장합니다: {csv_path}")
        
        try:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Time'] + sensor_names)
                writer.writerows(log_data)
            print("저장 완료!")
        except Exception as e:
            print(f"저장 실패: {e}")
