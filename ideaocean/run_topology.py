
import sys
import os
import subprocess
import time

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
    try:
        import mujoco
        import mujoco.viewer
    except ImportError:
        if is_running_in_target_env(target_python):
            print(f"오류: '{CONDA_ENV_NAME}' 환경({sys.executable})에서 실행 중이지만 'mujoco' 라이브러리를 찾을 수 없습니다.")
            print("해당 환경에 mujoco가 설치되어 있는지 확인해주세요.")
            exit(1)

        print(f"알림: 현재 Python에는 'mujoco'가 없습니다. '{CONDA_ENV_NAME}' 환경으로 전환을 시도합니다...")
        
        if target_python and os.path.exists(target_python):
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
    model_path = os.path.join(current_dir, "model", "topo10_sample0_new.xml")
    
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

    print("모델 로드 성공. 뷰어를 실행합니다...")

    # J1 joint의 range에서 왕복 운동 파라미터를 미리 계산 (시뮬 중 불변)
    j1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "J1")
    angle_min = model.jnt_range[j1_id, 0]
    angle_max = model.jnt_range[j1_id, 1]
    sweep = angle_max - angle_min  # rad
    speed = 2.0944  # 20 RPM = 120 deg/s ≈ 2.0944 rad/s
    half_period = sweep / speed
    period = 2.0 * half_period

    # create viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 초기 카메라: closeup 시점으로 freelook 시작
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = [0.0, 0.0, 0.0]
        viewer.cam.distance = 0.249
        viewer.cam.azimuth = 90.0
        viewer.cam.elevation = -45.0

        # Model element 시각화 활성화
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONSTRAINT] = True
        viewer.opt.label = mujoco.mjtLabel.mjLABEL_JOINT

        start = time.time()
        while viewer.is_running():
            step_start = time.time()

            cycle_time = data.time % period

            if cycle_time < half_period:
                data.ctrl[0] = angle_min + sweep * (cycle_time / half_period)
            else:
                data.ctrl[0] = angle_max - sweep * ((cycle_time - half_period) / half_period)

            mujoco.mj_step(model, data)
            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
