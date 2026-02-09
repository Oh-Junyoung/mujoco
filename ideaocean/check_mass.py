import sys
import os
import subprocess

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

if __name__ == "__main__":
    try:
        import mujoco
    except ImportError:
        target_python = get_conda_python_path(CONDA_ENV_NAME)
        if target_python and os.path.exists(target_python) and sys.executable != target_python:
            print(f"Switching to {CONDA_ENV_NAME} environment...")
            subprocess.run([target_python] + sys.argv)
            exit(0)
        else:
            print("Failed to find mujoco environment or library.")
            exit(1)

import numpy as np

# 모델 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "model", "4-bar", "four_bar.xml")

try:
    # 모델 로드
    model = mujoco.MjModel.from_xml_path(model_path)
except Exception as e:
    print(f"모델 로드 실패: {e}")
    exit(1)

print(f"Model: {model_path}")
print("-" * 50)
print(f"{'Body Name':<15} | {'Mass (kg)':<10} | {'Inertia (Diagonal)'}")
print("-" * 50)

total_mass = 0

for i in range(model.nbody):
    # Body 이름 가져오기
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    if name is None: 
        name = "WorldBody" if i == 0 else f"Body_{i}"
    
    mass = model.body_mass[i]
    inertia = model.body_inertia[i]
    
    print(f"{name:<15} | {mass:<10.5f} | {inertia}")
    
    # WorldBody 제외하고 질량 합산
    if i > 0:
        total_mass += mass

print("-" * 50)
print(f"Total Mechanism Mass: {total_mass:.5f} kg")
print("-" * 50)

# 밀도 정보 확인 (Geom에서 확인 가능)
print("\n[Geom Info]")
print("Geom density uses default (1000 kg/m^3) if not specified.")
for i in range(model.ngeom):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
    # geom의 body id 찾기
    body_id = model.geom_bodyid[i]
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    if body_name is None: body_name = "World"
    
    # size 확인 (capsule이면 radius, half-length 등)
    size = model.geom_size[i]
    print(f"Geom {i} (Body: {body_name}): Size={size}")
