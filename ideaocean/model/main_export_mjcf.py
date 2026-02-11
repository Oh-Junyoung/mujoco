# main_export_mjcf.py
"""
sim_results.pkl 에서 특정 (topology_index, sample_index) 를 꺼내
MuJoCo MJCF (.xml) 파일로 내보내는 스크립트.

사용법:
    1. TOPOLOGY_INDEX, SAMPLE_INDEX 를 원하는 값으로 수정
    2. python main_export_mjcf.py
"""
from __future__ import annotations

import os
import pickle

import numpy as np

from metheus_dynamics.topology.TopologyCalculator import TopologyDataLoader, TopologyCalculator
from metheus_dynamics.generators.PhysicalParametersGenerator import PhysicalParametersGenerator
from metheus_dynamics.generators.InitialDesignVariableGenerator import InitialDesignVariableGenerator

from MJCFConverter import MJCFConverter


# ======================================================================================
# USER SETTINGS
# ======================================================================================

PKL_PATH = "./sim_results.pkl"

# 변환할 샘플 선택
TOPOLOGY_INDEX = 10
SAMPLE_INDEX = 0

# 출력 경로 (스크립트와 같은 디렉토리)
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# PhysicalParametersGenerator 재생성용 설정
# (main_generate_sim_results_parallel.py 와 동일하게 맞추세요)
SEED = 420
COORD_RANGE_MM = (-50.0, 50.0)
LINEAR_DENSITY = 2.5
AREA_DENSITY = 10.0
LINK_THICKNESS = 0.05
COORD_UNIT = "mm"

# MJCF 시각화 설정
LINK_RADIUS = 0.001           # 링크 반지름 (m) → 직경 2 mm
MOTOR_KP = 10.0               # 모터 position servo 강성
FLOOR_HALF_EXTENT = 0.1       # 바닥 half-extent (m) → 200 mm × 200 mm
SHADOW_SIZE = 16384            # 그림자 해상도


# ======================================================================================
def main():
    # ── 1. Topology 로드
    loader = TopologyDataLoader()
    data = loader.load()
    if data is None:
        return

    # ── 2. sim_results.pkl 로드
    if not os.path.exists(PKL_PATH):
        print(f"✗ 파일을 찾을 수 없습니다: {PKL_PATH}")
        return

    with open(PKL_PATH, "rb") as f:
        db = pickle.load(f)

    if TOPOLOGY_INDEX not in db:
        print(f"✗ Topology {TOPOLOGY_INDEX} 이(가) pkl 에 존재하지 않습니다.")
        print(f"  사용 가능한 topology 인덱스: {sorted(db.keys())[:20]} ...")
        return

    samples = db[TOPOLOGY_INDEX]
    if SAMPLE_INDEX < 0 or SAMPLE_INDEX >= len(samples):
        print(f"✗ Sample {SAMPLE_INDEX} 범위 초과 (0..{len(samples)-1})")
        return

    sample = samples[SAMPLE_INDEX]

    # ── 3. full trajectory 관절 좌표
    q_all = np.asarray(sample["joint_positions"], dtype=float)  # (J_after, 2, T)
    if q_all.ndim != 3:
        print(f"✗ joint_positions shape 이상: {q_all.shape}")
        return

    # mm → m 변환 (full trajectory)
    if COORD_UNIT == "mm":
        q_all_m = q_all * 0.001
    else:
        q_all_m = q_all.copy()

    # ── 4. 물리 파라미터 재계산
    #   sim_results 에 attempt_idx 가 저장되지 않으므로,
    #   저장된 좌표(q0)를 직접 넘겨서 기하 기반으로 재계산한다.
    topo_calc = TopologyCalculator(data[TOPOLOGY_INDEX])
    J_orig = topo_calc.number_of_joints()
    q_orig_m = q_all_m[:J_orig, :, 0]  # original joints t=0 추출

    # Generator 인스턴스 (calculate_parameters_from_coords 호출용)
    dummy_gen = InitialDesignVariableGenerator(
        topology_data=data,
        num_samples_per_topology=1,
        coord_range=COORD_RANGE_MM,
        seed=SEED,
    )
    phys_gen = PhysicalParametersGenerator(
        topology_data=data,
        initial_design_generator=dummy_gen,
        linear_density=LINEAR_DENSITY,
        area_density=AREA_DENSITY,
        link_thickness=LINK_THICKNESS,
        coord_unit=COORD_UNIT,
    )

    params = phys_gen.calculate_parameters_from_coords(
        topology_idx=TOPOLOGY_INDEX,
        sample_idx=0,
        topology=data[TOPOLOGY_INDEX],
        calc_topo=topo_calc,
        joint_coords_m=q_orig_m,
    )

    # ── 5. MJCF 변환 (full trajectory 전달 → slider angle 자동 추출)
    converter = MJCFConverter(
        topology_info=topo_calc,
        topology_data=data[TOPOLOGY_INDEX],
        physical_params=params,
        joint_positions_m=q_all_m,
        topology_index=TOPOLOGY_INDEX,
        sample_index=SAMPLE_INDEX,
        link_radius=LINK_RADIUS,
        kp=MOTOR_KP,
        floor_half_extent=FLOOR_HALF_EXTENT,
        shadow_size=SHADOW_SIZE,
    )

    xml_str = converter.convert()

    # ── 6. 파일 저장
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"topo{TOPOLOGY_INDEX}_sample{SAMPLE_INDEX}.xml")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(xml_str)

    print(f"✓ MJCF 내보내기 완료: {out_path}")


if __name__ == "__main__":
    main()
