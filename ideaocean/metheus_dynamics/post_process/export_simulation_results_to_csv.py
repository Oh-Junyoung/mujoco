import csv
import os

def export_simulation_results_to_csv(
    results: dict,
    topology_index: int,
    sample_index: int,
    out_dir: str = "./csv_export",
    filename_prefix: str = "sim",
    include_after_pair: bool = False,
) -> str:
    """
    DynamicsSimulator.run() 결과(dict)를 PMKS+ 스타일(시간 + joint별 컬럼) CSV로 저장합니다.

    저장 컬럼(기본):
      - time_sec, step
      - j{j}_x_mm, j{j}_y_mm
      - v{j}_x_mmps, v{j}_y_mmps, v{j}_mag_mmps
      - a{j}_x_mmps2, a{j}_y_mmps2, a{j}_mag_mmps2
      - F{j}_x_N, F{j}_y_N, F{j}_mag_N
      - motor_torque_Nm

    Args:
        results: simulator.run() 결과 dict
        topology_index: topo index (파일명에 포함)
        sample_index: sample index (파일명에 포함)
        out_dir: 저장 폴더
        filename_prefix: 파일명 prefix
        include_after_pair: True면 after_pair 좌표/속도/가속도도 함께 export

    Returns:
        저장된 CSV 파일 경로(str)
    """
    import os
    import csv
    import numpy as np

    # ---- required fields
    time = np.asarray(results["time"], dtype=float).reshape(-1)  # (T,)
    joints = np.asarray(results["joint_coordinates_mm"], dtype=float)  # (J,2,T) mm
    vel = np.asarray(results["joint_velocities_mm"], dtype=float)  # (J,2,T) mm/s
    acc = np.asarray(results["joint_accelerations_mm"], dtype=float)  # (J,2,T) mm/s^2
    react = np.asarray(results["joint_reaction_forces"], dtype=float)  # (J,2,T) N

    # (MOD) motor torque key supports both "motor_torque" and "input_link_torque"
    # - prefer "motor_torque" if present
    if "motor_torque" in results:
        torque = np.asarray(results["motor_torque"], dtype=float).reshape(-1)  # (T,) N·m
        torque_key_used = "motor_torque"
    elif "input_link_torque" in results:
        # motor_torque = -input_link_torque (as defined in DynamicsSimulator)
        in_torque = np.asarray(results["input_link_torque"], dtype=float).reshape(-1)
        torque = -in_torque
        torque_key_used = "input_link_torque (negated)"
    else:
        raise KeyError("results must contain 'motor_torque' or 'input_link_torque'.")

    J = int(joints.shape[0])
    T = int(time.shape[0])

    # ---- shape checks
    if joints.shape != (J, 2, T):
        raise ValueError(f"joint_coordinates_mm expected (J,2,T) but got {joints.shape}")
    if vel.shape != (J, 2, T):
        raise ValueError(f"joint_velocities_mm expected (J,2,T) but got {vel.shape}")
    if acc.shape != (J, 2, T):
        raise ValueError(f"joint_accelerations_mm expected (J,2,T) but got {acc.shape}")
    if react.shape != (J, 2, T):
        raise ValueError(f"joint_reaction_forces expected (J,2,T) but got {react.shape}")
    if torque.shape[0] != T:
        raise ValueError(f"motor_torque expected length {T} but got {torque.shape}")

    # ---- optional after_pair export
    if include_after_pair:
        joints_ap = np.asarray(results["joint_coordinates_after_pair_mm"], dtype=float)  # (J_after,2,T)
        vel_ap = np.asarray(results["joint_velocities_after_pair_mm"], dtype=float)
        acc_ap = np.asarray(results["joint_accelerations_after_pair_mm"], dtype=float)

        J_after = int(joints_ap.shape[0])
        if joints_ap.shape != (J_after, 2, T):
            raise ValueError(f"joint_coordinates_after_pair_mm expected (J_after,2,T) but got {joints_ap.shape}")
        if vel_ap.shape != (J_after, 2, T):
            raise ValueError(f"joint_velocities_after_pair_mm expected (J_after,2,T) but got {vel_ap.shape}")
        if acc_ap.shape != (J_after, 2, T):
            raise ValueError(f"joint_accelerations_after_pair_mm expected (J_after,2,T) but got {acc_ap.shape}")

    # ---- output path
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{filename_prefix}_topo{topology_index}_sample{sample_index}.csv")

    # ---- header build
    header = ["time_sec", "step"]

    # original: position
    for j in range(J):
        header += [f"j{j}_x_mm", f"j{j}_y_mm"]

    # original: velocity (+mag)
    for j in range(J):
        header += [f"v{j}_x_mmps", f"v{j}_y_mmps", f"v{j}_mag_mmps"]

    # original: acceleration (+mag)
    for j in range(J):
        header += [f"a{j}_x_mmps2", f"a{j}_y_mmps2", f"a{j}_mag_mmps2"]

    # reactions (+mag)
    for j in range(J):
        header += [f"F{j}_x_N", f"F{j}_y_N", f"F{j}_mag_N"]

    # motor torque
    header += ["motor_torque_Nm"]

    # after_pair (optional)
    if include_after_pair:
        header += ["---after_pair---"]
        for j in range(J_after):
            header += [f"ap_j{j}_x_mm", f"ap_j{j}_y_mm"]
        for j in range(J_after):
            header += [f"ap_v{j}_x_mmps", f"ap_v{j}_y_mmps", f"ap_v{j}_mag_mmps"]
        for j in range(J_after):
            header += [f"ap_a{j}_x_mmps2", f"ap_a{j}_y_mmps2", f"ap_a{j}_mag_mmps2"]

    # ---- write CSV
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for k in range(T):
            row = [float(time[k]), int(k)]

            # positions
            for j in range(J):
                row += [float(joints[j, 0, k]), float(joints[j, 1, k])]

            # velocities + mag
            for j in range(J):
                vx = float(vel[j, 0, k])
                vy = float(vel[j, 1, k])
                row += [vx, vy, float((vx * vx + vy * vy) ** 0.5)]

            # accelerations + mag
            for j in range(J):
                ax = float(acc[j, 0, k])
                ay = float(acc[j, 1, k])
                row += [ax, ay, float((ax * ax + ay * ay) ** 0.5)]

            # reactions + mag
            for j in range(J):
                fx = float(react[j, 0, k])
                fy = float(react[j, 1, k])
                row += [fx, fy, float((fx * fx + fy * fy) ** 0.5)]

            # motor torque
            row += [float(torque[k]) if np.isfinite(torque[k]) else ""]

            # after_pair (optional)
            if include_after_pair:
                row += [""]  # separator column placeholder for "---after_pair---"
                for j in range(J_after):
                    row += [float(joints_ap[j, 0, k]), float(joints_ap[j, 1, k])]
                for j in range(J_after):
                    vx = float(vel_ap[j, 0, k])
                    vy = float(vel_ap[j, 1, k])
                    row += [vx, vy, float((vx * vx + vy * vy) ** 0.5)]
                for j in range(J_after):
                    ax = float(acc_ap[j, 0, k])
                    ay = float(acc_ap[j, 1, k])
                    row += [ax, ay, float((ax * ax + ay * ay) ** 0.5)]

            writer.writerow(row)

    print(f"[CSV EXPORT] saved: {out_path} (torque source: {torque_key_used})")
    return out_path
