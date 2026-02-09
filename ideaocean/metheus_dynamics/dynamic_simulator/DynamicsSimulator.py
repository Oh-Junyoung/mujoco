# DynamicsSimulator.py
import numpy as np
from typing import Dict, Optional, Tuple, List
from scipy.linalg import lstsq


class DynamicsSimulator:
    """
    ==========================================================================================================
    DynamicsSimulator
    ==========================================================================================================
    [목적]
    - 평면(2D) 강체 링크(Planar rigid bodies)로 구성된 1-자유도(1-DOF) 링크기구를,
      '구속식(holonomic constraints) + 라그랑주 승수(Lagrange multipliers)' 기반의
      DAE(Differential-Algebraic Equations) 형태로 이론적으로 일관되게 해석한다.
    - 구동 입력(각도 θ(t))은 "입력 링크의 각도(orientation)를 시간에 대해 지정하는 구동 구속식"으로 부여한다.
      이때 해당 구속식의 라그랑주 승수는 구동 토크(모터 토크)로 해석 가능하다.
    - 각 R joint(회전 조인트)의 구속식 라그랑주 승수는 해당 관절에서의 반력(2D force)로 해석 가능하다.
      P joint(프리즘틱)의 경우, 본 구현은 "마찰 없는 이상적 슬라이더" 모델로서 레일 법선 방향 반력만 갖는다.

    ----------------------------------------------------------------------------------------------------------
    [좌표계 / 일반화 좌표(Generalized coordinates)]
    - 링크 i(ground 제외)마다 3개의 좌표를 사용한다:
        q_i = [x_i, y_i, φ_i]
      여기서
        x_i, y_i : 링크 i의 질량중심(COM) 좌표 (m)
        φ_i      : 링크 i의 평면 내 자세(회전) 각도 (rad)
    - ground link는 완전히 고정된 강체로 취급하여 일반화 좌표에 포함하지 않는다.
    - 따라서 전체 일반화 좌표 벡터 q는 "움직이는 링크 개수 N_m"에 대해
        q ∈ R^(3*N_m)
      의 크기를 가진다.

    ----------------------------------------------------------------------------------------------------------
    [강체의 기구학(Rigid-body kinematics)]
    - 링크 i의 한 조인트 j의 공간 위치 p_{i,j}는,
        p_{i,j}(q) = r_i + R(φ_i) * s_{i,j}
      로 표현된다.
      여기서
        r_i = [x_i, y_i]^T : 링크 i COM 위치
        R(φ)              : 2D 회전행렬
        s_{i,j}           : 링크 i 좌표계(body-fixed frame)에서 조인트 j까지의 벡터(상수)

    - 2D 회전행렬:
        R(φ) = [[cosφ, -sinφ],
                [sinφ,  cosφ]]

    - 2D에서 각속도 ω가 z축 방향(평면 밖)을 향할 때, 벡터 v를 90도 회전시키는 연산자를
        E = [[0, -1],
             [1,  0]]
      로 두면,
        d/dt (R(φ) s) = ω * E * (R(φ) s)
      가 성립한다.
      따라서
        p_dot = r_dot + ω * E * (R s)
        p_ddot = r_ddot + α * E * (R s) - ω^2 * (R s)
      (α는 각가속도) 로 정확히 계산된다.

    ----------------------------------------------------------------------------------------------------------
    [동역학 방정식(Dynamics)]
    - 각 링크 i(ground 제외)는 평면 강체로서 뉴턴-오일러(Newton–Euler) 방정식을 따른다.
      일반화 좌표(q_i = [x_i, y_i, φ_i]) 기준으로 질량행렬(M)은 블록 대각 형태:
        M_i = diag(m_i, m_i, I_i)
      여기서
        m_i : 링크 질량(kg)
        I_i : 링크 COM 기준 평면 밖(z축) 관성모멘트(kg·m^2)

    - 전체 시스템(모든 moving links)의 질량행렬:
        M = blockdiag(M_i) ∈ R^(nq x nq)

    - 외력(일반화 힘) Q는 여기서는 중력만 적용:
        Q_i = [0, -m_i*g, 0]^T
      (COM에 작용하는 중력은 토크 성분이 0)

    ----------------------------------------------------------------------------------------------------------
    [구속식(Constraints)과 DAE]
    - 링크기구는 조인트에 의해 기하학적 구속식 Φ(q,t)=0을 만족해야 한다.
    - 본 구현은 다음 구속을 포함:
      (1) R joint(회전 조인트) : 두 링크의 동일 조인트 점이 공간상에서 일치(점-점 일치)
          p_b(q) - p_a(q) = 0  (2개 스칼라 방정식: x, y)
      (2) P joint(프리즘틱) : 레일을 따라 슬라이더 점이 이동
          - 본 데이터/의도에 따라 P joint는 ground에 연결되어 있다고 가정.
          - "마찰 없는 이상적 프리즘틱" 모델로:
            (a) 점-선 구속: 레일의 법선 방향으로는 움직일 수 없음
                n^T (p(q) - p0) = 0   (1개)
            (b) 자세 구속: 슬라이더(해당 링크)의 orientation을 레일 방향으로 고정(회전 금지)
                φ_link - (α + offset) = 0  (1개)
            여기서
              α : 레일 방향각 (slider_angles로 입력)
              n = [-sinα, cosα]^T : 레일 법선 단위벡터
              p0 : 레일 위 기준점(초기 P joint 위치를 사용)
              offset : 초기 자세 일치 위해 offset = φ0_link - α 로 설정
      (3) 구동(Driving) 구속: 입력 링크의 자세를 시간함수 θ(t)로 지정
            φ_input - (φ_input0 + θ(t)) = 0  (1개)
          이 구속식의 라그랑주 승수는 구동 토크(모터 토크)로 해석 가능하다(아래 참조).

    - 구속식을 모아서:
        Φ(q,t) = 0
      그 야코비안:
        J = Φ_q = ∂Φ/∂q

    ----------------------------------------------------------------------------------------------------------
    [속도 수준 / 가속도 수준 구속식]
    - 구속식을 시간에 대해 미분하면,
        (1) 속도 수준:
            Φ_q(q,t) * q_dot + Φ_t(q,t) = 0
        (2) 가속도 수준:
            Φ_q(q,t) * q_ddot + (d/dt(Φ_q)*q_dot + Φ_tt) = 0
      일반적으로
        Φ_q q_ddot + γ = 0
      로 쓰며,
        γ = Φ_qdot*q_dot + Φ_tt  (속도에 의해 발생하는 항 포함)

    - 본 구현에서 γ의 핵심은 "강체 점의 원심(centripetal) 항 -ω^2(Rs)"로부터 발생한다.
      예를 들어 R joint의 점-점 일치:
        p_b - p_a = 0
      를 가속도 수준에서 쓰면
        (p_b_ddot - p_a_ddot) = 0
      여기서 p_ddot 식에는 -ω^2(Rs)가 포함되므로,
      이 -ω^2(Rs) 항이 γ에 들어가고, q_ddot에 대한 항은 J q_ddot에 들어간다.
      (이 구현에서는 gamma 배열에 해당 항을 정확히 채운다.)

    ----------------------------------------------------------------------------------------------------------
    [DAE(증강 시스템) 해법 - 라그랑주 승수법]
    - 구속계의 동역학은
        M q_ddot = Q + J^T λ
        Φ(q,t)   = 0
      와 같이 라그랑주 승수 λ를 도입하여 표현된다.
    - 가속도 수준에서 Φ를 만족시키기 위해:
        J q_ddot + γ = 0
    - 두 식을 결합한 선형 시스템(증강 시스템):
        [ M   J^T ] [ q_ddot ] = [ Q   ]
        [ J    0  ] [  λ     ]   [ -γ  ]
      를 각 시간 스텝에서 풀면 q_ddot와 λ를 동시에 얻는다.

    ----------------------------------------------------------------------------------------------------------
    [반력과 모터 토크의 물리적 해석]
    - R joint 구속식이 p_b - p_a = 0 형태로 구성된 경우,
      해당 2개 스칼라 구속식의 라그랑주 승수 (λx, λy)는
      "joint에서 link_b가 link_a로부터 받는 반력" (global frame force vector)로 해석된다.
      따라서 결과로 각 joint의 반력을
        F_j = [λx, λy]
      로 기록한다.
      (반대 방향으로 link_a에는 -F_j가 작용)

    - Driving constraint(φ_input - (φ_input0 + θ(t)) = 0)의 라그랑주 승수 λ_drive는
      일반화 좌표 φ_input에 대응하는 구속 모멘트(토크)이다.
      즉, 이상적인 구동기(모터)가 입력 링크의 회전을 강제하기 위해 공급해야 하는 토크로 해석 가능하다.
      결과로:
        input_link_torque[t] = λ_drive[t]

    - (NEW) motor_torque는 input_link_torque의 부호를 뒤집은 값으로 정의한다.
      결과로:
        motor_torque[t] = -input_link_torque[t]

    - Prismatic constraint에서
        n^T(p - p0)=0
      의 라그랑주 승수 λ_n은 법선 방향 구속력 크기이며,
      반력 벡터는
        F = λ_n * n
      으로 기록한다.
      orientation lock 구속의 라그랑주 승수는 모멘트(N·m)이므로 force 벡터에는 포함하지 않는다.

    ----------------------------------------------------------------------------------------------------------
    [pair 의미(시각화용 after_pair 출력)]
    - 원본 데이터셋에서 P joint는 pair를 통해 "복제된 joint"를 만들고,
      슬라이더의 움직임을 그 복제 joint의 움직임으로 표현한다는 의도가 있다.
    - 따라서 after_pair 출력은 다음 규칙을 따른다:
        P joint 원본 j : 레일의 기준점(ground에 고정된 anchor) → 위치는 시간에 대해 고정
        dup = pair[j]  : 레일을 따라 움직이는 슬라이더 점 → 원본 j의 실제 운동 궤적을 복사
      pos/vel/acc 모두 동일하게 규정한다.

    ----------------------------------------------------------------------------------------------------------
    [참고 문헌/키워드]
    - Constrained Multibody Dynamics, Lagrange multiplier method
    - DAE formulation: Shabana, "Dynamics of Multibody Systems"
    - Velocity/Acceleration constraint stabilization(본 코드는 NR로 position을 맞추고,
      velocity/acceleration은 선형계로 맞추는 'index-3 DAE'의 전형적 절차를 따른다.)

    ==========================================================================================================
    """

    def __init__(
        self,
        topology_info,
        initial_coords: np.ndarray,
        physical_params: Dict[str, np.ndarray],
        input_motion: np.ndarray,
        slider_angles: Optional[np.ndarray] = None,
        gravity: float = 9.81,
        tolerance: float = 1e-10,
        max_iterations: int = 60,
        time_step: Optional[float] = None,
        topology_data: Optional[dict] = None,
        coord_unit: str = "mm",

        # (NEW) omega별 max torque 분석 옵션
        enable_omega_sweep: bool = False,
        omega_sweep_values: Optional[np.ndarray] = None,

        # (NEW) 모터가 일정 각속도로 구동한다고 가정하는 옵션
        # - input_rpm 또는 input_omega가 주어지면, theta_dot은 모든 step에서 상수,
        #   theta_ddot은 0으로 강제하며, input_motion(angle array)도 theta(t)=theta0+omega*t로 재구성한다.
        input_rpm: Optional[float] = None,
        input_omega: Optional[float] = None,
    ):
        self.topo = topology_info
        self.topology_data = topology_data if topology_data is not None else {}

        # (NEW) omega sweep 옵션 값 보관 (기존 주석 수정 없이 기능 추가)
        self.enable_omega_sweep = bool(enable_omega_sweep)
        self.omega_sweep_values = None if omega_sweep_values is None else np.asarray(omega_sweep_values, dtype=float).reshape(-1)

        # (NEW) omega sweep을 위해 원본 입력들을 보관 (기존 주석 수정 없이 기능 추가)
        self._omega_raw_initial_coords = np.asarray(initial_coords, dtype=float).copy()
        self._omega_raw_input_motion = np.asarray(input_motion, dtype=float).copy()
        self._omega_raw_slider_angles = None if slider_angles is None else np.asarray(slider_angles, dtype=float).copy()
        self._omega_raw_gravity = float(gravity)
        self._omega_raw_tolerance = float(tolerance)
        self._omega_raw_max_iterations = int(max_iterations)
        self._omega_raw_topology_data = topology_data
        self._omega_raw_coord_unit = coord_unit
        self.physical_params = physical_params  # omega sweep에서 재사용

        # ------------------------------------------------------------------------------------------------------
        # (1) Units
        # ------------------------------------------------------------------------------------------------------
        # initial_coords는 main에서 mm로 주어질 수 있으므로, 내부에서는 meters로 통일한다.
        # 물성(masses, inertias)은 SI(kg, kg·m^2), centers_of_mass는 m로 들어온다고 가정한다.
        self.coord_scale = 0.001 if coord_unit == "mm" else 1.0

        # ------------------------------------------------------------------------------------------------------
        # (2) Topology (original)
        # ------------------------------------------------------------------------------------------------------
        self.L = int(self.topo.number_of_links())
        self.J_original = int(self.topo.number_of_joints())
        self.edges_original = np.asarray(self.topo.links_connected_by_joints_original(), dtype=int)

        # ground / input link
        if "index_of_ground_link" not in self.topology_data or "input_link_index" not in self.topology_data:
            raise KeyError("topology_data must contain 'index_of_ground_link' and 'input_link_index'.")
        self.ground_idx = int(self.topology_data["index_of_ground_link"])
        self.input_idx = int(self.topology_data["input_link_index"])
        if self.input_idx == self.ground_idx:
            raise ValueError("input_link_index cannot be the ground link.")

        # joint types (1=R, 2=P)
        if "joint_type_list" not in self.topology_data:
            raise KeyError("topology_data must contain 'joint_type_list' (1=R, 2=P).")
        jt = np.asarray(self.topology_data["joint_type_list"], dtype=int).reshape(-1)
        if jt.shape[0] != self.J_original:
            raise ValueError(f"joint_type_list length mismatch. Expected {self.J_original} but got {jt.shape[0]}.")
        self.joint_type = jt.copy()

        # ground-input joint는 반드시 R로 강제 (구동축이 되는 조인트를 P로 두지 않기 위함)
        self.ground_input_joint = self._find_ground_input_joint_original()
        self.joint_type[self.ground_input_joint] = 1

        # ------------------------------------------------------------------------------------------------------
        # (3) after_pair (visualization/animator compatibility)
        # ------------------------------------------------------------------------------------------------------
        # pair[j] == 0이면 R, pair[j] != 0이면 P joint이며 pair[j]가 duplicated joint index.
        self.pair = np.asarray(self.topo.pair(), dtype=int).reshape(-1)
        self.edges_after = np.asarray(self.topo.links_connected_by_joints_after_pair(), dtype=int)
        self.J_after = int(self.edges_after.shape[0])

        # ------------------------------------------------------------------------------------------------------
        # (4) Time
        # ------------------------------------------------------------------------------------------------------
        self.input_angles = np.asarray(input_motion, dtype=float).reshape(-1)
        self.num_steps = int(self.input_angles.shape[0])

        if time_step is None:
            # 입력각이 0..2π를 돈다고 가정하는 기본 시간 스케일(임의 스케일)
            total_time = 2.0 * np.pi
            self.dt = float(total_time / max(self.num_steps - 1, 1))
        else:
            self.dt = float(time_step)

        self.time = np.arange(self.num_steps, dtype=float) * self.dt

        # (NEW) 모터 상수 각속도 입력 처리
        # - input_rpm 또는 input_omega가 주어지면, 입력각/각속도/각가속도를 "시간 기반"으로 일관되게 재구성한다.
        self._has_constant_speed_input = False
        self._constant_omega = None
        if input_omega is not None:
            om = float(input_omega)
            if (not np.isfinite(om)) or (om <= 0.0):
                raise ValueError(f"input_omega must be a positive finite value. Got {input_omega}")
            self._has_constant_speed_input = True
            self._constant_omega = om
        elif input_rpm is not None:
            rpm = float(input_rpm)
            if (not np.isfinite(rpm)) or (rpm <= 0.0):
                raise ValueError(f"input_rpm must be a positive finite value. Got {input_rpm}")
            self._has_constant_speed_input = True
            self._constant_omega = (2.0 * np.pi) * (rpm / 60.0)

        # θ_dot, θ_ddot (유한차분)
        self.theta_dot = np.zeros(self.num_steps, dtype=float)
        self.theta_ddot = np.zeros(self.num_steps, dtype=float)

        if self._has_constant_speed_input:
            # (NEW) 이상적인 모터가 일정한 각속도로 구동하는 경우
            # - theta(t) = theta0 + omega * t
            # - theta_dot(t) = omega (상수)
            # - theta_ddot(t) = 0
            theta0 = float(self.input_angles[0]) if self.num_steps > 0 else 0.0
            omega = float(self._constant_omega)
            self.input_angles = theta0 + omega * self.time
            self.theta_dot[:] = omega
            self.theta_ddot[:] = 0.0
        else:
            for k in range(1, self.num_steps):
                self.theta_dot[k] = (self.input_angles[k] - self.input_angles[k - 1]) / self.dt
            for k in range(1, self.num_steps):
                self.theta_ddot[k] = (self.theta_dot[k] - self.theta_dot[k - 1]) / self.dt

            # (NEW) 경계(step=0)에서 theta_dot이 0으로 고정되는 문제를 완화하기 위한 보정
            # - PMKS+처럼 t=0에서도 이미 일정 각속도로 "구동 중"인 해석을 원할 경우,
            #   step0에서 theta_dot이 0이면 속도장이 왜곡될 수 있다.
            # - 여기서는 최소한 theta_dot[0]을 theta_dot[1]로 맞춰 초기 스텝에서 정지로 떨어지는 문제를 줄인다.
            if self.num_steps >= 2:
                self.theta_dot[0] = self.theta_dot[1]
                self.theta_ddot[0] = self.theta_ddot[1]

        # ------------------------------------------------------------------------------------------------------
        # (5) Initial joint coordinates (meters)
        # ------------------------------------------------------------------------------------------------------
        joint_pos0 = np.asarray(initial_coords, dtype=float)
        if joint_pos0.shape != (self.J_original, 2):
            raise ValueError(f"initial_coords must be (J_original,2)=({self.J_original},2) but got {joint_pos0.shape}.")
        self.joint_pos0 = joint_pos0 * self.coord_scale  # meters

        # ground에 연결된 조인트의 초기 좌표는 ground frame의 고정 기준점으로 사용한다.
        self.ground_joint_pos = {}
        for j in range(self.J_original):
            a, b = int(self.edges_original[j, 0]), int(self.edges_original[j, 1])
            if a == self.ground_idx or b == self.ground_idx:
                self.ground_joint_pos[int(j)] = self.joint_pos0[int(j)].copy()

        # ------------------------------------------------------------------------------------------------------
        # (6) Physical parameters (SI)
        # ------------------------------------------------------------------------------------------------------
        self.masses = np.asarray(physical_params["masses"], dtype=float).reshape(-1)
        self.inertias = np.asarray(physical_params["inertias"], dtype=float).reshape(-1)
        self.coms = np.asarray(physical_params["centers_of_mass"], dtype=float)

        if self.masses.shape[0] != self.L or self.inertias.shape[0] != self.L:
            raise ValueError(f"masses/inertias length mismatch. Expected L={self.L}.")
        if self.coms.shape != (self.L, 2):
            raise ValueError(f"centers_of_mass shape mismatch. Expected ({self.L},2) but got {self.coms.shape}.")

        self.g = float(gravity)
        self.tol = float(tolerance)
        self.max_iter = int(max_iterations)

        # ------------------------------------------------------------------------------------------------------
        # (7) Link->joints mapping (original)
        # ------------------------------------------------------------------------------------------------------
        self.link_joints = self._compute_link_joints_original()

        # ------------------------------------------------------------------------------------------------------
        # (8) Initial orientation phi0 for each link
        # ------------------------------------------------------------------------------------------------------
        # 링크 i에 대해 가능한 경우 첫 두 조인트를 이용해 방향을 초기 추정:
        #   φ0 = atan2( (p2-p1)_y, (p2-p1)_x )
        # 이는 이후 s_local 추정(바디 좌표에서의 부착점 벡터)에 사용된다.
        self.phi0 = np.zeros(self.L, dtype=float)
        for i in range(self.L):
            if i == self.ground_idx:
                self.phi0[i] = 0.0
                continue
            joints = self.link_joints.get(i, [])
            if len(joints) >= 2:
                j1, j2 = joints[0], joints[1]
                d = self.joint_pos0[j2] - self.joint_pos0[j1]
                self.phi0[i] = float(np.arctan2(d[1], d[0]))
            else:
                self.phi0[i] = 0.0

        # ------------------------------------------------------------------------------------------------------
        # (9) Local attachment vectors s_{i,j} (body-fixed)
        # ------------------------------------------------------------------------------------------------------
        # 조인트 위치 표현:
        #   p_{i,j} = r_i + R(φ_i) s_{i,j}
        # 초기 시점(t=0)에서:
        #   p0 = r0 + R(φ0) s  =>  s = R(-φ0) (p0 - r0)
        self.s_local: Dict[Tuple[int, int], np.ndarray] = {}
        for i in range(self.L):
            if i == self.ground_idx:
                continue
            r0 = self.coms[i].copy()  # meters
            RiT = self._R(-self.phi0[i])
            for j in self.link_joints.get(i, []):
                p0 = self.joint_pos0[int(j)].copy()
                self.s_local[(i, int(j))] = RiT @ (p0 - r0)

        # ------------------------------------------------------------------------------------------------------
        # (10) Prismatic joint specs
        # ------------------------------------------------------------------------------------------------------
        # P joint 인덱스: joint_type==2 (단, ground-input은 강제 R)
        self.p_joint_indices = np.where(self.joint_type == 2)[0].astype(int)

        # slider_angles는 P joint 개수와 동일해야 한다.
        # slider_angles[k]는 p_joint_indices[k] 조인트에 대한 레일 방향각 α
        if slider_angles is None:
            self.slider_angles = np.zeros((len(self.p_joint_indices),), dtype=float)
        else:
            self.slider_angles = np.asarray(slider_angles, dtype=float).reshape(-1)
            if self.slider_angles.shape[0] != len(self.p_joint_indices):
                raise ValueError(
                    f"slider_angles length mismatch. Expected {len(self.p_joint_indices)} but got {self.slider_angles.shape[0]}."
                )

        # P joint는 데이터 규칙상 ground와 직접 연결되어야 한다.
        # 본 모델에서는:
        #   (a) n^T(p - p0) = 0  (법선 방향 변위 0)
        #   (b) φ_link - (α + offset) = 0  (회전 금지, 레일 방향 유지)
        self.prismatic_specs = []
        for k, j in enumerate(self.p_joint_indices):
            li, lj = int(self.edges_original[j, 0]), int(self.edges_original[j, 1])
            if li != self.ground_idx and lj != self.ground_idx:
                raise ValueError(f"P joint {j} is not connected to ground. edges[{j}]={self.edges_original[j].tolist()}")

            other = lj if li == self.ground_idx else li

            alpha = float(self.slider_angles[k])
            n = np.array([-np.sin(alpha), np.cos(alpha)], dtype=float)  # unit normal

            p0 = self.ground_joint_pos[int(j)].copy()  # reference point on rail
            offset_phi = float(self.phi0[other] - alpha)  # initial alignment

            self.prismatic_specs.append({
                "joint": int(j),
                "link": int(other),
                "alpha": alpha,
                "n": n,
                "p0": p0,
                "offset_phi": offset_phi,
            })

        # ------------------------------------------------------------------------------------------------------
        # (11) Driving (motor) constraint setup
        # ------------------------------------------------------------------------------------------------------
        # 입력 링크의 orientation을 θ(t)로 지정:
        #   φ_input(t) = φ_input0 + θ(t)
        # Driving constraint:
        #   Φ_drive = φ_input - (φ_input0 + θ(t)) = 0
        # 이때 라그랑주 승수 λ_drive는 입력에 필요한 토크로 해석 가능.
        self.phi_input0 = float(self.phi0[self.input_idx])

        # ------------------------------------------------------------------------------------------------------
        # (12) Generalized coordinates indexing (moving links only)
        # ------------------------------------------------------------------------------------------------------
        # ground 제외 링크들을 나열하고, 각 링크 i에 대해 q의 시작 인덱스를 부여:
        #   q[k:k+3] = [x_i, y_i, φ_i]
        self.moving_links = [i for i in range(self.L) if i != self.ground_idx]
        self.link_to_qidx = {i: 3 * k for k, i in enumerate(self.moving_links)}
        self.nq = 3 * len(self.moving_links)

        # ------------------------------------------------------------------------------------------------------
        # (13) Mass matrix and applied forces
        # ------------------------------------------------------------------------------------------------------
        # M = blockdiag(diag(m_i, m_i, I_i))
        self.M = np.zeros((self.nq, self.nq), dtype=float)
        for i in self.moving_links:
            k = self.link_to_qidx[i]
            m = float(self.masses[i])
            I = float(self.inertias[i])
            self.M[k + 0, k + 0] = m
            self.M[k + 1, k + 1] = m
            self.M[k + 2, k + 2] = I

        # Q_applied: 중력만
        self.Q_applied = np.zeros((self.nq,), dtype=float)
        for i in self.moving_links:
            k = self.link_to_qidx[i]
            self.Q_applied[k + 1] = -float(self.masses[i]) * self.g

        # ------------------------------------------------------------------------------------------------------
        # (14) Constraint bookkeeping
        # ------------------------------------------------------------------------------------------------------
        # R joint 인덱스: joint_type==1
        self.rev_joint_indices = np.where(self.joint_type == 1)[0].astype(int)

        # 제약식 개수:
        #   - R joint: 2 eq each
        #   - P joint: 2 eq each
        #   - driving: 1 eq
        self.n_constraints = 2 * len(self.rev_joint_indices) + 2 * len(self.prismatic_specs) + 1

        # lambda_map은 각 조인트/구속식이 lambdas 벡터의 어느 행에 대응되는지 저장한다.
        self.lambda_map = {"revolute": {}, "prismatic": {}, "drive": None}

    # =============================================================================
    # run()
    # =============================================================================
    def run(self) -> Dict[str, np.ndarray]:
        """
        시간 스텝을 순회하며 다음을 수행한다:
        1) position solve: Φ(q,t)=0을 Newton-Raphson으로 만족시키는 q를 찾는다.
        2) velocity solve: Φ_q qdot + Φ_t = 0 선형계로 qdot을 구한다.
        3) accel+lambda solve: 증강 시스템으로 qddot 및 λ를 구한다.
        4) 조인트 위치/속도/가속도를 강체 기구학으로 계산한다.
        5) λ로부터 관절 반력 및 모터 토크를 계산한다.
        6) 에너지를 계산한다.

        반환 dict는 main/animator 호환을 위해 original/after_pair 좌표를 모두 포함한다.
        """
        joint_pos_hist = np.full((self.J_original, 2, self.num_steps), np.nan, dtype=float)
        joint_vel_hist = np.full((self.J_original, 2, self.num_steps), np.nan, dtype=float)
        joint_acc_hist = np.full((self.J_original, 2, self.num_steps), np.nan, dtype=float)

        joint_react_hist = np.full((self.J_original, 2, self.num_steps), np.nan, dtype=float)
        input_link_torque_hist = np.full((self.num_steps,), np.nan, dtype=float)
        motor_torque_hist = np.full((self.num_steps,), np.nan, dtype=float)

        KE_hist = np.full((self.num_steps,), np.nan, dtype=float)
        PE_hist = np.full((self.num_steps,), np.nan, dtype=float)

        lambdas_hist = np.full((self.n_constraints, self.num_steps), np.nan, dtype=float)
        is_valid = np.zeros((self.num_steps,), dtype=bool)

        # 초기 q 추정: coms와 phi0 사용
        q_prev = np.zeros((self.nq,), dtype=float)
        for i in self.moving_links:
            k = self.link_to_qidx[i]
            q_prev[k + 0] = float(self.coms[i, 0])
            q_prev[k + 1] = float(self.coms[i, 1])
            q_prev[k + 2] = float(self.phi0[i])

        for step in range(self.num_steps):
            theta = float(self.input_angles[step])
            theta_dot = float(self.theta_dot[step])
            theta_ddot = float(self.theta_ddot[step])

            # (1) Position
            try:
                q = self._position_solve(q_prev, theta)
            except Exception:
                break

            # (2) Velocity
            try:
                qdot = self._velocity_solve(q, theta, theta_dot)
            except Exception:
                qdot = np.zeros_like(q)

            # (3) Acceleration + lambdas
            try:
                qddot, lam = self._accel_lambda_solve(q, qdot, theta, theta_dot, theta_ddot)
            except Exception:
                qddot = np.zeros_like(q)
                lam = np.full((self.n_constraints,), np.nan, dtype=float)

            # (4) Joint kinematics
            jp, jv, ja = self._compute_joint_kinematics(q, qdot, qddot)
            joint_pos_hist[:, :, step] = jp
            joint_vel_hist[:, :, step] = jv
            joint_acc_hist[:, :, step] = ja

            # (5) Reaction forces & motor torque
            joint_react_hist[:, :, step] = self._extract_joint_reaction_forces(lam)
            drive_row = self.lambda_map["drive"]
            input_link_torque_hist[step] = float(lam[drive_row]) if (drive_row is not None and np.isfinite(lam[drive_row])) else np.nan
            motor_torque_hist[step] = -float(input_link_torque_hist[step]) if np.isfinite(input_link_torque_hist[step]) else np.nan

            # (6) Energies
            KE_hist[step], PE_hist[step] = self._compute_energies(q, qdot)

            lambdas_hist[:, step] = lam
            is_valid[step] = True
            q_prev = q.copy()

        # original -> mm
        joint_pos_mm = joint_pos_hist / self.coord_scale
        joint_vel_mm = joint_vel_hist / self.coord_scale
        joint_acc_mm = joint_acc_hist / self.coord_scale

        # after_pair arrays (pair meaning)
        joint_pos_after = self._build_after_pair_array_pair_meaning(joint_pos_hist, kind="pos")
        joint_vel_after = self._build_after_pair_array_pair_meaning(joint_vel_hist, kind="vel")
        joint_acc_after = self._build_after_pair_array_pair_meaning(joint_acc_hist, kind="acc")

        joint_pos_after_mm = joint_pos_after / self.coord_scale
        joint_vel_after_mm = joint_vel_after / self.coord_scale
        joint_acc_after_mm = joint_acc_after / self.coord_scale

        # (NEW) PMKS+처럼 "초속도/초가속도(크기)"를 별도로 보고 싶을 수 있으므로 magnitude도 함께 제공
        joint_speed_hist = np.full((self.J_original, self.num_steps), np.nan, dtype=float)
        joint_acc_mag_hist = np.full((self.J_original, self.num_steps), np.nan, dtype=float)
        for t in range(self.num_steps):
            if np.all(np.isfinite(joint_vel_hist[:, :, t])):
                joint_speed_hist[:, t] = np.linalg.norm(joint_vel_hist[:, :, t], axis=1)
            if np.all(np.isfinite(joint_acc_hist[:, :, t])):
                joint_acc_mag_hist[:, t] = np.linalg.norm(joint_acc_hist[:, :, t], axis=1)

        joint_speed_mm = joint_speed_hist / self.coord_scale
        joint_acc_mag_mm = joint_acc_mag_hist / self.coord_scale

        results = {
            "time": self.time,

            # original
            "joint_coordinates": joint_pos_hist,
            "joint_coordinates_mm": joint_pos_mm,
            "joint_velocities": joint_vel_hist,
            "joint_velocities_mm": joint_vel_mm,
            "joint_accelerations": joint_acc_hist,
            "joint_accelerations_mm": joint_acc_mm,

            # after_pair (anchor fixed, dup moving)
            "joint_coordinates_after_pair": joint_pos_after,
            "joint_coordinates_after_pair_mm": joint_pos_after_mm,
            "joint_velocities_after_pair": joint_vel_after,
            "joint_velocities_after_pair_mm": joint_vel_after_mm,
            "joint_accelerations_after_pair": joint_acc_after,
            "joint_accelerations_after_pair_mm": joint_acc_after_mm,

            # (NEW) speed/acc magnitude
            "joint_speeds": joint_speed_hist,
            "joint_speeds_mm": joint_speed_mm,
            "joint_acc_magnitudes": joint_acc_mag_hist,
            "joint_acc_magnitudes_mm": joint_acc_mag_mm,

            "slider_angles": self.slider_angles.copy(),

            "joint_reaction_forces": joint_react_hist,
            "input_link_torque": input_link_torque_hist,
            "motor_torque": motor_torque_hist,
            "constraint_lambdas": lambdas_hist,

            "kinetic_energy": KE_hist,
            "potential_energy": PE_hist,
            "total_energy": KE_hist + PE_hist,

            "is_valid": is_valid,
        }

        # (NEW) omega sweep 결과를 run 결과에 추가
        if self.enable_omega_sweep:
            sweep = self.analyze_max_torque_over_omega(self.omega_sweep_values)
            results.update(sweep)

        return results

    def analyze_max_torque_over_omega(self, omega_values: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        # (NEW) omega별 max torque 분석 메서드 (기존 주석 수정 없이 기능 추가)
        if omega_values is None:
            omega_values = np.array([0.5, 1.0, 2.0, 5.0, 10.0], dtype=float)
        else:
            omega_values = np.asarray(omega_values, dtype=float).reshape(-1)

        if omega_values.size == 0:
            return {
                "omega_sweep_omegas": np.zeros((0,), dtype=float),
                "omega_sweep_max_abs_torque": np.zeros((0,), dtype=float),
                "omega_sweep_max_torque": np.zeros((0,), dtype=float),
                "omega_sweep_min_torque": np.zeros((0,), dtype=float),
                "omega_sweep_max_abs_input_link_torque": np.zeros((0,), dtype=float),
                "omega_sweep_max_input_link_torque": np.zeros((0,), dtype=float),
                "omega_sweep_min_input_link_torque": np.zeros((0,), dtype=float),
            }

        omegas = []
        max_abs = []
        max_signed = []
        min_signed = []

        # (NEW) input_link_torque도 함께 저장 (기존 주석 수정 없이 기능 추가)
        max_abs_in = []
        max_in = []
        min_in = []

        N = int(len(self._omega_raw_input_motion))
        denom = max(N - 1, 1)

        for om in omega_values:
            om = float(om)
            if (not np.isfinite(om)) or (om <= 0.0):
                continue

            dt = (2.0 * np.pi) / (om * denom)

            sim = DynamicsSimulator(
                topology_info=self.topo,
                initial_coords=self._omega_raw_initial_coords,
                physical_params=self.physical_params,
                input_motion=self._omega_raw_input_motion,
                slider_angles=self._omega_raw_slider_angles,
                gravity=self._omega_raw_gravity,
                tolerance=self._omega_raw_tolerance,
                max_iterations=self._omega_raw_max_iterations,
                time_step=dt,
                topology_data=self._omega_raw_topology_data,
                coord_unit=self._omega_raw_coord_unit,
                enable_omega_sweep=False,
                omega_sweep_values=None,
            )

            res = sim.run()
            in_torque = np.asarray(res.get("input_link_torque", np.full((N,), np.nan)), dtype=float).reshape(-1)
            torque = np.asarray(res.get("motor_torque", np.full((N,), np.nan)), dtype=float).reshape(-1)
            valid = np.asarray(res.get("is_valid", np.zeros((N,), dtype=bool)), dtype=bool).reshape(-1)

            if torque.size != N:
                torque = torque[:N] if torque.size > N else np.pad(torque, (0, N - torque.size), constant_values=np.nan)
            if in_torque.size != N:
                in_torque = in_torque[:N] if in_torque.size > N else np.pad(in_torque, (0, N - in_torque.size), constant_values=np.nan)
            if valid.size != N:
                valid = valid[:N] if valid.size > N else np.pad(valid, (0, N - valid.size), constant_values=False)

            mask = valid & np.isfinite(torque)
            mask_in = valid & np.isfinite(in_torque)
            omegas.append(om)

            if not np.any(mask):
                max_abs.append(np.nan)
                max_signed.append(np.nan)
                min_signed.append(np.nan)
            else:
                tsel = torque[mask]
                max_abs.append(float(np.max(np.abs(tsel))))
                max_signed.append(float(np.max(tsel)))
                min_signed.append(float(np.min(tsel)))

            if not np.any(mask_in):
                max_abs_in.append(np.nan)
                max_in.append(np.nan)
                min_in.append(np.nan)
            else:
                tsel_in = in_torque[mask_in]
                max_abs_in.append(float(np.max(np.abs(tsel_in))))
                max_in.append(float(np.max(tsel_in)))
                min_in.append(float(np.min(tsel_in)))

        return {
            "omega_sweep_omegas": np.asarray(omegas, dtype=float),
            "omega_sweep_max_abs_torque": np.asarray(max_abs, dtype=float),
            "omega_sweep_max_torque": np.asarray(max_signed, dtype=float),
            "omega_sweep_min_torque": np.asarray(min_signed, dtype=float),

            # (NEW) input_link_torque sweep
            "omega_sweep_max_abs_input_link_torque": np.asarray(max_abs_in, dtype=float),
            "omega_sweep_max_input_link_torque": np.asarray(max_in, dtype=float),
            "omega_sweep_min_input_link_torque": np.asarray(min_in, dtype=float),
        }

    # =============================================================================
    # Pair-meaning after_pair builder
    # =============================================================================
    def _build_after_pair_array_pair_meaning(self, arr: np.ndarray, kind: str) -> np.ndarray:
        """
        시각화/after_pair 호환용 배열 생성.

        입력:
            arr: (J_original,2,steps)  (original joint의 물리적 운동 궤적)

        출력:
            out: (J_after,2,steps)

        규칙:
            - 기본적으로 out[0:J_original] = arr
            - P joint j (pair[j]!=0) 에 대해:
                * anchor(원본 j) = ground-fixed:
                    pos: 시간 전체를 초기 위치로 고정
                    vel/acc: 0
                * dup = pair[j] = moving slider point:
                    pos/vel/acc 모두 arr[j]를 복사
        """
        if arr.ndim != 3 or arr.shape[0] != self.J_original or arr.shape[1] != 2:
            raise ValueError(f"Expected (J_original,2,steps) but got {arr.shape}")

        steps = int(arr.shape[2])
        out = np.full((self.J_after, 2, steps), np.nan, dtype=float)
        out[: self.J_original, :, :] = arr

        p_idx = np.where(self.pair != 0)[0].astype(int)
        for j in p_idx:
            dup = int(self.pair[j])
            if not (0 <= dup < self.J_after):
                continue

            if kind == "pos":
                anchor = arr[int(j), :, 0].copy()
                out[int(j), :, :] = anchor.reshape(2, 1)
                out[int(dup), :, :] = arr[int(j), :, :]
            elif kind in ("vel", "acc"):
                out[int(j), :, :] = 0.0
                out[int(dup), :, :] = arr[int(j), :, :]
            else:
                raise ValueError("kind must be 'pos', 'vel', or 'acc'")

        return out

    # =============================================================================
    # Topology helpers
    # =============================================================================
    def _find_ground_input_joint_original(self) -> int:
        """ground link와 input link를 잇는 R joint(구동축) 찾기"""
        for j in range(self.J_original):
            a, b = int(self.edges_original[j, 0]), int(self.edges_original[j, 1])
            if (a == self.ground_idx and b == self.input_idx) or (b == self.ground_idx and a == self.input_idx):
                return int(j)
        raise ValueError(f"Cannot find ground-input joint: ground={self.ground_idx}, input={self.input_idx}")

    def _compute_link_joints_original(self) -> Dict[int, List[int]]:
        """각 링크에 연결된 joint index 목록 생성"""
        link_joints: Dict[int, List[int]] = {i: [] for i in range(self.L)}
        for j in range(self.J_original):
            a, b = int(self.edges_original[j, 0]), int(self.edges_original[j, 1])
            link_joints[a].append(int(j))
            link_joints[b].append(int(j))
        for i in range(self.L):
            link_joints[i] = sorted(link_joints[i])
        return link_joints

    # =============================================================================
    # Rigid body kinematics
    # =============================================================================
    @staticmethod
    def _R(phi: float) -> np.ndarray:
        """2D rotation matrix"""
        c = float(np.cos(phi))
        s = float(np.sin(phi))
        return np.array([[c, -s], [s, c]], dtype=float)

    @staticmethod
    def _E() -> np.ndarray:
        """2D skew matrix: E v = [-v_y, v_x]"""
        return np.array([[0.0, -1.0], [1.0, 0.0]], dtype=float)

    def _get_link_state(self, q: np.ndarray, link: int) -> Tuple[np.ndarray, float]:
        """q에서 링크 link의 (r, phi) 추출"""
        k = self.link_to_qidx[link]
        r = np.array([q[k + 0], q[k + 1]], dtype=float)
        phi = float(q[k + 2])
        return r, phi

    def _joint_point_on_link(self, q: np.ndarray, link: int, joint: int) -> np.ndarray:
        """링크 link가 가지는 joint의 global 위치"""
        if link == self.ground_idx:
            return self.ground_joint_pos[int(joint)].copy()
        r, phi = self._get_link_state(q, link)
        s = self.s_local[(link, int(joint))]
        return r + self._R(phi) @ s

    # =============================================================================
    # Constraint assembly
    # =============================================================================
    def _build_constraints(self, q: np.ndarray, theta: float, theta_dot: float, theta_ddot: float):
        """
        Φ(q,t), J(q,t), Φ_t(q,t), γ(q,qdot,t) 구성.

        - Φ는 position-level constraints.
        - J = ∂Φ/∂q
        - Φ_t는 velocity-level에서 필요한 시간 미분 항 (driving constraint만 존재)
        - γ는 acceleration-level에서 J qddot + γ = 0 형태의 잔여항.
          여기서는 centripetal term(-ω^2 Rs)을 정확히 반영한다.
        """
        m = self.n_constraints
        Phi = np.zeros((m,), dtype=float)
        J = np.zeros((m, self.nq), dtype=float)
        Phi_t = np.zeros((m,), dtype=float)
        gamma = np.zeros((m,), dtype=float)

        E = self._E()
        row = 0

        # (1) Revolute joints: p_b - p_a = 0
        self.lambda_map["revolute"].clear()
        for j in self.rev_joint_indices:
            a, b = int(self.edges_original[j, 0]), int(self.edges_original[j, 1])
            pb = self._joint_point_on_link(q, b, int(j))
            pa = self._joint_point_on_link(q, a, int(j))
            c = pb - pa

            Phi[row + 0] = float(c[0])
            Phi[row + 1] = float(c[1])

            # Jacobian contributions
            if b != self.ground_idx:
                kb = self.link_to_qidx[b]
                _, phib = self._get_link_state(q, b)
                sb = self.s_local[(b, int(j))]
                rbs = self._R(phib) @ sb
                J[row:row + 2, kb + 0:kb + 2] += np.eye(2)
                J[row:row + 2, kb + 2] += (E @ rbs)

            if a != self.ground_idx:
                ka = self.link_to_qidx[a]
                _, phia = self._get_link_state(q, a)
                sa = self.s_local[(a, int(j))]
                ras = self._R(phia) @ sa
                J[row:row + 2, ka + 0:ka + 2] -= np.eye(2)
                J[row:row + 2, ka + 2] -= (E @ ras)

            self.lambda_map["revolute"][int(j)] = (row + 0, row + 1)
            row += 2

        # (2) Prismatic joints: point-on-line + orientation lock
        self.lambda_map["prismatic"].clear()
        for spec in self.prismatic_specs:
            j = int(spec["joint"])
            link = int(spec["link"])
            nvec = spec["n"]
            p0 = spec["p0"]
            alpha = float(spec["alpha"])
            offset_phi = float(spec["offset_phi"])

            # (a) n^T(p - p0) = 0
            p = self._joint_point_on_link(q, link, j)
            Phi[row] = float(np.dot(p - p0, nvec))

            kL = self.link_to_qidx[link]
            _, phiL = self._get_link_state(q, link)
            sL = self.s_local[(link, j)]
            rLs = self._R(phiL) @ sL

            J[row, kL + 0:kL + 2] = nvec
            J[row, kL + 2] = float(np.dot(nvec, E @ rLs))
            idx_normal = row
            row += 1

            # (b) φ_link - (α + offset) = 0
            Phi[row] = float(phiL - (alpha + offset_phi))
            J[row, kL + 2] = 1.0
            idx_lock = row
            row += 1

            self.lambda_map["prismatic"][int(j)] = (idx_normal, idx_lock)

        # (3) Driving constraint: φ_input - (φ_input0 + θ(t)) = 0
        kIn = self.link_to_qidx[self.input_idx]
        phi_in = float(q[kIn + 2])
        Phi[row] = float(phi_in - (self.phi_input0 + theta))
        J[row, kIn + 2] = 1.0
        Phi_t[row] = float(-theta_dot)
        gamma[row] = float(-theta_ddot)
        self.lambda_map["drive"] = row
        row += 1

        if row != m:
            raise RuntimeError("Constraint assembly size mismatch.")
        return Phi, J, Phi_t, gamma

    # =============================================================================
    # Solvers
    # =============================================================================
    def _position_solve(self, q_guess: np.ndarray, theta: float) -> np.ndarray:
        """
        Newton-Raphson로 Φ(q,t)=0을 만족시키는 q를 찾는다.
        반복:
            dq = argmin ||J dq + Φ||
            q <- q + dq
        """
        q = q_guess.copy()
        for _ in range(self.max_iter):
            Phi, J, _, _ = self._build_constraints(q, theta, 0.0, 0.0)
            if float(np.linalg.norm(Phi)) < self.tol:
                return q
            dq, _, _, _ = lstsq(J, -Phi)
            q = q + dq

        Phi, _, _, _ = self._build_constraints(q, theta, 0.0, 0.0)
        raise RuntimeError(f"Position solve did not converge: ||Phi||={float(np.linalg.norm(Phi)):.3e}")

    def _velocity_solve(self, q: np.ndarray, theta: float, theta_dot: float) -> np.ndarray:
        """
        속도 수준 구속식:
            J qdot + Φ_t = 0
        를 최소제곱으로 풀어 qdot을 구한다.
        """
        _, J, Phi_t, _ = self._build_constraints(q, theta, theta_dot, 0.0)
        qdot, _, _, _ = lstsq(J, -Phi_t)
        return qdot

    def _accel_lambda_solve(self, q: np.ndarray, qdot: np.ndarray, theta: float, theta_dot: float, theta_ddot: float):
        """
        가속도 및 라그랑주 승수 동시 해:
            [ M  J^T ] [qddot] = [ Q ]
            [ J   0  ] [  λ  ]   [-γ]
        여기서 γ는 구속 가속도 수준 잔여항(원심항 포함).
        """
        _, J, _, gamma = self._build_constraints(q, theta, theta_dot, theta_ddot)

        # γ에 원심항(-ω^2 Rs) 기여를 채운다.
        #  - Revolute: p_b - p_a = 0
        #  - Prismatic: point-on-line n^T(p - p0)=0
        for j in self.rev_joint_indices:
            rx, ry = self.lambda_map["revolute"][int(j)]
            a, b = int(self.edges_original[j, 0]), int(self.edges_original[j, 1])

            gb = np.zeros(2, dtype=float)
            ga = np.zeros(2, dtype=float)

            if b != self.ground_idx:
                kb = self.link_to_qidx[b]
                wb = float(qdot[kb + 2])
                _, phib = self._get_link_state(q, b)
                sb = self.s_local[(b, int(j))]
                rbs = self._R(phib) @ sb
                gb = - (wb ** 2) * rbs

            if a != self.ground_idx:
                ka = self.link_to_qidx[a]
                wa = float(qdot[ka + 2])
                _, phia = self._get_link_state(q, a)
                sa = self.s_local[(a, int(j))]
                ras = self._R(phia) @ sa
                ga = - (wa ** 2) * ras

            gamma[rx] = float(gb[0] - ga[0])
            gamma[ry] = float(gb[1] - ga[1])

        for spec in self.prismatic_specs:
            j = int(spec["joint"])
            link = int(spec["link"])
            nvec = spec["n"]
            idx_normal, _ = self.lambda_map["prismatic"][int(j)]

            kL = self.link_to_qidx[link]
            w = float(qdot[kL + 2])
            _, phiL = self._get_link_state(q, link)
            sL = self.s_local[(link, int(j))]
            rLs = self._R(phiL) @ sL
            gamma[idx_normal] = float(np.dot(nvec, - (w ** 2) * rLs))

        # 증강 시스템 구성 후 최소제곱으로 풂
        m = self.n_constraints
        n = self.nq

        A = np.zeros((n + m, n + m), dtype=float)
        b = np.zeros((n + m,), dtype=float)

        A[:n, :n] = self.M
        A[:n, n:] = J.T
        A[n:, :n] = J

        b[:n] = self.Q_applied
        b[n:] = -gamma

        sol, _, _, _ = lstsq(A, b)
        qddot = sol[:n]
        lam = sol[n:]
        return qddot, lam

    # =============================================================================
    # Kinematics & reactions & energies
    # =============================================================================
    def _compute_joint_kinematics(self, q: np.ndarray, qdot: np.ndarray, qddot: np.ndarray):
        """
        강체 기구학으로 joint의 위치/속도/가속도 계산.

        p = r + R s
        p_dot = r_dot + ω E (R s)
        p_ddot = r_ddot + α E (R s) - ω^2 (R s)
        """
        E = self._E()
        jp = np.zeros((self.J_original, 2), dtype=float)
        jv = np.zeros((self.J_original, 2), dtype=float)
        ja = np.zeros((self.J_original, 2), dtype=float)

        for j in range(self.J_original):
            a, b = int(self.edges_original[j, 0]), int(self.edges_original[j, 1])
            link = b if a == self.ground_idx else a
            if link == self.ground_idx:
                link = b

            if link == self.ground_idx:
                jp[j] = self.joint_pos0[j]
                jv[j] = 0.0
                ja[j] = 0.0
                continue

            k = self.link_to_qidx[link]
            r = np.array([q[k + 0], q[k + 1]], dtype=float)
            v = np.array([qdot[k + 0], qdot[k + 1]], dtype=float)
            a_lin = np.array([qddot[k + 0], qddot[k + 1]], dtype=float)

            phi = float(q[k + 2])
            w = float(qdot[k + 2])
            alpha = float(qddot[k + 2])

            s = self.s_local[(link, int(j))]
            Rs = self._R(phi) @ s

            jp[j] = r + Rs
            jv[j] = v + w * (E @ Rs)
            ja[j] = a_lin + alpha * (E @ Rs) - (w ** 2) * Rs

        return jp, jv, ja
    
    def _is_joint_connected_to_ground(self, joint_index: int) -> bool:
        """joint가 ground link에 연결되어 있는지 여부"""
        j = int(joint_index)
        a, b = int(self.edges_original[j, 0]), int(self.edges_original[j, 1])
        return (a == self.ground_idx) or (b == self.ground_idx)

    def _extract_joint_reaction_forces(self, lam: np.ndarray) -> np.ndarray:
        """
        라그랑주 승수로부터 관절 반력(force vector)을 산출.

        - R joint: p_b - p_a = 0 형태로 제약을 구성했으므로,
          해당 2개 방정식의 λ = [Fx, Fy] 는 link_b가 받는 반력.
        - P joint: point-on-line 구속의 λ_n은 법선방향 구속력 크기
          반력 벡터 = λ_n * n
          (orientation lock의 λ는 모멘트이므로 force 벡터에는 포함하지 않음)

        (NEW)
        - ground link에 연결된 joint(= ground를 구성하는 joint)의 반력 방향은 현재 정의가 맞으므로 그대로 둔다.
        - ground에 연결되어 있지 않은 joint들(두 링크 모두 ground가 아닌 joint)은
          모든 경우에 대해 반력 벡터를 -1 곱하여 반대 방향으로 출력한다.
        """
        jr = np.zeros((self.J_original, 2), dtype=float)

        # -------------------------
        # 1) Revolute joints
        # -------------------------
        for j in self.rev_joint_indices:
            rx, ry = self.lambda_map["revolute"][int(j)]
            fx = float(lam[rx])
            fy = float(lam[ry])

            # (NEW) ground에 연결되지 않은 joint는 부호를 뒤집는다.
            if not self._is_joint_connected_to_ground(int(j)):
                fx = -fx
                fy = -fy

            jr[int(j), 0] = fx
            jr[int(j), 1] = fy

        # -------------------------
        # 2) Prismatic joints
        # -------------------------
        for spec in self.prismatic_specs:
            j = int(spec["joint"])
            nvec = spec["n"]
            idx_normal, _ = self.lambda_map["prismatic"][int(j)]
            fx = float(lam[idx_normal]) * float(nvec[0])
            fy = float(lam[idx_normal]) * float(nvec[1])

            # (NEW) ground에 연결되지 않은 joint는 부호를 뒤집는다.
            # (현재 구현에서는 P joint가 ground에 연결되어 있어야 하므로 보통은 여기서 뒤집히지 않는다.)
            if not self._is_joint_connected_to_ground(int(j)):
                fx = -fx
                fy = -fy

            jr[j, 0] = fx
            jr[j, 1] = fy

        return jr

    def _compute_energies(self, q: np.ndarray, qdot: np.ndarray) -> Tuple[float, float]:
        """
        시스템 에너지 계산(정확한 강체 운동 에너지 + 중력 퍼텐셜).

        - Kinetic:
            T = Σ (1/2 m (vx^2+vy^2) + 1/2 I ω^2)
        - Potential:
            U = Σ (m g y_com)
        """
        KE = 0.0
        PE = 0.0
        for i in self.moving_links:
            k = self.link_to_qidx[i]
            vx = float(qdot[k + 0])
            vy = float(qdot[k + 1])
            w = float(qdot[k + 2])
            m = float(self.masses[i])
            I = float(self.inertias[i])

            KE += 0.5 * m * (vx * vx + vy * vy) + 0.5 * I * (w * w)
            PE += m * self.g * float(q[k + 1])
        return float(KE), float(PE)
