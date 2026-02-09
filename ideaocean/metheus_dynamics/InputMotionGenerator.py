# InputMotionGenerator.py
import numpy as np
from typing import Optional


class InputMotionGenerator:
    """
    입력 링크의 회전 입력(input motion)을 생성하고 관리하는 클래스
    
    기능:
    - 0 ~ 2π 범위의 회전 각도를 균등 분배
    - 위상별, 샘플별로 동일한 입력 적용
    - 스텝 수 조절 가능 (기본: 72 스텝)
    
    사용 예:
        # 기본 72 스텝
        input_generator = InputMotionGenerator(
            num_topologies=5,
            num_samples_per_topology=100
        )
        
        # 커스텀 스텝 수
        input_generator = InputMotionGenerator(
            num_topologies=5,
            num_samples_per_topology=100,
            num_steps=120  # 120 스텝으로 변경
        )
        
        # 특정 위상, 특정 샘플의 입력 가져오기
        input_angles = input_generator.get(topology_idx=0, sample_idx=5)
        # shape: (72,) - 0 ~ 2π를 72등분한 각도 배열
        
        # 한 스텝의 각도
        angle_at_step_10 = input_generator.get_step(topology_idx=0, sample_idx=5, step=10)
    """
    
    def __init__(
        self,
        num_topologies: int,
        num_samples_per_topology: int,
        num_steps: int = 72,
        include_endpoint: bool = False,

        # (NEW) 시간 기반 입력 생성을 위한 옵션
        # - total_time을 지정하면 "0~2π를 total_time(초)에 걸쳐" 회전하도록 해석 가능
        # - input_rpm 또는 input_omega를 지정하면 모터가 일정 각속도(상수 ω)로 구동한다고 가정 가능
        # - time_step을 지정하면 dt(초)를 고정하고, num_steps는 total_time과 time_step에 의해 자동 계산 가능
        total_time: Optional[float] = None,
        time_step: Optional[float] = None,
        input_rpm: Optional[float] = None,
        input_omega: Optional[float] = None,
        start_angle: float = 0.0,
    ):
        """
        Args:
            num_topologies: 전체 위상 개수
            num_samples_per_topology: 각 위상당 샘플 개수 
                                      (InitialDesignVariableGenerator와 동일하게)
            num_steps: 0 ~ 2π를 나눌 스텝 수 (기본값: 72)
            include_endpoint: True면 2π 포함, False면 2π 미포함
                             (기본값: False, 주기 함수이므로 보통 끝점 제외)

            total_time: 0~2π 한 바퀴 회전에 해당하는 총 시간(초)
                       (None이면 기존처럼 "각도만 생성"해도 되지만, time_step 기반으로 사용 가능)
            time_step: 시간 스텝(dt, 초). 지정하면 시간축이 명확해지고, 상수 ω 입력과 잘 맞는다.
            input_rpm: 모터 입력 속도(rpm). 지정하면 omega=2π*rpm/60으로 상수 각속도 구동 입력을 생성한다.
            input_omega: 모터 입력 각속도(rad/s). input_rpm보다 우선한다.
            start_angle: 시작 각도(rad). 기본 0.0
        """
        self.num_topologies = num_topologies
        self.num_samples = num_samples_per_topology
        self.num_steps = num_steps
        self.include_endpoint = include_endpoint

        # (NEW) 시간/각속도 옵션 저장
        self.total_time = total_time
        self.time_step = time_step
        self.input_rpm = input_rpm
        self.input_omega = input_omega
        self.start_angle = float(start_angle)

        # 입력 각도 배열 생성 (모든 위상, 모든 샘플이 동일한 입력 사용)
        self._time, self._input_angles = self._generate_time_and_angles()
    
    def _generate_input_angles(self) -> np.ndarray:
        """
        0 ~ 2π를 균등 분배한 각도 배열 생성
        
        Returns:
            shape (num_steps,)의 각도 배열 (단위: radian)
        """
        return np.linspace(
            0, 
            2 * np.pi, 
            self.num_steps, 
            endpoint=self.include_endpoint
        )

    def _generate_time_and_angles(self) -> tuple[np.ndarray, np.ndarray]:
        """
        (NEW) time 축과 입력 각도 배열을 함께 생성

        Returns:
            time: shape (num_steps,)의 시간 배열 (단위: sec)
            angles: shape (num_steps,)의 각도 배열 (단위: radian)
        """
        # 기본값: 기존 로직 유지(각도만 생성) + time은 "step index" 기반으로 0..N-1
        # 다만 total_time/time_step/rpm/omega가 지정되면 시간 기반으로 재구성한다.

        # omega 결정
        omega = None
        if self.input_omega is not None:
            omega = float(self.input_omega)
            if (not np.isfinite(omega)) or (omega <= 0.0):
                raise ValueError(f"input_omega must be a positive finite value. Got {self.input_omega}")
        elif self.input_rpm is not None:
            rpm = float(self.input_rpm)
            if (not np.isfinite(rpm)) or (rpm <= 0.0):
                raise ValueError(f"input_rpm must be a positive finite value. Got {self.input_rpm}")
            omega = (2.0 * np.pi) * (rpm / 60.0)

        # total_time 결정
        total_time = None
        if self.total_time is not None:
            total_time = float(self.total_time)
            if (not np.isfinite(total_time)) or (total_time <= 0.0):
                raise ValueError(f"total_time must be a positive finite value. Got {self.total_time}")
        elif omega is not None:
            # omega가 주어졌는데 total_time이 없다면, "한 바퀴(2π)를 도는 시간"을 total_time으로 둔다.
            total_time = (2.0 * np.pi) / omega

        # time_step이 명시되면 num_steps를 time_step 기반으로 구성 가능
        if (total_time is not None) and (self.time_step is not None):
            dt = float(self.time_step)
            if (not np.isfinite(dt)) or (dt <= 0.0):
                raise ValueError(f"time_step must be a positive finite value. Got {self.time_step}")

            if self.include_endpoint:
                # endpoint 포함이면 t=0..total_time을 포함하도록 num_steps를 결정
                num_steps = int(np.floor(total_time / dt + 0.5)) + 1
                time = np.arange(num_steps, dtype=float) * dt
                # 마지막이 정확히 total_time이 아닐 수 있으니 마지막 값을 total_time으로 맞춘다.
                time[-1] = total_time
            else:
                # endpoint 미포함이면 total_time 직전까지 샘플링
                num_steps = int(np.floor(total_time / dt + 1e-12))
                if num_steps <= 0:
                    num_steps = 1
                time = np.arange(num_steps, dtype=float) * dt

            # 내부 상태 업데이트
            self.num_steps = num_steps

            if omega is None:
                # total_time만 주고 omega를 안 준 경우: 한 바퀴 회전하도록 omega 결정
                omega = (2.0 * np.pi) / total_time

            angles = self.start_angle + omega * time
            return time, angles

        # total_time이 주어졌지만 time_step은 없고 num_steps 기반이라면 dt를 total_time에 맞춘다.
        if total_time is not None:
            if self.include_endpoint:
                dt = total_time / max(self.num_steps - 1, 1)
            else:
                dt = total_time / max(self.num_steps, 1)
            time = np.arange(self.num_steps, dtype=float) * dt

            if omega is None:
                omega = (2.0 * np.pi) / total_time

            angles = self.start_angle + omega * time
            return time, angles

        # 아무 것도 안 주면 기존 각도 생성
        angles = self._generate_input_angles() + self.start_angle
        time = np.arange(self.num_steps, dtype=float)
        return time, angles
    
    def get(self, topology_idx: int, sample_idx: int) -> np.ndarray:
        """
        특정 위상의 특정 샘플에 대한 전체 입력 각도 배열 가져오기
        
        Args:
            topology_idx: 위상 인덱스 (0 ~ num_topologies-1)
            sample_idx: 샘플 인덱스 (0 ~ num_samples-1)
            
        Returns:
            shape (num_steps,)의 각도 배열 (단위: radian)
            
        Note:
            현재는 모든 위상, 모든 샘플이 동일한 입력을 사용하지만,
            나중에 위상별/샘플별로 다른 입력이 필요하면 확장 가능
        """
        # 인덱스 유효성 검사
        if not (0 <= topology_idx < self.num_topologies):
            raise IndexError(
                f"topology_idx {topology_idx}가 범위를 벗어났습니다. "
                f"(0 ~ {self.num_topologies-1})"
            )
        
        if not (0 <= sample_idx < self.num_samples):
            raise IndexError(
                f"sample_idx {sample_idx}가 범위를 벗어났습니다. "
                f"(0 ~ {self.num_samples-1})"
            )
        
        # 현재는 모두 동일한 입력 반환
        # 향후 위상별/샘플별 다른 입력이 필요하면 여기서 처리
        return self._input_angles.copy()

    def get_time(self) -> np.ndarray:
        """
        (NEW) time 배열 가져오기 (모든 위상/샘플 공통)

        Returns:
            shape (num_steps,)의 시간 배열 (단위: sec)
        """
        return self._time.copy()
    
    def get_step(self, topology_idx: int, sample_idx: int, step: int) -> float:
        """
        특정 스텝의 입력 각도 하나만 가져오기
        
        Args:
            topology_idx: 위상 인덱스
            sample_idx: 샘플 인덱스
            step: 스텝 인덱스 (0 ~ num_steps-1)
            
        Returns:
            해당 스텝의 각도 (단위: radian)
        """
        if not (0 <= step < self.num_steps):
            raise IndexError(
                f"step {step}이 범위를 벗어났습니다. "
                f"(0 ~ {self.num_steps-1})"
            )
        
        angles = self.get(topology_idx, sample_idx)
        return angles[step]
    
    def get_all_angles(self) -> np.ndarray:
        """
        기본 입력 각도 배열 가져오기 (모든 위상/샘플 공통)
        
        Returns:
            shape (num_steps,)의 각도 배열
        """
        return self._input_angles.copy()
    
    def get_angular_velocity(self) -> float:
        """
        각 스텝 사이의 각도 변화량 (각속도)
        
        Returns:
            Δθ (radian/step)
        """
        if self.include_endpoint:
            return 2 * np.pi / (self.num_steps - 1)
        else:
            return 2 * np.pi / self.num_steps
    
    def get_step_size(self) -> float:
        """
        get_angular_velocity()의 별칭
        """
        return self.get_angular_velocity()

    def get_constant_omega(self) -> Optional[float]:
        """
        (NEW) 상수 각속도(rad/s)를 반환
        - input_omega 또는 input_rpm 또는 total_time 기반으로 time/angles를 생성한 경우에만 유효
        """
        if self.total_time is not None:
            T = float(self.total_time)
            if np.isfinite(T) and T > 0.0:
                return (2.0 * np.pi) / T
        if self.input_omega is not None:
            return float(self.input_omega)
        if self.input_rpm is not None:
            return (2.0 * np.pi) * (float(self.input_rpm) / 60.0)
        return None
    
    def to_degrees(self, angles: np.ndarray) -> np.ndarray:
        """
        라디안을 도(degree)로 변환
        
        Args:
            angles: 라디안 각도 배열
            
        Returns:
            도(degree) 각도 배열
        """
        return np.degrees(angles)
    
    def info(self):
        """생성기 정보 출력"""
        print("=" * 60)
        print("Input Motion Generator Info")
        print("=" * 60)
        print(f"총 위상 개수: {self.num_topologies}")
        print(f"위상당 샘플 개수: {self.num_samples}")
        print(f"스텝 수: {self.num_steps}")
        print(f"끝점 포함: {self.include_endpoint}")
        print(f"각도 범위: 0 ~ 2π (0 ~ 360°)")
        print(f"각도 간격: {self.get_step_size():.6f} rad ({np.degrees(self.get_step_size()):.2f}°)")
        if self.total_time is not None:
            print(f"총 시간: {float(self.total_time):.6f} sec")
        if self.time_step is not None:
            print(f"시간 스텝(dt): {float(self.time_step):.6f} sec")
        om = self.get_constant_omega()
        if om is not None:
            print(f"상수 각속도(omega): {float(om):.6f} rad/s")
            print(f"상수 각속도(rpm): {float(om) * 60.0 / (2.0*np.pi):.6f} rpm")
        print("\n입력 각도 미리보기 (처음 10개):")
        preview = self._input_angles[:10]
        for i, angle in enumerate(preview):
            print(f"  Step {i}: {angle:.6f} rad ({np.degrees(angle):.2f}°)")
        if self.num_steps > 10:
            print(f"  ... (총 {self.num_steps}개)")
        print("=" * 60)
