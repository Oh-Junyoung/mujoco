import numpy as np
from typing import Callable, Optional, Tuple, Dict
from topology_data_load.TopologyCalculator import TopologyCalculator
from .GenerationStrategies import GenerationStrategies


class InitialDesignVariableGenerator:
    """
    각 위상에 대한 초기 설계 변수(joint 좌표)를 생성하는 클래스
    
    기능:
    - 위상별로 원하는 개수만큼의 초기 좌표 셋 생성
    - 다양한 생성 전략 지원 (GenerationStrategies 사용)
    - Pair 적용 전 원본 joint 개수 기준으로 생성
    
    사용 예:
        from GenerationStrategies import GenerationStrategies
        
        # 기본 균등 분포 난수
        generator = InitialDesignVariableGenerator(
            data, 
            num_samples_per_topology=100
        )
        
        # 원형 배치 전략 사용
        generator = InitialDesignVariableGenerator(
            data,
            num_samples_per_topology=100,
            generation_strategy=GenerationStrategies.circular_arrangement(radius=15)
        )
        
        # 0번 위상, 5번째 좌표 셋
        coords = generator.get(topology_idx=0, sample_idx=5)
        print(coords.shape)  # (J, 2)
    """
    
    def __init__(
        self,
        topology_data: list,
        num_samples_per_topology: int = 100,
        generation_strategy: Optional[Callable] = None,
        coord_range: Tuple[float, float] = (-50.0, 50.0),
        seed: Optional[int] = None,

        # (NEW) 사용자가 수동으로 초기 좌표를 부여하는 시스템
        # - manual_joint_coords를 지정하면 "모든 위상/샘플"에서 동일한 초기 joint 좌표를 사용한다.
        # - manual_joint_coords_map을 지정하면 (topology_idx, sample_idx)별로 다른 초기 joint 좌표를 사용한다.
        manual_joint_coords: Optional[np.ndarray] = None,
        manual_joint_coords_map: Optional[Dict[Tuple[int, int], np.ndarray]] = None,
    ):
        """
        Args:
            topology_data: 위상 데이터 리스트 (TopologyDataLoader.load() 결과)
            num_samples_per_topology: 각 위상당 생성할 샘플 개수
            generation_strategy: 좌표 생성 전략 함수 
                                 signature: (num_joints, rng) -> np.ndarray(J, 2)
                                 None이면 기본 균등 분포 난수 사용
            coord_range: 좌표 범위 (min, max) - 기본 전략 사용 시
            seed: 난수 시드 (재현성을 위해)
        """
        self.topology_data = topology_data
        self.num_topologies = len(topology_data)
        self.num_samples = num_samples_per_topology
        self.coord_range = coord_range
        self.seed = seed
        
        # 랜덤 생성기 초기화
        self.rng = np.random.default_rng(seed)
        
        # 생성 전략 설정 - GenerationStrategies 사용
        if generation_strategy is None:
            self.generation_strategy = GenerationStrategies.uniform_random(coord_range)
        else:
            self.generation_strategy = generation_strategy
        
        # 각 위상의 joint 개수 미리 계산 (캐싱)
        self._num_joints_per_topology = self._compute_num_joints()
        
        # 좌표 저장소 (lazy initialization)
        # {(topology_idx, sample_idx): np.ndarray}
        self._coordinate_cache = {}

        # (NEW) 수동 좌표 저장소
        self._manual_joint_coords: Optional[np.ndarray] = None
        self._manual_joint_coords_map: Dict[Tuple[int, int], np.ndarray] = {}

        # (NEW) 수동 좌표 입력 처리
        if manual_joint_coords is not None:
            arr = np.asarray(manual_joint_coords, dtype=float)
            if arr.ndim != 2 or arr.shape[1] != 2:
                raise ValueError(f"manual_joint_coords expected shape (J,2) but got {arr.shape}")
            self._manual_joint_coords = arr.copy()

        if manual_joint_coords_map is not None:
            if not isinstance(manual_joint_coords_map, dict):
                raise ValueError("manual_joint_coords_map must be a dict {(topology_idx, sample_idx): np.ndarray}")
            for k, v in manual_joint_coords_map.items():
                if (not isinstance(k, tuple)) or len(k) != 2:
                    raise ValueError("manual_joint_coords_map keys must be (topology_idx, sample_idx) tuples")
                ti = int(k[0])
                si = int(k[1])
                vv = np.asarray(v, dtype=float)
                if vv.ndim != 2 or vv.shape[1] != 2:
                    raise ValueError(f"manual_joint_coords_map[{k}] expected shape (J,2) but got {vv.shape}")
                self._manual_joint_coords_map[(ti, si)] = vv.copy()
    
    def _compute_num_joints(self) -> list:
        """각 위상의 joint 개수 계산"""
        num_joints_list = []
        
        for topology in self.topology_data:
            calc = TopologyCalculator(topology)
            num_joints = calc.number_of_joints()
            num_joints_list.append(num_joints)
        
        return num_joints_list
    
    def get(self, topology_idx: int, sample_idx: int) -> np.ndarray:
        """
        특정 위상의 특정 샘플 좌표 가져오기
        
        Args:
            topology_idx: 위상 인덱스 (0 ~ num_topologies-1)
            sample_idx: 샘플 인덱스 (0 ~ num_samples-1)
            
        Returns:
            shape (J, 2)의 좌표 배열
            - J: 해당 위상의 joint 개수
            - [:, 0]: x 좌표
            - [:, 1]: y 좌표
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

        # (NEW) 수동 좌표 우선 적용
        # - 케이스별 수동 좌표가 있으면 그것을 사용
        # - 없으면 공통 수동 좌표가 있으면 그것을 사용
        # - 둘 다 없으면 기존 generation_strategy로 생성
        num_joints = self._num_joints_per_topology[topology_idx]
        manual = self._get_manual_coordinates(topology_idx, sample_idx)
        if manual is not None:
            if manual.shape != (num_joints, 2):
                raise ValueError(
                    f"수동 좌표의 shape이 잘못되었습니다. "
                    f"예상: ({num_joints}, 2), 실제: {manual.shape}"
                )
            return manual.copy()
        
        # 캐시 확인
        cache_key = (topology_idx, sample_idx)
        if cache_key in self._coordinate_cache:
            return self._coordinate_cache[cache_key]
        
        # 좌표 생성
        # 각 샘플마다 고유한 시드 생성 (재현성 유지)
        sample_seed = self._get_sample_seed(topology_idx, sample_idx)
        sample_rng = np.random.default_rng(sample_seed)
        
        coords = self.generation_strategy(num_joints, sample_rng)
        
        # 검증
        if coords.shape != (num_joints, 2):
            raise ValueError(
                f"생성된 좌표의 shape이 잘못되었습니다. "
                f"예상: ({num_joints}, 2), 실제: {coords.shape}"
            )
        
        # 캐싱
        self._coordinate_cache[cache_key] = coords
        
        return coords

    # =============================================================================
    # (NEW) Manual coordinate system helpers
    # =============================================================================
    def _get_manual_coordinates(self, topology_idx: int, sample_idx: int) -> Optional[np.ndarray]:
        ti = int(topology_idx)
        si = int(sample_idx)
        if (ti, si) in self._manual_joint_coords_map:
            return self._manual_joint_coords_map[(ti, si)]
        if self._manual_joint_coords is not None:
            return self._manual_joint_coords
        return None

    def has_manual_coordinates(self) -> bool:
        return (self._manual_joint_coords is not None) or (len(self._manual_joint_coords_map) > 0)

    def set_manual_coordinates(self, manual_joint_coords: np.ndarray):
        arr = np.asarray(manual_joint_coords, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"manual_joint_coords expected shape (J,2) but got {arr.shape}")
        self._manual_joint_coords = arr.copy()
        # 전략 변경 시 캐시 초기화
        self.clear_cache()

    def set_manual_coordinates_for_case(self, topology_idx: int, sample_idx: int, manual_joint_coords: np.ndarray):
        ti = int(topology_idx)
        si = int(sample_idx)
        arr = np.asarray(manual_joint_coords, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"manual_joint_coords expected shape (J,2) but got {arr.shape}")
        self._manual_joint_coords_map[(ti, si)] = arr.copy()
        # 전략 변경 시 캐시 초기화
        self.clear_cache()

    def clear_manual_coordinates(self):
        self._manual_joint_coords = None
        self._manual_joint_coords_map = {}
        # 전략 변경 시 캐시 초기화
        self.clear_cache()

    def set_manual_coordinates_for_case_4bar_rrrr(
        self,
        topology_idx: int,
        sample_idx: int,
        J2: Tuple[float, float],
        J0: Tuple[float, float],
        J3: Tuple[float, float],
        J1: Tuple[float, float],
    ):
        """
        (NEW) 기본적인 4절 rrrr (J=4) 형태에만 작동하도록 만든 수동 좌표 세터

        사용자가 지정한:
            J2 = (x2, y2)
            J0 = (x0, y0)
            J3 = (x3, y3)
            J1 = (x1, y1)
        를 joint index 배열(0..3)에 맞춰서 coords[0]=J0, coords[1]=J1, coords[2]=J2, coords[3]=J3 형태로 구성한다.

        Note:
            - 이 함수는 joint index가 0,1,2,3인 4절 rrrr 위상에만 맞춘 편의 함수다.
            - 다른 위상에서는 오류가 나도 상관 없다고 하셨으므로, 최소한의 shape 체크만 수행한다.
        """
        ti = int(topology_idx)
        si = int(sample_idx)
        num_joints = self._num_joints_per_topology[ti]
        if num_joints != 4:
            raise ValueError(f"4절 rrrr 전용 함수입니다. Expected J=4 but got J={num_joints}")

        coords = np.zeros((4, 2), dtype=float)
        coords[0, :] = np.asarray(J0, dtype=float)
        coords[1, :] = np.asarray(J1, dtype=float)
        coords[2, :] = np.asarray(J2, dtype=float)
        coords[3, :] = np.asarray(J3, dtype=float)

        self.set_manual_coordinates_for_case(ti, si, coords)
    
    def _get_sample_seed(self, topology_idx: int, sample_idx: int) -> int:
        """
        각 샘플마다 고유하면서도 재현 가능한 시드 생성
        
        Args:
            topology_idx: 위상 인덱스
            sample_idx: 샘플 인덱스
            
        Returns:
            고유한 시드 값
        """
        if self.seed is None:
            # 시드가 없으면 완전 랜덤
            return None
        
        # 고유 시드 생성: base_seed + topology_offset + sample_offset
        return self.seed + topology_idx * 10000 + sample_idx
    
    def get_batch(
        self, 
        topology_idx: int, 
        sample_indices: Optional[list] = None
    ) -> np.ndarray:
        """
        특정 위상의 여러 샘플 좌표를 배치로 가져오기
        
        Args:
            topology_idx: 위상 인덱스
            sample_indices: 샘플 인덱스 리스트. None이면 전체.
            
        Returns:
            shape (N, J, 2)의 배열
            - N: 샘플 개수
            - J: joint 개수
        """
        if sample_indices is None:
            sample_indices = range(self.num_samples)
        
        coords_list = []
        for sample_idx in sample_indices:
            coords = self.get(topology_idx, sample_idx)
            coords_list.append(coords)
        
        return np.array(coords_list)
    
    def get_num_joints(self, topology_idx: int) -> int:
        """특정 위상의 joint 개수 조회"""
        if not (0 <= topology_idx < self.num_topologies):
            raise IndexError(
                f"topology_idx {topology_idx}가 범위를 벗어났습니다."
            )
        return self._num_joints_per_topology[topology_idx]
    
    def clear_cache(self):
        """캐시 초기화 (메모리 절약)"""
        self._coordinate_cache.clear()
    
    def set_generation_strategy(self, strategy: Callable):
        """
        생성 전략 변경
        
        Args:
            strategy: 새로운 생성 전략 함수
                     signature: (num_joints, rng) -> np.ndarray(J, 2)
        """
        self.generation_strategy = strategy
        # 전략 변경 시 캐시 초기화
        self.clear_cache()
    
    def info(self):
        """생성기 정보 출력"""
        print("=" * 60)
        print("Initial Design Variable Generator Info")
        print("=" * 60)
        print(f"총 위상 개수: {self.num_topologies}")
        print(f"위상당 샘플 개수: {self.num_samples}")
        print(f"좌표 범위: {self.coord_range}")
        print(f"시드: {self.seed}")
        print(f"캐시된 샘플 수: {len(self._coordinate_cache)}")
        if self.has_manual_coordinates():
            print(f"\n수동 초기 좌표 설정: True")
            if self._manual_joint_coords is not None:
                print(f"  공통 수동 좌표: shape={self._manual_joint_coords.shape}")
            if len(self._manual_joint_coords_map) > 0:
                print(f"  케이스별 수동 좌표: {len(self._manual_joint_coords_map)}개")
        else:
            print(f"\n수동 초기 좌표 설정: False")
        print("\n각 위상의 joint 개수:")
        for i, num_joints in enumerate(self._num_joints_per_topology):
            print(f"  위상 {i}: {num_joints}개 joints")
        print("=" * 60)
