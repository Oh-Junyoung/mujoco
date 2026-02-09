# SliderDirectionGenerator.py
from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional, Sequence, Union


class SliderDirectionGenerator:
    """
    P joint(Prismatic joint) slider 방향(각도)을 샘플별로 생성/관리하는 클래스.

    - 각 topology의 joint_type_list에서 값==2 인 joint들을 P joint로 간주
    - topology_index, sample_index에 대해 해당 topology의 모든 P joint에 대한
      slider 방향각(theta) 배열을 반환
    - theta는 [0, 2π) 범위의 난수(균일분포)

    사용 예:
        slider_dir_gen = SliderDirectionGenerator(
            topology_data=data,
            num_topologies=number_of_topologies,
            num_samples_per_topology=num_samples_per_topology,
            seed=123
        )

        slider_angles = slider_dir_gen.get(topology_index, sample_index)
        # slider_angles.shape == (num_P_joints_in_that_topology,)

    주의:
    - 반환하는 slider_angles는 "P joint의 방향각"만 제공합니다.
      (DynamicsSimulator/constraint 쪽에서 이 각도를 사용해 P joint 제약을 구성해야 합니다.)
    """

    def __init__(
        self,
        topology_data: Union[List[Dict[str, Any]], Dict[str, Any]],
        num_topologies: int,
        num_samples_per_topology: int,
        seed: Optional[int] = None,
        dtype: Any = np.float64,
    ):
        self.topology_data = topology_data
        self.num_topologies = int(num_topologies)
        self.num_samples_per_topology = int(num_samples_per_topology)
        self.dtype = dtype

        self.rng = np.random.default_rng(seed)

        # topology별 P joint index 목록
        self.p_joint_indices_by_topology: List[np.ndarray] = []
        # topology별, sample별 slider angles 저장 (가변 길이)
        self._angles_by_topology: List[List[np.ndarray]] = []

        self._build()

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def get(self, topology_index: int, sample_index: int) -> np.ndarray:
        """
        해당 (topology_index, sample_index)에 대한 slider angles 반환.
        반환 shape: (num_P_joints_in_that_topology,)
        """
        ti = int(topology_index)
        si = int(sample_index)
        self._validate_indices(ti, si)
        return self._angles_by_topology[ti][si].copy()

    def get_p_joint_indices(self, topology_index: int) -> np.ndarray:
        """해당 topology의 P joint index 목록(원본 joint 인덱스 기준)"""
        ti = int(topology_index)
        if ti < 0 or ti >= self.num_topologies:
            raise IndexError(f"topology_index out of range: {ti}")
        return self.p_joint_indices_by_topology[ti].copy()

    def regenerate(
        self,
        seed: Optional[int] = None,
        topology_indices: Optional[Sequence[int]] = None,
    ) -> None:
        """
        slider angles를 재생성.
        - seed를 주면 RNG를 재시드
        - topology_indices를 주면 해당 topology만 재생성
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if topology_indices is None:
            topology_indices = list(range(self.num_topologies))

        for ti in topology_indices:
            ti = int(ti)
            if ti < 0 or ti >= self.num_topologies:
                raise IndexError(f"topology_index out of range: {ti}")

            p_count = int(self.p_joint_indices_by_topology[ti].size)
            self._angles_by_topology[ti] = []
            for _ in range(self.num_samples_per_topology):
                if p_count == 0:
                    self._angles_by_topology[ti].append(np.zeros((0,), dtype=self.dtype))
                else:
                    angles = self.rng.uniform(
                        low=0.0, high=2.0 * np.pi, size=(p_count,)
                    ).astype(self.dtype)
                    self._angles_by_topology[ti].append(angles)

    # ---------------------------------------------------------------------
    # Internal
    # ---------------------------------------------------------------------

    def _build(self) -> None:
        # 1) topology별 P joint 목록 만들기
        self.p_joint_indices_by_topology = []
        for ti in range(self.num_topologies):
            topo_dict = self._get_topology_dict(ti)
            joint_type_list = self._extract_joint_type_list(topo_dict)

            # P joint: value == 2
            p_idx = np.where(joint_type_list == 2)[0].astype(int)
            self.p_joint_indices_by_topology.append(p_idx)

        # 2) topology별/샘플별 angles 생성
        self._angles_by_topology = []
        for ti in range(self.num_topologies):
            p_count = int(self.p_joint_indices_by_topology[ti].size)
            per_topo: List[np.ndarray] = []
            for _ in range(self.num_samples_per_topology):
                if p_count == 0:
                    per_topo.append(np.zeros((0,), dtype=self.dtype))
                else:
                    angles = self.rng.uniform(
                        low=0.0, high=2.0 * np.pi, size=(p_count,)
                    ).astype(self.dtype)
                    per_topo.append(angles)
            self._angles_by_topology.append(per_topo)

    def _validate_indices(self, topology_index: int, sample_index: int) -> None:
        if topology_index < 0 or topology_index >= self.num_topologies:
            raise IndexError(
                f"topology_index out of range: {topology_index} (0..{self.num_topologies-1})"
            )
        if sample_index < 0 or sample_index >= self.num_samples_per_topology:
            raise IndexError(
                f"sample_index out of range: {sample_index} (0..{self.num_samples_per_topology-1})"
            )

    def _get_topology_dict(self, topology_index: int) -> Dict[str, Any]:
        """
        topology_data 형태가 여러 가지일 수 있어 최대한 유연하게 접근.
        - list[dict] 인 경우: topology_data[topology_index]
        - dict 인 경우:
            - topology_data['topologies'][topology_index] 또는
            - topology_data['data'][topology_index] 등을 시도
        """
        td = self.topology_data

        if isinstance(td, list):
            if topology_index >= len(td):
                raise IndexError(
                    f"topology_data list length={len(td)}, but requested index={topology_index}"
                )
            topo_dict = td[topology_index]
            if not isinstance(topo_dict, dict):
                raise TypeError(f"topology_data[{topology_index}] is not a dict.")
            return topo_dict

        if isinstance(td, dict):
            for key in ("topologies", "data", "topology_list", "items"):
                if key in td and isinstance(td[key], list):
                    lst = td[key]
                    if topology_index >= len(lst):
                        raise IndexError(
                            f"topology_data['{key}'] length={len(lst)}, but requested index={topology_index}"
                        )
                    topo_dict = lst[topology_index]
                    if not isinstance(topo_dict, dict):
                        raise TypeError(f"topology_data['{key}'][{topology_index}] is not a dict.")
                    return topo_dict

            # 마지막 fallback: dict 자체가 단일 topology일 수도 있으니 topology_index==0만 허용
            if topology_index == 0:
                return td

        raise TypeError("Unsupported topology_data format. Must be list[dict] or dict.")

    def _extract_joint_type_list(self, topo_dict: Dict[str, Any]) -> np.ndarray:
        """
        topo_dict에서 joint_type_list를 np.ndarray(int)로 추출.
        - 키 이름: 'joint_type_list' (권장)
        """
        if "joint_type_list" not in topo_dict:
            raise KeyError(
                "topology_data must contain 'joint_type_list' for each topology."
            )

        jt = topo_dict["joint_type_list"]
        arr = np.asarray(jt, dtype=int).reshape(-1)
        return arr
