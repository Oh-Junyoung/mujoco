import numpy as np
from typing import Dict, Optional, Tuple
from topology_data_load.TopologyCalculator import TopologyCalculator


class PhysicalParametersGenerator:
    """
    각 링크의 물리적 파라미터(질량, 관성모멘트, 무게중심)를 생성하는 클래스

    파라미터 생성 규칙:
    1. Binary link: 선밀도(linear density) 기반 질량 계산
    2. Ternary+ link: 면밀도(area density) 기반 질량 계산
    3. 무게중심: 기하학적 중심 (joint 좌표의 평균)
    4. 관성모멘트: 링크 형태에 따른 표준 공식/기하 기반 계산

    단위: SI 단위계 (kg, m, kg·m²)

    ✅ 중요:
    - initial_design_generator가 반환하는 joint_coords가 'mm'일 수 있으므로,
      coord_unit에 따라 내부에서 반드시 'm'로 변환 후 계산합니다.
    """

    def __init__(
        self,
        topology_data: list,
        initial_design_generator,
        linear_density: float = 1.0,      # kg/m (binary link)
        area_density: float = 10.0,       # kg/m² (ternary+ link)
        custom_masses: Optional[Dict] = None,
        custom_inertias: Optional[Dict] = None,
        link_thickness: float = 0.05,     # m (현재 모델에서는 사용하지 않음. 보존용)
        coord_unit: str = "mm",           # ✅ 추가: design_generator 좌표 단위 ('mm' or 'm')
    ):
        """
        Args:
            topology_data: 위상 데이터 리스트
            initial_design_generator: InitialDesignVariableGenerator 인스턴스
            linear_density: 선밀도 (kg/m) - binary link용
            area_density: 면밀도 (kg/m²) - ternary+ link용
            custom_masses: 사용자 지정 질량 {(topo_idx, sample_idx, link_idx): mass}
            custom_inertias: 사용자 지정 관성모멘트 {(topo_idx, sample_idx, link_idx): inertia}
            link_thickness: 링크 두께 (보존용)
            coord_unit: initial_design_generator 좌표 단위 ('mm' or 'm')
        """
        self.topology_data = topology_data
        self.design_generator = initial_design_generator
        self.num_topologies = len(topology_data)

        # 물리 파라미터
        self.linear_density = float(linear_density)
        self.area_density = float(area_density)
        self.link_thickness = float(link_thickness)

        # 좌표 단위 변환
        coord_unit = str(coord_unit).lower().strip()
        if coord_unit not in ("mm", "m"):
            raise ValueError("coord_unit must be 'mm' or 'm'.")
        self.coord_unit = coord_unit
        self.coord_scale = 0.001 if self.coord_unit == "mm" else 1.0  # design coords -> meters

        # 사용자 커스텀 값
        self.custom_masses = custom_masses if custom_masses else {}
        self.custom_inertias = custom_inertias if custom_inertias else {}

        # 캐시
        self._params_cache = {}

    def get(self, topology_idx: int, sample_idx: int) -> Dict[str, np.ndarray]:
        """
        특정 위상의 특정 샘플에 대한 물리 파라미터 가져오기

        Returns:
            - 'masses': shape (L,) - 각 링크의 질량 (kg)
            - 'inertias': shape (L,) - 각 링크의 관성모멘트 (kg·m²)
            - 'centers_of_mass': shape (L, 2) - 각 링크의 무게중심 좌표 (m)
            - 'link_lengths': shape (L,) - binary link의 길이 (m)
        """
        cache_key = (int(topology_idx), int(sample_idx))
        if cache_key in self._params_cache:
            return self._params_cache[cache_key]

        topology = self.topology_data[int(topology_idx)]
        calc_topo = TopologyCalculator(topology)

        # 초기 좌표: design_generator는 mm일 수 있으므로 -> m로 변환
        joint_coords_raw = np.asarray(self.design_generator.get(int(topology_idx), int(sample_idx)), dtype=float)
        if joint_coords_raw.ndim != 2 or joint_coords_raw.shape[1] != 2:
            raise ValueError(f"initial joint coords expected (J,2) but got {joint_coords_raw.shape}")

        joint_coords_m = joint_coords_raw * self.coord_scale  # ✅ meters

        params = self._calculate_parameters(
            int(topology_idx),
            int(sample_idx),
            topology,
            calc_topo,
            joint_coords_m
        )

        self._params_cache[cache_key] = params
        return params

    def _calculate_parameters(
        self,
        topology_idx: int,
        sample_idx: int,
        topology: dict,
        calc_topo: TopologyCalculator,
        joint_coords_m: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """물리 파라미터 계산 (joint_coords_m는 meters)"""

        num_links = int(calc_topo.number_of_links())
        joint_connections = np.asarray(calc_topo.links_connected_by_joints_original(), dtype=int)

        # 링크 분류
        n_binary = int(topology["number_of_binary_links"])
        n_ternary = int(topology["number_of_ternary_links"])
        n_quaternary = int(topology["number_of_quaternary_links"])

        if n_binary + n_ternary + n_quaternary != num_links:
            # 데이터가 항상 이 규칙을 만족한다고 가정했지만, 방어적으로 체크
            # (불일치가 있어도 계산은 하되 타입 판별이 틀릴 수 있음)
            pass

        # 초기화
        masses = np.zeros(num_links, dtype=float)
        inertias = np.zeros(num_links, dtype=float)
        centers_of_mass = np.zeros((num_links, 2), dtype=float)
        link_lengths = np.zeros(num_links, dtype=float)

        for link_idx in range(num_links):
            # 해당 링크에 연결된 joint들 (joint index들)
            connected_joints = np.where(
                (joint_connections[:, 0] == link_idx) |
                (joint_connections[:, 1] == link_idx)
            )[0].astype(int)

            # 이 구현에서는 joint index 자체가 좌표 row index라고 가정(TopologyCalculator 정의와 일치)
            joint_indices_on_link = connected_joints.tolist()
            link_joint_coords = joint_coords_m[joint_indices_on_link] if len(joint_indices_on_link) > 0 else np.zeros((0, 2), dtype=float)
            num_joints_on_link = int(len(joint_indices_on_link))

            # 링크 타입 판별 (데이터셋의 link index ordering 가정)
            if link_idx < n_binary:
                link_type = "binary"
            elif link_idx < n_binary + n_ternary:
                link_type = "ternary"
            else:
                link_type = "quaternary"

            custom_key = (topology_idx, sample_idx, link_idx)

            # 질량
            if custom_key in self.custom_masses:
                mass = float(self.custom_masses[custom_key])
            else:
                mass = float(self._calculate_mass(link_type, link_joint_coords, num_joints_on_link))

            # 무게중심 (meters)
            com = self._calculate_center_of_mass(link_joint_coords)

            # 관성모멘트
            if custom_key in self.custom_inertias:
                inertia = float(self.custom_inertias[custom_key])
            else:
                inertia = float(self._calculate_inertia(link_type, mass, link_joint_coords, com))

            # 링크 길이 (binary)
            if link_type == "binary" and num_joints_on_link == 2:
                length = float(np.linalg.norm(link_joint_coords[1] - link_joint_coords[0]))
            else:
                length = 0.0

            masses[link_idx] = mass
            inertias[link_idx] = inertia
            centers_of_mass[link_idx] = com
            link_lengths[link_idx] = length

        return {
            "masses": masses,
            "inertias": inertias,
            "centers_of_mass": centers_of_mass,
            "link_lengths": link_lengths,
        }

    def _calculate_mass(
        self,
        link_type: str,
        joint_coords_m: np.ndarray,
        num_joints: int
    ) -> float:
        """
        링크 질량 계산 (SI: kg)

        - binary: m = linear_density [kg/m] * length [m]
        - ternary/quaternary: m = area_density [kg/m^2] * area [m^2]
        """
        if link_type == "binary":
            if num_joints >= 2:
                length = float(np.linalg.norm(joint_coords_m[1] - joint_coords_m[0]))
                return self.linear_density * length
            return self.linear_density * 1.0  # fallback (1m)

        # ternary/quaternary
        area = float(self._calculate_polygon_area(joint_coords_m))
        return self.area_density * area

    def _calculate_polygon_area(self, vertices_m: np.ndarray) -> float:
        """
        다각형 면적 계산 (Shoelace formula) - 입력은 meters, 출력 m^2

        주의:
        - vertices 순서가 실제 폴리곤 순서가 아니면 면적이 의미 없을 수 있음.
          (현재는 joint index 순서 그대로 사용)
        """
        if vertices_m.shape[0] < 3:
            return 0.0

        x = vertices_m[:, 0]
        y = vertices_m[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        return float(area)

    def _calculate_center_of_mass(self, joint_coords_m: np.ndarray) -> np.ndarray:
        """
        무게중심 계산 (기하학적 중심 = joint 좌표 평균), meters
        """
        if joint_coords_m.shape[0] == 0:
            return np.zeros((2,), dtype=float)
        return np.mean(joint_coords_m, axis=0)

    def _calculate_inertia(
        self,
        link_type: str,
        mass: float,
        joint_coords_m: np.ndarray,
        com_m: np.ndarray
    ) -> float:
        """
        관성모멘트 계산 (kg·m²)

        - binary: slender rod about COM, perpendicular to plane:
            I = (1/12) m L^2
        - ternary/quaternary: 현재 구현은 "점 집합" 기반의 면내 분포 근사:
            I ≈ m * mean(||p_i - com||^2)
          (입력 좌표 단위가 m로 맞아야 스케일이 정상)
        """
        if link_type == "binary":
            if joint_coords_m.shape[0] >= 2:
                length = float(np.linalg.norm(joint_coords_m[1] - joint_coords_m[0]))
                return (1.0 / 12.0) * mass * (length ** 2)
            return 0.0

        if joint_coords_m.shape[0] == 0:
            return 0.0

        distances_sq = np.sum((joint_coords_m - com_m) ** 2, axis=1)
        avg_radius_sq = float(np.mean(distances_sq))
        return mass * avg_radius_sq

    def set_custom_mass(self, topology_idx: int, sample_idx: int, link_idx: int, mass: float):
        key = (int(topology_idx), int(sample_idx), int(link_idx))
        self.custom_masses[key] = float(mass)
        cache_key = (int(topology_idx), int(sample_idx))
        if cache_key in self._params_cache:
            del self._params_cache[cache_key]

    def set_custom_inertia(self, topology_idx: int, sample_idx: int, link_idx: int, inertia: float):
        key = (int(topology_idx), int(sample_idx), int(link_idx))
        self.custom_inertias[key] = float(inertia)
        cache_key = (int(topology_idx), int(sample_idx))
        if cache_key in self._params_cache:
            del self._params_cache[cache_key]

    def clear_cache(self):
        self._params_cache.clear()

    def info(self):
        print("=" * 60)
        print("Physical Parameters Generator Info")
        print("=" * 60)
        print(f"총 위상 개수: {self.num_topologies}")
        print(f"\n물리 파라미터:")
        print(f"  선밀도 (binary): {self.linear_density} kg/m")
        print(f"  면밀도 (ternary+): {self.area_density} kg/m²")
        print(f"  링크 두께(보존용): {self.link_thickness} m")
        print(f"  design coords unit: {self.coord_unit} -> internal meters scale={self.coord_scale}")
        print(f"\n커스텀 값:")
        print(f"  커스텀 질량: {len(self.custom_masses)}개")
        print(f"  커스텀 관성모멘트: {len(self.custom_inertias)}개")
        print(f"  캐시된 샘플: {len(self._params_cache)}개")
        print("=" * 60)
