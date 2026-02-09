import numpy as np
from typing import Tuple


class GenerationStrategies:
    """
    초기 설계 변수(joint 좌표) 생성을 위한 다양한 전략 모음
    
    각 전략은 (num_joints, rng) -> np.ndarray(J, 2) 형태의 함수를 반환합니다.
    
    사용 예:
        from GenerationStrategies import GenerationStrategies
        from InitialDesignVariableGenerator import InitialDesignVariableGenerator
        
        # 균등 분포 난수 전략
        strategy = GenerationStrategies.uniform_random(coord_range=(-100, 100))
        generator = InitialDesignVariableGenerator(
            data,
            generation_strategy=strategy
        )
        
        # 원형 배치 전략
        strategy = GenerationStrategies.circular_arrangement(radius=20.0)
        generator.set_generation_strategy(strategy)
    """
    
    @staticmethod
    def uniform_random(coord_range: Tuple[float, float] = (-50.0, 50.0)):
        """
        균등 분포 난수 생성 전략
        
        Args:
            coord_range: 좌표 범위 (min, max)
            
        Returns:
            생성 전략 함수
            
        예시:
            strategy = GenerationStrategies.uniform_random((-100, 100))
            coords = strategy(num_joints=4, rng=np.random.default_rng(42))
            # coords.shape = (4, 2)
        """
        def strategy(num_joints: int, rng: np.random.Generator) -> np.ndarray:
            min_val, max_val = coord_range
            return rng.uniform(min_val, max_val, size=(num_joints, 2))
        
        return strategy
    
    @staticmethod
    def normal_random(mean: float = 0.0, std: float = 20.0):
        """
        정규 분포 난수 생성 전략
        
        Args:
            mean: 평균값
            std: 표준편차
            
        Returns:
            생성 전략 함수
            
        예시:
            strategy = GenerationStrategies.normal_random(mean=0, std=15)
            coords = strategy(num_joints=4, rng=np.random.default_rng(42))
        """
        def strategy(num_joints: int, rng: np.random.Generator) -> np.ndarray:
            return rng.normal(mean, std, size=(num_joints, 2))
        
        return strategy
    
    @staticmethod
    def circular_arrangement(
        radius: float = 10.0, 
        center: Tuple[float, float] = (0.0, 0.0),
        noise_level: float = 0.0
    ):
        """
        원형 배치 생성 전략
        
        Joint들을 원 위에 균등하게 배치하고 선택적으로 노이즈 추가
        
        Args:
            radius: 원의 반지름
            center: 원의 중심 좌표 (x, y)
            noise_level: 노이즈 크기 (0이면 노이즈 없음)
            
        Returns:
            생성 전략 함수
            
        예시:
            # 반지름 15, 중심 (0,0), 약간의 노이즈
            strategy = GenerationStrategies.circular_arrangement(
                radius=15.0, 
                noise_level=2.0
            )
        """
        def strategy(num_joints: int, rng: np.random.Generator) -> np.ndarray:
            # 균등하게 각도 분배
            angles = np.linspace(0, 2*np.pi, num_joints, endpoint=False)
            
            if noise_level > 0:
                # 각도에 노이즈 추가
                angle_noise = noise_level / radius  # 상대적 노이즈
                angles += rng.uniform(-angle_noise, angle_noise, num_joints)
            
            # 원형 좌표 계산
            x = center[0] + radius * np.cos(angles)
            y = center[1] + radius * np.sin(angles)
            coords = np.column_stack([x, y])
            
            if noise_level > 0:
                # 위치에 직접 노이즈 추가
                position_noise = rng.uniform(
                    -noise_level, 
                    noise_level, 
                    size=(num_joints, 2)
                )
                coords += position_noise
            
            return coords
        
        return strategy
    
    @staticmethod
    def grid_based(spacing: float = 10.0, noise_level: float = 2.0):
        """
        격자 기반 배치 생성 전략
        
        Joint들을 격자 형태로 배치하고 노이즈 추가
        
        Args:
            spacing: 격자 간격
            noise_level: 노이즈 크기
            
        Returns:
            생성 전략 함수
            
        예시:
            strategy = GenerationStrategies.grid_based(spacing=15.0, noise_level=3.0)
        """
        def strategy(num_joints: int, rng: np.random.Generator) -> np.ndarray:
            # 정사각형에 가까운 격자 생성
            grid_size = int(np.ceil(np.sqrt(num_joints)))
            
            coords = []
            for i in range(num_joints):
                row = i // grid_size
                col = i % grid_size
                
                x = col * spacing
                y = row * spacing
                
                # 노이즈 추가
                if noise_level > 0:
                    x += rng.uniform(-noise_level, noise_level)
                    y += rng.uniform(-noise_level, noise_level)
                
                coords.append([x, y])
            
            return np.array(coords)
        
        return strategy
    
    @staticmethod
    def constrained_random(
        min_distance: float = 5.0,
        coord_range: Tuple[float, float] = (-50.0, 50.0),
        max_attempts: int = 1000
    ):
        """
        최소 거리 제약을 가진 난수 생성 전략
        
        Joint들 사이에 최소 거리를 유지하도록 생성
        
        Args:
            min_distance: joint 간 최소 거리
            coord_range: 좌표 범위 (min, max)
            max_attempts: 각 joint당 최대 시도 횟수
            
        Returns:
            생성 전략 함수
            
        예시:
            # 최소 8.0 거리 유지
            strategy = GenerationStrategies.constrained_random(
                min_distance=8.0,
                coord_range=(-100, 100)
            )
        """
        def strategy(num_joints: int, rng: np.random.Generator) -> np.ndarray:
            min_val, max_val = coord_range
            coords = []
            
            for i in range(num_joints):
                attempts = 0
                while attempts < max_attempts:
                    # 랜덤 좌표 생성
                    new_coord = rng.uniform(min_val, max_val, 2)
                    
                    # 첫 번째 좌표는 무조건 추가
                    if i == 0:
                        coords.append(new_coord)
                        break
                    
                    # 기존 좌표들과의 거리 확인
                    distances = np.linalg.norm(
                        np.array(coords) - new_coord, 
                        axis=1
                    )
                    
                    if np.all(distances >= min_distance):
                        coords.append(new_coord)
                        break
                    
                    attempts += 1
                
                # 최대 시도 횟수 초과 시 경고하고 그냥 추가
                if attempts >= max_attempts:
                    print(f"Warning: joint {i}에 대해 최소 거리 제약을 만족하지 못했습니다.")
                    coords.append(rng.uniform(min_val, max_val, 2))
            
            return np.array(coords)
        
        return strategy
    
    @staticmethod
    def line_arrangement(
        start: Tuple[float, float] = (-20.0, 0.0),
        end: Tuple[float, float] = (20.0, 0.0),
        noise_level: float = 0.0
    ):
        """
        직선 배치 생성 전략
        
        Joint들을 직선 위에 균등하게 배치
        
        Args:
            start: 시작 좌표 (x, y)
            end: 끝 좌표 (x, y)
            noise_level: 수직 방향 노이즈 크기
            
        Returns:
            생성 전략 함수
            
        예시:
            strategy = GenerationStrategies.line_arrangement(
                start=(-30, 0),
                end=(30, 0),
                noise_level=2.0
            )
        """
        def strategy(num_joints: int, rng: np.random.Generator) -> np.ndarray:
            # 직선 위에 균등 분포
            t = np.linspace(0, 1, num_joints)
            
            x = start[0] + t * (end[0] - start[0])
            y = start[1] + t * (end[1] - start[1])
            coords = np.column_stack([x, y])
            
            if noise_level > 0:
                # 직선에 수직 방향으로 노이즈 추가
                dx = end[0] - start[0]
                dy = end[1] - start[1]
                length = np.sqrt(dx**2 + dy**2)
                
                # 수직 벡터 (normalized)
                perp_x = -dy / length
                perp_y = dx / length
                
                # 노이즈 생성
                noise_mag = rng.uniform(-noise_level, noise_level, num_joints)
                
                coords[:, 0] += noise_mag * perp_x
                coords[:, 1] += noise_mag * perp_y
            
            return coords
        
        return strategy
    
    @staticmethod
    def polygon_arrangement(
        num_sides: int = 4,
        radius: float = 10.0,
        center: Tuple[float, float] = (0.0, 0.0)
    ):
        """
        정다각형 배치 생성 전략
        
        Joint들을 정다각형의 꼭짓점에 배치
        
        Args:
            num_sides: 다각형 변의 개수
            radius: 외접원 반지름
            center: 중심 좌표 (x, y)
            
        Returns:
            생성 전략 함수
            
        예시:
            # 정육각형 배치
            strategy = GenerationStrategies.polygon_arrangement(
                num_sides=6,
                radius=12.0
            )
        """
        def strategy(num_joints: int, rng: np.random.Generator) -> np.ndarray:
            if num_joints <= num_sides:
                # joint 수가 적으면 다각형의 일부만 사용
                angles = np.linspace(0, 2*np.pi, num_sides, endpoint=False)[:num_joints]
            else:
                # joint 수가 많으면 다각형을 여러 겹으로
                base_angles = np.linspace(0, 2*np.pi, num_sides, endpoint=False)
                num_layers = int(np.ceil(num_joints / num_sides))
                
                angles = []
                radii = []
                for layer in range(num_layers):
                    layer_radius = radius * (1 + layer * 0.5)
                    for angle in base_angles:
                        if len(angles) < num_joints:
                            angles.append(angle)
                            radii.append(layer_radius)
                
                angles = np.array(angles)
                radii = np.array(radii)
                
                x = center[0] + radii * np.cos(angles)
                y = center[1] + radii * np.sin(angles)
                
                return np.column_stack([x, y])
            
            # 단일 레이어
            x = center[0] + radius * np.cos(angles)
            y = center[1] + radius * np.sin(angles)
            
            return np.column_stack([x, y])
        
        return strategy