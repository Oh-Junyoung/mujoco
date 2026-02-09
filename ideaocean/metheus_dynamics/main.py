# main.py
from topology_data_load.TopologyDataLoader import TopologyDataLoader
from topology_data_load.TopologyCalculator import TopologyCalculator
from model_generator.InitialDesignVariableGenerator import InitialDesignVariableGenerator
from model_generator.InputMotionGenerator import InputMotionGenerator
from model_generator.PhysicalParametersGenerator import PhysicalParametersGenerator
from dynamic_simulator.DynamicsSimulator import DynamicsSimulator
from model_generator.GenerationStrategies import GenerationStrategies
from post_process.MechanismAnimator import MechanismAnimator
from model_generator.SliderDirectionGenerator import SliderDirectionGenerator
from post_process.export_simulation_results_to_csv import export_simulation_results_to_csv

import numpy as np

## -------------------------------------------------------------------------------------------------------------------------
# 기구 위상 정보 로드
## -------------------------------------------------------------------------------------------------------------------------

# (Fixed) 스크립트 실행 위치와 무관하게 ./data 폴더를 찾을 수 있도록 절대 경로로 변환
import os
from pathlib import Path

current_dir = Path(__file__).parent
data_folder = current_dir / "data"

topology_loader         = TopologyDataLoader(data_folder=data_folder)   # 위상 데이터 셋을 불러오는 인스턴스 생성
data                    = topology_loader.load()    # 위상 데이터 로드
number_of_topologies    = 4                         # (Modified) 테스트를 위해 위상 개수를 4개로 제한 

## -------------------------------------------------------------------------------------------------------------------------
# 초기 설계 변수 생성기 초기화
## -------------------------------------------------------------------------------------------------------------------------
"""
전체 위상에 대한 초기 좌표 셋을 미리 생성
- 각 위상당 num_samples_per_topology 개의 좌표 셋 생성
- 생성 전략 선택 가능 (기본: 균등 분포 난수)
"""

num_samples_per_topology = 1  # 각 위상당 샘플 개수
seed = 420
# seed = 1

# 기본 전략 (균등 분포 난수)
initial_design_generator = InitialDesignVariableGenerator(
    topology_data=data,
    num_samples_per_topology=num_samples_per_topology,  # 각 위상당 100개 샘플
    coord_range=(-50.0, 50.0),
    seed=seed  # 재현성을 위한 시드
)

# 또는 다른 전략 사용 예시:
# initial_design_generator = InitialDesignVariableGenerator(
#     topology_data=data,
#     num_samples_per_topology=100,
#     generation_strategy=GenerationStrategies.circular_arrangement(radius=20.0, noise_level=3.0),
#     seed=seed
# )

# 입력 모션 생성기 (동일한 샘플 개수로)
# (NEW) PMKS+처럼 "모터가 일정 rpm으로 구동"한다고 가정하려면 total_time 또는 rpm을 명시해주는 것이 안전하다.
# - 예: 0~2π 한 바퀴를 5.76초에 돌면 rpm = 60/5.76
total_time_for_one_rev = 3
input_rpm = 20

input_motion_generator = InputMotionGenerator(
    num_topologies=number_of_topologies,
    num_samples_per_topology=num_samples_per_topology,
    num_steps=73,  # 0 ~ 2π를 72등분

    # (NEW) 상수 각속도 입력 옵션
    total_time=total_time_for_one_rev,
    input_rpm=input_rpm,
)

# 물리 파라미터 생성기
physical_params_generator = PhysicalParametersGenerator(
    topology_data=data,
    initial_design_generator=initial_design_generator,
    linear_density=2.5,    # kg/m (binary link)
    area_density=10.0,     # kg/m² (ternary+ link)
    link_thickness=0.05    # m (면적 계산용)
)

# P joint slider 방향 생성기 (topology/sample마다 P joint 방향각을 [0, 2π)로 난수 생성)
# 초기 설계 변수 랜덤으로 뿌리는 기능에 포함시킬 수도 있지 않을까?
slider_direction_generator = SliderDirectionGenerator(
    topology_data=data,
    num_topologies=number_of_topologies,
    num_samples_per_topology=num_samples_per_topology,
    seed=seed,        # 재현 가능하게 하려면 고정 seed 권장
)

## -------------------------------------------------------------------------------------------------------------------------
# 한 위상에 대한 기본 정보 추출, 부가 정보 계산 및 동역학 분석 시행
## -------------------------------------------------------------------------------------------------------------------------
for topology_index in range(number_of_topologies):
    topology            = data[topology_index]          # 한 인덱스에 대한 위상 호출
    calculated_topology = TopologyCalculator(topology)  # 위상 부가 정보 계산을 위한 인스턴스 생성
    
    ## -------------------------------------------------------------------------------------------------------------------------
    # 위상의 기본 정보 추출
    ## -------------------------------------------------------------------------------------------------------------------------
    
    """
    [위상 기본 정보]

    기구 위상(topology)의 구조를 정의하는 기본 정보.
    이 정보들은 pkl 파일에서 직접 로드되며, 파생 정보 계산의 기초.

    -------------------------------------------------------------------------
    1. number_of_binary_links (int)
    - 2개의 joint를 가진 링크(직선형 링크)의 개수
    - 기구학에서 binary link는 두 개의 연결점(joint)만 가지는 링크
    - Grounded non-isomorphic topology 생성 시 계수법(enumeration)의 기준
    - 예: 4-bar 기구는 4개의 binary link로 구성
    - 참고: Tsai (2001) - Enumeration of Kinematic Structures

    2. number_of_ternary_links (int)
    - 3개의 joint를 가진 링크(삼각형 링크)의 개수
    - 복잡한 기구에서 분기점(branch point) 역할
    - 예: Stephenson 6-bar 기구는 1개의 ternary link 포함
    - 참고: Sharma et al. (2014) - Path Matrix method

    3. number_of_quaternary_links (int)
    - 4개의 joint를 가진 링크(사각형 링크)의 개수
    - 더 복잡한 기구 위상에서 사용 (8-bar 이상)
    - 예: 특정 8-bar 기구에서 사용
    
    ※ 위 1~3번은 kinematic chain의 구조적 분류 기준입니다.
        - 전체 링크 개수 = binary + ternary + quaternary
        - Grounded non-isomorphic topology 생성 시 이 조합으로 위상 구분
        - 참고: Soh & McCarthy (2007) - "Synthesis of Eight-Bar Linkages"
                위키피디아 - Eight-bar linkage enumeration

    4. number_of_joints_of_ground_link (int)
    - ground link(고정 링크)에 연결된 joint의 개수
    - ground link = 기구의 프레임, 고정된 기준 링크
    - 일반적으로 2개 이상 (4-bar: 2개, 6-bar: 2-3개)
    - Mobility 계산 및 inversion 분석에 중요한 파라미터

    5. index_of_ground_link (int)
    - ground link의 인덱스 번호 (0-based indexing)
    - 모든 운동학적 계산의 기준점(reference frame)
    - Inversion 시 서로 다른 링크를 ground로 고정하여 다른 기구 생성
    - 예: 0 (일반적으로 첫 번째 링크)

    6. adjacency_matrix (numpy.ndarray, shape: (L, L), dtype: int)
    - 링크 간 연결 관계를 나타내는 인접 행렬 (link-link adjacency)
    - adjacency_matrix[i][j] ≠ 0 ⟺ 링크 i와 링크 j 사이에 joint 존재
    - 그래프 이론에서 undirected graph로 표현
    - 특성:
        * 대칭 행렬 (symmetric): A[i][j] = A[j][i]
        * 대각선 = 0 (no self-loops): A[i][i] = 0
    - 예: Watt chain (6-bar)
            [[0,1,0,1,0,1],
            [1,0,1,0,0,0],
            [0,1,0,1,0,0],
            [1,0,1,0,1,0],
            [0,0,0,1,0,1],
            [1,0,0,0,1,0]]
    - 참고: Sharma et al. (2014) - Isomorphism detection using adjacency

    7. index_of_input_link (int)
    - 구동(actuated) 링크의 인덱스
    - 모터/액츄에이터가 연결되어 회전/이동을 발생시키는 링크
    - ground-input joint가 구동 joint (driving joint)
    - DOF = 1인 경우 하나의 input으로 전체 기구 구동
    - 예: 1

    8. list_of_end_effector_links (numpy.ndarray, shape: (K,), dtype: int)
    - End effector 부착 가능한 링크들의 인덱스 배열
    - End effector = 작업 도구, 기구의 출력단
    - Function generation, path generation에 따라 선택
    - 여러 링크가 후보가 될 수 있음 (다중 출력 가능)
    - 예: array([2, 3])

    9. list_of_rockers (numpy.ndarray, shape: (M,), dtype: int)
    - Rocker가 될 수 있는 링크들의 인덱스 배열
    - Rocker = 제한된 각도 범위 내에서 왕복 회전하는 링크
    - Crank-rocker 기구, double-rocker 기구 설계 시 중요
    - Grashof 조건과 밀접한 관련
    - 예: array([2]) → 링크 2가 rocker

    10. list_of_joint_type (numpy.ndarray, shape: (J,), dtype: int)
        - 각 joint의 타입 정보 배열
        - 1 = R joint (Revolute joint, 회전 관절)
        - 2 = P joint (Prismatic joint, 직선 관절/슬라이더)
        - Joint 순서는 adjacency matrix의 edge 순서와 일치
        - 예: array([1, 1, 2, 1]) → 0,1,3번 R joint, 2번 P joint
        
        ※ Joint index 정의:
        adjacency matrix에서 (i < j)인 edge들을 추출하여
        (i, j) 순으로 정렬한 순서가 joint index입니다.
        
        예시:
        adjacency_matrix가 링크 0-1, 1-2, 2-3, 0-3을 연결
        → edges = [(0,1), (0,3), (1,2), (2,3)]  (정렬됨)
        → joint 0 = (0,1), joint 1 = (0,3), 
            joint 2 = (1,2), joint 3 = (2,3)

    -------------------------------------------------------------------------
    [자료형 요약]
    - int (스칼라): 1, 2, 3, 4, 5, 7
    - numpy.ndarray (행렬/배열): 6, 8, 9, 10

    [위상 분류 체계]
    본 데이터는 grounded non-isomorphic kinematic chain의 분류를 따릅니다:
    - Non-isomorphic: 구조적으로 동일하지 않은(서로 다른) 위상
    - Grounded: ground link가 지정된 상태
    - 링크 조합(binary/ternary/quaternary)으로 위상 구분

    [참고 문헌]
    - Tsai, L.W. (2001). "Enumeration of Kinematic Structures According 
    to Function", CRC Press
    - Soh, G.S. & McCarthy, J.M. (2007). "Synthesis of Eight-Bar Linkages 
    as Mechanically Constrained Parallel Robots", IFToMM World Congress
    - Sharma, A.K. et al. (2014). "An Evolution of New Methodology for 
    the Detection of Isomorphism", IJERT
    - Eight-bar linkage: https://en.wikipedia.org/wiki/Eight-bar_linkage
    """
    
    number_of_binary_links          = topology["number_of_binary_links"]            
    number_of_ternary_links         = topology["number_of_ternary_links"]           
    number_of_quaternary_links      = topology["number_of_quaternary_links"]        
    number_of_joints_of_ground_link = topology["number_of_joints_of_ground_link"]   
    index_of_ground_link            = topology["index_of_ground_link"]              
    adjacency_matrix                = topology["array_of_adjacency_matrices"]       
    index_of_input_link             = topology["input_link_index"]                  
    list_of_end_effector_links      = topology["end_effector_link_list"]            
    list_of_rockers                 = topology["rocker_list"]                       
    list_of_joint_type              = topology["joint_type_list"] 
                      

    ## -------------------------------------------------------------------------------------------------------------------------
    # 위상에 대한 부가 정보 계산
    ## -------------------------------------------------------------------------------------------------------------------------
    
    """
    [위상 부가 정보 계산 프로세스]

    기본 위상 정보로부터 파생되는 정보들을 계산하는 과정.
    각 계산은 adjacency matrix와 joint type 정보를 기반으로 수행.

    -------------------------------------------------------------------------

    1. number_of_links (int)
    [계산 방법]
    - binary_links + ternary_links + quaternary_links의 단순 합산
    
    [수식]
    L = L_binary + L_ternary + L_quaternary
    
    [예시]
    - 6-bar Stephenson chain: 4 + 2 + 0 = 6개 링크
    
    [의미]
    - 기구를 구성하는 전체 링크(강체) 개수
    - Pair 적용 전, 물리적으로 존재하는 링크만 계산

    -------------------------------------------------------------------------

    2. number_of_joints (int)
    [계산 방법]
    - adjacency matrix에서 (i < j)인 모든 edge 추출
    - edge를 (i, j) 순으로 정렬하여 joint index 부여
    
    [알고리즘]
    1) adjacency matrix를 순회
    2) A[i][j] ≠ 0 이고 i < j인 모든 (i,j) 쌍 추출
    3) (i,j) 사전순으로 정렬
    4) 정렬된 순서 = joint index
    
    [예시]
    adjacency matrix:
    [[0,1,0,1],
        [1,0,1,0],
        [0,1,0,1],
        [1,0,1,0]]
    
    → edges: (0,1), (0,3), (1,2), (2,3)  [정렬됨]
    → joint 0=(0,1), joint 1=(0,3), joint 2=(1,2), joint 3=(2,3)
    → number_of_joints = 4
    
    [의미]
    - 기구의 자유도 계산에 사용: DOF = 3(L-1) - 2J (평면 기구)
    - Joint index 순서는 이후 모든 계산의 기준

    -------------------------------------------------------------------------

    3. pair (numpy.ndarray, shape: (J,), dtype: int)
    [계산 방법]
    - joint_type_list를 기반으로 각 joint의 pair 값 결정
    - P joint에 대해서만 복제 joint index 할당
    
    [알고리즘]
    1) ground-input joint 찾기 → 무조건 pair[j] = 0
    2) 나머지 joint 순회:
        - joint_type_list[j] == 1 (R) → pair[j] = 0
        - joint_type_list[j] == 2 (P) → pair[j] = J + count_P
    3) count_P를 증가시켜 다음 P joint의 복제 index 생성
    
    [출력 형식]
    - pair[j] == 0: j번 joint는 R joint (회전)
    - pair[j] == k (k ≥ J): j번 joint는 P joint, k는 복제 joint index
    
    [예시]
    joint_type_list = [1, 2, 1, 1]  (J=4)
    ground=0, input=1 → ground-input joint = joint 0
    
    계산 과정:
    - joint 0: ground-input → pair[0] = 0
    - joint 1: type=2 (P) → pair[1] = 4  (첫 P joint)
    - joint 2: type=1 (R) → pair[2] = 0
    - joint 3: type=1 (R) → pair[3] = 0
    
    결과: pair = [0, 4, 0, 0]
    
    [의미]
    - P joint를 두 개의 R joint로 변환하는 메커니즘
    - 원래 P joint는 ground에 고정, 복제 joint로 운동 전달

    -------------------------------------------------------------------------

    4. links_connected_by_joints_original (numpy.ndarray, shape: (J, 2), dtype: int)
    [계산 방법]
    - adjacency matrix의 edge들을 2D 배열로 변환
    - 각 row = [link_i, link_j]
    
    [알고리즘]
    1) _joint_edges()로 edge 리스트 추출 (정렬됨)
    2) List[Tuple[int,int]]를 numpy array로 변환
    
    [출력 형식]
    - row j: j번 joint가 연결하는 두 링크의 인덱스
    - 예: row 0 = [0, 1] → joint 0이 link 0과 link 1 연결
    
    [예시]
    edges = [(0,1), (0,3), (1,2), (2,3)]
    
    결과:
    [[0, 1],   ← joint 0: link 0 ↔ link 1
        [0, 3],   ← joint 1: link 0 ↔ link 3
        [1, 2],   ← joint 2: link 1 ↔ link 2
        [2, 3]]   ← joint 3: link 2 ↔ link 3
    
    [의미]
    - Joint-link 연결 관계의 명시적 표현
    - Pair 적용 전 원본 위상 구조

    -------------------------------------------------------------------------

    5. links_connected_by_joints_after_pair (numpy.ndarray, shape: (J+P, 2), dtype: int)
    [계산 방법]
    - P joint에 대해 가상 링크(virtual link) 생성 및 재배치
    - R joint는 원본 유지
    
    [알고리즘]
    1) pair 배열에서 P joint 인덱스들 추출 (pair[j] ≠ 0)
    2) 각 P joint j에 대해:
        a) 가상 링크 v 생성 (v = -1, -2, -3, ...)
        b) edges[j]에서 ground와 연결 확인
        c) 원래 joint j: [v, ground]로 수정
        d) 복제 joint dup: [v, other] 추가 (other = ground 반대편)
    3) 최종 shape = (J + num_P, 2)
    
    [가상 링크 개념]
    - 음수 인덱스 (-1, -2, ...): P joint의 slider를 표현
    - 각 P joint마다 고유한 가상 링크 생성
    - 원래 joint와 복제 joint가 같은 가상 링크를 공유
    
    [출력 형식]
    - row 0~(J-1): original joint (R은 유지, P는 수정)
    - row J~(J+P-1): 복제된 P joint들
    - 음수 값 = 가상 링크 인덱스
    
    [예시]
    Original: [[0,1], [1,2], [2,3], [0,3]]
    pair = [0, 4, 0, 0]  (joint 1이 P joint)
    ground = 0
    
    계산 과정:
    - joint 1: edges[1] = [1,2], ground=0과 무관 → 에러!
    
    올바른 예시 (joint 1이 ground 연결):
    Original: [[0,1], [0,2], [1,3], [2,3]]
    pair = [0, 4, 0, 0]  (joint 1이 P joint)
    ground = 0
    
    joint 1 처리:
    - edges[1] = [0,2], ground=0 포함
    - other = 2
    - v = -1 (첫 번째 가상 링크)
    - joint 1 수정: [0,2] → [-1,0]
    - joint 4 추가: [-1,2]
    
    결과:
    [[0, 1],    ← joint 0 (R, 유지)
        [-1, 0],   ← joint 1 (P, 수정: 가상↔ground)
        [1, 3],    ← joint 2 (R, 유지)
        [2, 3],    ← joint 3 (R, 유지)
        [-1, 2]]   ← joint 4 (복제: 가상↔other)
    
    [의미]
    - P joint를 kinematically equivalent한 RR chain으로 변환
    - 가상 링크로 slider 운동 표현

    -------------------------------------------------------------------------

    6. list_of_joints_of_ground_link (numpy.ndarray, shape: (K,), dtype: int)
    [계산 방법]
    - links_connected_by_joints_original에서 ground 포함 joint 필터링
    
    [알고리즘]
    1) ground_idx 가져오기
    2) edges에서 각 row 검사
    3) edges[j,0]==ground OR edges[j,1]==ground인 j 수집
    
    [출력 형식]
    - 1D 배열, 각 원소 = ground에 연결된 joint의 index
    
    [예시]
    edges = [[0,1], [0,3], [1,2], [2,3]]
    ground = 0
    
    검사:
    - joint 0: [0,1] → 0 포함 ✓
    - joint 1: [0,3] → 0 포함 ✓
    - joint 2: [1,2] → 0 없음 ✗
    - joint 3: [2,3] → 0 없음 ✗
    
    결과: [0, 1]
    
    [의미]
    - Ground link의 자유도 계산
    - Inversion 분석 시 필요 (ground 변경 가능성 판단)

    -------------------------------------------------------------------------

    7. number_of_joints_after_pair (int)
    [계산 방법]
    - links_connected_by_joints_after_pair의 row 개수
    
    [수식]
    J_after = J + P
    (J: original joints, P: P joint 개수)
    
    [예시]
    - Original: 4개 joint
    - P joint 1개
    - After: 4 + 1 = 5개
    
    [의미]
    - 확장된 위상의 전체 joint 개수

    -------------------------------------------------------------------------

    8. number_of_links_including_virtual_links (int)
    [계산 방법]
    - original link 개수 + 생성된 virtual link 개수
    
    [알고리즘]
    1) edges_after = links_connected_by_joints_after_pair
    2) virtual_links = edges_after에서 음수 값만 추출
    3) unique_virtuals = 중복 제거
    4) result = L_original + len(unique_virtuals)
    
    [출력 형식]
    - int: 가상 링크 포함 총 링크 수
    
    [예시]
    Original links: 4개
    edges_after: [[-1,0], [1,2], [2,3], [0,3], [-1,2]]
    
    음수 값 추출: -1, -1
    중복 제거: -1 (1개)
    
    결과: 4 + 1 = 5개
    
    복수 P joint 예시:
    edges_after: [[-1,0], [-2,0], [1,2], [-1,1], [-2,2]]
    음수 값: -1, -2, -1, -2
    중복 제거: -1, -2 (2개)
    결과: L_original + 2
    
    [의미]
    - 확장된 위상의 전체 링크 개수
    - 질량 행렬, 관성 행렬 크기 결정

    -------------------------------------------------------------------------

    [핵심 개념 정리]

    1. Joint Index 규칙
    - adjacency matrix의 (i<j) edge를 (i,j) 사전순 정렬
    - 이 순서가 모든 계산의 기준

    2. Pair 메커니즘
    - P joint → 2개의 R joint로 분해 (original + duplicate)
    - Virtual link로 slider 운동 표현
    - Ground 연결 필수 조건

    3. Virtual Link 표기
    - 음수 인덱스: -1, -2, -3, ...
    - 각 P joint마다 고유한 가상 링크
    - Original과 duplicate joint가 공유

    4. 자료형 일관성
    - 개수: int
    - 인덱스 배열: numpy.ndarray (1D)
    - 연결 정보: numpy.ndarray (2D, shape: (N,2))
    """
    
    number_of_links                         = calculated_topology.number_of_links()                             
    number_of_joints                        = calculated_topology.number_of_joints()                            
    pair                                    = calculated_topology.pair()                                        
    links_connected_by_joints_original      = calculated_topology.links_connected_by_joints_original()         
    links_connected_by_joints_after_pair    = calculated_topology.links_connected_by_joints_after_pair()        
    list_of_joints_of_ground_link           = calculated_topology.joints_list_of_ground_link()                 
    number_of_joints_after_pair             = calculated_topology.number_of_joints_after_pair()                 
    number_of_links_including_virtual_links = calculated_topology.number_of_links_including_virtual_links()

    ## -------------------------------------------------------------------------------------------------------------------------
    # 초기 설계 변수 가져오기
    ## -------------------------------------------------------------------------------------------------------------------------

    number_of_initial_design_samples = initial_design_generator.num_samples

    for sample_index in range(number_of_initial_design_samples):
        
        # 초기 좌표

        initial_design_generator.set_manual_coordinates_for_case_4bar_rrrr(
            topology_idx=0,
            sample_idx=0,
            J2=(0.0, 400.0),
            J0=(0.0, 0.0),
            J3=(918.3281, 795.8201),
            J1=(1000.0, 0.0),
        )
        initial_coordinates = initial_design_generator.get(topology_index, sample_index)

        print(initial_coordinates)

        # P joint slider 방향각들 (0 ~ 2π, P joint 개수만큼)
        slider_angles = slider_direction_generator.get(topology_index, sample_index)
        
        # 입력 각도 (0 ~ 2π, 72 스텝)
        input_angles = input_motion_generator.get(topology_index, sample_index)
        
        # 물리 파라미터
        physical_params = physical_params_generator.get(topology_index, sample_index)
        
        """
        physical_params 구조:
        {
            'masses': np.ndarray shape (L,) - 각 링크의 질량 (kg)
            'inertias': np.ndarray shape (L,) - 각 링크의 관성모멘트 (kg·m²)
            'centers_of_mass': np.ndarray shape (L, 2) - 각 링크의 무게중심 좌표 (m)
            'link_lengths': np.ndarray shape (L,) - binary link의 길이 (m)
        }
        """
        
        link_masses = physical_params['masses']
        link_inertias = physical_params['inertias']
        link_coms = physical_params['centers_of_mass']
        link_lengths = physical_params['link_lengths']

        # (NEW) 상수 rpm 입력을 쓸 때, DynamicsSimulator에도 input_rpm을 넘겨서 theta_dot/theta_ddot을 일관되게 만든다.
        # 또한 total_time을 맞추기 위해 time_step도 함께 지정한다.
        dt_for_one_rev = total_time_for_one_rev / max(len(input_angles) - 1, 1)

        simulator = DynamicsSimulator(
            topology_info=calculated_topology,
            initial_coords=initial_coordinates,
            physical_params=physical_params,
            input_motion=input_angles,
            slider_angles=slider_angles,
            gravity=9.81,
            topology_data=topology,
            coord_unit='mm',

            # (NEW) PMKS+처럼 모터가 일정 rpm으로 구동한다고 가정
            time_step=dt_for_one_rev,
            input_rpm=input_rpm,

            enable_omega_sweep=False,
            omega_sweep_values=np.array([0.5, 1.0, 2.0, 5.0, 10.0])
        )

        results = simulator.run()
        csv_path = export_simulation_results_to_csv(
                results=results,
                topology_index=topology_index,
                sample_index=sample_index,
                out_dir=os.path.join(current_dir, "csv_export"),
                filename_prefix="sim",
                include_after_pair=False,
            )

        print("CSV saved:", csv_path)

        # Animator 생성
        animator = MechanismAnimator(
            topology_info=calculated_topology,
            simulation_results=results,
            topology_data=topology,
            coord_unit='mm',  # 또는 'm'
            topology_id=topology_index
        )

        # animator.get_info()

        animator.animate(
            interval=50,           # 50ms per frame
            show_trajectory=False,  # 궤적 표시
            show_velocity=False,    # 속도 벡터 표시
            show_reaction_plot=True,   # joint reaction force subplot 표시
            show_torque_plot=True,     # input torque subplot 표시
            show_omega_sweep_plot=False, # omega vs. max torque subplot 표시
        )