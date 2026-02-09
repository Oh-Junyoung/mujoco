
import numpy as np
from typing import List, Tuple


class TopologyCalculator:
    """
    기본 위상 정보로부터 파생 정보들을 계산하는 클래스
    
    입력 정보:
        - number_of_binary_links
        - number_of_ternary_links
        - number_of_quaternary_links
        - number_of_joints_of_ground_link
        - index_of_ground_link
        - adjacency_matrix (array_of_adjacency_matrices)
        - index_of_input_link (input_link_index)
        - list_of_end_effector_links (end_effector_link_list)
        - list_of_rockers (rocker_list)
        - list_of_joint_type (joint_type_list)
    
    사용 예:
        calc = TopologyCalculator(topology_dict)
        
        # 기본 계산
        n_links = calc.number_of_links()
        n_joints = calc.number_of_joints()
        
        # Pair 계산
        pair_array = calc.pair()
        
        # Joint-Link 연결 정보
        original_conn = calc.links_connected_by_joints_original()
        after_pair_conn = calc.links_connected_by_joints_after_pair()
        
        # Ground link 정보
        ground_joints = calc.joints_list_of_ground_link()
    """
    
    def __init__(self, topology: dict):
        """
        Args:
            topology: 위상 정보를 담은 dictionary
        """
        # 기본 정보 저장
        self.n_binary = int(topology["number_of_binary_links"])
        self.n_ternary = int(topology["number_of_ternary_links"])
        self.n_quaternary = int(topology["number_of_quaternary_links"])
        self.n_joints_ground = int(topology["number_of_joints_of_ground_link"])
        self.ground_idx = int(topology["index_of_ground_link"])
        self.adjacency = np.array(topology["array_of_adjacency_matrices"])
        self.input_idx = int(topology["input_link_index"])
        self.end_effector_list = topology["end_effector_link_list"]
        self.rocker_list = topology["rocker_list"]
        self.joint_type_list = np.array(topology["joint_type_list"], dtype=int)
        
        # 내부 계산용 캐시
        self._adjacency_normalized = None
        self._joint_edges = None
        self._pair_cache = None
    
    # ========================================
    # Internal Helper Methods
    # ========================================
    
    def _get_adjacency_normalized(self) -> np.ndarray:
        """
        인접 행렬을 {0,1} 대칭/대각0 형태로 정규화
        """
        if self._adjacency_normalized is not None:
            return self._adjacency_normalized
        
        A = self.adjacency.copy()
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"adjacency matrix는 정사각 행렬이어야 합니다. shape={A.shape}")
        
        # Boolean 변환 후 대칭화
        A = (A != 0)
        A = (A | A.T)
        np.fill_diagonal(A, 0)
        
        self._adjacency_normalized = A.astype(np.uint8)
        return self._adjacency_normalized
    
    def _get_joint_edges(self) -> List[Tuple[int, int]]:
        """
        인접 행렬에서 edge(joint) 리스트를 추출하여 정렬
        각 edge (i, j)는 i < j 형태로 저장
        이 순서가 joint index 순서가 됨
        """
        if self._joint_edges is not None:
            return self._joint_edges
        
        A = self._get_adjacency_normalized()
        L = A.shape[0]
        edges = []
        
        for i in range(L):
            for j in range(i + 1, L):
                if A[i, j] != 0:
                    edges.append((i, j))
        
        edges.sort(key=lambda e: (e[0], e[1]))
        self._joint_edges = edges
        return self._joint_edges
    
    def _find_joint_index(self, link_a: int, link_b: int) -> int:
        """
        두 링크를 연결하는 joint의 index를 찾음
        """
        edges = self._get_joint_edges()
        i, j = (link_a, link_b) if link_a < link_b else (link_b, link_a)
        
        try:
            return edges.index((i, j))
        except ValueError:
            raise ValueError(f"링크 {link_a}와 {link_b}를 연결하는 joint가 없습니다.")
    
    # ========================================
    # Public Calculation Methods
    # ========================================
    
    def number_of_links(self) -> int:
        """
        전체 링크 개수 (binary + ternary + quaternary)
        """
        return self.n_binary + self.n_ternary + self.n_quaternary
    
    def number_of_joints(self) -> int:
        """
        전체 joint 개수 (pair 적용 전)
        """
        edges = self._get_joint_edges()
        return len(edges)
    
    def pair(self) -> np.ndarray:
        """
        각 joint의 pair 정보를 계산
        
        반환:
            shape (J,) 의 np.ndarray[int]
            - pair[j] == 0: j번 joint는 R joint
            - pair[j] != 0: j번 joint는 P joint이며, 
                           pair[j]는 복제된 joint의 index (J 이상)
        
        규칙:
            - ground-input joint는 무조건 R joint
            - joint_type_list[j] == 1: R joint
            - joint_type_list[j] == 2: P joint
        """
        if self._pair_cache is not None:
            return self._pair_cache
        
        edges = self._get_joint_edges()
        J = len(edges)
        
        if self.joint_type_list.shape[0] != J:
            raise ValueError(
                f"joint_type_list 길이({self.joint_type_list.shape[0]})와 "
                f"joint 개수({J})가 일치하지 않습니다."
            )
        
        # ground-input joint index 찾기
        gi_idx = self._find_joint_index(self.ground_idx, self.input_idx)
        
        pair = np.zeros(J, dtype=int)
        next_dup_idx = J  # 복제 joint index 시작값
        
        for j in range(J):
            if j == gi_idx:
                # ground-input joint는 무조건 R
                pair[j] = 0
                continue
            
            jtype = self.joint_type_list[j]
            
            if jtype == 1:  # R joint
                pair[j] = 0
            elif jtype == 2:  # P joint
                pair[j] = next_dup_idx
                next_dup_idx += 1
            else:
                raise ValueError(
                    f"joint_type_list[{j}] 값이 1(R) 또는 2(P)가 아닙니다: {jtype}"
                )
        
        self._pair_cache = pair
        return pair
    
    def links_connected_by_joints_original(self) -> np.ndarray:
        """
        각 joint가 연결하는 링크 쌍 (pair 적용 전)
        
        반환:
            shape (J, 2) 의 np.ndarray[int]
            row j = [link_i, link_j]
        """
        edges = self._get_joint_edges()
        if len(edges) == 0:
            return np.zeros((0, 2), dtype=int)
        
        return np.array(edges, dtype=int)
    
    def links_connected_by_joints_after_pair(self) -> np.ndarray:
        """
        pair를 반영한 joint-link 연결 관계
        
        규칙:
        - P joint는 ground에 연결되어 있어야 함
        - 각 P joint j에 대해:
          (1) 가상 링크 v (음수: -1, -2, ...) 생성
          (2) 원래 joint j: [v, ground]로 변경
          (3) 복제 joint dup: [v, other]로 설정
              (other = ground가 아닌 반대편 링크)
        
        반환:
            shape (J + num_P, 2) 의 np.ndarray[int]
        """
        edges = self.links_connected_by_joints_original()
        J = edges.shape[0]
        
        pair_array = self.pair()
        g = self.ground_idx
        
        # P joint들 찾기
        p_joint_indices = np.where(pair_array != 0)[0]
        num_p = len(p_joint_indices)
        
        # 출력 배열 초기화 (R joint는 그대로 복사)
        out = np.zeros((J + num_p, 2), dtype=int)
        out[:J, :] = edges
        
        next_virtual = -1  # 가상 링크 인덱스: -1, -2, -3, ...
        
        for j in p_joint_indices:
            dup = int(pair_array[j])  # 복제된 joint index
            
            if not (J <= dup < J + num_p):
                raise ValueError(
                    f"pair[{j}]={dup}가 예상 범위({J}~{J+num_p-1})를 벗어났습니다."
                )
            
            a, b = int(edges[j, 0]), int(edges[j, 1])
            
            # P joint는 ground를 포함해야 함
            if a == g:
                other = b
            elif b == g:
                other = a
            else:
                raise ValueError(
                    f"P joint {j}가 ground link({g})에 연결되어 있지 않습니다. "
                    f"edges[{j}] = [{a}, {b}]"
                )
            
            v = next_virtual
            next_virtual -= 1
            
            # (1) 원래 joint j: [virtual, ground]
            out[j, 0] = v
            out[j, 1] = g
            
            # (2) 복제 joint dup: [virtual, other]
            out[dup, 0] = v
            out[dup, 1] = other
        
        return out
    
    def joints_list_of_ground_link(self) -> np.ndarray:
        """
        ground link에 연결된 joint들의 인덱스 리스트
        
        반환:
            shape (K,) 의 np.ndarray[int]
            K = ground link에 연결된 joint 개수
        """
        edges = self.links_connected_by_joints_original()
        if edges.size == 0:
            return np.zeros((0,), dtype=int)
        
        g = self.ground_idx
        
        # ground를 포함하는 joint index 찾기
        joint_indices = np.where((edges[:, 0] == g) | (edges[:, 1] == g))[0]
        
        return joint_indices.astype(int)
    
    def number_of_joints_after_pair(self) -> int:
        """
        pair 적용 후 전체 joint 개수
        = original joint 개수 + P joint 개수
        """
        edges_after = self.links_connected_by_joints_after_pair()
        return edges_after.shape[0]
    
    def number_of_links_including_virtual_links(self) -> int:
        """
        가상 링크를 포함한 전체 링크 개수
        = original link 개수 + virtual link 개수
        """
        num_original = self.number_of_links()
        
        edges_after = self.links_connected_by_joints_after_pair()
        if edges_after.size == 0:
            return num_original
        
        # 음수 값 = virtual link
        virtual_links = edges_after[edges_after < 0]
        
        if virtual_links.size == 0:
            return num_original
        
        # 중복 제거하여 virtual link 개수 계산
        num_virtual = len(np.unique(virtual_links))
        
        return num_original + num_virtual