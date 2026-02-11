# MJCFConverter.py
"""
Topology + physical parameters + joint positions → MuJoCo MJCF XML.

Usage (standalone):
    converter = MJCFConverter(topology_info, topology_data, physical_params,
                              joint_positions_m, topology_index, sample_index)
    xml_str = converter.convert()
"""
from __future__ import annotations

from collections import defaultdict, deque
from itertools import combinations

import numpy as np


class MJCFConverter:
    """PKL 시뮬레이션 데이터를 MuJoCo MJCF XML 로 변환."""

    # ── Configurable defaults ──
    DEFAULT_LINK_RADIUS = 0.001       # 링크 반지름 (m) → 직경 2 mm
    DEFAULT_KP = 10.0                 # 모터 position servo 강성
    DEFAULT_FLOOR_HALF_EXTENT = 0.1   # 바닥 half-extent (m) → 200 mm × 200 mm
    DEFAULT_SHADOW_SIZE = 16384
    DEFAULT_TIMESTEP = 0.001
    DEFAULT_VIRTUAL_MASS = 0.0016     # virtual body 질량 (kg)
    DEFAULT_SLIDER_RAIL_HALF_LEN = 0.033  # slider rail 반쪽 길이 (m)

    def __init__(
        self,
        topology_info,          # TopologyCalculator instance
        topology_data: dict,    # raw topology dict
        physical_params: dict,  # masses, inertias, centers_of_mass, link_lengths
        joint_positions_m,      # (J_after, 2, T) in meters
        topology_index: int,
        sample_index: int,
        *,
        link_radius: float | None = None,
        kp: float | None = None,
        floor_half_extent: float | None = None,
        shadow_size: int | None = None,
    ):
        self.topo_info = topology_info
        self.topo_data = topology_data
        self.params = physical_params
        self.q_all = np.asarray(joint_positions_m, dtype=float)
        self.topo_idx = topology_index
        self.sample_idx = sample_index

        # settings
        self.link_r = link_radius or self.DEFAULT_LINK_RADIUS
        self.kp = kp or self.DEFAULT_KP
        self.floor_half = floor_half_extent or self.DEFAULT_FLOOR_HALF_EXTENT
        self.shadow_size = shadow_size or self.DEFAULT_SHADOW_SIZE

        # derived
        self.q0 = self.q_all[:, :, 0]  # initial config (J_after, 2)
        self.ground_idx = int(self.topo_data["index_of_ground_link"])
        self.input_link_idx = int(self.topo_data["input_link_index"])

        self.edges_after = self.topo_info.links_connected_by_joints_after_pair()
        self.J_after = len(self.edges_after)
        self.pair = self.topo_info.pair()
        self.J_orig = len(self.pair)

        # joint type (original joints only): 1=R, 2=P
        self.jtype = np.asarray(self.topo_data["joint_type_list"], dtype=int)

        # build link → joints adjacency
        self._link_joints: dict[int, list[int]] = defaultdict(list)
        for j in range(self.J_after):
            la, lb = int(self.edges_after[j][0]), int(self.edges_after[j][1])
            self._link_joints[la].append(j)
            self._link_joints[lb].append(j)

        # build spanning tree
        self.tree_edges, self.constraint_joints = self._build_spanning_tree()
        self.children: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for j_idx, parent, child in self.tree_edges:
            self.children[parent].append((j_idx, child))

        # identify input joint (ground joint on input link)
        self.input_joint_idx = self._find_input_joint()

        # global origins for each body (filled during XML generation)
        self._body_global_origin: dict[int, np.ndarray] = {}

    # ══════════════════════════════════════════════════════════════════
    # PUBLIC
    # ══════════════════════════════════════════════════════════════════

    def convert(self) -> str:
        lines: list[str] = []
        self._add_header(lines)
        self._add_worldbody(lines)
        self._add_equality(lines)
        self._add_actuator(lines)
        lines.append("</mujoco>")
        return "\n".join(lines) + "\n"

    # ══════════════════════════════════════════════════════════════════
    # TREE DECOMPOSITION
    # ══════════════════════════════════════════════════════════════════

    def _build_spanning_tree(self):
        """BFS from ground link → spanning tree + loop-closure joints."""
        adjacency: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for j in range(self.J_after):
            la, lb = int(self.edges_after[j][0]), int(self.edges_after[j][1])
            adjacency[la].append((j, lb))
            adjacency[lb].append((j, la))

        visited = {self.ground_idx}
        tree = []
        constraints = []
        used_joints: set[int] = set()

        # BFS — start from ground neighbours
        queue: deque[int] = deque()
        ground_joints = self.topo_info.joints_list_of_ground_link()
        for gj in sorted(ground_joints):
            for j, nb in adjacency[self.ground_idx]:
                if j == gj and j not in used_joints:
                    used_joints.add(j)
                    if nb not in visited:
                        visited.add(nb)
                        tree.append((j, self.ground_idx, nb))
                        queue.append(nb)
                    else:
                        constraints.append((j, self.ground_idx, nb))

        while queue:
            cur = queue.popleft()
            for j, nb in adjacency[cur]:
                if j in used_joints:
                    continue
                used_joints.add(j)
                if nb not in visited:
                    visited.add(nb)
                    tree.append((j, cur, nb))
                    queue.append(nb)
                else:
                    constraints.append((j, cur, nb))

        return tree, constraints

    def _find_input_joint(self) -> int:
        """Find the joint between ground and input link."""
        ground_joints = set(self.topo_info.joints_list_of_ground_link())
        for j, parent, child in self.tree_edges:
            if parent == self.ground_idx and child == self.input_link_idx:
                return j
            # input link might be reached through virtual link
            if parent == self.ground_idx:
                for j2, p2, c2 in self.tree_edges:
                    if p2 == child and c2 == self.input_link_idx:
                        return j  # ground joint is the one connecting to virtual
        # fallback: look for input link joints that are ground joints
        input_joints = set(self._link_joints.get(self.input_link_idx, []))
        match = input_joints & ground_joints
        if match:
            return min(match)
        return 0

    def _is_virtual_link(self, link_idx: int) -> bool:
        return link_idx < 0

    def _is_p_joint_original(self, j_idx: int) -> bool:
        """Check if original joint j_idx is prismatic."""
        if j_idx < len(self.jtype):
            return self.jtype[j_idx] == 2
        return False

    def _other_link(self, j_idx: int, this_link: int) -> int:
        la, lb = int(self.edges_after[j_idx][0]), int(self.edges_after[j_idx][1])
        return lb if la == this_link else la

    # ══════════════════════════════════════════════════════════════════
    # SLIDER DIRECTION
    # ══════════════════════════════════════════════════════════════════

    def _compute_slider_direction(self, p_joint_idx: int) -> np.ndarray:
        """Compute unit slider axis from joint trajectory (PCA)."""
        traj = self.q_all[p_joint_idx, :, :]  # (2, T)
        mean = traj.mean(axis=1, keepdims=True)
        centered = traj - mean
        U, S, _ = np.linalg.svd(centered, full_matrices=False)
        direction = U[:, 0]
        # consistent sign: positive Y preferred
        if direction[1] < 0:
            direction = -direction
        return direction

    # ══════════════════════════════════════════════════════════════════
    # INPUT ANGLE RANGE
    # ══════════════════════════════════════════════════════════════════

    def _compute_input_angle_range(self) -> tuple[float, float]:
        """Compute the input joint angular range from trajectory."""
        # find two joints on input link
        link_joints = self._link_joints[self.input_link_idx]
        # find the ground joint and the other joint
        ground_joints_set = set(self.topo_info.joints_list_of_ground_link())
        gj = None
        other_j = None
        for jj in link_joints:
            if jj in ground_joints_set or jj == self.input_joint_idx:
                gj = jj
            else:
                other_j = jj
        if gj is None or other_j is None:
            return (0.0, 2.0 * np.pi)

        # link direction over time
        gj_traj = self.q_all[gj, :, :]    # (2, T)
        oj_traj = self.q_all[other_j, :, :]  # (2, T)
        vec = oj_traj - gj_traj            # (2, T)
        angles = np.arctan2(vec[1], vec[0])  # (T,)
        # unwrap for continuous range
        angles = np.unwrap(angles)
        rel = angles - angles[0]
        return (float(rel.min()), float(rel.max()))

    # ══════════════════════════════════════════════════════════════════
    # XML GENERATION
    # ══════════════════════════════════════════════════════════════════

    def _fmt(self, v: float, prec: int = 9) -> str:
        """Format float, strip trailing zeros."""
        s = f"{v:.{prec}f}".rstrip("0").rstrip(".")
        return s if s != "-0" else "0"

    def _add_header(self, L: list[str]):
        model_name = f"topo{self.topo_idx}_s{self.sample_idx}"
        texrepeat = int(self.floor_half * 500)

        L.append(f'<?xml version="1.0" ?>')
        L.append(f'<mujoco model="{model_name}">')
        L.append(f'  <compiler angle="radian" inertiafromgeom="false"/>')
        L.append(f'  <option timestep="{self.DEFAULT_TIMESTEP}" gravity="0 0 -9.81"'
                 f' integrator="implicit" iterations="500" tolerance="1e-12"'
                 f' noslip_iterations="10" noslip_tolerance="1e-8"/>')
        L.append(f'  <visual>')
        L.append(f'    <quality shadowsize="{self.shadow_size}"/>')
        L.append(f'  </visual>')
        L.append(f'  <asset>')
        L.append(f'    <texture name="grid" type="2d" builtin="checker"'
                 f' rgb1="0.9 0.9 0.9" rgb2="0.8 0.8 0.8" width="100" height="100"/>')
        L.append(f'    <material name="grid_mat" texture="grid"'
                 f' texrepeat="{texrepeat} {texrepeat}" texuniform="true"/>')
        L.append(f'  </asset>')
        L.append(f'  <default>')
        L.append(f'    <geom rgba="0.3 0.6 0.85 1" contype="0" conaffinity="0"/>')
        L.append(f'    <joint limited="false" damping="0.01"/>')
        L.append(f'    <equality solref="0.002 1" solimp="0.99 0.999 0.001 0.5 2"/>')
        L.append(f'  </default>')

    def _add_worldbody(self, L: list[str]):
        r = self._fmt(self.link_r)
        fh = self._fmt(self.floor_half)

        L.append(f'  <worldbody>')
        L.append(f'    <light pos="0 0 5" dir="0 0 -1" diffuse="1 1 1"/>')
        L.append(f'    <geom name="floor" type="plane" pos="0 0 -0.01"'
                 f' size="{fh} {fh} 0.01" material="grid_mat"/>')
        L.append(f'')

        # ── XYZ Axes ──
        L.append(f'    <!-- XYZ Axes at origin -->')
        L.append(f'    <geom type="capsule" fromto="0 0 0 0.12 0 0"'
                 f' size="0.0003" rgba="0.9 0.2 0.2 1" contype="0" conaffinity="0"/>')
        L.append(f'    <geom type="capsule" fromto="0 0 0 0 0.12 0"'
                 f' size="0.0003" rgba="0.2 0.9 0.2 1" contype="0" conaffinity="0"/>')
        L.append(f'    <geom type="capsule" fromto="0 0 0 0 0 0.05"'
                 f' size="0.0003" rgba="0.2 0.2 0.9 1" contype="0" conaffinity="0"/>')

        # ── Ground link triangle (base frame) ──
        ground_joints = sorted(self.topo_info.joints_list_of_ground_link())
        gj_positions = {gj: self.q0[gj] for gj in ground_joints}
        for a, b in combinations(ground_joints, 2):
            pa, pb = gj_positions[a], gj_positions[b]
            L.append(f'    <geom type="capsule"'
                     f' fromto="{self._fmt(pa[0])} {self._fmt(pa[1])} 0'
                     f' {self._fmt(pb[0])} {self._fmt(pb[1])} 0"'
                     f' size="{r}" rgba="0.6 0.6 0.6 0.5" contype="0" conaffinity="0"/>')

        # ── Slider rails (fixed to worldbody) ──
        for j_idx, parent, child in self.tree_edges:
            if parent != self.ground_idx:
                continue
            if not self._is_virtual_link(child):
                continue
            # this is a P joint chain: ground → virtual (slide)
            anchor = self.q0[j_idx]
            direction = self._compute_slider_direction(j_idx)
            hl = self.DEFAULT_SLIDER_RAIL_HALF_LEN
            start = anchor - direction * hl
            end = anchor + direction * hl
            L.append(f'    <!-- Slider rail (fixed to ground) -->')
            L.append(f'    <geom type="cylinder"'
                     f' fromto="{self._fmt(start[0])} {self._fmt(start[1])} 0'
                     f' {self._fmt(end[0])} {self._fmt(end[1])} 0"'
                     f' size="{r}" rgba="0.4 0.4 0.4 0.5" contype="0" conaffinity="0"/>')

        # ── Bodies (tree children of ground) ──
        for j_idx, child in self.children[self.ground_idx]:
            origin = np.zeros(2)  # worldbody origin = (0,0)
            self._add_body(L, child, j_idx, origin, indent="    ")

        # ── Ground reference sites ──
        for gj in ground_joints:
            p = self.q0[gj]
            L.append(f'    <site name="site_ground_j{gj}"'
                     f' pos="{self._fmt(p[0])} {self._fmt(p[1])} 0" size="{r}"/>')

        # ── Camera (auto-computed) ──
        cx = float(self.q0[:, 0].mean())
        cy = float(self.q0[:, 1].mean())
        dist = 0.18
        L.append(f'    <camera name="closeup"'
                 f' pos="{self._fmt(cx)} {self._fmt(cy - dist)} {self._fmt(dist)}"'
                 f' xyaxes="1 0 0 0 0.707 0.707"/>')

        L.append(f'  </worldbody>')

    def _add_body(self, L: list[str], link_idx: int, joint_idx: int,
                  parent_global_origin: np.ndarray, indent: str):
        """Recursively add a body and its children."""
        joint_pos = self.q0[joint_idx]  # global position of connecting joint
        body_pos = joint_pos - parent_global_origin
        self._body_global_origin[link_idx] = joint_pos.copy()

        r = self._fmt(self.link_r)
        is_virtual = self._is_virtual_link(link_idx)

        if is_virtual:
            body_name = f"virtual_{abs(link_idx)}"
            L.append(f'{indent}<body name="{body_name}"'
                     f' pos="{self._fmt(body_pos[0])} {self._fmt(body_pos[1])} 0">')

            # inertial (small)
            vm = self.DEFAULT_VIRTUAL_MASS
            vi = vm * 1e-4
            L.append(f'{indent}  <inertial pos="0 0 0"'
                     f' mass="{self._fmt(vm)}" diaginertia="{self._fmt(vi)} {self._fmt(vi)} {self._fmt(vi)}"/>')

            # slide joint
            direction = self._compute_slider_direction(joint_idx)
            slider_angle = float(np.arctan2(direction[1], direction[0]))
            L.append(f'{indent}  <joint name="P{joint_idx}" type="slide"'
                     f' axis="{self._fmt(direction[0])} {self._fmt(direction[1])} 0"/>')

            # box geom (green slider marker)
            L.append(f'{indent}  <geom type="box" size="0.003 0.005 0.003"'
                     f' euler="0 0 {self._fmt(slider_angle)}"'
                     f' rgba="0.25 0.85 0.25 1"/>')

            # sites on virtual body
            self._add_sites_for_link(L, link_idx, joint_pos, indent + "  ")

        else:
            body_name = f"link_{link_idx}"
            L.append(f'{indent}<body name="{body_name}"'
                     f' pos="{self._fmt(body_pos[0])} {self._fmt(body_pos[1])} 0">')

            # inertial
            mass = float(self.params["masses"][link_idx])
            inertia = float(self.params["inertias"][link_idx])
            com_global = self.params["centers_of_mass"][link_idx]
            com_local = com_global - joint_pos
            L.append(f'{indent}  <inertial'
                     f' pos="{self._fmt(com_local[0])} {self._fmt(com_local[1])} 0"'
                     f' mass="{self._fmt(mass)}"'
                     f' diaginertia="{self._fmt(inertia)} {self._fmt(inertia)} {self._fmt(inertia)}"/>')

            # joint
            is_input = (link_idx == self.input_link_idx)
            if is_input:
                lo, hi = self._compute_input_angle_range()
                L.append(f'{indent}  <joint name="J{joint_idx}" type="hinge" axis="0 0 1"'
                         f' limited="true" range="{self._fmt(lo)} {self._fmt(hi)}"/>')
            else:
                # check if this is a paired hinge (D-type)
                paired = False
                for orig_j in range(self.J_orig):
                    if self.pair[orig_j] != 0 and int(self.pair[orig_j]) == joint_idx:
                        paired = True
                        break
                jname = f"D{joint_idx}" if paired else f"J{joint_idx}"
                L.append(f'{indent}  <joint name="{jname}" type="hinge" axis="0 0 1"/>')

            # geom(s)
            self._add_link_geoms(L, link_idx, joint_pos, is_input, indent + "  ")

            # sites
            self._add_sites_for_link(L, link_idx, joint_pos, indent + "  ")

        # children
        for child_j, child_link in self.children.get(link_idx, []):
            self._add_body(L, child_link, child_j, joint_pos, indent + "  ")

        L.append(f'{indent}</body>')

    def _add_link_geoms(self, L: list[str], link_idx: int,
                        body_origin: np.ndarray, is_input: bool, indent: str):
        """Add capsule geoms for a link's edges."""
        r = self._fmt(self.link_r)
        rgba = ' rgba="0.85 0.25 0.25 1"' if is_input else ""

        # joints on this link
        link_js = self._link_joints[link_idx]
        joint_positions_local = {}
        for jj in link_js:
            p = self.q0[jj] - body_origin
            joint_positions_local[jj] = p

        if len(joint_positions_local) <= 1:
            # single joint → just a sphere
            L.append(f'{indent}<geom type="sphere" size="{r}"{rgba}/>')
            return

        if len(joint_positions_local) == 2:
            # binary link → one capsule
            js = list(joint_positions_local.values())
            L.append(f'{indent}<geom type="capsule"'
                     f' fromto="{self._fmt(js[0][0])} {self._fmt(js[0][1])} 0'
                     f' {self._fmt(js[1][0])} {self._fmt(js[1][1])} 0"'
                     f' size="{r}"{rgba}/>')
        else:
            # ternary+ link → edges between all pairs + sphere at origin
            L.append(f'{indent}<geom type="sphere" size="{r}"{rgba}/>')
            pairs = list(combinations(joint_positions_local.values(), 2))
            for pa, pb in pairs:
                L.append(f'{indent}<geom type="capsule"'
                         f' fromto="{self._fmt(pa[0])} {self._fmt(pa[1])} 0'
                         f' {self._fmt(pb[0])} {self._fmt(pb[1])} 0"'
                         f' size="{r}"{rgba}/>')

    def _add_sites_for_link(self, L: list[str], link_idx: int,
                            body_origin: np.ndarray, indent: str):
        """Add site markers for each joint on this link."""
        r = self._fmt(self.link_r)
        body_name = f"virtual_{abs(link_idx)}" if self._is_virtual_link(link_idx) else f"link_{link_idx}"
        for jj in self._link_joints[link_idx]:
            p = self.q0[jj] - body_origin
            L.append(f'{indent}<site name="site_{body_name}_j{jj}"'
                     f' pos="{self._fmt(p[0])} {self._fmt(p[1])} 0" size="{r}"/>')

    def _add_equality(self, L: list[str]):
        """Add loop-closure connect constraints."""
        if not self.constraint_joints:
            return
        L.append(f'  <equality>')
        for j_idx, link_a, link_b in self.constraint_joints:
            name_a = f"virtual_{abs(link_a)}" if self._is_virtual_link(link_a) else f"link_{link_a}"
            name_b = f"virtual_{abs(link_b)}" if self._is_virtual_link(link_b) else f"link_{link_b}"
            site1 = f"site_{name_a}_j{j_idx}"
            site2 = f"site_{name_b}_j{j_idx}"
            L.append(f'    <connect name="loop_j{j_idx}" site1="{site1}" site2="{site2}"/>')
        L.append(f'  </equality>')

    def _add_actuator(self, L: list[str]):
        """Add position servo on the input joint."""
        lo, hi = self._compute_input_angle_range()
        L.append(f'  <actuator>')
        L.append(f'    <position name="input_servo" joint="J{self.input_joint_idx}"'
                 f' kp="{self._fmt(self.kp)}" ctrlrange="{self._fmt(lo)} {self._fmt(hi)}"/>')
        L.append(f'  </actuator>')
