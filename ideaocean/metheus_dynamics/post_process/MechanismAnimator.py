import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle
from matplotlib import transforms
from typing import Dict, Tuple, Optional


class MechanismAnimator:
    """
    링키지 기구의 운동 애니메이션 생성기 (pair/after_pair 자동 반영, 에너지 플롯 제거)

    기능:
    - 기구 애니메이션 (after_pair 좌표 자동 사용)
    - P joint 레일/슬라이더 블록 시각화 (pair 의미: P는 anchor, dup는 moving)
    - (옵션) 모든 joint 반력(|F|) subplot (x=time[s], y=N)
    - (옵션) input link 구동 토크 subplot (x=time[s], y=N·m)

    요구되는 simulator 결과 키:
    - joint_reaction_forces: (J_original, 2, steps) [N]
    - motor_torque: (steps,) [N·m]
    - (권장) slider_angles: (num_P,) [rad]  (없으면 기본 0)
    """

    def __init__(
        self,
        topology_info,
        simulation_results: Dict,
        topology_data: Dict,
        coord_unit: str = "mm",
        slider_angles: Optional[np.ndarray] = None,
        show_p_joint_rail: bool = True,
        show_p_joint_block: bool = True,
        topology_id: Optional[int] = None,
    ):
        self.topology_id = topology_id
        self.topo = topology_info
        self.results = simulation_results
        self.topology_data = topology_data
        self.coord_unit = coord_unit

        self.show_p_joint_rail = bool(show_p_joint_rail)
        self.show_p_joint_block = bool(show_p_joint_block)

        # time / valid
        self.is_valid = np.asarray(simulation_results["is_valid"], dtype=bool)
        self.valid_steps = int(np.sum(self.is_valid))
        self.time = np.asarray(simulation_results["time"], dtype=float)

        # 좌표/속도: after_pair 우선 선택
        self.use_after_pair = False

        if coord_unit == "mm":
            if "joint_coordinates_after_pair_mm" in simulation_results:
                self.coords = np.asarray(simulation_results["joint_coordinates_after_pair_mm"], dtype=float)
                self.velocities = simulation_results.get("joint_velocities_after_pair_mm", None)
                self.use_after_pair = True
            else:
                self.coords = np.asarray(simulation_results["joint_coordinates_mm"], dtype=float)
                self.velocities = simulation_results.get("joint_velocities_mm", None)
                self.use_after_pair = False
        else:
            if "joint_coordinates_after_pair" in simulation_results:
                self.coords = np.asarray(simulation_results["joint_coordinates_after_pair"], dtype=float)
                self.velocities = simulation_results.get("joint_velocities_after_pair", None)
                self.use_after_pair = True
            else:
                self.coords = np.asarray(simulation_results["joint_coordinates"], dtype=float)
                self.velocities = simulation_results.get("joint_velocities", None)
                self.use_after_pair = False

        if self.coords.ndim != 3 or self.coords.shape[1] != 2:
            raise ValueError(f"Unexpected coords shape: {self.coords.shape}. Expected (J,2,steps).")

        self.J = int(self.coords.shape[0])
        self.num_steps = int(self.coords.shape[2])

        # edges: 좌표 모드에 맞춰 자동 선택
        if self.use_after_pair:
            if not hasattr(self.topo, "links_connected_by_joints_after_pair"):
                raise ValueError(
                    "Animator selected after_pair coords, but topology_info has no links_connected_by_joints_after_pair()."
                )
            self.edges = np.asarray(self.topo.links_connected_by_joints_after_pair(), dtype=int)
        else:
            self.edges = np.asarray(self.topo.links_connected_by_joints_original(), dtype=int)

        if self.edges.shape[0] != self.J:
            raise ValueError(
                f"Mismatch between coords J={self.J} and edges rows={self.edges.shape[0]}. "
                f"Animator needs edges that match the simulated joint count."
            )

        # link ids (after_pair면 virtual link 포함 가능)
        link_ids = np.unique(self.edges.flatten())
        self.link_ids = [int(x) for x in link_ids.tolist()]

        # Ground / input link indices
        self.ground_idx = int(topology_data["index_of_ground_link"])
        self.input_idx = int(topology_data["input_link_index"])

        # pair 정보
        self.pair = None
        self.dup_joint_set = set()
        self.p_joint_set = set()

        if hasattr(self.topo, "pair"):
            pr = np.asarray(self.topo.pair(), dtype=int).reshape(-1)
            self.pair = pr
            p_joint_indices = np.where(pr != 0)[0].astype(int)
            for j in p_joint_indices:
                dup = int(pr[j])
                self.p_joint_set.add(int(j))
                self.dup_joint_set.add(int(dup))

        # slider_angles(레일 방향)
        self.slider_angles = None
        self.p_joint_indices = None

        if self.pair is not None:
            self.p_joint_indices = np.where(self.pair != 0)[0].astype(int)
            num_p = int(len(self.p_joint_indices))

            if slider_angles is None:
                sa = simulation_results.get("slider_angles", None)
                if sa is None:
                    self.slider_angles = np.zeros((num_p,), dtype=float)
                else:
                    sa = np.asarray(sa, dtype=float).reshape(-1)
                    self.slider_angles = sa if sa.shape[0] == num_p else np.zeros((num_p,), dtype=float)
            else:
                sa = np.asarray(slider_angles, dtype=float).reshape(-1)
                self.slider_angles = sa if sa.shape[0] == num_p else np.zeros((num_p,), dtype=float)

        # Plot data: reaction forces / motor torque
        self.reaction_forces = simulation_results.get("joint_reaction_forces", None)  # (J_original,2,steps)
        if self.reaction_forces is not None:
            self.reaction_forces = np.asarray(self.reaction_forces, dtype=float)

        self.motor_torque = simulation_results.get("motor_torque", None)  # (steps,)
        if self.motor_torque is not None:
            self.motor_torque = np.asarray(self.motor_torque, dtype=float).reshape(-1)

        # (NEW) omega sweep result (optional)
        self.omega_sweep_omegas = simulation_results.get("omega_sweep_omegas", None)
        self.omega_sweep_max_abs_torque = simulation_results.get("omega_sweep_max_abs_torque", None)
        self.omega_sweep_max_torque = simulation_results.get("omega_sweep_max_torque", None)
        self.omega_sweep_min_torque = simulation_results.get("omega_sweep_min_torque", None)

        if self.omega_sweep_omegas is not None:
            self.omega_sweep_omegas = np.asarray(self.omega_sweep_omegas, dtype=float).reshape(-1)
        if self.omega_sweep_max_abs_torque is not None:
            self.omega_sweep_max_abs_torque = np.asarray(self.omega_sweep_max_abs_torque, dtype=float).reshape(-1)
        if self.omega_sweep_max_torque is not None:
            self.omega_sweep_max_torque = np.asarray(self.omega_sweep_max_torque, dtype=float).reshape(-1)
        if self.omega_sweep_min_torque is not None:
            self.omega_sweep_min_torque = np.asarray(self.omega_sweep_min_torque, dtype=float).reshape(-1)

        # 링크 연결 정보 계산
        self._compute_link_info()

        # plot handles
        self.fig = None
        self.ax_main = None
        self.ax_react = None
        self.ax_torque = None
        self.ax_omega = None
        self.animation = None

    def _compute_link_info(self):
        """각 링크의 joint 연결 정보 계산 (현재 self.edges 기준)"""
        self.link_joints = {}
        for link_idx in self.link_ids:
            joints = []
            for j in range(self.J):
                if int(self.edges[j, 0]) == link_idx or int(self.edges[j, 1]) == link_idx:
                    joints.append(int(j))
            if len(joints) > 0:
                self.link_joints[int(link_idx)] = joints

    def animate(
        self,
        interval: int = 50,
        show_trajectory: bool = False,
        show_velocity: bool = False,
        figsize: Tuple[int, int] = (14, 9),
        show_reaction_plot: bool = False,
        show_torque_plot: bool = False,
        show_reaction_legend: bool = True,
        reaction_legend_cols: int = 6,

        # (NEW) omega sweep plot option
        show_omega_sweep_plot: bool = False,
        show_omega_sweep_legend: bool = True,
    ):
        want_react = bool(show_reaction_plot)
        want_torque = bool(show_torque_plot)
        want_omega = bool(show_omega_sweep_plot)

        # Figure layout (dynamic rows)
        # main is always shown.
        n_rows = 1 + int(want_react) + int(want_torque) + int(want_omega)

        self.fig = plt.figure(figsize=figsize)
        height_ratios = [3] + [1] * (n_rows - 1)
        gs = self.fig.add_gridspec(n_rows, 1, height_ratios=height_ratios, hspace=0.28)

        row = 0
        self.ax_main = self.fig.add_subplot(gs[row, 0]); row += 1
        self.ax_react = self.fig.add_subplot(gs[row, 0]) if want_react else None
        if want_react: row += 1
        self.ax_torque = self.fig.add_subplot(gs[row, 0]) if want_torque else None
        if want_torque: row += 1
        self.ax_omega = self.fig.add_subplot(gs[row, 0]) if want_omega else None

        # Bounds for main view
        all_coords = self.coords[:, :, :self.valid_steps].reshape(-1, self.valid_steps)
        x_min, x_max = np.nanmin(all_coords[0::2]), np.nanmax(all_coords[0::2])
        y_min, y_max = np.nanmin(all_coords[1::2]), np.nanmax(all_coords[1::2])
        margin = 0.1 * max(x_max - x_min, y_max - y_min) if self.valid_steps > 1 else 1.0

        def _setup_main_axes():
            self.ax_main.set_xlim(x_min - margin, x_max + margin)
            self.ax_main.set_ylim(y_min - margin, y_max + margin)
            self.ax_main.set_aspect("equal")
            self.ax_main.grid(True, alpha=0.3)
            self.ax_main.set_xlabel(f"X [{self.coord_unit}]")
            self.ax_main.set_ylabel(f"Y [{self.coord_unit}]")

        def _link_style(link_idx: int):
            if link_idx == self.ground_idx:
                return dict(color="black", lw=4, alpha=1.0, ls="-")
            if link_idx == self.input_idx:
                return dict(color="red", lw=3, alpha=0.9, ls="-")
            if link_idx < 0:
                return dict(color="gray", lw=2.0, alpha=0.7, ls="--")
            return dict(color="blue", lw=2.5, alpha=0.8, ls="-")

        # Trajectory storage
        self.trajectories = {j: {"x": [], "y": []} for j in range(self.J)}

        # Plot time window
        t_plot = self.time[:self.valid_steps]

        # Reaction plot data (ALL joints, magnitude)
        react_mag = None
        if want_react:
            if self.reaction_forces is None:
                raise ValueError("show_reaction_plot=True but results has no 'joint_reaction_forces'.")
            if self.reaction_forces.ndim != 3 or self.reaction_forces.shape[1] != 2:
                raise ValueError(f"joint_reaction_forces expected (J_original,2,steps) but got {self.reaction_forces.shape}")
            rf = self.reaction_forces[:, :, :self.valid_steps]
            react_mag = np.sqrt(rf[:, 0, :] ** 2 + rf[:, 1, :] ** 2)

        # Torque plot data (input link only)
        torque_series = None
        if want_torque:
            if self.motor_torque is None:
                raise ValueError("show_torque_plot=True but results has no 'motor_torque'.")
            torque_series = self.motor_torque[:self.valid_steps].reshape(-1)

        # Reaction subplot
        react_vline = None
        if self.ax_react is not None:
            self.ax_react.clear()
            self.ax_react.set_xlim(t_plot[0], t_plot[-1])
            self.ax_react.set_xlabel("Time [s]")
            self.ax_react.set_ylabel("Reaction |F| [N]")
            self.ax_react.grid(True, alpha=0.3)

            J0 = int(react_mag.shape[0])
            cmap = plt.get_cmap("tab20", J0) if J0 <= 20 else plt.get_cmap("hsv", J0)

            handles = []
            labels = []

            for j in range(J0):
                (ln,) = self.ax_react.plot(
                    t_plot, react_mag[j, :],
                    linewidth=1.2,
                    alpha=0.85,
                    color=cmap(j),
                )
                handles.append(ln)
                labels.append(f"J{j}")

            react_vline = self.ax_react.axvline(t_plot[0], linestyle="--", alpha=0.8)

            if show_reaction_legend:
                ncols = max(1, int(reaction_legend_cols))
                self.ax_react.legend(
                    handles,
                    labels,
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.25),
                    ncol=ncols,
                    fontsize=8,
                    frameon=False,
                )

        # Torque subplot
        torque_vline = None
        if self.ax_torque is not None:
            self.ax_torque.clear()
            self.ax_torque.set_xlim(t_plot[0], t_plot[-1])
            self.ax_torque.set_xlabel("Time [s]")
            self.ax_torque.set_ylabel("Motor torque [N·m]")
            self.ax_torque.grid(True, alpha=0.3)

            self.ax_torque.plot(t_plot, torque_series, linewidth=2.0, alpha=0.95)
            torque_vline = self.ax_torque.axvline(t_plot[0], linestyle="--", alpha=0.8)

        # Omega sweep subplot (static)
        if self.ax_omega is not None:
            if self.omega_sweep_omegas is None or self.omega_sweep_max_abs_torque is None:
                raise ValueError("show_omega_sweep_plot=True but results has no omega sweep data.")
            self.ax_omega.clear()
            self.ax_omega.set_xlabel("Omega [rad/s]")
            self.ax_omega.set_ylabel("Max |torque| [N·m]")
            self.ax_omega.grid(True, alpha=0.3)

            (ln_abs,) = self.ax_omega.plot(
                self.omega_sweep_omegas,
                self.omega_sweep_max_abs_torque,
                linewidth=2.0,
                alpha=0.95,
                label="max |torque|",
            )

            extra_handles = [ln_abs]
            extra_labels = ["max |torque|"]

            if self.omega_sweep_max_torque is not None and self.omega_sweep_max_torque.shape == self.omega_sweep_omegas.shape:
                (ln_max,) = self.ax_omega.plot(
                    self.omega_sweep_omegas,
                    self.omega_sweep_max_torque,
                    linewidth=1.5,
                    alpha=0.8,
                    linestyle="--",
                    label="max torque",
                )
                extra_handles.append(ln_max)
                extra_labels.append("max torque")

            if self.omega_sweep_min_torque is not None and self.omega_sweep_min_torque.shape == self.omega_sweep_omegas.shape:
                (ln_min,) = self.ax_omega.plot(
                    self.omega_sweep_omegas,
                    self.omega_sweep_min_torque,
                    linewidth=1.5,
                    alpha=0.8,
                    linestyle="--",
                    label="min torque",
                )
                extra_handles.append(ln_min)
                extra_labels.append("min torque")

            if show_omega_sweep_legend:
                self.ax_omega.legend(extra_handles, extra_labels, loc="best", fontsize=9, frameon=False)

        # P joint visualization (pair meaning)
        def _draw_p_joint_visuals(q_frame: np.ndarray):
            if (not self.use_after_pair) or (self.pair is None) or (self.slider_angles is None) or (self.p_joint_indices is None):
                return

            rail_half = margin * 0.8
            block_len = margin * 0.18
            block_thk = margin * 0.10

            for k, j in enumerate(self.p_joint_indices):
                dup = int(self.pair[int(j)])
                alpha = float(self.slider_angles[k])
                tdir = np.array([np.cos(alpha), np.sin(alpha)], dtype=float)

                anchor = q_frame[int(j), :]      # fixed anchor
                slider = q_frame[int(dup), :]    # moving slider

                if self.show_p_joint_rail:
                    p1 = anchor - rail_half * tdir
                    p2 = anchor + rail_half * tdir
                    self.ax_main.plot(
                        [p1[0], p2[0]],
                        [p1[1], p2[1]],
                        linestyle=":",
                        linewidth=2.0,
                        alpha=0.7,
                        color="purple",
                        zorder=1,
                    )

                if self.show_p_joint_block:
                    rect = Rectangle(
                        (slider[0] - block_len / 2.0, slider[1] - block_thk / 2.0),
                        block_len,
                        block_thk,
                        linewidth=1.5,
                        edgecolor="purple",
                        facecolor="none",
                        alpha=0.85,
                        zorder=6,
                    )
                    tr = transforms.Affine2D().rotate_around(slider[0], slider[1], alpha) + self.ax_main.transData
                    rect.set_transform(tr)
                    self.ax_main.add_patch(rect)

        def init():
            return []

        def update(frame):
            self.ax_main.clear()
            _setup_main_axes()

            q = self.coords[:, :, frame]

            # P joint visuals
            _draw_p_joint_visuals(q)

            # links
            for link_idx, joints in self.link_joints.items():
                st = _link_style(int(link_idx))
                if len(joints) == 2:
                    j1, j2 = joints
                    self.ax_main.plot(
                        [q[j1, 0], q[j2, 0]],
                        [q[j1, 1], q[j2, 1]],
                        color=st["color"],
                        linewidth=st["lw"],
                        alpha=st["alpha"],
                        linestyle=st["ls"],
                        marker="o",
                        markersize=8,
                        markerfacecolor="white",
                        markeredgecolor=st["color"],
                        markeredgewidth=2,
                        zorder=3,
                    )
                else:
                    coords_link = q[joints]
                    coords_closed = np.vstack([coords_link, coords_link[0]])
                    self.ax_main.fill(
                        coords_closed[:, 0],
                        coords_closed[:, 1],
                        color=st["color"],
                        alpha=0.25 if link_idx >= 0 else 0.15,
                        edgecolor=st["color"],
                        linewidth=st["lw"],
                        linestyle=st["ls"],
                        zorder=2,
                    )
                    self.ax_main.scatter(
                        coords_link[:, 0],
                        coords_link[:, 1],
                        s=100,
                        c="white",
                        edgecolors=st["color"],
                        linewidths=2,
                        zorder=5,
                    )

            # joint markers / labels
            for j in range(self.J):
                if j in self.dup_joint_set:
                    self.ax_main.scatter([q[j, 0]], [q[j, 1]], s=90, marker="X",
                                         c="lightgreen", edgecolors="green", linewidths=1.2, zorder=12)
                    label = f"D{j}"
                    face = "lightgreen"
                elif j in self.p_joint_set:
                    self.ax_main.scatter([q[j, 0]], [q[j, 1]], s=90, marker="o",
                                         c="lightskyblue", edgecolors="blue", linewidths=1.2, zorder=11)
                    label = f"P{j}"
                    face = "lightskyblue"
                else:
                    label = f"J{j}"
                    face = "yellow"

                self.ax_main.text(
                    q[j, 0],
                    q[j, 1] + margin * 0.05,
                    label,
                    fontsize=8,
                    ha="center",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor=face, alpha=0.7),
                    zorder=20,
                )

            title = f"Step: {frame}/{self.valid_steps - 1} | Time: {self.time[frame]:.3f} s"
            if self.topology_id is not None:
                title = f"[Topology ID: {self.topology_id}] " + title
            self.ax_main.set_title(title)

            # update vertical markers
            if react_vline is not None:
                react_vline.set_xdata([self.time[frame], self.time[frame]])
            if torque_vline is not None:
                torque_vline.set_xdata([self.time[frame], self.time[frame]])

            return []

        self.animation = FuncAnimation(
            self.fig,
            update,
            frames=self.valid_steps,
            init_func=init,
            interval=interval,
            blit=False,
            repeat=True,
        )

        plt.tight_layout()
        plt.show()
        return self.animation
