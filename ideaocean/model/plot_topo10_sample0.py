"""
topo10_sample0_data.pkl 시각화 스크립트.

Usage:
    python mjcf_export/plot_topo10_sample0.py
"""
from __future__ import annotations

import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── 데이터 로드 ──
PKL_PATH = os.path.join(os.path.dirname(__file__), "topo10_sample0_data.pkl")

with open(PKL_PATH, "rb") as f:
    sample = pickle.load(f)

dt = sample["time_step"]
pos = sample["joint_positions"]                     # (8, 2, 72) mm
theta = sample["rotation_angle_of_input_link"]      # (72,) rad
vel = sample["joint_velocities"]                    # (8, 2, 72) mm/s
vel_mag = sample["joint_velocity_magnitudes"]       # (8, 72) mm/s
acc = sample["joint_accelerations"]                 # (8, 2, 72) mm/s²
acc_mag = sample["joint_acceleration_magnitudes"]   # (8, 72) mm/s²
rf = sample["joint_reaction_forces"]                # (8, 2, 72) N
rf_mag = sample["joint_reaction_force_magnitudes"]  # (8, 72) N
torque = sample["motor_torque"]                     # (72,) N·m

J, _, T = pos.shape
time = np.arange(T) * dt

# ── Topology 10 관절-링크 매핑 ──
edge_labels = [
    "J0: virtual_1–ground (P)",
    "J1: ground–link_4 (input)",
    "J2: ground–link_5",
    "J3: link_1–link_3",
    "J4: link_1–link_5",
    "J5: link_2–link_3",
    "J6: link_2–link_4",
    "J7: virtual_1–link_3",
]
colors = plt.cm.tab10(np.arange(J))

# ══════════════════════════════════════════════════
# Figure 1: 관절 궤적 (XY 평면)
# ══════════════════════════════════════════════════
fig1, ax = plt.subplots(figsize=(8, 7))
for j in range(J):
    ax.plot(pos[j, 0], pos[j, 1], "-o", color=colors[j],
            markersize=2, label=edge_labels[j])
    ax.plot(pos[j, 0, 0], pos[j, 1, 0], "s", color=colors[j], markersize=7)
ax.set_xlabel("X (mm)")
ax.set_ylabel("Y (mm)")
ax.set_title("Joint Trajectories (XY Plane)")
ax.set_aspect("equal")
ax.legend(fontsize=7, loc="upper left")
ax.grid(True, alpha=0.3)
fig1.tight_layout()

# ══════════════════════════════════════════════════
# Figure 2: 입력 각도 & 모터 토크
# ══════════════════════════════════════════════════
fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

ax1.plot(time, np.degrees(theta), "b-o", markersize=3)
ax1.set_ylabel("Input Angle (deg)")
ax1.set_title("Input Link Rotation & Motor Torque")
ax1.grid(True, alpha=0.3)

ax2.plot(time, torque * 1000, "r-o", markersize=3)  # mN·m 로 표시
ax2.axhline(0, color="k", linewidth=0.5)
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Motor Torque (mN·m)")
ax2.grid(True, alpha=0.3)
fig2.tight_layout()

# ══════════════════════════════════════════════════
# Figure 3: 관절 속도 크기
# ══════════════════════════════════════════════════
fig3, ax = plt.subplots(figsize=(10, 5))
for j in range(J):
    ax.plot(time, vel_mag[j], color=colors[j], label=edge_labels[j])
ax.set_xlabel("Time (s)")
ax.set_ylabel("Velocity Magnitude (mm/s)")
ax.set_title("Joint Velocity Magnitudes")
ax.legend(fontsize=7, loc="upper right")
ax.grid(True, alpha=0.3)
fig3.tight_layout()

# ══════════════════════════════════════════════════
# Figure 4: 관절 가속도 크기
# ══════════════════════════════════════════════════
fig4, ax = plt.subplots(figsize=(10, 5))
for j in range(J):
    ax.plot(time, acc_mag[j], color=colors[j], label=edge_labels[j])
ax.set_xlabel("Time (s)")
ax.set_ylabel("Acceleration Magnitude (mm/s²)")
ax.set_title("Joint Acceleration Magnitudes")
ax.legend(fontsize=7, loc="upper right")
ax.grid(True, alpha=0.3)
fig4.tight_layout()

# ══════════════════════════════════════════════════
# Figure 5: 관절 반력 크기
# ══════════════════════════════════════════════════
fig5, ax = plt.subplots(figsize=(10, 5))
for j in range(J):
    ax.plot(time, rf_mag[j], color=colors[j], label=edge_labels[j])
ax.set_xlabel("Time (s)")
ax.set_ylabel("Reaction Force Magnitude (N)")
ax.set_title("Joint Reaction Force Magnitudes")
ax.legend(fontsize=7, loc="upper right")
ax.grid(True, alpha=0.3)
fig5.tight_layout()

# ══════════════════════════════════════════════════
# Figure 6: 반력 Fx, Fy 성분 (관절별 서브플롯)
# ══════════════════════════════════════════════════
fig6, axes = plt.subplots(4, 2, figsize=(14, 12), sharex=True)
fig6.suptitle("Joint Reaction Forces — Fx / Fy Components", fontsize=13)

for j in range(J):
    ax = axes[j // 2, j % 2]
    ax.plot(time, rf[j, 0], label="Fx", color="steelblue")
    ax.plot(time, rf[j, 1], label="Fy", color="indianred")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_ylabel("Force (N)")
    ax.set_title(edge_labels[j], fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

for ax in axes[-1]:
    ax.set_xlabel("Time (s)")
fig6.tight_layout()

# ══════════════════════════════════════════════════
# Figure 7: 종합 대시보드
# ══════════════════════════════════════════════════
fig7 = plt.figure(figsize=(16, 10))
fig7.suptitle("Topo10 Sample0 — Overview Dashboard", fontsize=14)
gs = GridSpec(3, 2, figure=fig7, hspace=0.35, wspace=0.3)

# (0,0) 궤적
ax = fig7.add_subplot(gs[0, 0])
for j in range(J):
    ax.plot(pos[j, 0], pos[j, 1], "-", color=colors[j], linewidth=1)
    ax.plot(pos[j, 0, 0], pos[j, 1, 0], "s", color=colors[j], markersize=5)
ax.set_xlabel("X (mm)")
ax.set_ylabel("Y (mm)")
ax.set_title("Joint Trajectories")
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)

# (0,1) 입력 각도
ax = fig7.add_subplot(gs[0, 1])
ax.plot(time, np.degrees(theta), "b-", linewidth=1.5)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Angle (deg)")
ax.set_title("Input Rotation (const ω)")
ax.grid(True, alpha=0.3)

# (1,0) 속도
ax = fig7.add_subplot(gs[1, 0])
for j in range(J):
    ax.plot(time, vel_mag[j], color=colors[j], linewidth=1)
ax.set_xlabel("Time (s)")
ax.set_ylabel("mm/s")
ax.set_title("Velocity Magnitudes")
ax.grid(True, alpha=0.3)

# (1,1) 가속도
ax = fig7.add_subplot(gs[1, 1])
for j in range(J):
    ax.plot(time, acc_mag[j], color=colors[j], linewidth=1)
ax.set_xlabel("Time (s)")
ax.set_ylabel("mm/s²")
ax.set_title("Acceleration Magnitudes")
ax.grid(True, alpha=0.3)

# (2,0) 반력
ax = fig7.add_subplot(gs[2, 0])
for j in range(J):
    ax.plot(time, rf_mag[j], color=colors[j], linewidth=1)
ax.set_xlabel("Time (s)")
ax.set_ylabel("N")
ax.set_title("Reaction Force Magnitudes")
ax.grid(True, alpha=0.3)

# (2,1) 토크
ax = fig7.add_subplot(gs[2, 1])
ax.plot(time, torque * 1000, "r-", linewidth=1.5)
ax.axhline(0, color="k", linewidth=0.5)
ax.set_xlabel("Time (s)")
ax.set_ylabel("mN·m")
ax.set_title("Motor Torque")
ax.grid(True, alpha=0.3)

fig7.tight_layout()

plt.show()
