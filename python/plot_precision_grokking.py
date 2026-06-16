#!/usr/bin/env python3
"""
IFO Grokking: Precision-Dependent Dynamics Plotter
Emmy Team, Amoy Studio (2026-06-11)

Generates publication-quality figures demonstrating:
1. Sisyphus Loop in Float32 vs. Laminar Condensation in Float64.
2. The exact alignment of Float32 collapses with the Softmax Overflow Limit (88.7).
"""

import matplotlib.pyplot as plt
import numpy as np

# =====================================================================
# 1. Hardcoded Experimental Data (From 50,000 Step Run)
# =====================================================================

steps = np.array([
    0, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 
    11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000, 
    21000, 22000, 23000, 24000, 25000, 26000, 27000, 28000, 29000, 30000, 
    31000, 32000, 33000, 34000, 35000, 36000, 37000, 38000, 39000, 40000, 
    41000, 42000, 43000, 44000, 45000, 46000, 47000, 48000, 49000, 50000
])

# --- Float32 Data ---
f32_val_acc = np.array([
    0.0382, 0.0000, 0.0000, 0.0000, 0.0000, 0.0035, 0.0278, 0.0556, 0.0833, 0.1215, 
    0.1597, 0.1875, 0.2083, 0.2222, 0.2396, 0.2500, 0.2778, 0.3160, 0.3681, 0.4028, 
    0.4062, 0.4062, 0.3611, 0.1806, 0.0556, 0.0868, 0.1215, 0.1528, 0.1736, 0.2465, 
    0.2951, 0.3229, 0.3507, 0.3403, 0.3368, 0.3160, 0.2847, 0.2778, 0.2674, 0.2639, 
    0.2639, 0.2743, 0.2812, 0.2882, 0.2882, 0.3125, 0.3229, 0.3333, 0.3021, 0.2674, 
    0.2118, 0.1840
])

f32_ccc = np.array([
    0.0642, 0.2795, 0.2760, 0.2812, 0.2743, 0.2899, 0.2969, 0.3108, 0.3316, 0.3524, 
    0.3767, 0.3993, 0.4097, 0.4167, 0.4236, 0.4306, 0.4479, 0.4757, 0.5000, 0.5295, 
    0.5382, 0.5417, 0.5087, 0.4010, 0.3194, 0.3229, 0.3628, 0.3889, 0.4115, 0.4549, 
    0.4670, 0.4913, 0.5156, 0.5122, 0.5052, 0.4948, 0.4670, 0.4549, 0.4479, 0.4462, 
    0.4444, 0.4549, 0.4514, 0.4497, 0.4514, 0.4653, 0.4740, 0.4861, 0.4688, 0.4410, 
    0.3993, 0.3733
])

f32_logit_range = np.array([
    1.6203, 21.9519, 22.2571, 22.2646, 23.2595, 24.4037, 25.4454, 26.0615, 26.4151, 26.9810, 
    27.7244, 28.8122, 29.8712, 30.8845, 32.1494, 33.1116, 34.1389, 34.6304, 35.2905, 35.7814, 
    37.1537, 39.3208, 57.4288, 179.9147, 1008.9596, 2110.9744, 1210.4069, 681.4344, 392.5067, 231.3696, 
    141.8839, 92.7437, 66.3640, 54.0627, 50.3387, 49.0090, 48.5750, 49.5091, 51.6417, 53.1500, 
    54.0867, 54.9419, 55.7647, 57.1011, 57.8635, 58.5466, 58.0621, 62.5071, 76.5710, 125.4415, 
    314.7757, 1241.7218
])

# --- Float64 Data ---
f64_val_acc = np.array([
    0.0347, 0.0000, 0.0000, 0.0000, 0.0139, 0.0208, 0.0417, 0.0833, 0.1250, 0.1354, 
    0.1493, 0.1562, 0.1632, 0.1806, 0.1910, 0.1944, 0.2049, 0.2153, 0.2153, 0.2257, 
    0.2639, 0.2812, 0.3264, 0.3542, 0.3889, 0.4201, 0.4653, 0.4896, 0.5139, 0.5660, 
    0.6042, 0.6319, 0.6562, 0.6910, 0.7188, 0.7431, 0.7778, 0.7917, 0.8194, 0.8368, 
    0.8507, 0.8715, 0.8750, 0.8993, 0.9167, 0.9201, 0.9306, 0.9340, 0.9479, 0.9549, 
    0.9618, 0.9653
])

f64_ccc = np.array([
    0.0590, 0.2847, 0.2812, 0.2917, 0.3056, 0.3056, 0.3090, 0.3403, 0.3646, 0.3733, 
    0.3819, 0.3785, 0.3819, 0.3906, 0.3958, 0.4028, 0.4132, 0.4184, 0.4167, 0.4167, 
    0.4462, 0.4479, 0.4670, 0.4931, 0.5243, 0.5486, 0.5764, 0.5920, 0.6111, 0.6510, 
    0.6823, 0.6997, 0.7205, 0.7413, 0.7639, 0.7847, 0.8108, 0.8212, 0.8385, 0.8542, 
    0.8663, 0.8837, 0.8837, 0.9080, 0.9253, 0.9288, 0.9358, 0.9375, 0.9497, 0.9566, 
    0.9618, 0.9653
])

f64_logit_range = np.array([
    1.8000, 22.0796, 22.0003, 22.1964, 22.8488, 24.1231, 25.9323, 27.9994, 29.7453, 31.1876, 
    32.4954, 33.5262, 34.6603, 35.8923, 37.0478, 37.9369, 38.7250, 39.4264, 39.9742, 40.2848, 
    40.3317, 40.0937, 39.5567, 38.7407, 38.3044, 37.7909, 37.3176, 37.2171, 37.0912, 37.0847, 
    37.1784, 37.2356, 37.3941, 37.5856, 37.7567, 37.9061, 38.0336, 38.1356, 38.2694, 38.4065, 
    38.5641, 38.7107, 38.8565, 38.9495, 39.0308, 39.1370, 39.2148, 39.3047, 39.3998, 39.4577, 
    39.4949, 39.5379
])

# =====================================================================
# 2. Plotting Configuration (Academic Style)
# =====================================================================

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'font.family': 'serif',
    'text.usetex': False  # Set to True if LaTeX is installed
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# ---------------------------------------------------------------------
# Left Plot: Performance & Causal Closure (Val Acc & CCC)
# ---------------------------------------------------------------------
# Float64 (Stable Condensation)
ax1.plot(steps, f64_val_acc, color='#1f77b4', linestyle='-', linewidth=2, label='Float64 Val Acc')
ax1.plot(steps, f64_ccc, color='#aec7e8', linestyle='--', linewidth=2, label='Float64 CCC')

# Float32 (Sisyphus Loops)
ax1.plot(steps, f32_val_acc, color='#d62728', linestyle='-', linewidth=2, label='Float32 Val Acc')
ax1.plot(steps, f32_ccc, color='#ff9896', linestyle='--', linewidth=2, label='Float32 CCC')

ax1.set_title("A. Causal & Generalization Dynamics")
ax1.set_xlabel("Training Steps")
ax1.set_ylabel("Metric Value")
ax1.set_ylim(-0.02, 1.05)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='lower right', framealpha=0.9)

# Annotate Sisyphus Collapses
ax1.annotate('1st Collapse', xy=(23000, 0.05), xytext=(27000, 0.15),
             arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6))
ax1.annotate('2nd Collapse', xy=(50000, 0.18), xytext=(42000, 0.28),
             arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6))

# ---------------------------------------------------------------------
# Right Plot: Logit Range & Softmax Overflow Limit (Log Scale)
# ---------------------------------------------------------------------
ax2.plot(steps, f64_logit_range, color='#1f77b4', linestyle='-', linewidth=2.5, label='Float64 Logit Range')
ax2.plot(steps, f32_logit_range, color='#d62728', linestyle='-', linewidth=2.5, label='Float32 Logit Range')

# Draw the critical Softmax Overflow Limit line (ln(3.4e38) ≈ 88.7)
overflow_limit = np.log(3.4e38)
ax2.axhline(y=overflow_limit, color='black', linestyle=':', linewidth=2, 
            label=f'Softmax Overflow Limit ({overflow_limit:.1f})')

ax2.set_yscale('log')
ax2.set_title("B. Microscopic Numerical Bounds (Log Scale)")
ax2.set_xlabel("Training Steps")
ax2.set_ylabel("Logit Range (max - min)")
ax2.grid(True, which="both", linestyle=':', alpha=0.6)
ax2.legend(loc='upper left', framealpha=0.9)

# Annotate the Overflow triggers
ax2.annotate('Overflow Triggered', xy=(23000, 1008), xytext=(12000, 300),
             arrowprops=dict(facecolor='red', shrink=0.08, width=1, headwidth=6))
ax2.annotate('Overflow Triggered', xy=(50000, 1241), xytext=(35000, 400),
             arrowprops=dict(facecolor='red', shrink=0.08, width=1, headwidth=6))

# =====================================================================
# 3. Save and Show
# =====================================================================
plt.tight_layout()
output_filename = "ifo_grokking_precision_dynamics.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"\n[Success] Academic-grade plot saved to: {output_filename}")
plt.show()