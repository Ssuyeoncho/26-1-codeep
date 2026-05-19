import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import datetime

matplotlib.rcParams['font.size'] = 12

# ─── Output directory & run numbering ───
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

existing = [f for f in os.listdir(OUT_DIR) if f.startswith("run_") and f.endswith(".png")]
run_num = len(existing) + 1
run_tag = f"run_{run_num:03d}"
png_path = os.path.join(OUT_DIR, f"{run_tag}_mode_separation.png")
log_path = os.path.join(OUT_DIR, f"{run_tag}_log.txt")

# ─── Parameters ───
d = 3.0      # half-distance between modes: modes at -d and +d
sigma = 1.0  # std of each mode

# ─── R(alpha_bar): mode separation ratio ───
# R = 2*sqrt(alpha_bar)*d / sqrt(alpha_bar*sigma^2 + 1 - alpha_bar)
def R(alpha_bar, d=3.0, sigma=1.0):
    numerator = 2 * np.sqrt(alpha_bar) * d
    denominator = np.sqrt(alpha_bar * sigma**2 + (1 - alpha_bar))
    return numerator / denominator

# ─── Schedule definitions ───
# Linear: alpha_bar decreases linearly from 1 to ~0
def linear_schedule(t, T=1000):
    beta_start, beta_end = 0.0001, 0.02
    betas = np.linspace(beta_start, beta_end, T)
    idx = np.clip((t * T).astype(int), 0, T-1)
    alpha_bars = np.cumprod(1 - betas)
    return alpha_bars[idx]

# Cosine: alpha_bar = cos^2(pi*t/2)
def cosine_schedule(t):
    return np.cos(np.pi * t / 2) ** 2

# ─── alpha_bar and lambda arrays ───
ab = np.linspace(1e-4, 1 - 1e-4, 1000)
lam = np.log(ab / (1 - ab))  # lambda = log SNR

t_arr = np.linspace(0, 1 - 1e-4, 1000)
ab_linear = linear_schedule(t_arr)
ab_cosine = cosine_schedule(t_arr)

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# ──────────────────────────────
# Plot 1: R vs alpha_bar
# ──────────────────────────────
ax = axes[0]
ax.plot(ab, R(ab), color='#534AB7', linewidth=2.5)
ax.axhline(y=2, color='#888780', linestyle='--', alpha=0.6, linewidth=1)
ax.text(0.15, 2.15, 'R=2: modes start overlapping', fontsize=10, color='#888780')
ax.set_xlabel(r'$\bar{\alpha}$  (signal retention)', fontsize=13)
ax.set_ylabel(r'$R(\bar{\alpha})$  (mode separation ratio)', fontsize=13)
ax.set_title(r'Mode separation vs $\bar{\alpha}$', fontsize=14, fontweight='medium')
ax.set_xlim(0, 1)
ax.set_ylim(0, R(1.0) * 1.1)
ax.grid(True, alpha=0.2)

# ──────────────────────────────
# Plot 2: R vs lambda (log SNR)
# ──────────────────────────────
ax = axes[1]
ax.plot(lam, R(ab), color='#1D9E75', linewidth=2.5)
ax.axvline(x=0, color='#D85A30', linestyle='--', alpha=0.7, linewidth=1.5,
           label=r'$\lambda=0$ (signal = noise)')
ax.axhline(y=2, color='#888780', linestyle='--', alpha=0.6, linewidth=1)

# Shade the critical transition zone
mask = (lam > -3) & (lam < 3)
ax.fill_between(lam[mask], 0, R(ab[mask]), color='#D85A30', alpha=0.08)
ax.text(0.3, R(0.5) + 0.3, 'Critical zone', fontsize=10, color='#D85A30')

ax.set_xlabel(r'$\lambda = \log$ SNR', fontsize=13)
ax.set_ylabel(r'$R(\lambda)$  (mode separation ratio)', fontsize=13)
ax.set_title(r'Mode separation vs $\lambda$ (log SNR)', fontsize=14, fontweight='medium')
ax.set_xlim(-10, 10)
ax.set_ylim(0, R(1.0) * 1.1)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2)

# ──────────────────────────────
# Plot 3: R vs t (schedule comparison)
# ──────────────────────────────
ax = axes[2]
ax.plot(t_arr, R(ab_linear), color='#378ADD', linewidth=2.5, label='Linear schedule')
ax.plot(t_arr, R(ab_cosine), color='#D85A30', linewidth=2.5, label='Cosine schedule')
ax.axhline(y=2, color='#888780', linestyle='--', alpha=0.6, linewidth=1)

# Find where R crosses 2 for each schedule
lin_cross_idx = np.where(R(ab_linear) < 2)[0]
cos_cross_idx = np.where(R(ab_cosine) < 2)[0]
if len(lin_cross_idx) > 0:
    t_lin = t_arr[lin_cross_idx[0]]
    ax.axvline(x=t_lin, color='#378ADD', linestyle=':', alpha=0.5, linewidth=1)
    ax.text(t_lin + 0.02, 1.0, f't={t_lin:.2f}', fontsize=10, color='#378ADD')
if len(cos_cross_idx) > 0:
    t_cos = t_arr[cos_cross_idx[0]]
    ax.axvline(x=t_cos, color='#D85A30', linestyle=':', alpha=0.5, linewidth=1)
    ax.text(t_cos + 0.02, 0.5, f't={t_cos:.2f}', fontsize=10, color='#D85A30')

ax.set_xlabel(r'$t$  (timestep)', fontsize=13)
ax.set_ylabel(r'$R(t)$  (mode separation ratio)', fontsize=13)
ax.set_title('Linear vs Cosine: Structure collapse speed', fontsize=14, fontweight='medium')
ax.set_xlim(0, 1)
ax.set_ylim(0, R(1.0) * 1.1)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.2)

plt.tight_layout(pad=2.0)
plt.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# ─── Key values ───
r_at_1   = R(1.0)
r_at_05  = R(0.5)
r_at_0   = R(1e-8)
t_lin    = t_arr[lin_cross_idx[0]]  if len(lin_cross_idx) > 0 else None
t_cos    = t_arr[cos_cross_idx[0]]  if len(cos_cross_idx) > 0 else None

lines = [
    f"=== Run: {run_tag} ===",
    f"Timestamp : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    f"Output PNG: {png_path}",
    "",
    f"--- Parameters ---",
    f"d (half-distance between modes) = {d}",
    f"sigma (std of each mode)        = {sigma}",
    "",
    f"--- Key R values ---",
    f"R(alpha_bar=1.0) = {r_at_1:.4f}  (원본 분리도, pure signal)",
    f"R(alpha_bar=0.5) = {r_at_05:.4f}  (중간 노이즈)",
    f"R(alpha_bar->0 ) = {r_at_0:.6f}  (완전 노이즈)",
    "",
    f"--- Schedule crossover (R=2 도달 시점) ---",
    f"Linear schedule : t = {t_lin:.4f}" if t_lin is not None else "Linear schedule : R=2 미도달",
    f"Cosine schedule : t = {t_cos:.4f}" if t_cos is not None else "Cosine schedule : R=2 미도달",
]
if t_lin is not None and t_cos is not None:
    lines.append(f"→ Cosine이 Linear보다 t={t_cos - t_lin:.4f} 만큼 더 오래 구조 보존")

log_text = "\n".join(lines)

# Print to terminal
print(log_text)
print(f"\n[저장 완료] PNG → {png_path}")
print(f"[저장 완료] LOG → {log_path}")

# Save log file
with open(log_path, "w", encoding="utf-8") as f:
    f.write(log_text + "\n")