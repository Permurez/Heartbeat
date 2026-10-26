# === Progress figure for class: one subject, four panels ===
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rppg import Config
import rppg.pipeline as pipeline
from rppg.groundtruth import hr_spectral, bandpass

SUBJECT = 'subject1'
cfg = Config()
raw = pipeline.load_subject(cfg, SUBJECT)
res = pipeline.resample_subject(raw, cfg)

# --- HR per 8 s window: reference vs each ROI vs ROI average ---
W, step = int(cfg.window_s * cfg.fs), int(cfg.fs)
green = res['rgb_lin'][:, :, 1]
green_norm = green / green.mean(axis=0)

def hr(x):
    return hr_spectral(bandpass(x, cfg.fs, 0.7, 3.0), cfg.fs)

rows = []
for s in range(0, len(res['t']) - W + 1, step):
    sl = slice(s, s + W)
    rows.append({'t': res['t'][s] + cfg.window_s / 2,          # plot at window centre
                 'gt': hr(res['ppg'][sl]),
                 'forehead': hr(green[sl, 0]), 'left cheek': hr(green[sl, 1]),
                 'right cheek': hr(green[sl, 2]), 'all ROIs': hr(green_norm[sl].mean(axis=1))})
hr_df = pd.DataFrame(rows)
mae = {c: float((hr_df[c] - hr_df['gt']).abs().mean()) for c in ['forehead', 'left cheek', 'right cheek', 'all ROIs']}

# --- Figure ---
fig = plt.figure(figsize=(16, 9), constrained_layout=True)
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1.3, 1.3])
fig.suptitle(f'rPPG data pipeline: UBFC-rPPG {SUBJECT}', fontsize=16, fontweight='bold')

# (a) ROIs on the face
ax = fig.add_subplot(gs[:, 0])
ax.imshow(plt.imread(os.path.join(cfg.cache_dir, 'debug', f'{SUBJECT}.png')))
ax.set_title('(a) Face landmarks (MediaPipe) and skin ROIs')
ax.axis('off')

# (b) Irregular camera timing
ax = fig.add_subplot(gs[0, 1])
dt_ms = np.diff(raw['t']) * 1000
ax.plot(raw['t'][1:], dt_ms, lw=0.8)
ax.axhline(1000 / 30, color='k', ls='--', lw=1, label='ideal 30 FPS (33.3 ms)')
ax.set_title(f'(b) Frame intervals: jitter and dropped frames (max {dt_ms.max():.0f} ms)')
ax.set_xlabel('time [s]'); ax.set_ylabel('interval [ms]'); ax.legend(loc='upper right')

# (c) Reference PPG vs camera signal (green inverted: more blood = darker skin)
ax = fig.add_subplot(gs[0, 2])
p = bandpass(res['ppg'], cfg.fs, 0.7, 3.0)
g = -bandpass(green_norm.mean(axis=1), cfg.fs, 0.7, 3.0)
t0 = min(res['t'][0] + 20, res['t'][-1] - 8)        # 8 s excerpt starting at ~20 s
m = (res['t'] >= t0) & (res['t'] < t0 + 8)
ax.plot(res['t'][m], p[m] / p[m].std(), label='finger oximeter (reference)')
ax.plot(res['t'][m], g[m] / g[m].std(), label='camera, green channel (inverted)', alpha=0.85)
ax.set_title('(c) Pulse wave: finger vs face video (8 s)')
ax.set_ylim(-3, 4.5); ax.set_xlabel('time [s]'); ax.set_yticks([])
ax.legend(loc='upper center', ncol=2, fontsize=8, framealpha=0.9)

# (d) Heart rate over time
ax = fig.add_subplot(gs[1, 1:])
ax.plot(hr_df['t'], hr_df['gt'], 'k-', lw=2.5, label='reference')
for c in ['forehead', 'left cheek', 'right cheek']:
    ax.plot(hr_df['t'], hr_df[c], ':', lw=1.2, label=f'{c} (MAE {mae[c]:.1f})')
ax.plot(hr_df['t'], hr_df['all ROIs'], '-', color='tab:red', lw=2,
        label=f'average of 3 ROIs (MAE {mae["all ROIs"]:.1f})')
ax.set_title('(d) Heart rate per 8 s window: single ROIs fail at different moments, their average does not')
ax.set_xlabel('time [s]'); ax.set_ylabel('heart rate [BPM]')
ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=9)

out = os.path.join(cfg.cache_dir, f'progress_{SUBJECT}.png')
fig.savefig(out, dpi=150)
plt.show()
print('Saved to', out)
print('MAE [BPM]:', {k: round(v, 1) for k, v in mae.items()})
