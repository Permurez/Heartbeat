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
window_length_samples, step_samples = int(cfg.window_s * cfg.sample_rate_hz), int(cfg.sample_rate_hz)
green_trace = res['rgb_lin'][:, :, 1]
green_norm = green_trace / green_trace.mean(axis=0)

def heart_rate_bpm(signal):
    return hr_spectral(bandpass(signal, cfg.sample_rate_hz, 0.7, 3.0), cfg.sample_rate_hz)

rows = []
for start in range(0, len(res['timestamps_s']) - window_length_samples + 1, step_samples):
    window_slice = slice(start, start + window_length_samples)
    rows.append({'t': res['timestamps_s'][start] + cfg.window_s / 2,          # plot at window centre
                 'gt': heart_rate_bpm(res['ppg'][window_slice]),
                 'forehead': heart_rate_bpm(green_trace[window_slice, 0]),
                 'left cheek': heart_rate_bpm(green_trace[window_slice, 1]),
                 'right cheek': heart_rate_bpm(green_trace[window_slice, 2]),
                 'all ROIs': heart_rate_bpm(green_norm[window_slice].mean(axis=1))})
hr_df = pd.DataFrame(rows)
mean_absolute_error_bpm = {c: float((hr_df[c] - hr_df['gt']).abs().mean())
                           for c in ['forehead', 'left cheek', 'right cheek', 'all ROIs']}

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
frame_interval_ms = np.diff(raw['timestamps_s']) * 1000
ax.plot(raw['timestamps_s'][1:], frame_interval_ms, lw=0.8)
ax.axhline(1000 / 30, color='k', ls='--', lw=1, label='ideal 30 FPS (33.3 ms)')
ax.set_title(f'(b) Frame intervals: jitter and dropped frames (max {frame_interval_ms.max():.0f} ms)')
ax.set_xlabel('time [s]'); ax.set_ylabel('interval [ms]'); ax.legend(loc='upper right')

# (c) Reference PPG vs camera signal (green inverted: more blood = darker skin)
ax = fig.add_subplot(gs[0, 2])
reference_filtered_signal = bandpass(res['ppg'], cfg.sample_rate_hz, 0.7, 3.0)
camera_filtered_signal = -bandpass(green_norm.mean(axis=1), cfg.sample_rate_hz, 0.7, 3.0)
excerpt_start_s = min(res['timestamps_s'][0] + 20, res['timestamps_s'][-1] - 8)   # 8 s excerpt starting at ~20 s
m = (res['timestamps_s'] >= excerpt_start_s) & (res['timestamps_s'] < excerpt_start_s + 8)
ax.plot(res['timestamps_s'][m], reference_filtered_signal[m] / reference_filtered_signal[m].std(),
        label='finger oximeter (reference)')
ax.plot(res['timestamps_s'][m], camera_filtered_signal[m] / camera_filtered_signal[m].std(),
        label='camera, green channel (inverted)', alpha=0.85)
ax.set_title('(c) Pulse wave: finger vs face video (8 s)')
ax.set_ylim(-3, 4.5); ax.set_xlabel('time [s]'); ax.set_yticks([])
ax.legend(loc='upper center', ncol=2, fontsize=8, framealpha=0.9)

# (d) Heart rate over time
ax = fig.add_subplot(gs[1, 1:])
ax.plot(hr_df['t'], hr_df['gt'], 'k-', lw=2.5, label='reference')
for c in ['forehead', 'left cheek', 'right cheek']:
    ax.plot(hr_df['t'], hr_df[c], ':', lw=1.2, label=f'{c} (MAE {mean_absolute_error_bpm[c]:.1f})')
ax.plot(hr_df['t'], hr_df['all ROIs'], '-', color='tab:red', lw=2,
        label=f'average of 3 ROIs (MAE {mean_absolute_error_bpm["all ROIs"]:.1f})')
ax.set_title('(d) Heart rate per 8 s window: single ROIs fail at different moments, their average does not')
ax.set_xlabel('time [s]'); ax.set_ylabel('heart rate [BPM]')
ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=9)

out = os.path.join(cfg.cache_dir, f'progress_{SUBJECT}.png')
fig.savefig(out, dpi=150)
plt.show()
print('Saved to', out)
print('MAE [BPM]:', {k: round(v, 1) for k, v in mean_absolute_error_bpm.items()})
