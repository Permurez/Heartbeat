"""Live, step-by-step rPPG demo: one window per processing stage.

Run it from the repo root (or anywhere):

    python scripts/live_demo.py                  # a small window asks: camera or video file
    python scripts/live_demo.py --source 0       # camera number 0, no question
    python scripts/live_demo.py --source clip.mp4

Windows (all text on screen is Polish, code and comments are English):
    1  face + the 478 MediaPipe landmarks (yellow = landmarks used to build the skin regions)
    2  the three skin regions (forehead, left cheek, right cheek) and how many pixels each has
    3  mean green value of every region over the last window_s seconds
    4  FFT of every region BEFORE the filter
    5  Butterworth band-pass: signal before, signal after, and what was cut out
    6  FFT of every region AFTER the filter + the chosen peak = heart rate

Keys: SPACE = pause / resume, Q or Esc = quit.

Everything numerical is done by the existing rppg package (roi, timeseries, groundtruth),
so this script only collects data and draws pictures.

Note: the demo plots the raw (gamma-encoded) green mean, 0-255, because it is easier to
read on screen. The notebooks use the linear-light version; both give the same pulse.
"""
import argparse
import sys
import time
from collections import deque
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import detrend

ROOT = Path(__file__).resolve().parents[1]      # repo root, so "import rppg" works from anywhere
sys.path.insert(0, str(ROOT))

from rppg import roi                                     # noqa: E402
from rppg.config import Config                           # noqa: E402
from rppg.groundtruth import bandpass, hr_spectral       # noqa: E402
from rppg.timeseries import resample_uniform             # noqa: E402

# --- Settings ---------------------------------------------------------------
MODEL_PATH = Path.home() / ".cache" / "heartbeat" / "face_landmarker.task"   # downloaded once (~4 MB)
VIDEO_W = 640                 # width of the two video windows
PLOT_W, PLOT_H = 640, 410     # size of the plot windows (window 6 is taller)
PLOT6_H = 460
ANALYSIS_EVERY_S = 0.25       # recompute spectra this often (source time)
FACE_LOST_RESET_S = 1.0       # forget collected data if the face is gone this long
SPEC_MAX_BPM = 240            # right edge of the spectrum plots
FFT_LENGTH = 2 ** 14           # zero-padding of the FFT (same as groundtruth.hr_spectral)
AGREE_BPM = 5.0               # regions "agree" if their peaks are within this many BPM

# Colours are BGR (OpenCV order). Text colours go through the same order and are flipped when drawn.
ROI_COLORS = {"forehead": (60, 160, 255), "left_cheek": (255, 200, 60), "right_cheek": (200, 90, 255)}
ROI_LABELS = {"forehead": "czoło", "left_cheek": "policzek L", "right_cheek": "policzek P"}  # L/P as seen in the image
LIME, WHITE, GREY, RED, BAND = (80, 230, 120), (240, 240, 240), (150, 150, 150), (90, 90, 255), (200, 140, 60)


# --- Text drawing (OpenCV fonts cannot draw Polish letters, Pillow can) ------
FONT_NAMES = ["DejaVuSans.ttf", "LiberationSans-Regular.ttf", "NotoSans-Regular.ttf", "Arial.ttf"]


@lru_cache(maxsize=None)
def get_font(size):
    for name in FONT_NAMES:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


class Canvas:
    """A BGR image plus a queue of texts. Shapes are drawn with OpenCV right away,
    texts are collected and drawn in one Pillow pass in render() (much faster)."""

    def __init__(self, img):
        self.img = img
        self.texts = []

    @classmethod
    def blank(cls, w, h, bg=(26, 26, 26)):
        return cls(np.full((h, w, 3), bg, np.uint8))

    def text(self, xy, text, size=14, color=WHITE, anchor="lt", outline=False):
        self.texts.append((xy, text, size, color, anchor, outline))

    def render(self):
        if not self.texts:
            return self.img
        pil = Image.fromarray(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        for xy, text, size, color, anchor, outline in self.texts:
            kw = {"stroke_width": 2, "stroke_fill": (0, 0, 0)} if outline else {}
            draw.text(xy, text, font=get_font(size), fill=color[::-1], anchor=anchor, **kw)
        return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


class Panel:
    """A plot area inside a Canvas. Converts data coordinates to pixels and draws on it."""

    def __init__(self, canvas, rect, xlim, ylim):
        self.c = canvas
        self.x0, self.y0, self.x1, self.y1 = rect
        self.xlim, self.ylim = xlim, ylim

    def px(self, xs, ys):
        xs, ys = np.asarray(xs, float), np.asarray(ys, float)
        fx = (xs - self.xlim[0]) / (self.xlim[1] - self.xlim[0])
        fy = (ys - self.ylim[0]) / (self.ylim[1] - self.ylim[0])
        return self.x0 + fx * (self.x1 - self.x0), self.y1 - fy * (self.y1 - self.y0)

    def frame(self):
        cv2.rectangle(self.c.img, (self.x0, self.y0), (self.x1, self.y1), (90, 90, 90), 1)

    def band(self, xa, xb, color, alpha=0.18):
        """Shade the vertical strip between data x-values xa and xb."""
        (pa, pb), _ = self.px([xa, xb], [0, 0])
        pa, pb = int(max(self.x0, pa)), int(min(self.x1, pb))
        if pb <= pa:
            return
        sub = self.c.img[self.y0:self.y1, pa:pb]
        sub[:] = (sub * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)

    def line(self, xs, ys, color, thickness=1):
        X, Y = self.px(xs, ys)
        Y = np.clip(Y, self.y0, self.y1)
        pts = np.round(np.stack([X, Y], axis=1)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(self.c.img, [pts], False, color, thickness, cv2.LINE_AA)

    def vline(self, x, color, thickness=1):
        (X,), _ = self.px([x], [0])
        cv2.line(self.c.img, (int(X), self.y0), (int(X), self.y1), color, thickness, cv2.LINE_AA)

    def dot(self, x, y, color, r=4):
        (X,), (Y,) = self.px([x], [y])
        cv2.circle(self.c.img, (int(X), int(Y)), r, color, -1, cv2.LINE_AA)

    def xticks(self, ticks, fmt="{:g}"):
        for v in ticks:
            (X,), _ = self.px([v], [0])
            cv2.line(self.c.img, (int(X), self.y1), (int(X), self.y1 + 4), (120, 120, 120), 1)
            self.c.text((int(X), self.y1 + 5), fmt.format(v), 11, GREY, "mt")

    def ylabels(self, fmt="{:.1f}"):
        self.c.text((self.x0 - 4, self.y0), fmt.format(self.ylim[1]), 11, GREY, "rt")
        self.c.text((self.x0 - 4, self.y1), fmt.format(self.ylim[0]), 11, GREY, "rb")


def auto_ylim(arrays, pad=0.15):
    lo = min(float(np.min(a)) for a in arrays)
    hi = max(float(np.max(a)) for a in arrays)
    if hi - lo < 1e-9:
        lo, hi = lo - 1, hi + 1
    margin = (hi - lo) * pad
    return lo - margin, hi + margin


# --- Signal processing -------------------------------------------------------
def spectrum(signal, sample_rate_hz, max_bpm=SPEC_MAX_BPM):
    """Magnitude spectrum with the same maths as groundtruth.hr_spectral.
    Returns (bpm_axis, magnitude). 1 Hz = 60 BPM."""
    signal = detrend(np.asarray(signal, float))
    magnitude = np.abs(np.fft.rfft(signal * np.hanning(len(signal)), n=FFT_LENGTH))
    bpm_axis = np.fft.rfftfreq(FFT_LENGTH, d=1.0 / sample_rate_hz) * 60.0
    keep = bpm_axis <= max_bpm
    return bpm_axis[keep], magnitude[keep]


def band_peak(bpm_axis, magnitude, low_bpm, high_bpm):
    """(BPM of the highest peak inside the band, how many times higher it is than the band's median)."""
    inside = (bpm_axis >= low_bpm) & (bpm_axis <= high_bpm)
    i = int(np.argmax(magnitude[inside]))
    return float(bpm_axis[inside][i]), float(magnitude[inside][i] / max(np.median(magnitude[inside]), 1e-12))


class Analyzer:
    """Collects (time, green mean of every ROI) and runs the whole signal chain on the last window."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.time_s = deque()
        self.green_trace = deque()

    def reset(self):
        self.time_s.clear()
        self.green_trace.clear()

    def add(self, time_s, green3):
        self.time_s.append(float(time_s))
        self.green_trace.append(np.asarray(green3, float))
        while self.time_s[-1] - self.time_s[0] > self.cfg.window_s + 1.0:      # keep one spare second
            self.time_s.popleft()
            self.green_trace.popleft()

    def span(self):
        return self.time_s[-1] - self.time_s[0] if len(self.time_s) >= 2 else 0.0

    def recent(self):
        """Raw samples of the last window_s seconds: (time, (N, 3) green means) or (None, None)."""
        if len(self.time_s) < 2:
            return None, None
        time_s, green_trace = np.asarray(self.time_s), np.asarray(self.green_trace)
        keep = time_s >= time_s[-1] - self.cfg.window_s
        return time_s[keep], green_trace[keep]

    def analyze(self):
        """Returns a dict with every intermediate result, or None until a full window is collected."""
        cfg = self.cfg
        if self.span() < cfg.window_s:
            return None
        time_s, green_trace = np.asarray(self.time_s), np.asarray(self.green_trace)
        # Step 5 of the pipeline: irregular camera timestamps -> even 30 Hz grid (cubic spline)
        resampled_time_s, resampled_green, valid, _ = resample_uniform(
            time_s, green_trace, cfg.sample_rate_hz, "spline", cfg.max_gap_s)
        n_samples = int(round(cfg.window_s * cfg.sample_rate_hz))
        if len(resampled_time_s) < n_samples:
            return None
        window_time_s = resampled_time_s[-n_samples:]
        raw = resampled_green[-n_samples:]
        valid = valid[-n_samples:]
        low_cutoff_hz, high_cutoff_hz = cfg.heart_rate_min_bpm / 60, cfg.heart_rate_max_bpm / 60

        detrended = detrend(raw, axis=0)                                   # remove the slow drift line
        filtered = np.stack([bandpass(detrended[:, i], cfg.sample_rate_hz, low_cutoff_hz, high_cutoff_hz)
                             for i in range(raw.shape[1])], axis=1)
        mean_detrended, mean_filtered = detrended.mean(axis=1), filtered.mean(axis=1)   # average of the 3 regions

        spectra_before = [spectrum(detrended[:, i], cfg.sample_rate_hz) for i in range(raw.shape[1])]
        spectra_after = [spectrum(filtered[:, i], cfg.sample_rate_hz) for i in range(raw.shape[1])]
        mean_spectrum = spectrum(mean_filtered, cfg.sample_rate_hz)
        roi_peak_bpm_list = [band_peak(bpm_axis, magnitude, cfg.heart_rate_min_bpm, cfg.heart_rate_max_bpm)[0]
                            for bpm_axis, magnitude in spectra_after]
        _, peak_prominence_ratio = band_peak(*mean_spectrum, cfg.heart_rate_min_bpm, cfg.heart_rate_max_bpm)
        return {
            "time_s": window_time_s - window_time_s[-1], "raw": raw, "detrended": detrended, "filtered": filtered,
            "mean_detrended": mean_detrended, "mean_filtered": mean_filtered,
            "spectra_before": spectra_before, "spectra_after": spectra_after, "mean_spectrum": mean_spectrum,
            "roi_peak_bpm_list": roi_peak_bpm_list, "peak_prominence_ratio": peak_prominence_ratio,
            "heart_rate_bpm": hr_spectral(mean_filtered, cfg.sample_rate_hz, cfg.heart_rate_min_bpm,
                                          cfg.heart_rate_max_bpm),
            "valid_fraction": float(valid.mean()),
        }


# --- Drawing the six windows -------------------------------------------------
def waiting(cv, analyzer, cfg):
    """'Collecting data' message with a progress bar, shown until a full window exists."""
    span = analyzer.span() if analyzer is not None else 0.0
    cv.text((PLOT_W // 2, 170), "Zbieram dane…", 20, WHITE, "mm")
    cv.text((PLOT_W // 2, 200), f"{min(span, cfg.window_s):.1f} / {cfg.window_s:.0f} s", 14, GREY, "mm")
    x0, x1, y = 120, PLOT_W - 120, 225
    cv2.rectangle(cv.img, (x0, y), (x1, y + 10), (90, 90, 90), 1)
    done = int((x1 - x0) * min(span / cfg.window_s, 1.0))
    cv2.rectangle(cv.img, (x0, y), (x0 + done, y + 10), LIME, -1)
    return cv.render()


def header(cv, title, subtitle):
    cv.text((10, 8), title, 18)
    cv.text((10, 33), subtitle, 12, GREY)


def build_green_window(analyzer, cfg):
    cv = Canvas.blank(PLOT_W, PLOT_H)
    header(cv, "3 · Zielony kanał w czasie",
           f"średnia zieleń (G, 0–255) wewnątrz każdego obszaru, ostatnie {cfg.window_s:.0f} s")
    time_s, green_trace = analyzer.recent()
    if time_s is None:
        cv.text((PLOT_W // 2, 200), "czekam na twarz…", 18, GREY, "mm")
        return cv.render()
    relative_time_s = time_s - time_s[-1]
    for i, name in enumerate(roi.ROI_NAMES):
        y0 = 58 + i * 108
        p = Panel(cv, (64, y0, PLOT_W - 16, y0 + 92), (-cfg.window_s, 0), auto_ylim([green_trace[:, i]]))
        p.frame()
        p.line(relative_time_s, green_trace[:, i], ROI_COLORS[name], 2)
        p.ylabels()
        cv.text((p.x0 + 6, p.y0 + 4), ROI_LABELS[name], 12, ROI_COLORS[name])
        if i == len(roi.ROI_NAMES) - 1:
            p.xticks(range(-int(cfg.window_s), 1, 2))
    cv.text((PLOT_W // 2, PLOT_H - 14), "czas [s] (0 = teraz)", 12, GREY, "mm")
    return cv.render()


def build_spectra_before_window(analysis, analyzer, cfg):
    cv = Canvas.blank(PLOT_W, PLOT_H)
    header(cv, "4 · Widmo FFT przed filtrem",
           "FFT rozkłada sygnał na rytmy: im wyższy słupek, tym mocniejszy rytm")
    if analysis is None:
        return waiting(cv, analyzer, cfg)
    ymax = max(float(magnitude.max()) for _, magnitude in analysis["spectra_before"])
    p = Panel(cv, (40, 64, PLOT_W - 16, 300), (0, SPEC_MAX_BPM), (0, ymax * 1.1))
    p.band(cfg.heart_rate_min_bpm, cfg.heart_rate_max_bpm, BAND)
    p.frame()
    for name, (bpm_axis, magnitude) in zip(roi.ROI_NAMES, analysis["spectra_before"]):
        p.line(bpm_axis, magnitude, ROI_COLORS[name], 2)
    p.xticks(range(0, SPEC_MAX_BPM + 1, 30))
    cv.text((PLOT_W // 2, 322), "częstotliwość w BPM (uderzenia na minutę = Hz × 60)", 12, GREY, "mm")
    x = 40
    for name in roi.ROI_NAMES:
        cv.text((x, 342), "■ " + ROI_LABELS[name], 12, ROI_COLORS[name])
        x += 120
    cv.text((10, 366), f"Niebieskie tło = pasmo {cfg.heart_rate_min_bpm:.0f}–{cfg.heart_rate_max_bpm:.0f} BPM, "
                      "które filtr przepuści.", 12)
    cv.text((10, 384), "Wszystko poza tłem (dryf światła, szum) zostanie wycięte.", 12)
    return cv.render()


def build_filter_window(analysis, analyzer, cfg):
    cv = Canvas.blank(PLOT_W, PLOT_H)
    header(cv, f"5 · Filtr Butterwortha {cfg.heart_rate_min_bpm / 60:.1f}–{cfg.heart_rate_max_bpm / 60:.1f} Hz: "
               "przed i po",
           "średnia z trzech obszarów, po usunięciu trendu")
    if analysis is None:
        return waiting(cv, analyzer, cfg)
    time_s = analysis["time_s"]
    lim = float(np.abs(analysis["mean_detrended"]).max()) * 1.15
    top = Panel(cv, (64, 58, PLOT_W - 16, 210), (-cfg.window_s, 0), (-lim, lim))
    top.frame()
    top.line(time_s, analysis["mean_detrended"], GREY, 1)
    top.line(time_s, analysis["mean_filtered"], LIME, 2)
    top.ylabels("{:+.2f}")
    cv.text((top.x0 + 6, top.y0 + 4), "szary = przed filtrem", 12, GREY)
    cv.text((top.x0 + 6, top.y0 + 20), "zielony = po filtrze (czysty puls)", 12, LIME)
    removed = analysis["mean_detrended"] - analysis["mean_filtered"]
    lim2 = max(float(np.abs(removed).max()) * 1.15, 1e-6)
    bot = Panel(cv, (64, 240, PLOT_W - 16, 340), (-cfg.window_s, 0), (-lim2, lim2))
    bot.frame()
    bot.line(time_s, removed, RED, 1)
    bot.ylabels("{:+.2f}")
    bot.xticks(range(-int(cfg.window_s), 1, 2))
    cv.text((bot.x0 + 6, bot.y0 + 4), "czerwony = to, co filtr wyciął", 12, RED)
    cv.text((PLOT_W // 2, 366), "czas [s] (0 = teraz)", 12, GREY, "mm")
    cv.text((10, 384), "Szary minus zielony = czerwony: wolne zmiany światła i szybki szum.", 12)
    return cv.render()


def build_spectra_after_window(analysis, analyzer, cfg):
    cv = Canvas.blank(PLOT_W, PLOT6_H)
    header(cv, "6 · Widmo FFT po filtrze i wynik", "każdy obszar osobno + średnia z trzech (biała)")
    if analysis is None:
        return waiting(cv, analyzer, cfg)
    heart_rate_bpm = analysis["heart_rate_bpm"]
    cv.text((PLOT_W - 12, 6), f"{heart_rate_bpm:.0f} BPM", 40, LIME, "rt")
    ymax = max(float(magnitude.max()) for _, magnitude in analysis["spectra_after"] + [analysis["mean_spectrum"]])
    p = Panel(cv, (40, 70, PLOT_W - 16, 330), (0, SPEC_MAX_BPM), (0, ymax * 1.15))
    p.band(cfg.heart_rate_min_bpm, cfg.heart_rate_max_bpm, BAND)
    p.frame()
    for name, (bpm_axis, magnitude), peak in zip(roi.ROI_NAMES, analysis["spectra_after"],
                                                 analysis["roi_peak_bpm_list"]):
        p.line(bpm_axis, magnitude, ROI_COLORS[name], 1)
        p.dot(peak, float(np.interp(peak, bpm_axis, magnitude)), ROI_COLORS[name])
    p.line(*analysis["mean_spectrum"], WHITE, 2)
    p.vline(heart_rate_bpm, LIME, 1)
    cv.text((int(p.px([heart_rate_bpm], [0])[0][0]) + 6, p.y0 + 4),
            f"{heart_rate_bpm:.1f} BPM = {heart_rate_bpm / 60:.2f} Hz", 12, LIME)
    p.xticks(range(0, SPEC_MAX_BPM + 1, 30))
    cv.text((PLOT_W // 2, 352), "częstotliwość w BPM", 12, GREY, "mm")
    roi_peak_bpm_list = analysis["roi_peak_bpm_list"]
    peaks_agree = max(roi_peak_bpm_list) - min(roi_peak_bpm_list) <= AGREE_BPM
    lines = [
        "Cienkie krzywe = jeden obszar po filtrze, biała = średnia z trzech.",
        f"Tętno = częstotliwość najwyższego szczytu w paśmie "
        f"{cfg.heart_rate_min_bpm:.0f}–{cfg.heart_rate_max_bpm:.0f} BPM.",
        f"Ten szczyt jest {analysis['peak_prominence_ratio']:.1f}× wyższy niż typowy poziom w paśmie.",
        "Szczyty obszarów: " + " · ".join(
            f"{ROI_LABELS[n]} {v:.0f}" for n, v in zip(roi.ROI_NAMES, roi_peak_bpm_list))
        + (f" — zgodne (±{AGREE_BPM:.0f} BPM)" if peaks_agree else " — NIEZGODNE, wynik niepewny"),
    ]
    if analysis["valid_fraction"] < 0.9:
        lines.append(f"Uwaga: tylko {analysis['valid_fraction'] * 100:.0f}% próbek zmierzonych "
                     "(reszta dorysowana).")
    for i, line_text in enumerate(lines):
        cv.text((10, 372 + i * 17), line_text, 12,
               RED if "NIEZGODNE" in line_text or line_text.startswith("Uwaga") else WHITE)
    return cv.render()


def build_landmarks_window(frame_bgr, landmarks_px, used_landmark_indices, status):
    """Window 1: the face with all 478 landmarks; the ones used for the regions are yellow."""
    scale = VIDEO_W / frame_bgr.shape[1]
    img = cv2.resize(frame_bgr, (VIDEO_W, int(round(frame_bgr.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    cv = Canvas(img)
    if landmarks_px is None:
        cv.text((12, 10), "1 · brak twarzy w kadrze", 18, RED, outline=True)
    else:
        scaled_points = np.round(landmarks_px * scale).astype(int)
        for i, (x, y) in enumerate(scaled_points):
            if i in used_landmark_indices:
                cv2.circle(img, (x, y), 3, (0, 255, 255), -1, cv2.LINE_AA)
            else:
                cv2.circle(img, (x, y), 1, (0, 200, 0), -1, cv2.LINE_AA)
        cv.text((12, 10), "1 · twarz i punkty MediaPipe", 18, WHITE, outline=True)
        cv.text((12, 34), f"{len(landmarks_px)} punktów, żółte = z nich budujemy obszary", 13,
               (0, 255, 255), outline=True)
    cv.text((12, img.shape[0] - 8), status, 12, WHITE, "lb", outline=True)
    return cv.render()


def build_roi_window(frame_bgr, roi_shapes, pixel_counts):
    """Window 2: the three skin regions filled in, with their pixel counts."""
    frame_height_px, frame_width_px = frame_bgr.shape[:2]
    scale = VIDEO_W / frame_width_px
    img = cv2.resize(frame_bgr, (VIDEO_W, int(round(frame_height_px * scale))), interpolation=cv2.INTER_AREA)
    cv = Canvas(img)
    if roi_shapes is None:
        cv.text((12, 10), "2 · brak twarzy w kadrze", 18, RED, outline=True)
        return cv.render()
    overlay = img.copy()
    scaled_shapes = {k: np.round(v.astype(np.float32) * scale).astype(np.int32) for k, v in roi_shapes.items()}
    for name, shape in scaled_shapes.items():
        cv2.fillPoly(overlay, [shape], ROI_COLORS[name])
    img[:] = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
    for name, shape in scaled_shapes.items():
        cv2.polylines(img, [shape], True, ROI_COLORS[name], 2, cv2.LINE_AA)
    cv.text((12, 10), "2 · obszary skóry", 18, WHITE, outline=True)
    cv.text((12, 34), "z każdego liczymy średnie R, G, B (każdy piksel w środku)", 13, WHITE, outline=True)
    for i, name in enumerate(roi.ROI_NAMES):
        cx, cy = scaled_shapes[name].reshape(-1, 2).mean(axis=0)
        cv.text((int(cx), int(cy) - 8), ROI_LABELS[name], 12, WHITE, "mm", outline=True)
        cv.text((int(cx), int(cy) + 8), f"{int(pixel_counts[i])} px", 12, WHITE, "mm", outline=True)
    return cv.render()


# --- Source selection --------------------------------------------------------
def choose_source():
    """Small window: camera (with a camera number) or a video file. Returns int, path string, or None."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        print("tkinter is not installed (sudo apt install python3-tk) - using camera 0.\n"
              "Or pass the source yourself: python scripts/live_demo.py --source clip.mp4")
        return 0
    result = {"value": None}
    root = tk.Tk()
    root.title("rPPG – wybierz źródło obrazu")
    root.resizable(False, False)
    tk.Label(root, text="Skąd brać obraz?", font=("TkDefaultFont", 12, "bold")).pack(padx=24, pady=(16, 8))
    row = tk.Frame(root)
    row.pack(padx=24, pady=4)
    tk.Label(row, text="Numer kamerki:").pack(side="left")
    cam_var = tk.StringVar(value="0")
    tk.Spinbox(row, from_=0, to=9, width=3, textvariable=cam_var).pack(side="left", padx=6)

    def use_camera():
        result["value"] = int(cam_var.get() or 0)
        root.destroy()

    def use_file():
        path = filedialog.askopenfilename(
            title="Wybierz plik wideo",
            filetypes=[("Wideo", "*.mp4 *.avi *.mov *.mkv"), ("Wszystkie pliki", "*.*")])
        if path:
            result["value"] = path
            root.destroy()

    tk.Button(root, text="Kamerka", width=24, command=use_camera).pack(padx=24, pady=(8, 4))
    tk.Button(root, text="Plik wideo…", width=24, command=use_file).pack(padx=24, pady=(4, 16))
    root.mainloop()
    return result["value"]


def open_source(source):
    cap = cv2.VideoCapture(source)
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


# --- Main loop ---------------------------------------------------------------
WINDOWS = [   # (title, column, row) - ASCII titles, because window titles are also the lookup keys
    ("1 twarz + punkty", 0, 0), ("2 obszary skory", 1, 0), ("3 zielen w czasie", 2, 0),
    ("4 FFT przed filtrem", 0, 1), ("5 Butterworth przed-po", 1, 1), ("6 FFT po filtrze + BPM", 2, 1),
]


def main():
    parser = argparse.ArgumentParser(description="Live step-by-step rPPG demo")
    parser.add_argument("--source", help="camera number (e.g. 0) or path to a video file; omit to get a chooser window")
    args = parser.parse_args()

    source = args.source
    if source is None:
        source = choose_source()
    elif source.isdigit():
        source = int(source)
    if source is None:
        print("No source chosen, exiting.")
        return
    is_file = not isinstance(source, int)

    try:
        import mediapipe as mp
    except ImportError:
        sys.exit("mediapipe is not installed for this Python. Try: pip install mediapipe")

    cfg = Config()
    print("Checking the face-landmark model (downloaded once, ~4 MB)...")
    roi.ensure_model(str(MODEL_PATH), cfg.model_url)
    landmarker = roi.create_landmarker(str(MODEL_PATH))
    cap = open_source(source)
    if not cap.isOpened():
        sys.exit(f"Cannot open the source: {source!r}")
    sample_rate_hz = cap.get(cv2.CAP_PROP_FPS) if is_file else 0.0
    sample_rate_hz = sample_rate_hz if sample_rate_hz and sample_rate_hz > 1 else 30.0

    lut = roi.srgb_to_linear_lut()
    used_landmark_indices = {i for indices in roi.ROI_LANDMARKS.values() for i in indices}
    analyzer = Analyzer(cfg)

    for title, col, row in WINDOWS:
        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow(title, col * (VIDEO_W + 20), row * 540)

    shown = {}                      # last image per window, so paused windows keep their picture
    playback_start_time_s, frame_index, last_frame_timestamp_ms = time.monotonic(), 0, -1
    last_analysis_time_s, analysis, face_lost_since = -1e9, None, None
    paused = ended = False
    landmarks_px = roi_shapes = pixel_counts = None
    status = "SPACJA = pauza · Q = wyjście"

    shown[WINDOWS[2][0]] = build_green_window(analyzer, cfg)
    shown[WINDOWS[3][0]] = build_spectra_before_window(None, analyzer, cfg)
    shown[WINDOWS[4][0]] = build_filter_window(None, analyzer, cfg)
    shown[WINDOWS[5][0]] = build_spectra_after_window(None, analyzer, cfg)

    try:
        while True:
            if paused and not is_file:
                cap.read()          # camera keeps running; drop frames so we do not resume on a stale one
            elif not paused and not ended:
                ok, frame_bgr = cap.read()
                if not ok:
                    if is_file:
                        ended, status = True, "koniec nagrania · Q = wyjście"
                    else:
                        print("Camera returned no frame, exiting.")
                        break
                else:
                    time_s = frame_index / sample_rate_hz if is_file else time.monotonic() - playback_start_time_s
                    frame_index += 1
                    frame_height_px, frame_width_px = frame_bgr.shape[:2]
                    frame_rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                    frame_timestamp_ms = max(int(round(time_s * 1000)), last_frame_timestamp_ms + 1)
                    last_frame_timestamp_ms = frame_timestamp_ms      # MediaPipe needs strictly rising times
                    result = landmarker.detect_for_video(
                        mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb), frame_timestamp_ms)

                    if result.face_landmarks:
                        landmarks_px = np.array([(p.x * frame_width_px, p.y * frame_height_px)
                                                 for p in result.face_landmarks[0]], np.float32)
                        roi_shapes = roi.roi_polygons(landmarks_px, cfg.roi_shrink)
                        mean_raw, _, pixel_counts, _ = roi.roi_stats(
                            frame_rgb, roi_shapes, lut, cfg.clip_low, cfg.clip_high)
                        if np.all(np.isfinite(mean_raw[:, 1])):
                            analyzer.add(time_s, mean_raw[:, 1])          # green channel of the 3 regions
                        face_lost_since = None
                    else:
                        landmarks_px = roi_shapes = None
                        face_lost_since = time_s if face_lost_since is None else face_lost_since
                        if time_s - face_lost_since > FACE_LOST_RESET_S:
                            analyzer.reset()
                            analysis = None

                    shown[WINDOWS[0][0]] = build_landmarks_window(frame_bgr, landmarks_px,
                                                                  used_landmark_indices, status)
                    shown[WINDOWS[1][0]] = build_roi_window(frame_bgr, roi_shapes, pixel_counts)
                    if time_s - last_analysis_time_s >= ANALYSIS_EVERY_S:
                        last_analysis_time_s = time_s
                        analysis = analyzer.analyze()
                        shown[WINDOWS[2][0]] = build_green_window(analyzer, cfg)
                        shown[WINDOWS[3][0]] = build_spectra_before_window(analysis, analyzer, cfg)
                        shown[WINDOWS[4][0]] = build_filter_window(analysis, analyzer, cfg)
                        shown[WINDOWS[5][0]] = build_spectra_after_window(analysis, analyzer, cfg)
                    if is_file:                                              # play a file at its natural speed
                        wait = (playback_start_time_s + time_s) - time.monotonic()
                        if wait > 0:
                            time.sleep(min(wait, 0.05))

            for title, _, _ in WINDOWS:
                if title in shown:
                    cv2.imshow(title, shown[title])
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                paused = not paused
    finally:
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
