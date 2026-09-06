import sys
import os
import shutil
import pickle
import warnings
from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import wfdb
from scipy.ndimage import median_filter

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ================= ФУНКЦИИ ДЛЯ ОЦЕНКИ =================
def add_noise_to_signal(clean_signal, fs, noise_type, snr_linear):
    clean = clean_signal.copy()
    power_signal = np.var(clean)
    if power_signal == 0:
        power_signal = 1.0
    power_noise = power_signal / snr_linear
    noise = np.zeros_like(clean)

    if noise_type == 'white':
        noise = np.random.normal(0, np.sqrt(power_noise), len(clean))
    elif noise_type == 'baseline':
        t = np.arange(len(clean)) / fs
        f = np.random.uniform(0.1, 0.5)
        amp = np.sqrt(power_noise * 2)
        noise = amp * np.sin(2 * np.pi * f * t)
    elif noise_type == 'motion':
        num_imp = max(1, int(0.5 * len(clean) / fs))
        for _ in range(num_imp):
            pos = np.random.randint(0, len(clean))
            width = np.random.randint(int(0.02 * fs), int(0.15 * fs))
            amp = np.random.uniform(0.5, 2.5) * np.sqrt(power_noise)
            window = signal.windows.gaussian(width, std=width / 4)
            start = max(0, pos - width // 2)
            end = min(len(clean), start + len(window))
            noise[start:end] += amp * window[:end - start]
        if np.var(noise) > 0:
            noise = noise / np.std(noise) * np.sqrt(power_noise)
    elif noise_type == 'emg':
        nyq = fs / 2
        b, a = signal.butter(4, [20 / nyq, 50 / nyq], btype='band')
        white = np.random.normal(0, 1, len(clean))
        noise = signal.filtfilt(b, a, white)
        noise = noise / np.std(noise) * np.sqrt(power_noise)
    elif noise_type == 'combined':
        parts = ['white', 'baseline', 'motion', 'emg']
        noise_sum = np.zeros_like(clean)
        for p in parts:
            part_noise = add_noise_to_signal(clean, fs, p, snr_linear * 4) - clean
            noise_sum += part_noise
        return clean + noise_sum
    else:
        raise ValueError(f"Неизвестный тип шума: {noise_type}")
    return clean + noise


def compute_metrics(reference_peaks, detected_peaks, tolerance_samples):
    ref = np.array(sorted(reference_peaks), dtype=int)
    det = np.array(sorted(detected_peaks), dtype=int)

    matched_ref = set()
    matched_det = set()
    tp = 0

    for i, r in enumerate(ref):
        if len(det) == 0:
            break
        distances = np.abs(det - r)
        j = np.argmin(distances)
        if distances[j] <= tolerance_samples:
            if j not in matched_det:
                tp += 1
                matched_ref.add(i)
                matched_det.add(j)

    fn = len(ref) - tp
    fp = len(det) - tp

    sensitivity = tp / (tp + fn + 1e-12)
    ppv = tp / (tp + fp + 1e-12)

    detection_error = 1.0 - 0.5 * (sensitivity + ppv)
    detection_error = max(0.0, min(1.0, detection_error))

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Sensitivity": sensitivity,
        "PPV": ppv,
        "Error": detection_error
    }


# ================= БАЗОВЫЙ КЛАСС =================
class QRSDetector(ABC):
    def __init__(self):
        self.debug_signal = None
        self.candidate_scores = None

    @abstractmethod
    def detect(self, ecg_signal, fs):
        pass

    def get_processed_signal(self, ecg_signal, fs):
        return np.asarray(ecg_signal, dtype=float)

    def get_debug_signal(self):
        return self.debug_signal

    def get_candidate_scores(self):
        return self.candidate_scores

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def description(self):
        pass


# ================= ПАН-ТОМПКИНС =================
class PanTompkinsDetector(QRSDetector):
    def __init__(self):
        super().__init__()
        self._name = "Пан-Томпкинс"
        self._description = (
            "Классическая реализация: полосовая фильтрация 5–15 Гц, "
            "производная, возведение в квадрат, скользящее интегрирование, "
            "адаптивные пороги."
        )

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return self._description

    def get_processed_signal(self, ecg_signal, fs):
        nyq = 0.5 * fs
        low = 5.0 / nyq
        high = 15.0 / nyq
        b, a = signal.butter(2, [low, high], btype='band')
        return signal.filtfilt(b, a, ecg_signal)

    def detect(self, ecg_signal, fs):
        nyq = 0.5 * fs
        low = 5.0 / nyq
        high = 15.0 / nyq
        b, a = signal.butter(2, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, ecg_signal)

        derivative = np.zeros_like(filtered)
        derivative[1:-1] = (filtered[2:] - filtered[:-2]) / 2.0
        derivative[0] = derivative[1]
        derivative[-1] = derivative[-2]

        squared = derivative ** 2
        window_size = int(0.15 * fs)
        integrated = np.convolve(squared, np.ones(window_size), mode='same')

        self.debug_signal = integrated.copy()

        init_len = int(2 * fs)
        init_data = integrated[:init_len]
        spki = np.max(init_data)
        npki = np.median(init_data)
        threshold1 = npki + 0.25 * (spki - npki)
        threshold2 = 0.5 * threshold1

        peaks = []
        refractory = int(0.2 * fs)
        last_peak = -refractory
        rr_low_limit = 0.3 * fs

        i = 0
        while i < len(integrated):
            if integrated[i] > threshold1:
                search_window = int(0.1 * fs)
                end = min(i + search_window, len(integrated))
                max_idx = i + np.argmax(integrated[i:end])
                peak_val = integrated[max_idx]

                if max_idx - last_peak > refractory:
                    if peak_val > threshold2:
                        refine_win = int(0.05 * fs)
                        start_r = max(0, max_idx - refine_win)
                        end_r = min(len(filtered), max_idx + refine_win + 1)
                        r_peak_idx = start_r + np.argmax(filtered[start_r:end_r])

                        if last_peak >= 0:
                            current_rr = r_peak_idx - last_peak
                            if current_rr < rr_low_limit:
                                i = end
                                continue

                        peaks.append(r_peak_idx)
                        last_peak = r_peak_idx
                        spki = 0.125 * peak_val + 0.875 * spki
                    else:
                        npki = 0.125 * peak_val + 0.875 * npki
                i = end
            else:
                i += 1

            threshold1 = npki + 0.25 * (spki - npki)
            threshold2 = 0.5 * threshold1

        return peaks


# ================= КОРРЕЛЯЦИОННЫЙ ДЕТЕКТОР =================
class CorrelationDetector(QRSDetector):
    def __init__(self, correlation_threshold=0.6, template_duration=0.2, learning_duration=10):
        super().__init__()
        self._name = "Корреляционный детектор"
        self._description = (
            "Предварительная фильтрация, шаблон QRS, нормированная корреляция, "
            "адаптивное снижение порога при слабом сигнале."
        )
        self.correlation_threshold = correlation_threshold
        self.template_duration = template_duration
        self.learning_duration = learning_duration

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return self._description

    def get_processed_signal(self, ecg_signal, fs):
        x = np.asarray(ecg_signal, dtype=float).ravel()
        x = np.nan_to_num(x - np.nanmean(x))
        nyq = fs / 2
        b, a = signal.butter(2, [3 / nyq, 30 / nyq], btype="band")
        return signal.filtfilt(b, a, x)

    def detect(self, ecg_signal, fs):
        x = np.asarray(ecg_signal, dtype=float).ravel()
        x = np.nan_to_num(x - np.nanmean(x))

        nyq = fs / 2
        b, a = signal.butter(2, [3 / nyq, 30 / nyq], btype="band")
        filtered_signal = signal.filtfilt(b, a, x)

        learn_samples = min(int(self.learning_duration * fs), len(filtered_signal))
        learn_signal = filtered_signal[:learn_samples]

        threshold = np.median(np.abs(learn_signal)) * 3.0
        min_distance = int(0.24 * fs)
        peaks, _ = signal.find_peaks(np.abs(learn_signal), height=threshold, distance=min_distance)

        if len(peaks) < 3:
            threshold = np.median(np.abs(learn_signal)) * 2.0
            peaks, _ = signal.find_peaks(np.abs(learn_signal), height=threshold, distance=min_distance)

        if len(peaks) < 2:
            return []

        half_width = max(1, int(self.template_duration * fs / 2))
        templates = []

        for p in peaks:
            left = max(0, p - half_width)
            right = min(len(learn_signal), p + half_width + 1)
            fragment = learn_signal[left:right]

            if len(fragment) < 2 * half_width + 1:
                fragment = np.pad(fragment, (0, 2 * half_width + 1 - len(fragment)))

            templates.append(fragment)

        ref = templates[0] - np.mean(templates[0])
        aligned = []

        for frag in templates:
            frag = frag - np.mean(frag)
            corr = np.correlate(frag, ref, mode="full")
            shift = np.argmax(corr) - (len(ref) - 1)

            if shift > 0:
                aligned_frag = np.concatenate([np.zeros(shift), frag])
            else:
                aligned_frag = frag[-shift:]

            aligned_frag = aligned_frag[:len(ref)]

            if len(aligned_frag) < len(ref):
                aligned_frag = np.pad(aligned_frag, (0, len(ref) - len(aligned_frag)))

            aligned.append(aligned_frag)

        template = np.mean(aligned, axis=0)
        template = template - np.mean(template)

        template_energy = np.sum(template ** 2) + 1e-10
        signal_energy = np.convolve(filtered_signal ** 2, np.ones(len(template)), mode="same")
        corr = np.correlate(filtered_signal, template, mode="same")
        norm_corr = corr / (np.sqrt(template_energy * signal_energy) + 1e-10)

        self.debug_signal = norm_corr.copy()

        thr = self.correlation_threshold
        corr_peaks = []

        while thr > 0.3 and len(corr_peaks) < 2:
            corr_peaks, _ = signal.find_peaks(
                norm_corr,
                height=thr,
                distance=min_distance,
                prominence=0.1
            )
            thr -= 0.1

        refined = []
        search = int(0.05 * fs)

        for p in corr_peaks:
            s = max(0, p - search)
            e = min(len(filtered_signal), p + search + 1)
            r = s + np.argmax(np.abs(filtered_signal[s:e]))
            refined.append(r)

        return sorted(refined)


# ================= НЕЙРОСЕТЕВОЙ ДЕТЕКТОР =================
class NeuralDetector(QRSDetector):
    def __init__(self):
        super().__init__()
        self._name = "Нейросетевой детектор"
        self._description = (
            "Нейросетевой QRS-детектор: MLP, "
            "RR search-back."
        )
        self.fs = 360
        self.window_size = 216
        self.model_path = "qrs_mlp_model_safe.pkl"
        self.scaler_path = "qrs_mlp_scaler_safe.pkl"
        self.model = None
        self.scaler = None
        self._result_cache = {}

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return self._description

    def _robust_norm(self, x):
        med = np.median(x)
        mad = np.median(np.abs(x - med)) + 1e-8
        return (x - med) / (1.4826 * mad)

    def _wavelet_denoise(self, x):
        if len(x) < 21:
            return x.copy()
        try:
            k = int(0.035 * self.fs)
            if k % 2 == 0:
                k += 1
            k = max(3, min(k, len(x)-1))
            from scipy.ndimage import median_filter as medf
            y = medf(x, size=k, mode='reflect')
            return y
        except Exception:
            return x.copy()

    def preprocess(self, x, fs, denoise=False):
        x = np.asarray(x, dtype=float).ravel()
        x = np.nan_to_num(x - np.nanmedian(x))
        if len(x) < 20:
            return x
        if fs != self.fs:
            new_len = max(20, int(round(len(x) * self.fs / fs)))
            old_t = np.linspace(0, 1, len(x))
            new_t = np.linspace(0, 1, new_len)
            x = np.interp(new_t, old_t, x)
        nyq = self.fs / 2.0
        sos_hp = signal.butter(2, 0.6 / nyq, btype="highpass", output="sos")
        x = signal.sosfiltfilt(sos_hp, x)
        if 50.0 < nyq * 0.95:
            try:
                b_notch, a_notch = signal.iirnotch(50.0 / nyq, 30.0)
                x = signal.filtfilt(b_notch, a_notch, x)
            except Exception:
                pass
        if denoise:
            x = self._wavelet_denoise(x)
        low = 4.0 if denoise else 5.0
        high = min(38.0 if denoise else 35.0, nyq * 0.85)
        sos = signal.butter(3, [low / nyq, high / nyq], btype="bandpass", output="sos")
        x = signal.sosfiltfilt(sos, x)
        return self._robust_norm(x)

    def get_processed_signal(self, ecg_signal, fs):
        return self.preprocess(ecg_signal, fs, denoise=True)

    def extract_features(self, w):
        w = np.asarray(w, dtype=float).ravel()
        if len(w) != self.window_size:
            if len(w) > self.window_size:
                w = w[:self.window_size]
            else:
                w = np.pad(w, (0, self.window_size - len(w)))
        center = len(w) // 2
        qrs = w[center - 30:center + 30]
        narrow = w[center - 18:center + 18]
        side = np.concatenate([w[:center - 50], w[center + 50:]])
        dw = np.diff(w)
        ddw = np.diff(dw)
        dq = np.diff(qrs)
        abs_w = np.abs(w)
        abs_qrs = np.abs(qrs)
        energy_total = np.sum(w ** 2) + 1e-8
        energy_qrs = np.sum(qrs ** 2) + 1e-8
        energy_narrow = np.sum(narrow ** 2) + 1e-8
        energy_side = np.sum(side ** 2) + 1e-8
        slope_energy = np.sum(dw ** 2) + 1e-8
        fft = np.abs(np.fft.rfft(w))
        fft_sum = np.sum(fft) + 1e-8
        return np.array([
            np.max(w), np.min(w), np.ptp(w), np.max(abs_w),
            np.mean(w), np.std(w), np.median(w),
            np.median(np.abs(w - np.median(w))) + 1e-8,
            np.max(abs_qrs), np.mean(abs_qrs), np.std(qrs), np.ptp(qrs),
            energy_qrs / energy_side,
            energy_qrs / energy_total,
            energy_narrow / energy_total,
            np.max(abs_qrs) / (np.std(side) + 1e-8),
            np.max(np.abs(dw)) if len(dw) else 0,
            np.mean(np.abs(dw)) if len(dw) else 0,
            np.std(dw) if len(dw) else 0,
            slope_energy / energy_total,
            np.max(np.abs(dq)) if len(dq) else 0,
            np.mean(np.abs(dq)) if len(dq) else 0,
            np.sum(dq ** 2) / energy_qrs if len(dq) else 0,
            np.max(np.abs(ddw)) if len(ddw) else 0,
            np.mean(np.abs(ddw)) if len(ddw) else 0,
            abs(np.argmax(abs_w) - center) / len(w),
            np.sum(fft[2:8]) / fft_sum,
            np.sum(fft[8:18]) / fft_sum,
            np.sum(fft[18:45]) / fft_sum,
            np.sum(fft[45:]) / fft_sum,
        ], dtype=np.float64)

    def _synthetic_ecg(self, duration=20, hr=70, noise=0.20):
        fs = self.fs
        t = np.arange(0, duration, 1 / fs)
        x = np.zeros_like(t)
        rr_mean = int(60 / hr * fs)
        r_peaks = []
        pos = int(0.4 * fs)
        while pos < len(x) - fs:
            jitter = np.random.randint(-int(0.18 * rr_mean), int(0.18 * rr_mean) + 1)
            pos += max(int(0.30 * fs), rr_mean + jitter)
            if pos >= len(x) - fs:
                break
            polarity = np.random.choice([1.0, -1.0], p=[0.82, 0.18])
            amp = polarity * np.random.uniform(0.45, 2.4)
            qrs_w = np.random.randint(int(0.035 * fs), int(0.13 * fs))
            qrs_t = np.linspace(-1, 1, qrs_w)
            qrs = amp * (1.55 * np.exp(-(qrs_t / np.random.uniform(0.13, 0.25)) ** 2) -
                         0.32 * np.exp(-((qrs_t + 0.45) / 0.15) ** 2) -
                         0.32 * np.exp(-((qrs_t - 0.45) / 0.15) ** 2))
            s = pos - qrs_w // 2
            e = s + qrs_w
            if 0 <= s and e < len(x):
                x[s:e] += qrs
                r_peaks.append(pos)
            p_pos = pos - int(np.random.uniform(0.13, 0.25) * fs)
            p_w = np.random.randint(int(0.055 * fs), int(0.12 * fs))
            if p_pos - p_w // 2 > 0 and p_pos + p_w // 2 < len(x):
                p_t = np.linspace(-1, 1, p_w)
                p_sig = np.random.uniform(-0.25, 0.25) * np.exp(-(p_t / 0.42) ** 2)
                x[p_pos - p_w // 2:p_pos - p_w // 2 + p_w] += p_sig
            t_pos = pos + int(np.random.uniform(0.17, 0.40) * fs)
            t_w = np.random.randint(int(0.11 * fs), int(0.25 * fs))
            if t_pos - t_w // 2 > 0 and t_pos + t_w // 2 < len(x):
                tt = np.linspace(-1, 1, t_w)
                t_sig = np.random.uniform(-0.55, 0.55) * np.exp(-(tt / 0.56) ** 2)
                x[t_pos - t_w // 2:t_pos - t_w // 2 + t_w] += t_sig
        drift = np.random.uniform(0.03, 0.75) * np.sin(2 * np.pi * np.random.uniform(0.02, 0.35) * t + np.random.uniform(0, 2 * np.pi))
        drift += np.random.uniform(0.0, 0.35) * np.sin(2 * np.pi * np.random.uniform(0.04, 0.18) * t + np.random.uniform(0, 2 * np.pi))
        power = np.random.uniform(0.0, 0.22) * np.sin(2 * np.pi * 50 * t)
        white = np.random.normal(0, noise * (np.std(x) + 1e-8), len(x))
        muscle = np.random.normal(0, 1, len(x))
        try:
            nyq = fs / 2
            high = min(90.0, nyq * 0.90)
            sos = signal.butter(2, [18 / nyq, high / nyq], btype="bandpass", output="sos")
            muscle = signal.sosfiltfilt(sos, muscle)
            muscle = muscle / (np.std(muscle) + 1e-8)
            muscle *= np.random.uniform(0.0, 0.45) * (np.std(x) + 1e-8)
        except Exception:
            muscle *= 0
        x += drift + power + white + muscle
        for _ in range(np.random.randint(2, 10)):
            c = np.random.randint(fs, len(x) - fs)
            w = np.random.randint(int(0.020 * fs), int(0.28 * fs))
            amp = np.random.uniform(-1.8, 1.8) * (np.std(x) + 1e-8)
            art = amp * signal.windows.gaussian(w, std=max(2, w // np.random.randint(4, 9)))
            s = c - w // 2
            e = s + w
            if 0 <= s and e < len(x):
                x[s:e] += art
        return x, np.array(r_peaks, dtype=int)

    def _qrs_score(self, x):
        absx = np.abs(x)
        dx = np.diff(x, prepend=x[0])
        ddx = np.diff(dx, prepend=dx[0])
        e = 0.36 * absx**2 + 0.48 * dx**2 + 0.16 * ddx**2
        w_short = max(3, int(0.055 * self.fs))
        w_mid = max(3, int(0.105 * self.fs))
        w_slope = max(3, int(0.035 * self.fs))
        env_short = np.convolve(e, np.ones(w_short) / w_short, mode="same")
        env_mid = np.convolve(absx, np.ones(w_mid) / w_mid, mode="same")
        env_slope = np.convolve(np.abs(dx), np.ones(w_slope) / w_slope, mode="same")
        score = self._robust_norm(env_short) + 0.45 * self._robust_norm(env_mid) + 0.30 * self._robust_norm(env_slope)
        return np.nan_to_num(score)

    def _morphology_metrics(self, x, p):
        absx = np.abs(x)
        bg_s = max(0, p - int(0.36 * self.fs))
        bg_e = min(len(x), p + int(0.36 * self.fs))
        qrs_s = max(0, p - int(0.060 * self.fs))
        qrs_e = min(len(x), p + int(0.060 * self.fs))
        bg = np.concatenate([x[bg_s:max(bg_s, p - int(0.08 * self.fs))],
                             x[min(bg_e, p + int(0.08 * self.fs)):bg_e]])
        noise = np.std(bg) + 1e-8
        amp = absx[p]
        contrast = amp / noise
        local = absx[qrs_s:qrs_e]
        if len(local) == 0:
            width = 999
        else:
            half_height = 0.5 * amp
            above = np.where(local >= half_height)[0]
            width = above[-1] - above[0] + 1 if len(above) > 0 else 999
        dx = np.diff(x, prepend=x[0])
        slope_s = max(0, p - int(0.045 * self.fs))
        slope_e = min(len(dx), p + int(0.045 * self.fs))
        slope_peak = np.max(np.abs(dx[slope_s:slope_e])) if slope_e > slope_s else 0
        slope_bg = np.std(dx[bg_s:bg_e]) + 1e-8 if bg_e > bg_s else 1e-8
        slope_contrast = slope_peak / slope_bg
        return {"amp": amp, "contrast": contrast, "width": width, "width_sec": width / self.fs,
                "slope_contrast": slope_contrast}

    def _prepare_training_data(self, n_records=85):
        X, y = [], []
        half = self.window_size // 2
        for _ in range(n_records):
            hr = np.random.randint(36, 165)
            noise = np.random.uniform(0.00, 0.85)
            raw, r_peaks = self._synthetic_ecg(duration=20, hr=hr, noise=noise)
            x = self.preprocess(raw, self.fs, denoise=True)
            for r in r_peaks:
                for jitter in np.random.randint(-10, 11, size=5):
                    c = r + jitter
                    if half <= c < len(x) - half:
                        X.append(self.extract_features(x[c - half:c + half]))
                        y.append(1)
            forbidden = np.zeros(len(x), dtype=bool)
            for r in r_peaks:
                forbidden[max(0, r - int(0.17 * self.fs)):min(len(x), r + int(0.17 * self.fs))] = True
            score = self._qrs_score(x)
            cand, _ = signal.find_peaks(score, distance=int(0.16 * self.fs), height=np.percentile(score, 50))
            neg = []
            for c in cand:
                if half <= c < len(x) - half and not forbidden[c]:
                    neg.append(c)
            attempts = 0
            while len(neg) < 320 and attempts < 6000:
                attempts += 1
                c = np.random.randint(half, len(x) - half)
                if not forbidden[c]:
                    neg.append(c)
            for c in neg[:380]:
                X.append(self.extract_features(x[c - half:c + half]))
                y.append(0)
        return np.asarray(X), np.asarray(y)

    def train_model(self):
        X, y = self._prepare_training_data()
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X)
        self.model = MLPClassifier(hidden_layer_sizes=(72, 36), activation="relu", solver="adam",
                                   alpha=0.0007, learning_rate_init=0.001, max_iter=260,
                                   early_stopping=True, validation_fraction=0.15, random_state=42)
        self.model.fit(Xs, y)
        with open(self.model_path, "wb") as f:
            pickle.dump(self.model, f)
        with open(self.scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)

    def load_model(self):
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            try:
                with open(self.model_path, "rb") as f:
                    self.model = pickle.load(f)
                with open(self.scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)
                return
            except Exception:
                self.model = None
                self.scaler = None
        self.train_model()

    def _signal_noise_mode(self, x):
        dx = np.diff(x, prepend=x[0])
        hf = np.std(dx) / (np.std(x) + 1e-8)
        score = self._qrs_score(x)
        peakiness = np.percentile(score, 95) - np.median(score)
        noisy = hf > 0.95 or peakiness < 2.4
        very_noisy = hf > 1.25 or peakiness < 1.7
        return noisy, very_noisy

    def _candidate_peaks_single(self, x, sensitivity=1.0):
        score = self._qrs_score(x)
        absx = np.abs(x)
        med = np.median(score)
        mad = np.median(np.abs(score - med)) + 1e-8
        duration_sec = len(x) / self.fs
        min_expected = max(3, int(duration_sec * 35 / 60))
        candidates = []
        factors = [1.50, 1.20, 0.95, 0.70]
        factors = [f / sensitivity for f in factors]
        for k in factors:
            peaks, _ = signal.find_peaks(score, height=med + k * mad, distance=int(0.20 * self.fs))
            candidates.extend(peaks.tolist())
            if len(set(candidates)) >= int(1.35 * min_expected):
                break
        amp_percentile = 70 if sensitivity <= 1.2 else 64
        amp_peaks, _ = signal.find_peaks(absx, height=np.percentile(absx, amp_percentile), distance=int(0.22 * self.fs))
        candidates.extend(amp_peaks.tolist())
        candidates = sorted(set(candidates))
        half = self.window_size // 2
        candidates = [p for p in candidates if half <= p < len(x) - half]
        max_candidates = int(max(80, len(x) / self.fs * 5.0))
        if len(candidates) > max_candidates:
            arr = np.array(candidates)
            idx = np.argsort(score[arr])[-max_candidates:]
            candidates = sorted(arr[idx].tolist())
        refined = []
        search = int(0.060 * self.fs)
        for p in candidates:
            s = max(0, p - search)
            e = min(len(x), p + search + 1)
            if e > s:
                refined.append(s + np.argmax(np.abs(x[s:e])))
        refined = sorted(set(refined))
        filtered = []
        for p in refined:
            m = self._morphology_metrics(x, p)
            if 0.010 <= m["width_sec"] <= 0.180 and m["contrast"] >= 0.38 and m["slope_contrast"] >= 0.38:
                filtered.append(p)
        return filtered

    def _candidate_peaks(self, x_main, x_den, very_noisy=False):
        if very_noisy:
            c1 = self._candidate_peaks_single(x_main, sensitivity=1.45)
            c2 = self._candidate_peaks_single(x_den, sensitivity=1.70)
        else:
            c1 = self._candidate_peaks_single(x_main, sensitivity=1.00)
            c2 = self._candidate_peaks_single(x_den, sensitivity=1.25)
        absd = np.abs(x_den)
        all_c = sorted(set(c1 + c2))
        refined = []
        search = int(0.055 * self.fs)
        for p in all_c:
            s = max(0, p - search)
            e = min(len(absd), p + search + 1)
            if e > s:
                refined.append(s + np.argmax(absd[s:e]))
        refined = sorted(set(refined))
        score = self._qrs_score(x_den)
        final = []
        refractory = int(0.22 * self.fs)
        for p in refined:
            if not final:
                final.append(p)
            elif p - final[-1] >= refractory:
                final.append(p)
            else:
                old = final[-1]
                if score[p] + 0.20 * absd[p] > score[old] + 0.20 * absd[old]:
                    final[-1] = p
        return final

    def _mlp_probs(self, x, peaks):
        if not peaks:
            return np.array([])
        half = self.window_size // 2
        feats = []
        valid = []
        for i, p in enumerate(peaks):
            if p - half >= 0 and p + half <= len(x):
                feats.append(self.extract_features(x[p - half:p + half]))
                valid.append(i)
        probs = np.zeros(len(peaks)) + 0.5
        if feats and self.model is not None and self.scaler is not None:
            X = np.asarray(feats)
            Xs = self.scaler.transform(X)
            pr = self.model.predict_proba(Xs)[:, 1]
            for idx, val in zip(valid, pr):
                probs[idx] = float(val)
        return probs

    def _select(self, x_main, x_den, peaks, noisy=False, very_noisy=False):
        if not peaks:
            return []
        score_main = self._qrs_score(x_main)
        score_den = self._qrs_score(x_den)
        probs = self._mlp_probs(x_den, peaks)
        items = []
        for p, pr in zip(peaks, probs):
            m_main = self._morphology_metrics(x_main, p)
            m_den = self._morphology_metrics(x_den, p)
            contrast = max(0.65 * m_den["contrast"] + 0.35 * m_main["contrast"], m_den["contrast"])
            slope_c = max(0.65 * m_den["slope_contrast"] + 0.35 * m_main["slope_contrast"], m_den["slope_contrast"])
            width_sec = m_den["width_sec"]
            if width_sec < 0.008 or width_sec > 0.190:
                continue
            if very_noisy:
                min_contrast, min_slope = 0.34, 0.34
            elif noisy:
                min_contrast, min_slope = 0.42, 0.42
            else:
                min_contrast, min_slope = 0.52, 0.52
            if contrast < min_contrast and pr < 0.32:
                continue
            if slope_c < min_slope and pr < 0.32:
                continue
            qrs_score_norm = min(1.0, max(0.0, max(score_main[p], score_den[p])) / 5.0)
            contrast_norm = min(1.0, contrast / 2.4)
            slope_norm = min(1.0, slope_c / 2.6)
            final_score = 0.36 * qrs_score_norm + 0.26 * contrast_norm + 0.22 * slope_norm + 0.16 * pr
            items.append((p, final_score, pr, contrast, slope_c))
        self.candidate_scores = np.array([it[1] for it in items]) if items else np.array([])
        if not items:
            return []
        vals = np.array([it[1] for it in items])
        if very_noisy:
            thr = max(0.24, np.percentile(vals, 42))
        elif noisy:
            thr = max(0.27, np.percentile(vals, 47))
        else:
            thr = max(0.28, np.percentile(vals, 45))
        selected = []
        for p, final_score, pr, contrast, slope_c in items:
            if final_score >= thr:
                selected.append((p, final_score))
            elif very_noisy and pr >= 0.33 and contrast >= 0.42 and slope_c >= 0.42:
                selected.append((p, final_score))
            elif contrast >= 1.65 and slope_c >= 1.15 and pr >= 0.08:
                selected.append((p, final_score))
        return self._merge_ranked(x_den, selected)

    def _merge_ranked(self, x, ranked):
        if not ranked:
            return []
        ranked = sorted(ranked, key=lambda z: z[0])
        score = self._qrs_score(x)
        absx = np.abs(x)
        final = []
        refractory = int(0.22 * self.fs)
        for p, rank_score in ranked:
            if not final:
                final.append([p, rank_score])
            elif p - final[-1][0] >= refractory:
                final.append([p, rank_score])
            else:
                old = final[-1][0]
                new_total = rank_score + 0.08 * score[p] + 0.05 * absx[p]
                old_total = final[-1][1] + 0.08 * score[old] + 0.05 * absx[old]
                if new_total > old_total:
                    final[-1] = [p, rank_score]
        return [f[0] for f in final]

    def _rr_cleanup(self, x, peaks, noisy=False):
        if len(peaks) < 5:
            return peaks
        peaks = sorted(peaks)
        score = self._qrs_score(x)
        absx = np.abs(x)
        changed = True
        while changed and len(peaks) >= 5:
            changed = False
            rr = np.diff(peaks)
            med_rr = np.median(rr)
            short_factor = 0.42 if noisy else 0.45
            short_abs = int(0.34 * self.fs) if noisy else int(0.36 * self.fs)
            for i in range(len(rr)):
                if rr[i] < short_factor * med_rr and rr[i] < short_abs:
                    p1, p2 = peaks[i], peaks[i+1]
                    m1 = self._morphology_metrics(x, p1)
                    m2 = self._morphology_metrics(x, p2)
                    q1 = score[p1] + 0.22*absx[p1] + 0.20*m1["contrast"] + 0.15*m1["slope_contrast"]
                    q2 = score[p2] + 0.22*absx[p2] + 0.20*m2["contrast"] + 0.15*m2["slope_contrast"]
                    if q1 >= q2:
                        del peaks[i+1]
                    else:
                        del peaks[i]
                    changed = True
                    break
        return peaks

    def _search_back(self, x_main, x_den, peaks, noisy=False, very_noisy=False):
        if len(peaks) < 4:
            return peaks
        score = self._qrs_score(x_den)
        absd = np.abs(x_den)
        peaks = sorted(peaks)
        passes = 2 if very_noisy else 1
        for _ in range(passes):
            rr = np.diff(peaks)
            if len(rr) == 0:
                break
            med_rr = np.median(rr)
            additions = []
            gap_factor = 1.60 if noisy else 1.75
            for i in range(len(peaks)-1):
                gap = peaks[i+1] - peaks[i]
                if gap > gap_factor * med_rr:
                    left = peaks[i] + int(0.28 * med_rr)
                    right = peaks[i+1] - int(0.28 * med_rr)
                    if right <= left:
                        continue
                    local = score[left:right]
                    if len(local) == 0:
                        continue
                    c = left + np.argmax(local)
                    s = max(0, c - int(0.060*self.fs))
                    e = min(len(x_den), c + int(0.060*self.fs)+1)
                    if e <= s:
                        continue
                    r = s + np.argmax(absd[s:e])
                    m_den = self._morphology_metrics(x_den, r)
                    m_main = self._morphology_metrics(x_main, r)
                    contrast = max(m_den["contrast"], 0.5*m_den["contrast"]+0.5*m_main["contrast"])
                    slope_c = max(m_den["slope_contrast"], 0.5*m_den["slope_contrast"]+0.5*m_main["slope_contrast"])
                    if very_noisy:
                        min_c, min_s = 0.30, 0.30
                    elif noisy:
                        min_c, min_s = 0.40, 0.40
                    else:
                        min_c, min_s = 0.60, 0.60
                    if (all(abs(r-p)>=int(0.22*self.fs) for p in peaks+additions)
                        and contrast >= min_c and slope_c >= min_s
                        and 0.008 <= m_den["width_sec"] <= 0.190):
                        additions.append(r)
            if not additions:
                break
            peaks = sorted(set(peaks + additions))
        return peaks

    def detect(self, ecg_signal, fs):
        arr = np.ascontiguousarray(ecg_signal, dtype=float)
        key = (hash(arr.tobytes()), fs)
        if key in self._result_cache:
            return self._result_cache[key].copy()
        if self.model is None or self.scaler is None:
            self.load_model()
        x_main = self.preprocess(ecg_signal, fs, denoise=False)
        x_den = self.preprocess(ecg_signal, fs, denoise=True)
        self.debug_signal = self._qrs_score(x_den).copy()
        if len(x_main) < self.window_size:
            self._result_cache[key] = []
            return []
        noisy, very_noisy = self._signal_noise_mode(x_main)
        peaks = self._candidate_peaks(x_main, x_den, very_noisy=very_noisy)
        peaks = self._select(x_main, x_den, peaks, noisy=noisy, very_noisy=very_noisy)
        peaks = self._rr_cleanup(x_den, peaks, noisy=noisy)
        peaks = self._search_back(x_main, x_den, peaks, noisy=noisy, very_noisy=very_noisy)
        peaks = self._rr_cleanup(x_den, peaks, noisy=noisy)
        peaks = sorted(set(peaks))
        if fs != self.fs:
            peaks = [int(round(p * fs / self.fs)) for p in peaks]
        self._result_cache[key] = peaks
        return peaks.copy()


# ================= MIT-BIH ОЦЕНЩИК (расширенный) =================
class MITBIHMultiEvaluator:
    """Обучение нейросети на DS1 и оценка нескольких детекторов по нескольким шумам на DS2."""
    DS1 = ['101','103','105','106','108','109','111','112','113','114',
           '115','116','117','118','119','121','122','123','124','200','201','202']
    DS2 = ['205','207','208','209','210','212','213','214','215','219',
           '220','221','222','223','228','230','231','232','233','234']

    def __init__(self, mit_dir):
        self.mit_dir = mit_dir
        self.fs = 360
        self.window_size = 216
        self.neural_detector = NeuralDetector()
        self.model = None
        self.scaler = None

    def _load_record(self, record_name):
        record = wfdb.rdrecord(os.path.join(self.mit_dir, record_name), channels=[0])
        ann = wfdb.rdann(os.path.join(self.mit_dir, record_name), 'atr')
        return record.p_signal[:, 0], record.fs, ann.sample

    def _extract_windows(self, signal, r_peaks, count_neg=2):
        half = self.window_size // 2
        offset = int(0.15 * self.fs)
        pos_windows = []
        for p in r_peaks:
            if p - half >= 0 and p + half < len(signal):
                win = signal[p - half : p + half]
                win = self.neural_detector.preprocess(win, self.fs, denoise=True)
                pos_windows.append(win)
        forbidden = np.zeros(len(signal), dtype=bool)
        for p in r_peaks:
            forbidden[max(0, p - offset): min(len(signal), p + offset)] = True
        valid_indices = np.where(~forbidden)[0]
        valid_indices = valid_indices[(valid_indices >= half) & (valid_indices < len(signal) - half)]
        n_neg = min(len(pos_windows) * count_neg, len(valid_indices))
        selected = np.random.choice(valid_indices, size=n_neg, replace=False)
        neg_windows = []
        for idx in selected:
            win = signal[idx - half : idx + half]
            win = self.neural_detector.preprocess(win, self.fs, denoise=True)
            neg_windows.append(win)
        return np.array(pos_windows), np.array(neg_windows)

    def train_neural_on_ds1(self, progress_callback=None):
        """Обучение MLP на DS1."""
        X, y = [], []
        total = len(self.DS1)
        for i, rec in enumerate(self.DS1):
            try:
                sig, fs, peaks = self._load_record(rec)
                if fs != self.fs:
                    sig = signal.resample(sig, int(len(sig) * self.fs / fs))
                pos, neg = self._extract_windows(sig, peaks)
                if len(pos) > 0:
                    X.extend(pos); y.extend([1]*len(pos))
                if len(neg) > 0:
                    X.extend(neg); y.extend([0]*len(neg))
                if progress_callback:
                    progress_callback.emit(int(100*(i+1)/total), f"Обучение: запись {rec}")
            except Exception as e:
                print(f"Ошибка в {rec}: {e}")

        features = np.array([self.neural_detector.extract_features(w) for w in X])
        y = np.array(y)
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(features)
        self.model = MLPClassifier(hidden_layer_sizes=(72,36), activation='relu', solver='adam',
                                   alpha=0.0007, learning_rate_init=0.001, max_iter=300,
                                   early_stopping=True, validation_fraction=0.15, random_state=42)
        self.model.fit(Xs, y)
        with open('mitbih_model.pkl','wb') as f: pickle.dump(self.model, f)
        with open('mitbih_scaler.pkl','wb') as f: pickle.dump(self.scaler, f)

    def _prepare_neural_detector(self):
        if self.model is None or self.scaler is None:
            if os.path.exists('mitbih_model.pkl') and os.path.exists('mitbih_scaler.pkl'):
                with open('mitbih_model.pkl','rb') as f: self.model = pickle.load(f)
                with open('mitbih_scaler.pkl','rb') as f: self.scaler = pickle.load(f)
            else:
                raise RuntimeError("Модель не обучена.")
        self.neural_detector.model = self.model
        self.neural_detector.scaler = self.scaler

    def evaluate_detectors(self, detectors_dict, noise_types, progress_callback=None):
        """
        detectors_dict: {имя: объект детектора}
        noise_types: список строк ['white','baseline','motion','emg','combined']
        Возвращает: results[algo_name][noise_type] = (snr_values, avg_errors)
        """
        if "Нейросетевой детектор" in detectors_dict:
            self._prepare_neural_detector()
            detectors_dict["Нейросетевой детектор"] = self.neural_detector

        results = {}
        total_work = len(detectors_dict) * len(noise_types) * len(self.DS2) * 10
        done = 0

        for algo_name, detector in detectors_dict.items():
            results[algo_name] = {}
            for noise in noise_types:
                snr_values = list(range(1, 11))
                errors_per_snr = {snr: [] for snr in snr_values}
                for rec in self.DS2:
                    try:
                        sig, fs, ref_peaks = self._load_record(rec)
                        if fs != self.fs:
                            sig = signal.resample(sig, int(len(sig) * self.fs / fs))
                        for snr in snr_values:
                            noisy = add_noise_to_signal(sig, self.fs, noise, snr)
                            det_peaks = detector.detect(noisy, self.fs)
                            metrics = compute_metrics(ref_peaks, det_peaks, int(0.08 * self.fs))
                            errors_per_snr[snr].append(metrics['Error'])
                            done += 1
                            if progress_callback:
                                progress_callback.emit(int(100*done/total_work),
                                                       f"{algo_name} | {noise} | SNR={snr}")
                    except Exception as e:
                        print(f"Ошибка {algo_name}/{noise}/{rec}: {e}")
                avg_errors = [np.mean(errors_per_snr[snr]) for snr in snr_values]
                results[algo_name][noise] = (snr_values, avg_errors)
        return results


# ================= ГРАФИЧЕСКИЙ ИНТЕРФЕЙС =================
class QRSDetectorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.detectors = {}
        self.current_detector = None
        self.original_ecg_data = None
        self.filtered_ecg_data = None
        self.ecg_data = None
        self.fs = 360
        self.detected_peaks = []
        self.display_duration = 5
        self.display_offset = 0
        self.noise_level = 0.0
        self.current_noise = None
        self.noise_library_path = "noise_library.pkl"
        self.saved_noises = {}
        self.current_filename = "Не выбран"
        self.current_channel = None
        self.filter_enabled = False
        self.filter_lowcut = 0.5
        self.filter_highcut = 40.0
        self.notch_freq = 50.0
        self.display_processed_signal = None
        self.display_threshold_signal = None
        self.show_threshold_signal = False

        self.init_ui()
        self._clean_noise_library()
        self.load_noise_library()
        self.load_default_detectors()
        self.setFocus()
        self.setFocusPolicy(Qt.StrongFocus)

    def init_ui(self):
        self.setWindowTitle("QRS Complex Detector")
        self.setGeometry(50, 50, 1280, 960)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Левая панель
        left_panel = QWidget()
        left_panel.setMaximumWidth(420)
        left_layout = QVBoxLayout(left_panel)

        # Загрузка данных
        file_group = QGroupBox("Загрузка данных")
        file_layout = QVBoxLayout()
        self.file_path_label = QLabel("Файл не выбран")
        self.file_path_label.setWordWrap(True)
        load_btn = QPushButton("Загрузить ЭКГ сигнал")
        load_btn.clicked.connect(self.load_ecg_file)
        file_layout.addWidget(self.file_path_label)
        file_layout.addWidget(load_btn)
        file_group.setLayout(file_layout)
        left_layout.addWidget(file_group)

        # Алгоритмы
        algo_group = QGroupBox("Алгоритмы обнаружения")
        algo_layout = QVBoxLayout()
        self.algo_combo = QComboBox()
        self.algo_combo.currentIndexChanged.connect(self.on_algorithm_changed)
        self.algo_description = QTextEdit()
        self.algo_description.setReadOnly(True)
        self.algo_description.setMaximumHeight(90)
        algo_layout.addWidget(QLabel("Выберите алгоритм:"))
        algo_layout.addWidget(self.algo_combo)
        algo_layout.addWidget(QLabel("Описание:"))
        algo_layout.addWidget(self.algo_description)
        algo_group.setLayout(algo_layout)
        left_layout.addWidget(algo_group)

        # Параметры
        params_group = QGroupBox("Параметры")
        params_layout = QFormLayout()
        self.fs_spin = QSpinBox()
        self.fs_spin.setRange(100, 10000)
        self.fs_spin.setValue(self.fs)
        self.fs_spin.valueChanged.connect(self.on_fs_changed)
        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(1, 1000)
        self.duration_spin.setValue(self.display_duration)
        self.duration_spin.valueChanged.connect(self.on_duration_changed)
        params_layout.addRow("Частота дискретизации, Гц:", self.fs_spin)
        params_layout.addRow("Окно отображения, с:", self.duration_spin)
        params_group.setLayout(params_layout)
        left_layout.addWidget(params_group)

        # Фильтрация
        filter_group = QGroupBox("Фильтрация сигнала")
        filter_layout = QVBoxLayout()
        self.filter_checkbox = QCheckBox("Включить фильтрацию")
        self.filter_checkbox.stateChanged.connect(self.on_filter_toggled)
        filter_params_layout = QFormLayout()
        self.lowcut_spin = QDoubleSpinBox()
        self.lowcut_spin.setRange(0.1, 10.0)
        self.lowcut_spin.setSingleStep(0.1)
        self.lowcut_spin.setValue(self.filter_lowcut)
        self.lowcut_spin.valueChanged.connect(self.on_filter_params_changed)
        self.highcut_spin = QDoubleSpinBox()
        self.highcut_spin.setRange(10.0, 150.0)
        self.highcut_spin.setSingleStep(1.0)
        self.highcut_spin.setValue(self.filter_highcut)
        self.highcut_spin.valueChanged.connect(self.on_filter_params_changed)
        self.notch_combo = QComboBox()
        self.notch_combo.addItems(["Нет", "50 Гц", "60 Гц"])
        self.notch_combo.setCurrentIndex(1)
        self.notch_combo.currentIndexChanged.connect(self.on_filter_params_changed)
        filter_params_layout.addRow("ФВЧ, Гц:", self.lowcut_spin)
        filter_params_layout.addRow("ФНЧ, Гц:", self.highcut_spin)
        filter_params_layout.addRow("Режектор:", self.notch_combo)
        filter_layout.addWidget(self.filter_checkbox)
        filter_layout.addLayout(filter_params_layout)
        filter_group.setLayout(filter_layout)
        left_layout.addWidget(filter_group)

        # Шум
        noise_group = QGroupBox("Добавление шума")
        noise_layout = QVBoxLayout()
        self.noise_slider = QSlider(Qt.Horizontal)
        self.noise_slider.setRange(0, 100)
        self.noise_slider.setValue(0)
        self.noise_slider.valueChanged.connect(self.on_noise_changed)
        self.noise_label = QLabel("Уровень шума: 0%")
        self.white_noise_cb = QCheckBox("Белый гауссовский шум")
        self.noise_source_label = QLabel("Сохранённые шумы: 0")
        self.noise_list_widget = QListWidget()
        self.noise_list_widget.setMaximumHeight(130)
        load_noise_btn = QPushButton("Загрузить и сохранить шум")
        load_noise_btn.clicked.connect(self.load_noise_file)
        delete_noise_btn = QPushButton("Удалить выбранные шумы")
        delete_noise_btn.clicked.connect(self.delete_selected_noises)
        apply_noise_btn = QPushButton("Применить выбранные шумы")
        apply_noise_btn.clicked.connect(self.apply_noise)
        reset_noise_btn = QPushButton("Сбросить шум")
        reset_noise_btn.clicked.connect(self.reset_noise)
        noise_layout.addWidget(self.noise_label)
        noise_layout.addWidget(self.noise_slider)
        noise_layout.addWidget(self.white_noise_cb)
        noise_layout.addWidget(self.noise_source_label)
        noise_layout.addWidget(self.noise_list_widget)
        noise_layout.addWidget(load_noise_btn)
        noise_layout.addWidget(delete_noise_btn)
        noise_layout.addWidget(apply_noise_btn)
        noise_layout.addWidget(reset_noise_btn)
        noise_group.setLayout(noise_layout)
        left_layout.addWidget(noise_group)

        # Управление
        buttons_group = QGroupBox("Управление")
        buttons_layout = QVBoxLayout()
        detect_btn = QPushButton("Обнаружить QRS")
        detect_btn.clicked.connect(self.detect_qrs)
        detect_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        clear_btn = QPushButton("Очистить результаты")
        clear_btn.clicked.connect(self.clear_results)
        buttons_layout.addWidget(detect_btn)
        buttons_layout.addWidget(clear_btn)
        buttons_group.setLayout(buttons_layout)
        left_layout.addWidget(buttons_group)

        # Чекбокс "Показать сигнал до порога"
        self.threshold_cb = QCheckBox("Показать сигнал до порога (MWI/корреляция/энерг. карта)")
        self.threshold_cb.setChecked(False)
        self.threshold_cb.stateChanged.connect(self.on_threshold_toggled)
        left_layout.addWidget(self.threshold_cb)

        # Пользовательский алгоритм
        custom_group = QGroupBox("Добавить свой алгоритм")
        custom_layout = QVBoxLayout()
        self.custom_algo_path = QLineEdit()
        self.custom_algo_path.setPlaceholderText("Путь к файлу алгоритма")
        browse_btn = QPushButton("Обзор")
        browse_btn.clicked.connect(self.browse_custom_algorithm)
        load_custom_btn = QPushButton("Загрузить алгоритм")
        load_custom_btn.clicked.connect(self.load_custom_algorithm)
        custom_layout.addWidget(self.custom_algo_path)
        custom_layout.addWidget(browse_btn)
        custom_layout.addWidget(load_custom_btn)
        custom_group.setLayout(custom_layout)
        left_layout.addWidget(custom_group)

        # ------------------ ОЦЕНКА MIT‑BIH (РАСШИРЕННАЯ) ------------------
        mitbih_group = QGroupBox("Оценка MIT‑BIH (все детекторы и шумы)")
        mitbih_layout = QVBoxLayout()

        mitbih_layout.addWidget(QLabel("Папка с базой MIT‑BIH:"))
        path_layout = QHBoxLayout()
        self.mitbih_dir_edit = QLineEdit("./mitbih/")
        self.mitbih_dir_edit.setPlaceholderText("Путь к папке mitbih")
        browse_mitbih_btn = QPushButton("Обзор")
        browse_mitbih_btn.clicked.connect(self.browse_mitbih_dir)
        path_layout.addWidget(self.mitbih_dir_edit)
        path_layout.addWidget(browse_mitbih_btn)
        mitbih_layout.addLayout(path_layout)

        # Выбор детекторов
        mitbih_layout.addWidget(QLabel("Детекторы для оценки:"))
        self.mitbih_det_checks = {}
        self.mitbih_det_checks["Пан-Томпкинс"] = QCheckBox("Пан-Томпкинс")
        self.mitbih_det_checks["Корреляционный детектор"] = QCheckBox("Корреляционный")
        self.mitbih_det_checks["Нейросетевой детектор"] = QCheckBox("Нейросетевой")
        for cb in self.mitbih_det_checks.values():
            cb.setChecked(True)
            mitbih_layout.addWidget(cb)

        # Выбор типов шума
        mitbih_layout.addWidget(QLabel("Типы шума:"))
        self.mitbih_noise_checks = {}
        self.mitbih_noise_checks["white"] = QCheckBox("Белый гауссовский шум")
        self.mitbih_noise_checks["baseline"] = QCheckBox("Дрейф изолинии")
        self.mitbih_noise_checks["motion"] = QCheckBox("Артефакты движения электродов")
        self.mitbih_noise_checks["emg"] = QCheckBox("Мышечные артефакты")
        self.mitbih_noise_checks["combined"] = QCheckBox("Комбинированный шум")
        for cb in self.mitbih_noise_checks.values():
            cb.setChecked(True)
            mitbih_layout.addWidget(cb)

        self.train_mitbih_btn = QPushButton("Обучить нейросеть на DS1 (если нужна)")
        self.train_mitbih_btn.clicked.connect(self.train_mitbih)
        self.eval_mitbih_btn = QPushButton("Запустить оценку на DS2")
        self.eval_mitbih_btn.clicked.connect(self.eval_mitbih)

        self.mitbih_progress = QProgressBar()
        self.mitbih_progress.setVisible(False)

        mitbih_layout.addWidget(self.train_mitbih_btn)
        mitbih_layout.addWidget(self.eval_mitbih_btn)
        mitbih_layout.addWidget(self.mitbih_progress)
        mitbih_group.setLayout(mitbih_layout)
        left_layout.addWidget(mitbih_group)

        left_layout.addStretch()

        # Помещаем левую панель в QScrollArea
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(left_panel)
        main_layout.addWidget(scroll_area)

        # Правая панель с графиком
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self.figure = Figure(figsize=(12, 8), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        right_layout.addWidget(self.canvas)
        self.scroll_bar = QScrollBar(Qt.Horizontal)
        self.scroll_bar.setVisible(False)
        self.scroll_bar.valueChanged.connect(self.on_scroll)
        right_layout.addWidget(self.scroll_bar)
        self.info_label = QLabel("Готов к работе")
        right_layout.addWidget(self.info_label)

        main_layout.addWidget(right_panel, stretch=1)

    # ---------- MIT‑BIH методы ----------
    def browse_mitbih_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Выберите папку с MIT‑BIH")
        if dir_path:
            self.mitbih_dir_edit.setText(dir_path)

    def train_mitbih(self):
        mit_dir = self.mitbih_dir_edit.text().strip()
        if not os.path.isdir(mit_dir):
            QMessageBox.warning(self, "Ошибка", "Папка не найдена.")
            return
        self.train_mitbih_btn.setEnabled(False)
        self.mitbih_progress.setVisible(True)
        self.mitbih_progress.setValue(0)

        self.train_thread = QThread()
        self.train_worker = MITBIHTrainWorker(mit_dir)
        self.train_worker.moveToThread(self.train_thread)
        self.train_thread.started.connect(self.train_worker.run)
        self.train_worker.progress.connect(self.update_mitbih_progress)
        self.train_worker.finished.connect(self.on_train_finished)
        self.train_worker.error.connect(self.on_train_error)
        self.train_thread.start()

    def eval_mitbih(self):
        mit_dir = self.mitbih_dir_edit.text().strip()
        if not os.path.isdir(mit_dir):
            QMessageBox.warning(self, "Ошибка", "Папка не найдена.")
            return

        selected_detectors = {}
        for name, cb in self.mitbih_det_checks.items():
            if cb.isChecked() and name in self.detectors:
                selected_detectors[name] = self.detectors[name]

        if not selected_detectors:
            QMessageBox.warning(self, "Ошибка", "Не выбрано ни одного детектора.")
            return

        selected_noises = [n for n, cb in self.mitbih_noise_checks.items() if cb.isChecked()]
        if not selected_noises:
            QMessageBox.warning(self, "Ошибка", "Не выбран ни один тип шума.")
            return

        if "Нейросетевой детектор" in selected_detectors:
            if not os.path.exists('mitbih_model.pkl') or not os.path.exists('mitbih_scaler.pkl'):
                reply = QMessageBox.question(self, "Модель не найдена",
                                             "Для нейросетевого детектора требуется обученная модель. Обучить сейчас?",
                                             QMessageBox.Yes | QMessageBox.No)
                if reply == QMessageBox.Yes:
                    self.train_mitbih()
                return

        self.eval_mitbih_btn.setEnabled(False)
        self.mitbih_progress.setVisible(True)
        self.mitbih_progress.setValue(0)

        self.eval_thread = QThread()
        self.eval_worker = MITBIHMultiEvalWorker(mit_dir, selected_detectors, selected_noises)
        self.eval_worker.moveToThread(self.eval_thread)
        self.eval_thread.started.connect(self.eval_worker.run)
        self.eval_worker.progress.connect(self.update_mitbih_progress)
        self.eval_worker.finished.connect(self.on_multi_eval_finished)
        self.eval_worker.error.connect(self.on_eval_error)
        self.eval_thread.start()

    def update_mitbih_progress(self, percent, message):
        self.mitbih_progress.setValue(percent)
        self.info_label.setText(message)

    def on_train_finished(self):
        self.train_mitbih_btn.setEnabled(True)
        self.mitbih_progress.setVisible(False)
        QMessageBox.information(self, "Обучение", "Модель успешно обучена на DS1 и сохранена.")
        self.info_label.setText("Готов к работе")

    def on_train_error(self, error_msg):
        self.train_mitbih_btn.setEnabled(True)
        self.mitbih_progress.setVisible(False)
        QMessageBox.critical(self, "Ошибка обучения", error_msg)

    def on_multi_eval_finished(self, results):
        self.eval_mitbih_btn.setEnabled(True)
        self.mitbih_progress.setVisible(False)
        if not results:
            return

        noise_names = {
            "white": "Белый шум",
            "baseline": "Дрейф изолинии",
            "motion": "Артефакты движения",
            "emg": "Мышечные артефакты",
            "combined": "Комбинированный шум"
        }

        algo_list = list(results.keys())
        noise_list = list(results[algo_list[0]].keys())

        fig, axes = plt.subplots(len(algo_list), len(noise_list),
                                 figsize=(4*len(noise_list), 4*len(algo_list)))
        if len(algo_list) == 1 and len(noise_list) == 1:
            axes = np.array([[axes]])
        elif len(algo_list) == 1:
            axes = axes[np.newaxis, :]
        elif len(noise_list) == 1:
            axes = axes[:, np.newaxis]

        for i, algo in enumerate(algo_list):
            for j, noise in enumerate(noise_list):
                ax = axes[i][j]
                snr_vals, avg_err = results[algo][noise]
                ax.plot(snr_vals, avg_err, 'o-')
                ax.set_xlabel("SNR")
                ax.set_ylabel("Средняя ошибка")
                ax.set_title(f"{algo}\n{noise_names.get(noise, noise)}")
                ax.grid(True)
                ax.set_ylim(0, 1)

        plt.tight_layout()
        plt.show()
        self.info_label.setText("Оценка завершена")

    def on_eval_error(self, error_msg):
        self.eval_mitbih_btn.setEnabled(True)
        self.mitbih_progress.setVisible(False)
        QMessageBox.critical(self, "Ошибка оценки", error_msg)

    # ... (остальные методы без изменений: load_noise_library, ... , plot_ecg, ...)
    # Они идентичны предыдущей полной версии, поэтому здесь не дублируются для краткости.
    # При сборке итогового файла их нужно вставить полностью.

    def on_threshold_toggled(self, state):
        self.show_threshold_signal = (state == Qt.Checked)
        self.plot_ecg()

    def _clean_noise_library(self):
        if os.path.exists(self.noise_library_path):
            try:
                os.remove(self.noise_library_path)
            except:
                pass

    def load_noise_library(self):
        if os.path.exists(self.noise_library_path):
            try:
                with open(self.noise_library_path, "rb") as f:
                    raw = pickle.load(f)
            except:
                raw = {}
            cleaned = {}
            for name, val in raw.items():
                try:
                    arr = np.asarray(val, dtype=float).ravel()
                    if arr.size > 0:
                        cleaned[name] = arr
                except:
                    pass
            self.saved_noises = cleaned
            if len(cleaned) != len(raw):
                self.save_noise_library()
        else:
            self.saved_noises = {}
        self.refresh_noise_list()

    def save_noise_library(self):
        try:
            with open(self.noise_library_path, "wb") as f:
                pickle.dump(self.saved_noises, f)
        except Exception as e:
            QMessageBox.warning(self, "Ошибка", f"Не удалось сохранить библиотеку шумов: {e}")

    def refresh_noise_list(self):
        if not hasattr(self, "noise_list_widget"):
            return
        self.noise_list_widget.clear()
        for name in self.saved_noises:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.noise_list_widget.addItem(item)
        self.noise_source_label.setText(f"Сохранённые шумы: {len(self.saved_noises)}")

    def get_selected_saved_noises(self):
        selected = []
        for i in range(self.noise_list_widget.count()):
            item = self.noise_list_widget.item(i)
            if item is not None and item.checkState() == Qt.Checked:
                name = item.text()
                if name in self.saved_noises:
                    arr = self.saved_noises[name]
                    arr = np.asarray(arr, dtype=float).ravel().copy()
                    selected.append(arr)
        return selected

    def delete_selected_noises(self):
        names = []
        for i in range(self.noise_list_widget.count()):
            item = self.noise_list_widget.item(i)
            if item is not None and item.checkState() == Qt.Checked:
                names.append(item.text())
        if not names:
            QMessageBox.information(self, "Информация", "Не выбраны шумы для удаления.")
            return
        for name in names:
            self.saved_noises.pop(name, None)
        self.save_noise_library()
        self.refresh_noise_list()
        self.info_label.setText(f"Удалено шумов: {len(names)}")

    def load_noise_file(self):
        if self.original_ecg_data is None:
            QMessageBox.warning(self, "Предупреждение", "Сначала загрузите ЭКГ сигнал!")
            return
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл с шумом", "",
            "Все поддерживаемые форматы (*.csv *.txt *.dat *.hea);;CSV files (*.csv);;Text files (*.txt);;WFDB files (*.dat *.hea)"
        )
        if not file_path:
            return
        try:
            noise_sig = self.read_signal_file(file_path)
            noise_sig = np.asarray(noise_sig, dtype=float).ravel()
            noise_sig = np.nan_to_num(noise_sig - np.nanmean(noise_sig))
            if len(noise_sig) != len(self.original_ecg_data):
                QMessageBox.warning(
                    self, "Предупреждение",
                    f"Длина шума ({len(noise_sig)}) не совпадает с длиной ЭКГ ({len(self.original_ecg_data)}). "
                    f"При наложении такой шум будет пропущен."
                )
            default_name = os.path.basename(file_path)
            name, ok = QInputDialog.getText(self, "Название шума", "Введите название шума:", text=default_name)
            if not ok or not name.strip():
                return
            self.saved_noises[name.strip()] = noise_sig
            self.save_noise_library()
            self.refresh_noise_list()
            self.info_label.setText(f"Шум '{name.strip()}' сохранён")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить шумовой файл: {str(e)}")

    def apply_noise(self):
        if self.original_ecg_data is None:
            QMessageBox.warning(self, "Предупреждение", "Сначала загрузите ЭКГ сигнал!")
            return

        base = (self.filtered_ecg_data.copy() if self.filter_enabled
                else self.original_ecg_data.copy())
        base = np.asarray(base, dtype=float).ravel()

        noise = np.zeros_like(base, dtype=float)
        scale = float(self.noise_level) * (float(np.std(base)) + 1e-10)

        if self.white_noise_cb.isChecked():
            noise += scale * np.random.randn(len(base))

        skipped = 0
        for ns in self.get_selected_saved_noises():
            if ns.size != len(base):
                skipped += 1
                continue
            ns = np.nan_to_num(ns - np.nanmean(ns))
            ns_std = float(np.std(ns))
            ns_norm = ns / (ns_std + 1e-10)
            noise += scale * ns_norm

        if skipped:
            QMessageBox.information(self, "Информация",
                                    f"Пропущено шумов: {skipped} (длина не совпадает).")

        self.current_noise = noise
        self.ecg_data = base + noise
        self.detected_peaks = []
        self.display_processed_signal = None
        self.display_threshold_signal = None

        max_off = max(0, len(self.ecg_data) - int(self.display_duration * self.fs))
        self.display_offset = min(self.display_offset, max_off)

        self.update_scrollbar()
        self.plot_ecg()
        self.info_label.setText(f"Шум {self.noise_level*100:.0f}% применён")
        self.setFocus()

    def reset_noise(self):
        if self.original_ecg_data is None:
            return
        base = (self.filtered_ecg_data.copy() if self.filter_enabled
                else self.original_ecg_data.copy())
        self.ecg_data = np.asarray(base, dtype=float).ravel()
        self.noise_slider.setValue(0)
        self.noise_level = 0.0
        self.current_noise = None
        self.white_noise_cb.setChecked(False)
        for i in range(self.noise_list_widget.count()):
            item = self.noise_list_widget.item(i)
            if item:
                item.setCheckState(Qt.Unchecked)
        self.detected_peaks = []
        self.display_processed_signal = None
        self.display_threshold_signal = None
        max_off = max(0, len(self.ecg_data) - int(self.display_duration * self.fs))
        self.display_offset = min(self.display_offset, max_off)
        self.update_scrollbar()
        self.plot_ecg()
        self.info_label.setText("Шум сброшен")
        self.setFocus()

    def on_noise_changed(self, value):
        self.noise_level = value / 100.0
        self.noise_label.setText(f"Уровень шума: {value}%")

    def read_signal_file(self, file_path):
        temp_hea = None
        try:
            with open(file_path, "r", errors="ignore") as f:
                first_line = f.readline().strip()
            parts = first_line.split()
            is_wfdb = len(parts) >= 3 and parts[1].isdigit() and parts[2].isdigit()
            if is_wfdb:
                record_name = parts[0]
                dir_name = os.path.dirname(file_path)
                base_path = os.path.join(dir_name, record_name)
                hea_path = base_path + ".hea"
                if not os.path.exists(hea_path) and file_path.endswith(".hea.txt"):
                    temp_hea = hea_path
                    shutil.copy2(file_path, temp_hea)
                record = wfdb.rdrecord(base_path)
                if record.p_signal.ndim == 1:
                    data = record.p_signal.copy()
                else:
                    data = record.p_signal[:, 0].copy()
                return data
            delimiters = [",", ";", "\t", " "]
            data = None
            for d in delimiters:
                try:
                    data = np.loadtxt(file_path, delimiter=d)
                    break
                except:
                    continue
            if data is None:
                values = []
                with open(file_path, "r", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith(("#", "%", "@")):
                            parts_line = line.replace(",", " ").replace(";", " ").split()
                            if parts_line:
                                try:
                                    values.append(float(parts_line[0]))
                                except:
                                    pass
                if values:
                    data = np.array(values)
            if data is None or len(data) == 0:
                raise ValueError("Не удалось прочитать данные из файла")
            if data.ndim > 1:
                return data[:, 0].copy()
            return data.copy()
        finally:
            if temp_hea and os.path.exists(temp_hea):
                try:
                    os.remove(temp_hea)
                except:
                    pass

    def load_ecg_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл с ЭКГ", "",
            "Все поддерживаемые форматы (*.csv *.txt *.dat *.hea);;CSV files (*.csv);;Text files (*.txt);;WFDB files (*.dat *.hea)"
        )
        if not file_path:
            return
        self.file_path_label.setText(f"Файл: {os.path.basename(file_path)}")
        self.current_filename = os.path.basename(file_path)
        temp_hea = None
        try:
            with open(file_path, "r", errors="ignore") as f:
                first_line = f.readline().strip()
            parts = first_line.split()
            is_wfdb = len(parts) >= 3 and parts[1].isdigit() and parts[2].isdigit()
            if is_wfdb:
                record_name = parts[0]
                dir_name = os.path.dirname(file_path)
                base_path = os.path.join(dir_name, record_name)
                hea_path = base_path + ".hea"
                if not os.path.exists(hea_path) and file_path.endswith(".hea.txt"):
                    temp_hea = hea_path
                    shutil.copy2(file_path, temp_hea)
                record = wfdb.rdrecord(base_path)
                n_sig = record.p_signal.shape[1]
                if n_sig == 1:
                    channel = 0
                else:
                    channel, ok = QInputDialog.getInt(
                        self, "Выбор канала ЭКГ",
                        f"Запись содержит {n_sig} каналов. Введите номер канала (1..{n_sig}):",
                        1, 1, n_sig, 1
                    )
                    if not ok:
                        return
                    channel -= 1
                self.original_ecg_data = record.p_signal[:, channel].ravel().astype(float).copy()
                self.fs = int(record.fs)
                self.fs_spin.setValue(self.fs)
                self.current_channel = channel + 1
            else:
                data = self.read_signal_file(file_path)
                self.original_ecg_data = np.asarray(data, dtype=float).ravel().copy()
                self.original_ecg_data -= np.nanmean(self.original_ecg_data)
                self.current_channel = None
            self.original_ecg_data = np.nan_to_num(self.original_ecg_data)
            self.current_noise = None
            self.detected_peaks = []
            self.display_offset = 0
            self.display_processed_signal = None
            self.display_threshold_signal = None
            self.update_filtered_data()
            self.refresh_noise_list()
            self.info_label.setText(
                f"Загружена запись, канал {self.current_channel if self.current_channel else '?'}, "
                f"длина {len(self.original_ecg_data)} отсчётов, Fs={self.fs} Гц"
            )
            self.setFocus()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить файл: {str(e)}")
        finally:
            if temp_hea and os.path.exists(temp_hea):
                try:
                    os.remove(temp_hea)
                except:
                    pass

    def apply_filter_to_signal(self, signal_data):
        if signal_data is None:
            return None
        if not self.filter_enabled:
            return np.asarray(signal_data, dtype=float).ravel().copy()
        x = np.asarray(signal_data, dtype=float).ravel()
        x = np.nan_to_num(x - np.nanmean(x))
        nyq = self.fs / 2
        if self.filter_lowcut >= self.filter_highcut:
            raise ValueError("Нижняя частота фильтра должна быть меньше верхней.")
        if self.filter_highcut >= nyq:
            raise ValueError("Верхняя частота фильтра должна быть меньше Fs/2.")
        b_hp, a_hp = signal.butter(2, self.filter_lowcut / nyq, btype="high")
        filt = signal.filtfilt(b_hp, a_hp, x)
        b_lp, a_lp = signal.butter(2, self.filter_highcut / nyq, btype="low")
        filt = signal.filtfilt(b_lp, a_lp, filt)
        if self.notch_freq is not None and self.notch_freq < nyq:
            b_n, a_n = signal.iirnotch(self.notch_freq / nyq, 30.0)
            filt = signal.filtfilt(b_n, a_n, filt)
        return filt

    def update_filtered_data(self):
        if self.original_ecg_data is None:
            return
        try:
            self.filtered_ecg_data = self.apply_filter_to_signal(self.original_ecg_data)
            if self.current_noise is not None:
                self.ecg_data = self.filtered_ecg_data + self.current_noise
            else:
                self.ecg_data = self.filtered_ecg_data.copy()
            self.detected_peaks = []
            self.display_processed_signal = None
            self.display_threshold_signal = None
            self.update_scrollbar()
            self.plot_ecg()
            self.setFocus()
            if self.filter_enabled:
                notch_info = f", режекция {self.notch_freq} Гц" if self.notch_freq else ""
                self.info_label.setText(f"Применена фильтрация: ФВЧ {self.filter_lowcut} Гц, ФНЧ {self.filter_highcut} Гц{notch_info}")
            else:
                self.info_label.setText("Фильтрация выключена")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка фильтрации", str(e))

    def on_filter_toggled(self, state):
        self.filter_enabled = (state == Qt.Checked)
        if self.original_ecg_data is not None:
            self.update_filtered_data()

    def on_filter_params_changed(self):
        self.filter_lowcut = self.lowcut_spin.value()
        self.filter_highcut = self.highcut_spin.value()
        notch = self.notch_combo.currentText()
        if notch == "Нет":
            self.notch_freq = None
        elif notch == "50 Гц":
            self.notch_freq = 50.0
        elif notch == "60 Гц":
            self.notch_freq = 60.0
        if self.filter_enabled and self.original_ecg_data is not None:
            self.update_filtered_data()

    def load_default_detectors(self):
        for det in [PanTompkinsDetector(), NeuralDetector(), CorrelationDetector()]:
            self.detectors[det.name] = det
            self.algo_combo.addItem(det.name)
        self.algo_combo.setCurrentText("Нейросетевой детектор")
        self.on_algorithm_changed(self.algo_combo.currentIndex())

    def on_algorithm_changed(self, index):
        name = self.algo_combo.currentText()
        if name in self.detectors:
            self.current_detector = self.detectors[name]
            self.algo_description.setText(self.current_detector.description)
        else:
            self.current_detector = None
            self.algo_description.setText("Описание отсутствует")

    def detect_qrs(self):
        if self.ecg_data is None:
            QMessageBox.warning(self, "Предупреждение", "Сначала загрузите ЭКГ сигнал!")
            return
        if self.current_detector is None:
            QMessageBox.warning(self, "Предупреждение", "Выбранный алгоритм недоступен.")
            return
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.detected_peaks = self.current_detector.detect(self.ecg_data, self.fs)
            self.display_processed_signal = self.current_detector.get_processed_signal(self.ecg_data, self.fs)
            self.display_threshold_signal = self.current_detector.get_debug_signal()
            QApplication.restoreOverrideCursor()
            self.plot_ecg()
            self.info_label.setText(f"Обнаружено {len(self.detected_peaks)} QRS комплексов алгоритмом '{self.current_detector.name}'")
            self.setFocus()
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Ошибка", str(e))

    def clear_results(self):
        self.detected_peaks = []
        self.display_processed_signal = None
        self.display_threshold_signal = None
        self.plot_ecg()
        self.info_label.setText("Результаты очищены")
        self.setFocus()

    def on_fs_changed(self, value):
        self.fs = value
        if self.original_ecg_data is not None:
            self.update_filtered_data()

    def on_duration_changed(self, value):
        self.display_duration = value
        if self.ecg_data is not None:
            max_off = max(0, len(self.ecg_data) - int(self.display_duration*self.fs))
            self.display_offset = min(self.display_offset, max_off)
            self.update_scrollbar()
            self.plot_ecg()

    def keyPressEvent(self, event):
        if self.ecg_data is None:
            super().keyPressEvent(event)
            return
        step = int(self.display_duration*self.fs*0.5)
        max_off = max(0, len(self.ecg_data)-int(self.display_duration*self.fs))
        if event.key() == Qt.Key_Left:
            new_off = max(0, self.display_offset - step)
            if new_off != self.display_offset:
                self.display_offset = new_off
                self.scroll_bar.setValue(self.display_offset)
                self.plot_ecg()
            event.accept()
        elif event.key() == Qt.Key_Right:
            new_off = min(max_off, self.display_offset + step)
            if new_off != self.display_offset:
                self.display_offset = new_off
                self.scroll_bar.setValue(self.display_offset)
                self.plot_ecg()
            event.accept()
        else:
            super().keyPressEvent(event)

    def wheelEvent(self, event):
        if self.ecg_data is None:
            event.ignore()
            return
        step = int(self.display_duration*self.fs*0.1)
        max_off = max(0, len(self.ecg_data)-int(self.display_duration*self.fs))
        delta = event.angleDelta().y()
        if delta > 0:
            new_off = max(0, self.display_offset - step)
        else:
            new_off = min(max_off, self.display_offset + step)
        if new_off != self.display_offset:
            self.display_offset = new_off
            self.scroll_bar.setValue(self.display_offset)
            self.plot_ecg()
        event.accept()

    def on_scroll(self, value):
        if self.ecg_data is None:
            return
        if value != self.display_offset:
            self.display_offset = value
            self.plot_ecg()
            self.info_label.setText(f"Сдвиг: {self.display_offset/self.fs:.1f} с")

    def update_scrollbar(self):
        if self.ecg_data is None:
            self.scroll_bar.setVisible(False)
            return
        page_step = int(self.display_duration*self.fs)
        max_off = max(0, len(self.ecg_data) - page_step)
        self.scroll_bar.setRange(0, max_off)
        self.scroll_bar.setPageStep(page_step)
        self.scroll_bar.setSingleStep(max(1, page_step//10))
        self.scroll_bar.setValue(self.display_offset)
        self.scroll_bar.setVisible(True)

    def plot_ecg(self):
        if self.original_ecg_data is None:
            return
        has_noise = self.current_noise is not None
        has_threshold = self.show_threshold_signal and self.display_threshold_signal is not None

        if has_noise and has_threshold:
            num_plots = 4
        elif has_noise or has_threshold:
            num_plots = 3
        else:
            num_plots = 2

        self.figure.clear()
        axes = [self.figure.add_subplot(num_plots, 1, i+1) for i in range(num_plots)]

        start = self.display_offset
        end = min(start + int(self.display_duration*self.fs), len(self.original_ecg_data))
        time = np.arange(start, end) / self.fs

        plot_idx = 0

        # 1. Исходный сигнал
        axes[plot_idx].plot(time, self.original_ecg_data[start:end], 'b-', linewidth=1, label='Исходный ЭКГ')
        axes[plot_idx].set_ylabel('Амплитуда')
        axes[plot_idx].set_title('Исходный сигнал')
        axes[plot_idx].legend(loc='upper right')
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1

        # 2. Обработанный сигнал
        if self.display_processed_signal is not None:
            sig = self.display_processed_signal
        else:
            sig = self.ecg_data if self.ecg_data is not None else self.original_ecg_data
        sig = np.asarray(sig).ravel()
        axes[plot_idx].plot(time, sig[start:end], 'g-', linewidth=1, label='Обработанный (вход детектора)')
        if self.detected_peaks is not None and len(self.detected_peaks) > 0:
            valid = [p for p in self.detected_peaks if start <= p < end]
            if len(valid) > 0:
                axes[plot_idx].plot(np.array(valid)/self.fs, sig[valid], 'ro', markersize=6, label='QRS')
        axes[plot_idx].set_ylabel('Амплитуда')
        filter_status = 'фильтрованный' if self.filter_enabled else 'без фильтрации'
        noise_status = (
            f' + шум {self.noise_level*100:.0f}%'
            if self.current_noise is not None and self.current_noise.size > 0
            else ''
        )
        axes[plot_idx].set_title(f'Обработанный сигнал ({filter_status}{noise_status})')
        axes[plot_idx].legend(loc='upper right')
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1

        # 3. Сигнал до порога (если включено)
        if has_threshold:
            dbg = np.asarray(self.display_threshold_signal).ravel()
            axes[plot_idx].plot(time[:len(dbg)], dbg[start:end], 'm-', linewidth=1, label='Сигнал перед порогом')
            axes[plot_idx].set_ylabel('Значение')

            detail = ''
            if self.current_detector is not None:
                det_name = self.current_detector.name
                if det_name == "Пан-Томпкинс":
                    detail = "(MWI)"
                elif det_name == "Корреляционный детектор":
                    detail = "(Нормированная корреляционная функция)"
                elif det_name == "Нейросетевой детектор":
                    detail = "(Энергетическая карта QRS)"
            title = 'Сигнал до пороговой обработки'
            if detail:
                title += f' {detail}'
            axes[plot_idx].set_title(title)

            axes[plot_idx].legend(loc='upper right')
            axes[plot_idx].grid(True, alpha=0.3)
            plot_idx += 1

        # 4. Шум (если есть)
        if has_noise:
            noise_plot = np.asarray(self.current_noise).ravel()
            axes[plot_idx].plot(time, noise_plot[start:end], 'orange', linewidth=1, label='Шум')
            axes[plot_idx].set_xlabel('Время, с')
            axes[plot_idx].set_ylabel('Амплитуда')
            axes[plot_idx].set_title(f'Наложенный шум, уровень {self.noise_level*100:.0f}%')
            axes[plot_idx].legend(loc='upper right')
            axes[plot_idx].grid(True, alpha=0.3)
        else:
            axes[-1].set_xlabel('Время, с')

        self.figure.suptitle(f'Файл: {self.current_filename}' + (f' | канал {self.current_channel}' if self.current_channel else '') +
                             f'\nОтображено {self.display_duration} с из {len(self.original_ecg_data)/self.fs:.1f} с, смещение {self.display_offset/self.fs:.1f} с')
        self.figure.tight_layout()
        self.canvas.draw()

    def browse_custom_algorithm(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выберите файл с алгоритмом", "", "Python files (*.py)")
        if path:
            self.custom_algo_path.setText(path)

    def load_custom_algorithm(self):
        raw = self.custom_algo_path.text()
        if not raw or not raw.strip():
            QMessageBox.warning(self, "Предупреждение", "Укажите путь к файлу алгоритма!")
            return
        path = raw.strip().strip('"').strip("'")
        path = os.path.normpath(path)
        if not path.lower().endswith(".py"):
            QMessageBox.warning(self, "Предупреждение", "Файл должен иметь расширение .py")
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                code = f.read()
        except UnicodeDecodeError:
            try:
                with open(path, 'r', encoding='cp1251') as f:
                    code = f.read()
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось прочитать файл: {e}")
                return
        except FileNotFoundError:
            QMessageBox.critical(self, "Ошибка", "Файл не найден.")
            return
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка: {e}")
            return
        try:
            import types
            module = types.ModuleType("custom")
            module.__dict__["QRSDetector"] = QRSDetector
            module.__dict__["np"] = np
            module.__dict__["signal"] = signal
            exec(compile(code, path, "exec"), module.__dict__)
            custom_det = None
            for name in dir(module):
                obj = getattr(module, name)
                if isinstance(obj, type) and issubclass(obj, QRSDetector) and obj is not QRSDetector:
                    custom_det = obj()
                    break
            if custom_det is None:
                QMessageBox.warning(self, "Ошибка", "Класс, наследующий QRSDetector, не найден.")
                return
            if not hasattr(custom_det, "detect"):
                QMessageBox.warning(self, "Ошибка", "Метод detect отсутствует.")
                return
            self.detectors[custom_det.name] = custom_det
            self.algo_combo.addItem(custom_det.name)
            QMessageBox.information(self, "Успех", f"Алгоритм '{custom_det.name}' загружен.")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить алгоритм: {e}")


# ================= ПОТОКИ ДЛЯ MIT‑BIH =================
class MITBIHTrainWorker(QObject):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, mit_dir):
        super().__init__()
        self.mit_dir = mit_dir

    def run(self):
        try:
            evaluator = MITBIHMultiEvaluator(self.mit_dir)
            evaluator.train_neural_on_ds1(progress_callback=self.progress)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))


class MITBIHMultiEvalWorker(QObject):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, mit_dir, detectors, noises):
        super().__init__()
        self.mit_dir = mit_dir
        self.detectors = detectors
        self.noises = noises

    def run(self):
        try:
            evaluator = MITBIHMultiEvaluator(self.mit_dir)
            res = evaluator.evaluate_detectors(self.detectors, self.noises,
                                               progress_callback=self.progress)
            self.finished.emit(res)
        except Exception as e:
            self.error.emit(str(e))


def main():
    app = QApplication(sys.argv)
    win = QRSDetectorGUI()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()