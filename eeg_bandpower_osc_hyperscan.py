#!/usr/bin/env python3
"""
EEG hyperscanning processor: LSL → ASR → bandpower → z/0-1 → calm ratio → OSC

This script reads all available LSL EEG streams, cleans them with ASR (asrpy),
computes bandpower in overlapping windows, normalizes per-participant/channel/band,
computes a calm ratio (alpha/beta), and sends per-channel, per-participant, and
group metrics over OSC to TouchDesigner/Max.

Design choices (aligned with requirements):
- Two-script approach: this is the processing/OSC script; streaming is handled by lsl_multi_streamer.py.
- Threads: one worker thread per headset to pull LSL data without blocking others.
- Windowing: 2.5 s Welch windows, 50% overlap, assuming 256 Hz (configurable).
- ASR: uses asrpy (DBSCAN/GEV per library defaults), 60 s calibration, 175 µV threshold.
- Baseline/Z: rolling baseline trial in the 10–60 s range; default here is 20 s; require ≥3 windows before use.
- Normalization: percentile-based 0–1 mapping (5th–95th of baseline) with fallback to z-clamp.
- Calm metric: alpha/beta ratio → ratio/(ratio+1) → 0–1.
- Synchrony: similarity = 1 − |Δ calm|; rolling correlation over 15 s.
- Channel weights for calm: AF7/AF8 = 0.3 each, TP9/TP10 = 0.2 each.
- Artifact flags: simple amplitude checks for blink/motion; log when ASR reconstructs vs clean.
- Logging: concise; startup shows participant mapping; optional debug for first OSC paths.

Usage:
    python -u eeg_bandpower_osc_hyperscan.py --osc-port 9005 --log-first-osc

Dependencies:
    pylsl, numpy, scipy, python-osc, asrpy (https://github.com/thiagorroque/asrpy)
"""

import argparse
import math
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
from pylsl import StreamInlet, resolve_byprop
from pythonosc.udp_client import SimpleUDPClient
from scipy.signal import welch, butter, lfilter

try:
    from asrpy import ASR, ASRCalibrated
except Exception:
    ASR = None  # type: ignore
    ASRCalibrated = None  # type: ignore

# -------------------- PARAMETERS --------------------
FS = 256
WIN_SEC = 2.5
OVERLAP = 0.5
UPDATE_HZ = 4
BASELINE_SEC = 20.0  # trial setting; adjust 10–60 s as needed
CALIB_SEC = 60.0
ASR_THRESHOLD_UV = 175.0
SYNCH_CORR_WIN_SEC = 15.0
PERCENTILE_LO = 5.0
PERCENTILE_HI = 95.0
EPS = 1e-12
OSC_PREFIX = "/eeg"
CHANNELS = ["tp9", "af7", "af8", "tp10"]
CHANNEL_WEIGHTS = {"tp9": 0.2, "af7": 0.3, "af8": 0.3, "tp10": 0.2}
BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}
# Pin specific headsets to participant IDs
PINNED_OSC_IDS = {
    "Muse-2A27": 1,
    "Muse-4730": 2,
    "C6F24F1D-7513-7F63-882D-60E3F832B871": 1,  # MAC for 2A27
    "336C8D9E-08D3-C640-673F-D7782318386A": 2,  # MAC for 4730
}

# Filters (explicit bandpass + notch before ASR)
def make_filters(fs: float, notch_hz: float = 60.0):
    nyq = 0.5 * fs
    b_bp, a_bp = butter(4, [1.0 / nyq, 50.0 / nyq], btype="band")
    b_notch, a_notch = butter(4, [(notch_hz - 2.0) / nyq, (notch_hz + 2.0) / nyq], btype="bandstop")
    return (b_bp, a_bp), (b_notch, a_notch)


def apply_filters(data: np.ndarray, filters: Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    (b_bp, a_bp), (b_notch, a_notch) = filters
    d = lfilter(b_bp, a_bp, data, axis=1)
    d = lfilter(b_notch, a_notch, d, axis=1)
    return d


def bandpower(freqs: np.ndarray, psd: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
    idx = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(idx):
        return np.zeros(psd.shape[0], dtype=np.float32)
    return psd[:, idx].mean(axis=1)


def percentile_01(val: float, history: deque) -> float:
    if len(history) < 5:
        return 0.5  # not enough history
    arr = np.array(history, dtype=np.float32)
    lo = np.percentile(arr, PERCENTILE_LO)
    hi = np.percentile(arr, PERCENTILE_HI)
    if hi - lo < EPS:
        return 0.5
    return float(np.clip((val - lo) / (hi - lo), 0.0, 1.0))


def z01(val: float, history: deque) -> float:
    if len(history) < 5:
        return 0.5
    mu = float(np.mean(history))
    sd = float(np.std(history) + EPS)
    z = (val - mu) / sd
    return float(np.clip((z + 2.0) / 4.0, 0.0, 1.0))


def calm_ratio(alpha: float, beta: float) -> float:
    ratio = (alpha + EPS) / (beta + EPS)
    return float(ratio / (ratio + 1.0))


def mean_pairwise_absdiff(vals: List[float]) -> float:
    if len(vals) < 2:
        return 0.0
    diffs = [abs(a - b) for a, b in combinations(vals, 2)]
    return float(np.mean(diffs)) if diffs else 0.0


def rolling_corr(x: deque, y: deque) -> Optional[float]:
    if len(x) < 5 or len(y) < 5:
        return None
    n = min(len(x), len(y))
    xa = np.array(list(x)[-n:], dtype=np.float32)
    ya = np.array(list(y)[-n:], dtype=np.float32)
    if xa.std() < EPS or ya.std() < EPS:
        return None
    return float(np.corrcoef(xa, ya)[0, 1])


@dataclass
class ParticipantState:
    osc_id: int
    inlet: StreamInlet
    sample_queue: queue.Queue = field(default_factory=queue.Queue)
    sample_buffer: deque = field(default_factory=deque)
    asr_model: Optional[ASRCalibrated] = None
    asr_needs_calib: bool = True
    asr_buffer: List[np.ndarray] = field(default_factory=list)
    baseline: Dict[Tuple[str, str], deque] = field(default_factory=dict)
    calm_hist: deque = field(default_factory=lambda: deque(maxlen=int(SYNCH_CORR_WIN_SEC * UPDATE_HZ)))
    smoothed: Dict[Tuple[str, str], float] = field(default_factory=dict)
    artifact_flags: Dict[str, bool] = field(default_factory=dict)


def lsl_worker(part: ParticipantState, stop_evt: threading.Event, max_queue_sec: float) -> None:
    max_queue = int(max_queue_sec * FS)
    inlet = part.inlet
    while not stop_evt.is_set():
        chunk, _ = inlet.pull_chunk(timeout=0.05, max_samples=64)
        if chunk:
            for s in chunk:
                try:
                    part.sample_queue.put_nowait(np.array(s[:4], dtype=np.float32))
                except queue.Full:
                    # Drop old samples to keep responsiveness
                    try:
                        while part.sample_queue.qsize() > max_queue // 2:
                            part.sample_queue.get_nowait()
                    except Exception:
                        pass
        else:
            time.sleep(0.01)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--osc-host", default="127.0.0.1")
    ap.add_argument("--osc-port", type=int, default=9005)
    ap.add_argument("--lsl-timeout", type=float, default=5.0)
    ap.add_argument("--update-hz", type=float, default=UPDATE_HZ)
    ap.add_argument("--max-queue-sec", type=float, default=2.0, help="Per-headset buffer before dropping samples")
    ap.add_argument("--log-first-osc", action="store_true", help="Print first send of each OSC path")
    args = ap.parse_args()

    client = SimpleUDPClient(args.osc_host, args.osc_port)

    print("Resolving LSL EEG streams…")
    streams = resolve_byprop("type", "EEG", timeout=args.lsl_timeout)
    if not streams:
        raise RuntimeError("No LSL EEG streams found.")

    def match_pinned_id(info) -> Optional[int]:
        name = info.name() or ""
        src = info.source_id() or ""
        for needle, pid in PINNED_OSC_IDS.items():
            if needle in name or needle in src:
                return pid
        return None

    pinned = []
    unpinned = []
    seen_ids = set()

    for s in streams:
        pid = match_pinned_id(s)
        if pid is not None:
            if pid in seen_ids:
                print(f"Warning: duplicate pinned id p{pid} for {s.name()} ({s.source_id()}); skipping pin.")
                unpinned.append(s)
            else:
                pinned.append((pid, s))
                seen_ids.add(pid)
        else:
            unpinned.append(s)

    participants: List[ParticipantState] = []
    # Add pinned in order
    for pid, info in sorted(pinned, key=lambda x: x[0]):
        inlet = StreamInlet(info, max_buflen=120, max_chunklen=64)
        ps = ParticipantState(
            osc_id=pid,
            inlet=inlet,
            baseline={(ch, b): deque(maxlen=int(BASELINE_SEC * args.update_hz)) for ch in CHANNELS for b in BANDS},
            smoothed={(ch, b): None for ch in CHANNELS for b in BANDS},  # type: ignore
            artifact_flags={"blink": False, "motion": False},
        )
        participants.append(ps)
        print(f"  LSL pinned -> p{ps.osc_id} ({info.name()} | {info.source_id()})")

    # Fill remaining in discovery order, skipping taken IDs
    next_pid = 1
    for info in unpinned:
        while next_pid in seen_ids:
            next_pid += 1
        inlet = StreamInlet(info, max_buflen=120, max_chunklen=64)
        ps = ParticipantState(
            osc_id=next_pid,
            inlet=inlet,
            baseline={(ch, b): deque(maxlen=int(BASELINE_SEC * args.update_hz)) for ch in CHANNELS for b in BANDS},
            smoothed={(ch, b): None for ch in CHANNELS for b in BANDS},  # type: ignore
            artifact_flags={"blink": False, "motion": False},
        )
        participants.append(ps)
        seen_ids.add(next_pid)
        print(f"  LSL index -> p{ps.osc_id} ({info.name()} | {info.source_id()})")
        next_pid += 1

    stop_evt = threading.Event()
    threads = []
    for ps in participants:
        t = threading.Thread(target=lsl_worker, args=(ps, stop_evt, args.max_queue_sec), daemon=True)
        t.start()
        threads.append(t)

    win = int(FS * WIN_SEC)
    overlap = int(win * OVERLAP)
    next_send = time.time()
    logged_paths: set[str] = set()

    # Per-participant rolling sample buffers sized to the analysis window
    for ps in participants:
        ps.sample_buffer = deque(maxlen=win)

    filters = make_filters(FS)

    print("Running processing → OSC…")

    try:
        while not stop_evt.is_set():
            if time.time() < next_send:
                time.sleep(0.005)
                continue
            next_send = time.time() + 1.0 / args.update_hz

            participant_calm: List[float] = []
            calm_history_map: Dict[int, deque] = {}

            for ps in participants:
                # Drain queue into buffer
                buf: List[np.ndarray] = []
                try:
                    while True:
                        buf.append(ps.sample_queue.get_nowait())
                except queue.Empty:
                    pass
                for s in buf:
                    ps.sample_buffer.append(s)
                if len(ps.sample_buffer) < win:
                    continue

                data = np.array(ps.sample_buffer, dtype=np.float32).T  # (ch, t)

                # Apply explicit bandpass + notch before ASR/Welch
                filtered_data = apply_filters(data, filters)

                # ASR calibration does not block output; we pass through until calibrated.
                if ASR is not None and ps.asr_needs_calib:
                    ps.asr_buffer.append(filtered_data)
                    calib_len = sum([d.shape[1] for d in ps.asr_buffer]) / FS
                    if calib_len >= CALIB_SEC:
                        concat = np.concatenate(ps.asr_buffer, axis=1)
                        try:
                            ps.asr_model = ASR(FS, cutoff=ASR_THRESHOLD_UV).fit(concat)
                            ps.asr_needs_calib = False
                            print(f"Calibration complete for p{ps.osc_id} with {calib_len:.1f}s")
                        except Exception as e:
                            print(f"ASR calibration failed for p{ps.osc_id}: {e}")
                            ps.asr_needs_calib = True
                            ps.asr_buffer = []
                    # Block outputs until calibration is done
                    continue

                # Apply ASR if available; otherwise pass through
                if ps.asr_model is not None:
                    try:
                        data = ps.asr_model.transform(filtered_data)
                        ps.artifact_flags["motion"] = False
                    except Exception:
                        ps.artifact_flags["motion"] = True
                else:
                    # Simple artifact flag: large amplitude spike
                    data = filtered_data
                    if np.max(np.abs(filtered_data)) > ASR_THRESHOLD_UV:
                        ps.artifact_flags["motion"] = True

                # Blink flag: spike in frontal channels
                frontal = data[[1, 2], :]
                ps.artifact_flags["blink"] = bool(np.max(np.abs(frontal)) > ASR_THRESHOLD_UV / 2)

                if data.shape[1] < win:
                    continue

                freqs, psd = welch(data, fs=FS, nperseg=win, noverlap=overlap, axis=1)
                psd = np.log10(psd + EPS)

                calm_vals = []
                calm_weights = []

                alpha_vals = {}
                beta_vals = {}

                for band, (fmin, fmax) in BANDS.items():
                    bp = bandpower(freqs, psd, fmin, fmax)  # per channel
                    for ci, ch in enumerate(CHANNELS):
                        key = (ch, band)
                        prev = ps.smoothed[key]
                        cur = float(bp[ci]) if prev is None else float(0.85 * prev + 0.15 * bp[ci])
                        ps.smoothed[key] = cur

                        hist = ps.baseline[key]
                        hist.append(cur)
                        if len(hist) < max(3, int(3 * args.update_hz)):
                            continue

                        val01 = percentile_01(cur, hist)
                        p_out = ps.osc_id
                        client.send_message(f"{OSC_PREFIX}/p{p_out}/{ch}/{band}/power", cur)
                        client.send_message(f"{OSC_PREFIX}/p{p_out}/{ch}/{band}/01", val01)
                        if args.log_first_osc:
                            for path in (
                                f"{OSC_PREFIX}/p{p_out}/{ch}/{band}/power",
                                f"{OSC_PREFIX}/p{p_out}/{ch}/{band}/01",
                            ):
                                if path not in logged_paths:
                                    print(f"OSC send (first): {path}")
                                    logged_paths.add(path)

                    # Participant band average
                    vals = [ps.smoothed[(ch, band)] for ch in CHANNELS if ps.smoothed[(ch, band)] is not None]
                    if vals:
                        p_out = ps.osc_id
                        client.send_message(f"{OSC_PREFIX}/p{p_out}/avg/{band}/power", float(np.mean(vals)))
                        vals01 = [
                            percentile_01(ps.smoothed[(ch, band)], ps.baseline[(ch, band)])
                            for ch in CHANNELS
                            if ps.smoothed[(ch, band)] is not None and len(ps.baseline[(ch, band)]) >= 5
                        ]
                        if vals01:
                            client.send_message(f"{OSC_PREFIX}/p{p_out}/avg/{band}/01", float(np.mean(vals01)))

                    if band == "alpha":
                        alpha_vals = {ch: ps.smoothed[(ch, band)] for ch in CHANNELS}
                    if band == "beta":
                        beta_vals = {ch: ps.smoothed[(ch, band)] for ch in CHANNELS}

                # Calm ratio (weighted)
                if alpha_vals and beta_vals:
                    alpha_w = []
                    beta_w = []
                    weights = []
                    for ch in CHANNELS:
                        if alpha_vals.get(ch) is None or beta_vals.get(ch) is None:
                            continue
                        w = CHANNEL_WEIGHTS[ch]
                        alpha_w.append(alpha_vals[ch] * w)
                        beta_w.append(beta_vals[ch] * w)
                        weights.append(w)
                    if alpha_w and beta_w:
                        a = sum(alpha_w) / sum(weights)
                        b = sum(beta_w) / sum(weights)
                        calm_val = calm_ratio(a, b)
                        p_out = ps.osc_id
                        client.send_message(f"{OSC_PREFIX}/p{p_out}/calm_ratio01", calm_val)
                        ps.calm_hist.append(calm_val)
                        calm_vals.append(calm_val)
                        calm_weights.append(1.0)

                # Artifact flags
                p_out = ps.osc_id
                client.send_message(f"{OSC_PREFIX}/p{p_out}/artifacts/blink", float(ps.artifact_flags["blink"]))
                client.send_message(f"{OSC_PREFIX}/p{p_out}/artifacts/motion", float(ps.artifact_flags["motion"]))

                if calm_vals:
                    participant_calm.append(float(np.mean(calm_vals)))
                    calm_history_map[p_out] = ps.calm_hist

            # Group metrics
            if participant_calm:
                similarity = 1.0 - mean_pairwise_absdiff(participant_calm)
                client.send_message(f"{OSC_PREFIX}/group/similarity", similarity)

            # Rolling correlation (pairwise) over SYNCH_CORR_WIN_SEC
            ids = list(calm_history_map.keys())
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    c1 = calm_history_map[ids[i]]
                    c2 = calm_history_map[ids[j]]
                    r = rolling_corr(c1, c2)
                    if r is not None:
                        client.send_message(f"{OSC_PREFIX}/group/correlation/p{ids[i]}_p{ids[j]}", r)
    except KeyboardInterrupt:
        stop_evt.set()
    finally:
        stop_evt.set()
        for t in threads:
            t.join(timeout=1.0)


if __name__ == "__main__":
    main()
