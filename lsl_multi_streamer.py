#!/usr/bin/env python3
"""
Multi-Headset LSL streamer (wrapper around muselsl CLI)

What this does:
- Launches one muselsl streaming process per headset MAC address.
- Auto-restarts a process if it dies.
- Prints a clear status line for each headset so you know what is running.

How to use:
    python lsl_multi_streamer.py --mac C6F24F1D-7513-7F63-882D-60E3F832B871 --mac 336C8D9E-08D3-C640-673F-D7782318386A

Notes:
- Requires the muselsl package installed and available on PATH (the same environment you normally use).
- This script is lightweight: it does not do any EEG processing, just manages the muselsl stream commands.
- Auto-reconnect: if a muselsl process exits, we wait a few seconds and relaunch.
"""

import argparse
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class StreamProcess:
    mac: str
    proc: subprocess.Popen | None
    restarts: int = 0


def launch_stream(mac: str, python_cmd: str) -> subprocess.Popen:
    """Launch muselsl stream for a given MAC. Returns the Popen handle."""
    # Use the muselsl CLI; this mirrors "muselsl stream --address <MAC>"
    cmd = [python_cmd, "-m", "muselsl", "stream", "--address", mac]
    print(f"[launcher] starting muselsl stream for {mac}: {' '.join(cmd)}")
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def monitor_streams(streams: Dict[str, StreamProcess], python_cmd: str, restart_delay: float) -> None:
    """Main loop: keep processes alive; restart if they exit."""
    try:
        while True:
            for mac, sp in list(streams.items()):
                if sp.proc is None or sp.proc.poll() is not None:
                    # Process not running; restart
                    sp.restarts += 1
                    if sp.proc and sp.proc.stdout:
                        # Drain any remaining output
                        try:
                            tail = sp.proc.stdout.read()
                            if tail:
                                print(f"[{mac}] last output before exit:\n{tail.rstrip()}")
                        except Exception:
                            pass
                    time.sleep(restart_delay)
                    sp.proc = launch_stream(mac, python_cmd)
                else:
                    # Emit concise status from muselsl output
                    try:
                        if sp.proc.stdout and sp.proc.stdout.readable():
                            line = sp.proc.stdout.readline()
                            if line:
                                lower = line.lower()
                                if "connecting to muse" in lower:
                                    print(f"[{mac}] Connecting…")
                                elif "streaming" in lower:
                                    print(f"[{mac}] Connected. Streaming EEG…")
                                elif "connected" in lower:
                                    print(f"[{mac}] Connected.")
                                elif "failed to connect" in lower:
                                    print(f"[{mac}] Failed to connect. Retrying…")
                                # Suppress chatty multicast/interface logs; uncomment below to see all
                                # else:
                                #     print(f"[{mac}] {line.rstrip()}")
                    except Exception:
                        pass
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Stopping all muselsl streamers…")
    finally:
        for mac, sp in streams.items():
            if sp.proc and sp.proc.poll() is None:
                try:
                    sp.proc.terminate()
                except Exception:
                    pass
        time.sleep(1.0)
        for mac, sp in streams.items():
            if sp.proc and sp.proc.poll() is None:
                try:
                    sp.proc.kill()
                except Exception:
                    pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mac",
        action="append",
        required=True,
        help="MAC address of a Muse headset (repeat flag per headset)",
    )
    ap.add_argument(
        "--python-cmd",
        default=sys.executable,
        help="Python executable that has muselsl installed (default: current interpreter)",
    )
    ap.add_argument(
        "--restart-delay",
        type=float,
        default=3.0,
        help="Seconds to wait before restarting a crashed muselsl process",
    )
    args = ap.parse_args()

    streams: Dict[str, StreamProcess] = {
        mac: StreamProcess(mac=mac, proc=None, restarts=0) for mac in args.mac
    }

    # Clean exit on SIGTERM/SIGINT
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    # Initial launch
    for mac, sp in streams.items():
        sp.proc = launch_stream(mac, args.python_cmd)

    monitor_streams(streams, args.python_cmd, args.restart_delay)


if __name__ == "__main__":
    main()
