#!/usr/bin/env python3
"""
OSC recorder with manual markers.

- Listens to OSC on a given host/port and writes all received messages to CSV.
- Lets you type marker labels while it runs; each marker is recorded with a timestamp.
- Designed to run alongside the processing script; does not affect OSC forwarding to TD.

Usage (default port 9005):
    python osc_recorder_with_markers.py --port 9005 --outfile session_log.csv

While running:
- Type a label and press Enter to drop a marker (e.g., "CONTROL_START").
- Press Ctrl+C to stop and finalize the CSV.
"""

import argparse
import csv
import sys
import threading
import time
from queue import Queue

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0", help="Host to bind")
    ap.add_argument("--port", type=int, default=9005, help="Port to listen for OSC")
    ap.add_argument("--outfile", default="session_log.csv", help="CSV output file")
    args = ap.parse_args()

    rows = []

    def osc_handler(addr, *values):
        ts = time.time()
        rows.append((ts, addr, values[0] if values else None))

    disp = Dispatcher()
    disp.set_default_handler(osc_handler)

    server = BlockingOSCUDPServer((args.host, args.port), disp)

    print(f"Recording OSC on {args.host}:{args.port} -> {args.outfile}")
    print("Press Ctrl+C to stop; you'll be asked for a marker label at stop time.")

    try:
        while True:
            # Handle one OSC packet (blocks until received)
            server.handle_request()
    except KeyboardInterrupt:
        stop_ts = time.time()
        try:
            marker_input = input("Enter marker label(s) (comma-separated) to tag stop time, or press Enter to skip: ").strip()
            if marker_input:
                for label in [m.strip() for m in marker_input.split(",") if m.strip()]:
                    rows.append((stop_ts, "MARKER", label))
        except Exception:
            pass
    finally:
        with open(args.outfile, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "address", "value"])
            w.writerows(rows)
        print(f"Wrote {len(rows)} rows to {args.outfile}")


if __name__ == "__main__":
    main()
