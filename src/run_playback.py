"""Stream ADC codes to playback_verify.ino and compare level C against level B.

Requires pyserial:  pip install pyserial

Usage
    python src/run_playback.py --port COM3 --list
    python src/run_playback.py --port COM3 --trial s1g16t6 --mode filter
    python src/run_playback.py --port COM3 --all

Levels
    A  pipeline.py         scipy / sklearn, float64
    B  embedded_reference  manual float32, AVR operation order
    C  playback_verify.ino physical ATmega328P

This script measures B->C, which isolates C++ transcription error. A->B was
already measured by export_playback.py.

Predetermined criteria, fixed before any hardware result was seen:

    ADC codes                 exact, verified by CRC-16 over received bytes
    ZC counts                 exact
    window count              exact
    predictions               exact
    filter output             < 0.5 ADC LSB
    standardized features     < 1e-3 training SD
    LDA scores                rtol 1e-4, atol 1e-5, and 2E < min_margin
                              as a sufficient worst-case argmax guarantee
"""

import argparse
import os
import struct
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PB = os.path.join(ROOT, "playback")

MAGIC = b"\xA5\x5A"
REC_READY, REC_ACK, REC_FILT, REC_WIN, REC_DONE, REC_INFO, REC_ERR = range(1, 8)

MODE_SUMMARY, MODE_FILTER = 0, 1

# The firmware sizes its receive buffer from CHUNK_SAMPLES. Sending a different
# amount per chunk would desynchronize the handshake, so this is enforced.
FIRMWARE_CHUNK_SAMPLES = 8


def crc16_ccitt(data, crc=0xFFFF):
    """CRC-16/CCITT-FALSE, matching crc16_update() in playback_verify.ino."""
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc

# tolerances, fixed in advance
TOL_FILT_LSB = 0.5
TOL_STD_SD = 1e-3
TOL_SCORE_RTOL = 1e-4
TOL_SCORE_ATOL = 1e-5


# ------------------------------------------------------------------ protocol

def read_record(ser):
    """Block until one complete record arrives. Returns (type, payload)."""
    while True:
        b = ser.read(1)
        if not b:
            raise TimeoutError("no data from device")
        if b != MAGIC[0:1]:
            continue
        b2 = ser.read(1)
        if b2 != MAGIC[1:2]:
            continue
        head = ser.read(3)
        if len(head) != 3:
            raise TimeoutError("truncated record header")
        rtype = head[0]
        length = head[1] | (head[2] << 8)
        payload = ser.read(length) if length else b""
        if len(payload) != length:
            raise TimeoutError("truncated record payload")
        return rtype, payload


def parse_window(payload):
    n_feat, n_class = 8, 3
    exp = (n_feat * 2 + n_class) * 4 + 1
    if len(payload) != exp:
        raise ValueError(f"window record is {len(payload)} B, expected {exp}")
    o = 0
    raw = np.frombuffer(payload, "<f4", n_feat, o); o += n_feat * 4
    scaled = np.frombuffer(payload, "<f4", n_feat, o); o += n_feat * 4
    score = np.frombuffer(payload, "<f4", n_class, o); o += n_class * 4
    pred = payload[o]
    return raw, scaled, score, pred


# ------------------------------------------------------------------ transfer

def run_trial(ser, codes, mode, chunk=8, verbose=False):
    n = codes.shape[0]
    ser.reset_input_buffer()
    ser.write(f"BEGIN {n} {mode}\n".encode())
    ser.flush()

    rtype, payload = read_record(ser)
    while rtype == REC_INFO:
        if verbose:
            print("   device:", payload.decode(errors="replace"))
        rtype, payload = read_record(ser)
    if rtype != REC_READY:
        raise RuntimeError(f"expected READY, got type {rtype}: {payload!r}")

    flat = codes.astype("<i2").reshape(-1)
    filt, wins = [], []
    sent = 0

    while sent < n:
        take = min(chunk, n - sent)
        ser.write(flat[sent * 2:(sent + take) * 2].tobytes())
        ser.flush()

        while True:
            rtype, payload = read_record(ser)
            if rtype == REC_ACK:
                break
            elif rtype == REC_FILT:
                filt.append(np.frombuffer(payload, "<f4"))
            elif rtype == REC_WIN:
                wins.append(parse_window(payload))
            elif rtype == REC_INFO:
                if verbose:
                    print("   device:", payload.decode(errors="replace"))
            elif rtype == REC_ERR:
                raise RuntimeError("device error: " +
                                   payload.decode(errors="replace"))
            else:
                raise RuntimeError(f"unexpected record type {rtype}")
        sent += take

    while True:
        rtype, payload = read_record(ser)
        if rtype == REC_DONE:
            break
        elif rtype == REC_WIN:
            wins.append(parse_window(payload))
        elif rtype == REC_FILT:
            filt.append(np.frombuffer(payload, "<f4"))
        elif rtype == REC_ERR:
            raise RuntimeError("device error: " +
                               payload.decode(errors="replace"))

    out = {}
    if rtype == REC_DONE and len(payload) == 2:
        out["crc"] = payload[0] | (payload[1] << 8)
    if filt:
        out["filtered"] = np.concatenate(filt).reshape(-1, 2)
    if wins:
        out["features"] = np.array([w[0] for w in wins], dtype=np.float32)
        out["scaled"] = np.array([w[1] for w in wins], dtype=np.float32)
        out["scores"] = np.array([w[2] for w in wins], dtype=np.float32)
        out["predictions"] = np.array([w[3] for w in wins], dtype=np.int8)
    return out


# ------------------------------------------------------------------ compare

def compare(key, C, ref, lsb, expect_crc=None):
    """Compare level C against level B. Returns (all_pass, lines)."""
    lines, ok = [], True

    if expect_crc is not None:
        got = C.get("crc")
        p = (got == expect_crc)
        ok &= p
        shown = "missing" if got is None else f"0x{got:04X}"
        lines.append(f"  ADC code CRC       {shown} vs 0x{expect_crc:04X}"
                     f"      (exact)   {'PASS' if p else 'FAIL'}")

    if "filtered" in C:
        B = ref[f"{key}_B_filtered"]
        n = min(len(B), len(C["filtered"]))
        if len(B) != len(C["filtered"]):
            lines.append(f"  filter sample count B={len(B)} C={len(C['filtered'])}")
            ok = False
        e = np.abs(B[:n] - C["filtered"][:n]).max() / lsb
        p = e < TOL_FILT_LSB
        ok &= p
        lines.append(f"  filter output      {e:9.5f} LSB   "
                     f"(< {TOL_FILT_LSB})  {'PASS' if p else 'FAIL'}")

    if "features" in C:
        Bf = ref[f"{key}_B_features"]
        Bs = ref[f"{key}_B_scaled"]
        Bsc = ref[f"{key}_B_scores"]
        Bp = ref[f"{key}_B_predictions"]

        if len(Bf) != len(C["features"]):
            lines.append(f"  window count       B={len(Bf)} C={len(C['features'])}  FAIL")
            return False, lines
        lines.append(f"  window count       {len(Bf)}   PASS")

        zc_cols = [3, 7]
        zc_bad = int(np.sum(np.abs(Bf[:, zc_cols] - C["features"][:, zc_cols]) > 1e-6))
        ok &= (zc_bad == 0)
        lines.append(f"  ZC mismatches      {zc_bad:9d}         "
                     f"(exact)   {'PASS' if zc_bad == 0 else 'FAIL'}")

        e = np.abs(Bs - C["scaled"]).max()
        p = e < TOL_STD_SD
        ok &= p
        lines.append(f"  standardized feat  {e:9.3e} SD    "
                     f"(< {TOL_STD_SD})  {'PASS' if p else 'FAIL'}")

        e = np.abs(Bsc - C["scores"]).max()
        p = np.allclose(Bsc, C["scores"], rtol=TOL_SCORE_RTOL, atol=TOL_SCORE_ATOL)
        ok &= p
        s = np.sort(Bsc, axis=1)
        margin = float((s[:, -1] - s[:, -2]).min())
        guard = 2 * e < margin
        lines.append(f"  LDA scores         {e:9.3e}       "
                     f"(rtol {TOL_SCORE_RTOL})  {'PASS' if p else 'FAIL'}")
        ok &= guard
        lines.append(f"  argmax guarantee   2E={2*e:9.3e} vs M={margin:.4f}   "
                     f"{'PASS' if guard else 'FAIL'}")

        agree = int(np.sum(Bp == C["predictions"]))
        p = agree == len(Bp)
        ok &= p
        lines.append(f"  predictions        {agree}/{len(Bp)}          "
                     f"(exact)   {'PASS' if p else 'FAIL'}")

    return ok, lines


# ------------------------------------------------------------------ main

def bin_files():
    out = {}
    for f in sorted(os.listdir(PB)):
        if f.startswith("codes_") and f.endswith(".bin"):
            path = os.path.join(PB, f)
            size = os.path.getsize(path)
            if size % 4 != 0:
                print(f"  {f}: {size} B is not divisible by 4 -- skipping")
                continue
            n = size // 4
            if size != 16000:
                print(f"  note: {f} is {size} B ({n} samples), "
                      f"not the usual 16000 B / 4000 samples")
            out[f[len("codes_"):-len(".bin")]] = (path, n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", help="serial port, e.g. COM3 or /dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--trial")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--mode", choices=["summary", "filter"], default="summary")
    ap.add_argument("--chunk", type=int, default=FIRMWARE_CHUNK_SAMPLES,
                    help="must match CHUNK_SAMPLES in the firmware")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    if args.chunk != FIRMWARE_CHUNK_SAMPLES:
        print(f"--chunk must be {FIRMWARE_CHUNK_SAMPLES}: the firmware sizes its "
              f"receive buffer from CHUNK_SAMPLES and both sides must agree. "
              f"Change CHUNK_SAMPLES in playback_verify.ino and "
              f"FIRMWARE_CHUNK_SAMPLES here together.")
        sys.exit(1)

    files = bin_files()
    if args.list or not (args.trial or args.all):
        print("available trials:")
        for k, (_, n) in files.items():
            print(f"  {k:<12} {n} samples")
        if args.list:
            return

    if not args.port:
        print("--port is required")
        sys.exit(1)

    try:
        import serial
    except ImportError:
        print("pyserial not installed:  pip install pyserial")
        sys.exit(1)

    ref = np.load(os.path.join(PB, "playback_vectors.npz"))
    lsb = None
    for line in open(os.path.join(PB, "playback_params.h")):
        if line.startswith("const float ADC_LSB"):
            lsb = float(line.split("=")[1].strip().rstrip("f;"))
    if lsb is None:
        print("could not read ADC_LSB from playback_params.h")
        sys.exit(1)

    keys = list(files) if args.all else [args.trial]
    mode = MODE_FILTER if args.mode == "filter" else MODE_SUMMARY

    ser = serial.Serial(args.port, args.baud, timeout=10)
    import time
    time.sleep(2.0)          # Arduino resets when the port opens

    all_ok = True
    for key in keys:
        if key not in files:
            print(f"unknown trial {key}")
            all_ok = False
            continue
        path, n = files[key]
        codes = np.fromfile(path, dtype="<i2").reshape(-1, 2)
        assert codes.shape[0] == n
        with open(path, "rb") as fh:
            expect_crc = crc16_ccitt(fh.read())

        print(f"\n{key}  ({n} samples, mode={args.mode})")
        C = run_trial(ser, codes, mode, chunk=args.chunk, verbose=args.verbose)
        ok, lines = compare(key, C, ref, lsb, expect_crc=expect_crc)
        for ln in lines:
            print(ln)
        all_ok &= ok

    ser.close()
    print()
    print("B->C VERIFIED" if all_ok else "B->C FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()