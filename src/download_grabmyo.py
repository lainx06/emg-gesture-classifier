"""Download the part of GRABMyo 1.1.0 this study uses.

Fetches only gestures 11, 16 and 17 (wrist extension, hand close, rest) for all
3 sessions, 43 participants and 7 trials: 1,806 recordings (.hea + .dat),
roughly a fifth of the 9.1 GB dataset. Files go to data/grabmyo/ in the layout
preprocess.load_trial expects, and each one is checked against PhysioNet's
published SHA256SUMS.txt.

Re-running is safe: files already present with the correct checksum are
skipped, so an interrupted download resumes where it stopped.

    python src/download_grabmyo.py
"""
import argparse
import hashlib
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "https://physionet.org/files/grabmyo/1.1.0/"
SESSIONS = (1, 2, 3)
PARTICIPANTS = range(1, 44)
GESTURES = (11, 16, 17)
TRIALS = range(1, 8)

HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(os.path.dirname(HERE), "data", "grabmyo")


def wanted_files():
    for s in SESSIONS:
        for p in PARTICIPANTS:
            for g in GESTURES:
                for t in TRIALS:
                    stem = (f"Session{s}/session{s}_participant{p}/"
                            f"session{s}_participant{p}_gesture{g}_trial{t}")
                    yield stem + ".hea"
                    yield stem + ".dat"


def fetch(url, retries=4):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def load_checksums(base_url):
    text = fetch(base_url + "SHA256SUMS.txt").decode("utf-8")
    sums = {}
    for line in text.splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            digest, name = parts
            sums[name.lstrip("*").lstrip("./")] = digest.lower()
    return sums


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def get_one(rel, sums, base_url, dest):
    path = os.path.join(dest, *rel.split("/"))
    expected = sums.get(rel)
    if expected is None:
        return rel, "no checksum listed"
    if os.path.exists(path) and sha256(path) == expected:
        return rel, "present"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = fetch(base_url + rel)
    if hashlib.sha256(data).hexdigest() != expected:
        return rel, "checksum mismatch"
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)
    return rel, "downloaded"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--base-url", default=BASE_URL, help=argparse.SUPPRESS)
    ap.add_argument("--dest", default=DEST, help=argparse.SUPPRESS)
    args = ap.parse_args()

    files = list(wanted_files())
    print(f"Fetching checksums from {args.base_url}SHA256SUMS.txt ...")
    sums = load_checksums(args.base_url)
    print(f"{len(files)} files to check into {args.dest}")

    counts, problems = {}, []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(get_one, rel, sums, args.base_url, args.dest): rel
                   for rel in files}
        for i, fut in enumerate(as_completed(futures), 1):
            rel = futures[fut]
            try:
                _, status = fut.result()
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                status = f"error: {e}"
            counts[status.split(":")[0]] = counts.get(status.split(":")[0], 0) + 1
            if status not in ("present", "downloaded"):
                problems.append(f"{rel}: {status}")
            if i % 200 == 0 or i == len(files):
                print(f"  {i}/{len(files)}  " +
                      ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))

    if problems:
        print(f"\nINCOMPLETE: {len(problems)} file(s) failed. First few:")
        for msg in problems[:10]:
            print(f"  {msg}")
        print("Run the script again to retry; finished files are kept.")
        sys.exit(1)
    print("\nDOWNLOAD COMPLETE: all files present and verified.")


if __name__ == "__main__":
    main()
