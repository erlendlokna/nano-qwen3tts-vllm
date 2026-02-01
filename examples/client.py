"""
Client for Qwen3-TTS server. Calls POST /v1/audio/speech and dumps the PCM stream to a WAV file.
Supports parallel requests to maximize GPU utilization.
"""

import argparse
import json
import sys
import time
import urllib.request
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Server returns 16-bit PCM mono at 24 kHz
SAMPLE_RATE = 24000
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit = 2 bytes


def _request_one(args, url: str, payload: dict, idx: int) -> tuple[int, bytes | Exception]:
    """Send one request; returns (idx, pcm_bytes) or (idx, exception)."""
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            pcm = resp.read()
        return (idx, pcm)
    except urllib.error.HTTPError as e:
        err_msg = f"{e.code} {e.reason}: {e.read().decode('utf-8', errors='replace')}"
        return (idx, RuntimeError(err_msg))
    except urllib.error.URLError as e:
        return (idx, RuntimeError(str(e.reason)))


def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS client: generate speech and save to WAV")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Server base URL")
    parser.add_argument("--text", default="Hi there, this is a test.", help="Text to synthesize")
    parser.add_argument("--language", default="English", help="Language of the text")
    parser.add_argument("--speaker", default="Vivian", help="Speaker name")
    parser.add_argument("-o", "--output", default="output.wav", help="Output WAV file path")
    parser.add_argument(
        "-j", "--parallel",
        type=int,
        default=1,
        metavar="N",
        help="Number of parallel requests (default: 1). Use >1 to maximize GPU utilization.",
    )
    parser.add_argument(
        "-n", "--requests",
        type=int,
        default=1,
        metavar="N",
        help="Total number of requests to send (default: 1). With -j, requests are batched.",
    )
    args = parser.parse_args()

    url = f"{args.base_url.rstrip('/')}/v1/audio/speech"
    payload = {
        "text": args.text,
        "language": args.language,
        "speaker": args.speaker,
    }

    n = args.requests
    njobs = min(args.parallel, n)
    out_path = Path(args.output)

    if n == 1 and njobs == 1:
        # Original single-request path
        _, result = _request_one(args, url, payload, 0)
        if isinstance(result, Exception):
            print(f"Request failed: {result}", file=sys.stderr)
            sys.exit(1)
        pcm = result
        with wave.open(args.output, "wb") as wav:
            wav.setnchannels(CHANNELS)
            wav.setsampwidth(SAMPLE_WIDTH)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes(pcm)
        print(f"Saved {len(pcm)} bytes ({len(pcm) // (SAMPLE_RATE * SAMPLE_WIDTH)} s) to {args.output}")
        return

    # Parallel path
    start = time.perf_counter()
    results: list[tuple[int, bytes | Exception]] = []
    with ThreadPoolExecutor(max_workers=njobs) as ex:
        futures = {ex.submit(_request_one, args, url, payload, i): i for i in range(n)}
        for fut in as_completed(futures):
            results.append(fut.result())

    # Sort by index and write outputs
    results.sort(key=lambda r: r[0])
    failed = 0
    for idx, result in results:
        stem = out_path.stem
        suffix = out_path.suffix
        path = out_path.parent / f"{stem}_{idx}{suffix}"
        if isinstance(result, Exception):
            print(f"Request {idx} failed: {result}", file=sys.stderr)
            failed += 1
            continue
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(CHANNELS)
            wav.setsampwidth(SAMPLE_WIDTH)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes(result)
        dur_s = len(result) / (SAMPLE_RATE * SAMPLE_WIDTH)
        print(f"Saved {len(result)} bytes ({dur_s:.2f}s) to {path}")

    elapsed = time.perf_counter() - start
    print(f"Total: {n} requests, {n - failed} ok, {failed} failed, {elapsed:.2f}s ({n / elapsed:.1f} req/s)")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
