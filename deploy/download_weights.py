"""Download public, generation-pinned openpi checkpoints without importing model code."""

import argparse
import base64
import concurrent.futures
import json
import pathlib
import shutil
import subprocess
import time
import urllib.parse
import urllib.request

import google_crc32c


def crc32c(path):
    digest = google_crc32c.Checksum()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return base64.b64encode(digest.digest()).decode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, required=True)
    parser.add_argument("--models", nargs="+", choices=("pi05_base", "pi05_libero"),
                        default=["pi05_base", "pi05_libero"])
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--chunk-mib", type=int, default=32)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("workers must be between 1 and 16")
    if not 1 <= args.chunk_mib <= 256:
        parser.error("chunk-mib must be between 1 and 256")
    args.root.mkdir(parents=True, exist_ok=True)
    items = []
    for model in args.models:
        query = {"prefix": f"checkpoints/{model}/", "maxResults": 1000}
        while True:
            url = "https://storage.googleapis.com/storage/v1/b/openpi-assets/o?" + urllib.parse.urlencode(query)
            with urllib.request.urlopen(url, timeout=60) as response:
                page = json.load(response)
            items.extend(page.get("items", []))
            if "nextPageToken" not in page:
                break
            query["pageToken"] = page["nextPageToken"]
    if not items:
        raise RuntimeError("Empty checkpoint listing")
    for item in items:
        relative = pathlib.PurePosixPath(item["name"]).relative_to("checkpoints")
        if ".." in relative.parts or not item.get("crc32c"):
            raise RuntimeError("Unsafe path or missing checksum")
    total = sum(int(item["size"]) for item in items)
    if shutil.disk_usage(args.root).free < total + 2 * 1024**3:
        raise RuntimeError("Insufficient free space for checkpoints plus 2 GiB reserve")
    (args.root / "download-manifest.json").write_text(json.dumps(items, indent=2) + "\n")
    print(f"Downloading/verifying {len(items)} files, {total / 1e9:.3f} GB", flush=True)

    plans = []
    jobs = []
    for item in items:
        relative = pathlib.PurePosixPath(item["name"]).relative_to("checkpoints")
        target = args.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        size = int(item["size"])
        if target.exists():
            if target.stat().st_size == size and crc32c(target) == item["crc32c"]:
                print(f"VERIFIED existing {relative}", flush=True)
                continue
            raise RuntimeError(f"Existing file does not match official checksum: {target}")
        part = target.with_name(target.name + ".part")
        url = "https://openpi-assets.storage.googleapis.com/"
        url += urllib.parse.quote(item["name"], safe="/")
        url += "?generation=" + item["generation"]
        prefix = part.stat().st_size if part.exists() else 0
        if prefix > size:
            raise RuntimeError(f"Partial file larger than expected: {part}")
        chunks = part.with_name(part.name + ".chunks")
        chunks.mkdir(exist_ok=True)
        pieces = []
        for start in range(prefix, size, args.chunk_mib * 1024**2):
            end = min(start + args.chunk_mib * 1024**2, size) - 1
            piece = chunks / f"{item['generation']}-{start}-{end}"
            pieces.append(piece)
            jobs.append((url, start, end, piece))
        plans.append((item, target, part, pieces, chunks))

    def download_range(job):
        url, start, end, piece = job
        expected = end - start + 1
        if piece.exists() and piece.stat().st_size == expected:
            return
        temporary = piece.with_name(piece.name + ".tmp")
        offset = temporary.stat().st_size if temporary.exists() else 0
        if offset > expected:
            raise RuntimeError(f"Partial range larger than expected: {temporary}")
        while offset < expected:
            length = min(8 * 1024**2, expected - offset)
            segment = temporary.with_name(temporary.name + ".segment")
            subprocess.run(["curl", "--http1.1", "--fail", "--location", "--silent", "--show-error",
                            "--retry", "5", "--retry-all-errors", "--retry-delay", "3",
                            "--retry-max-time", "180", "--connect-timeout", "30",
                            "--max-time", "90", "--max-filesize", str(length),
                            "--range", f"{start + offset}-{start + offset + length - 1}",
                            "--output", str(segment), url], check=True)
            if segment.stat().st_size != length:
                raise RuntimeError(f"Segment size mismatch: {segment}")
            with temporary.open("ab") as output, segment.open("rb") as source:
                shutil.copyfileobj(source, output)
            segment.unlink()
            offset += length
        if temporary.stat().st_size != expected:
            raise RuntimeError(f"Range size mismatch: {temporary}")
        temporary.replace(piece)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for count, _ in enumerate(pool.map(download_range, jobs), 1):
            if count % 16 == 0 or count == len(jobs):
                print(f"Downloaded ranges: {count}/{len(jobs)}", flush=True)
    for item, target, part, pieces, chunks in plans:
        with part.open("ab") as output:
            for piece in pieces:
                with piece.open("rb") as source:
                    shutil.copyfileobj(source, output, 8 * 1024**2)
                output.flush()
                piece.unlink()
        size = int(item["size"])
        if part.stat().st_size != size or crc32c(part) != item["crc32c"]:
            raise RuntimeError(f"Size/checksum mismatch: {part}; retained for inspection")
        part.replace(target)
        if not any(chunks.iterdir()):
            chunks.rmdir()
        print(f"VERIFIED {target.relative_to(args.root)} ({size} bytes)", flush=True)
    report = {"verified_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "models": args.models, "files": len(items), "bytes": total,
              "verification": "Every file matched official GCS size and CRC32C",
              "model_loaded": False}
    (args.root / "download-verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
