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
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("workers must be between 1 and 16")
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

    def download(item):
        relative = pathlib.PurePosixPath(item["name"]).relative_to("checkpoints")
        target = args.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        size = int(item["size"])
        if target.exists():
            if target.stat().st_size == size and crc32c(target) == item["crc32c"]:
                print(f"VERIFIED existing {relative}", flush=True)
                return
            raise RuntimeError(f"Existing file does not match official checksum: {target}")
        part = target.with_name(target.name + ".part")
        url = "https://storage.googleapis.com/openpi-assets/" + urllib.parse.quote(item["name"], safe="/")
        url += "?generation=" + item["generation"]
        if not part.exists() or part.stat().st_size != size:
            subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error",
                            "--retry", "5", "--retry-delay", "3", "--connect-timeout", "30",
                            "--speed-limit", "1024", "--speed-time", "120",
                            "--continue-at", "-", "--output", str(part), url], check=True)
        if part.stat().st_size != size or crc32c(part) != item["crc32c"]:
            raise RuntimeError(f"Size/checksum mismatch: {part}; retained for inspection")
        part.replace(target)
        print(f"VERIFIED {relative} ({size} bytes)", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(download, items))
    report = {"verified_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "models": args.models, "files": len(items), "bytes": total,
              "verification": "Every file matched official GCS size and CRC32C",
              "model_loaded": False}
    (args.root / "download-verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
