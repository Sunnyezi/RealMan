"""Download public pi05_base and verify the MD5 or CRC32C provided by GCS."""

import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.parse
import urllib.request

PREFIX = "checkpoints/pi05_base/params/"


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "work/pi05_setup/tools"))
    import google_crc32c

    cache = Path(os.environ.get("OPENPI_DATA_HOME", "~/.cache/openpi")).expanduser()
    target = cache / "openpi-assets" / PREFIX.rstrip("/")
    if target.exists():
        print(f"Existing cache: {target}", flush=True)
        return
    partial = target.with_name("params.download")
    partial.mkdir(parents=True, exist_ok=True)
    items = []
    query = {"prefix": PREFIX}
    while True:
        url = "https://storage.googleapis.com/storage/v1/b/openpi-assets/o?" + urllib.parse.urlencode(query)
        with urllib.request.urlopen(url, timeout=60) as response:
            listing = json.load(response)
        items.extend(listing["items"])
        if "nextPageToken" not in listing:
            break
        query["pageToken"] = listing["nextPageToken"]

    def fetch(item: dict) -> None:
        relative = Path(item["name"].removeprefix(PREFIX))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid object path")
        dest = partial / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists() or dest.stat().st_size != int(item["size"]):
            url = "https://storage.googleapis.com/openpi-assets/" + urllib.parse.quote(item["name"], safe="/")
            subprocess.run(["curl", "-fsSL", "--retry", "3", "--connect-timeout", "20",
                            "--max-time", "1800", url, "-o", str(dest)], check=True)
        checksum = hashlib.md5() if "md5Hash" in item else google_crc32c.Checksum()
        with dest.open("rb") as stream:
            while block := stream.read(8 * 1024 * 1024):
                checksum.update(block)
        expected = item.get("md5Hash", item.get("crc32c"))
        if expected is None or base64.b64encode(checksum.digest()).decode() != expected:
            raise RuntimeError(f"Checksum mismatch: {relative}")
        print(f"verified {relative} ({dest.stat().st_size} bytes)", flush=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, items))
    (partial / "download_manifest.json").write_text(json.dumps(items, indent=2) + "\n")
    if target.exists():
        raise FileExistsError(f"Refusing to replace existing cache: {target}")
    partial.rename(target)
    print(f"Ready: {target}", flush=True)


if __name__ == "__main__":
    main()
