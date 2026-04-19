"""Package a TensorBoard run directory as a self-contained tarball and
optionally upload it to a Slack channel / DM so the recipient can view
the run on their own machine.

Typical use
-----------

Just package (no upload):

    python scripts/ship_tb_run.py --run-dir runs/shapes
    # → /tmp/tb_run_shapes_<timestamp>.tar.gz

Package + ship to Slack:

    export SLACK_BOT_TOKEN=xoxb-...       # bot token with files:write + chat:write
    python scripts/ship_tb_run.py \\
        --run-dir runs/shapes \\
        --slack-channel C0123456789 \\
        --message "Shapes run, 50 epochs, purity 0.93"

Recipient
---------

The archive contains an ``ABOUT.md`` with extraction + viewing
instructions. The short version:

    tar -xzf tb_run_shapes_*.tar.gz
    pip install tensorboard
    tensorboard --logdir tb_run_shapes_*/

"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import mimetypes
import os
import secrets
import socket
import subprocess
import sys
import tarfile
import urllib.parse
import urllib.request
from pathlib import Path


SLACK_API = "https://slack.com/api"
# Slack's default per-workspace max file upload is 1 GB (free / pro)
# and 5 GB on enterprise. We warn above the conservative limit.
SLACK_SOFT_LIMIT_BYTES = 1 * 1024 * 1024 * 1024  # 1 GiB


# ---------------------------------------------------------------------------
# Metadata captured into the archive
# ---------------------------------------------------------------------------

def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _about_md(run_dir: Path, message: str) -> str:
    ts = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    size_h = _human_bytes(_dir_size(run_dir))
    note = f"\n\n> {message}\n" if message else ""
    return f"""\
# TensorBoard run: `{run_dir.name}`{note}

Packaged on **{ts}** from `{socket.gethostname()}` at git SHA `{_git_sha()}`.
Uncompressed run size: {size_h}.

## View it

```bash
# 1. extract
tar -xzf {run_dir.name}_<timestamp>.tar.gz

# 2. install tensorboard if you don't have it
pip install tensorboard

# 3. serve it (the --logdir points at the extracted run directory)
tensorboard --logdir {run_dir.name}/

# open the URL that prints (usually http://localhost:6006)
```

## What's inside

- `events.out.tfevents.*` — scalars, histograms, images.
- `config.yaml` (if present) — the exact runcard used for training.
- `architecture/` (if present) — Mermaid flowchart + torchinfo layer
  table emitted at run start.
- Any projector subfolders (e.g. `00000/`, `00500/`) — embedding data
  if the run produced any. Safe to keep; TB reads them automatically.

## Tabs worth checking

- **SCALARS** — `train/loss/*` and `val/loss/*` are the objective
  components; `train/grad_norm` is the post-clip gradient norm; the
  `train/beta` histogram tells you whether the model is learning to
  condense.
- **IMAGES** — `viz/grid` is a fixed-event panel (TRUTH / PRED / OC)
  refreshed every few hundred steps; watch convergence by scrubbing
  the step slider.
- **PROJECTOR** — `viz/oc_space`; stick to the PCA tab.
- **TEXT** — `0_overview` explains every scalar and panel in detail.
"""


def _human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


# ---------------------------------------------------------------------------
# Packaging
# ---------------------------------------------------------------------------

def build_tarball(run_dir: Path, out_path: Path, message: str) -> int:
    """Create ``out_path`` (.tar.gz). Returns the size of the archive in bytes."""
    if not run_dir.is_dir():
        raise SystemExit(f"run-dir not found or not a directory: {run_dir}")

    about = _about_md(run_dir, message).encode()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tarfile.open(out_path, "w:gz") as tar:
        # 1) about file at the root of the archive
        info = tarfile.TarInfo(name="ABOUT.md")
        info.size = len(about)
        info.mtime = int(dt.datetime.now().timestamp())
        tar.addfile(info, io.BytesIO(about))

        # 2) the run dir, kept under its own top-level name so extraction
        # doesn't splatter files into the recipient's cwd.
        tar.add(run_dir, arcname=run_dir.name, recursive=True)

    return out_path.stat().st_size


# ---------------------------------------------------------------------------
# Slack upload (three-step external-upload flow)
# ---------------------------------------------------------------------------

def _slack_post_form(endpoint: str, token: str, form: dict[str, str]) -> dict:
    req = urllib.request.Request(
        f"{SLACK_API}/{endpoint}",
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
        },
        data=urllib.parse.urlencode(form).encode(),
    )
    with urllib.request.urlopen(req) as r:
        payload = json.loads(r.read().decode())
    if not payload.get("ok"):
        raise RuntimeError(f"Slack {endpoint} failed: {payload}")
    return payload


def _slack_post_json(endpoint: str, token: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"{SLACK_API}/{endpoint}",
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        },
        data=json.dumps(body).encode(),
    )
    with urllib.request.urlopen(req) as r:
        payload = json.loads(r.read().decode())
    if not payload.get("ok"):
        raise RuntimeError(f"Slack {endpoint} failed: {payload}")
    return payload


def _upload_bytes(url: str, data: bytes, filename: str) -> None:
    # Slack's external upload URL expects a multipart/form-data POST.
    boundary = "----tb-ship-" + secrets.token_hex(8)
    mime = mimetypes.guess_type(filename)[0] or "application/gzip"
    parts = [
        f"--{boundary}\r\n".encode(),
        (f'Content-Disposition: form-data; name="file"; '
         f'filename="{filename}"\r\n').encode(),
        f"Content-Type: {mime}\r\n\r\n".encode(),
        data,
        f"\r\n--{boundary}--\r\n".encode(),
    ]
    body = b"".join(parts)
    req = urllib.request.Request(
        url,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        data=body,
    )
    urllib.request.urlopen(req).read()


def slack_upload(
    archive_path: Path,
    token: str,
    channel: str,
    title: str,
    comment: str,
) -> dict:
    """Three-step upload: get URL → PUT file → complete."""
    size = archive_path.stat().st_size
    if size > SLACK_SOFT_LIMIT_BYTES:
        print(
            f"warning: archive is {_human_bytes(size)}; "
            f"Slack's default cap is 1 GiB. Upload may be rejected.",
            file=sys.stderr,
        )

    step1 = _slack_post_form(
        "files.getUploadURLExternal", token,
        {"filename": archive_path.name, "length": str(size)},
    )
    upload_url = step1["upload_url"]
    file_id = step1["file_id"]

    with archive_path.open("rb") as f:
        _upload_bytes(upload_url, f.read(), archive_path.name)

    step3 = _slack_post_json(
        "files.completeUploadExternal", token,
        {
            "files": [{"id": file_id, "title": title}],
            "channel_id": channel,
            "initial_comment": comment,
        },
    )
    return step3


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--run-dir", type=Path, required=True,
                    help="path to the TensorBoard log directory to ship")
    ap.add_argument("--out", type=Path, default=None,
                    help="where to write the tarball (default: /tmp/tb_run_<name>_<ts>.tar.gz)")

    ap.add_argument("--slack-channel", type=str, default=None,
                    help="Slack channel ID (e.g. C012...) or user ID (e.g. U012...). "
                         "If omitted, only packages locally.")
    ap.add_argument("--slack-token", type=str, default=None,
                    help="Slack bot token. Defaults to the $SLACK_BOT_TOKEN env var. "
                         "Needs files:write and chat:write scopes.")
    ap.add_argument("--message", type=str, default="",
                    help="optional note included in the Slack message and in ABOUT.md.")
    ap.add_argument("--title", type=str, default=None,
                    help="file title shown in Slack (defaults to the archive name)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.out or Path(f"/tmp/tb_run_{run_dir.name}_{ts}.tar.gz")
    size = build_tarball(run_dir, out, args.message)
    print(f"packaged: {out}  ({_human_bytes(size)})")

    if args.slack_channel is None:
        return

    token = args.slack_token or os.environ.get("SLACK_BOT_TOKEN")
    if not token:
        raise SystemExit(
            "no Slack token: set $SLACK_BOT_TOKEN or pass --slack-token"
        )

    title = args.title or out.name
    comment = args.message or f"TensorBoard run `{run_dir.name}` — see ABOUT.md for how to view."
    resp = slack_upload(out, token, args.slack_channel, title, comment)
    permalink = (resp.get("files") or [{}])[0].get("permalink", "")
    print(f"uploaded to Slack channel {args.slack_channel}")
    if permalink:
        print(f"  {permalink}")


if __name__ == "__main__":
    main()
