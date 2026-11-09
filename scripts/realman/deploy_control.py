"""Deploy a verified control/calibration bundle; never start services or issue motion."""

import argparse
from pathlib import Path
import shlex
import subprocess

from release import ROOT, build_archive, verify_archive


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--host", default="realman@139.224.22.149")
    parser.add_argument("--port", type=int, default=7582)
    parser.add_argument("--root", default="/home/realman/aspire-control")
    parser.add_argument("--archive", type=Path, help="Deploy this previously built/reviewed bundle")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or args.host.startswith("-"):
        raise ValueError("invalid SSH endpoint")
    data = args.archive.read_bytes() if args.archive else build_archive()[0]
    verify_archive(data)
    installer = (ROOT / "scripts/realman/release.py").read_text()
    code = (
        "import sys,json\nnamespace={'__name__':'release_install','__file__':'/opt/aspire/scripts/realman/release.py'}\n"
        "exec("
        + repr(installer)
        + ",namespace)\n"
        + "print(json.dumps(namespace['install_archive'](sys.stdin.buffer.read(),"
        + repr(args.root)
        + ")))\n"
    )
    result = subprocess.run(
        [
            "ssh",
            "-T",
            "-p",
            str(args.port),
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            args.host,
            "python3 -c " + shlex.quote(code),
        ],
        input=data,
        capture_output=True,
        timeout=90,
        check=True,
    )
    print(result.stdout.decode())


if __name__ == "__main__":
    main()
