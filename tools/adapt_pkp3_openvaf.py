from pathlib import Path
import re
import sys


def adapt(src: Path, dst: Path) -> None:
    text = src.read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"\(\*\s*instance_parameter_list\s*=\s*\{[^}]+\}\s*\*\)\s*\n", "", text)
    replacements = {
        "real mx = 0.916 * MEL;": "real mx = 8.34356e-31;",
        "real mxprime = 0.190 * MEL;": "real mxprime = 1.7309e-31;",
        "real md = 0.190 * MEL;": "real md = 1.7309e-31;",
        "real mdprime = 0.417 * MEL;": "real mdprime = 3.79947e-31;",
        "real epsratio = EPSRSUB / EPSROX;": "real epsratio = 3.051282051282051;",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("用法：adapt_pkp3_openvaf.py 原文件 目标文件")
    adapt(Path(sys.argv[1]), Path(sys.argv[2]))
