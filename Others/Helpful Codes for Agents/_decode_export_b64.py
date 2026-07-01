import base64
import json
from pathlib import Path

nb = json.loads(Path(r"c:\Projects\Graduate Project\Fusion\Pose Extraction and Testing.ipynb").read_text(encoding="utf-8"))
for cell in nb["cells"]:
    src = "".join(cell.get("source", []))
    if "_STGNF_SCRIPT_B64" in src:
        start = src.index('("') + 2
        end = src.index('")', start)
        b64 = src[start:end]
        out = Path(r"c:\Projects\Graduate Project\STG-NF\stgnf_export_scores.py")
        out.write_bytes(base64.b64decode(b64))
        print("Wrote", out, "bytes", out.stat().st_size)
        break
else:
    raise SystemExit("b64 cell not found")
