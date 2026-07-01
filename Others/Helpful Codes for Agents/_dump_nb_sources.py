import json
from pathlib import Path

nb_path = Path(r"c:\Projects\Graduate Project\Fusion\Pose Extraction and Testing.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))
out = Path(r"c:\Projects\Graduate Project\Fusion\Others\Helpful Codes for Agents\_pose_nb_sources.txt")
lines = []
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    lines.append(f"\n{'='*80}\nCELL {i} ({cell['cell_type']})\n{'='*80}\n")
    lines.append(src)
out.write_text("".join(lines), encoding="utf-8")
print(f"Wrote {out} ({len(nb['cells'])} cells)")
