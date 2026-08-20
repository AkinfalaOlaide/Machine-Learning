"""Strip a Coursera lab notebook down to its code.

Usage:
    python3 strip_to_code.py <notebook.ipynb> [output]

Writes a code-only .ipynb by default; pass a .py output path to get a plain
script instead. Either way this drops all markdown cells, images and saved
outputs, and any code cell that is empty or only contains comments (the
"### START CODE HERE ###" scaffolding).

The .py conversion additionally comments out IPython magics (%matplotlib and
friends, which are not valid Python) and appends plt.show() to any cell that
builds a figure without one, since a script has no inline display.
"""

import json
import re
import sys
from pathlib import Path

MAGIC = re.compile(r"^\s*[%!]")
MAKES_FIGURE = re.compile(r"\bplt\.(subplots|figure|imshow|plot|scatter|hist)\b")


def is_meaningful(source):
    for line in source.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return True
    return False


def code_cells(nb):
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"]).rstrip("\n")
        if is_meaningful(source):
            yield source


def to_notebook(nb, sources, dst):
    nb["cells"] = [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": s.splitlines(keepends=True),
        }
        for s in sources
    ]
    dst.write_text(json.dumps(nb, indent=1))


def to_script(sources, dst):
    blocks = []
    for source in sources:
        lines = [
            "# " + line if MAGIC.match(line) else line
            for line in source.splitlines()
        ]
        block = "\n".join(lines)
        if MAKES_FIGURE.search(block) and "plt.show()" not in block:
            block += "\nplt.show()"
        blocks.append(block)
    dst.write_text("\n\n\n".join(blocks) + "\n")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)

    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_name(
        src.stem + "_code_only.ipynb"
    )

    nb = json.loads(src.read_text())
    sources = list(code_cells(nb))

    if dst.suffix == ".py":
        to_script(sources, dst)
    else:
        to_notebook(nb, sources, dst)

    print(f"{len(sources)} code cells -> {dst}")


if __name__ == "__main__":
    main()
