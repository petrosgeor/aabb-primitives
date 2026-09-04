"""Keep the documented quickstart and shape examples executable."""

import re
from pathlib import Path


def test_readme_python_examples() -> None:
    """Execute documentation examples in their displayed order."""
    readme = Path(__file__).resolve().parents[1] / "README.md"
    namespace = {}
    for block in re.findall(r"```python\n(.*?)```", readme.read_text(encoding="utf-8-sig"), re.S):
        exec(compile(block, str(readme), "exec"), namespace)
