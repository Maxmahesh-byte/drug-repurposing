from __future__ import annotations

import importlib
import platform
import sys

PACKAGES = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("pyarrow", "pyarrow"),
    ("scipy", "scipy"),
    ("sklearn", "scikit-learn"),
    ("torch", "torch"),
    ("torch_geometric", "torch-geometric"),
    ("rdkit", "rdkit"),
    ("fastapi", "fastapi"),
    ("streamlit", "streamlit"),
    ("mlflow", "mlflow"),
    ("dvc", "dvc"),
]

def main() -> None:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"Machine: {platform.machine()}")
    print()

    failures = []
    for module_name, display_name in PACKAGES:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "unknown")
            print(f"[OK] {display_name}: {version}")
        except Exception as exc:
            failures.append((display_name, str(exc)))
            print(f"[FAIL] {display_name}: {exc}")

    if failures:
        raise SystemExit(f"{len(failures)} package import(s) failed.")
    print("Environment check passed.")

if __name__ == "__main__":
    main()
