"""List selected ninfer artifact objects (ASCII source, path via argv).
Usage: python list_objects.py <artifact.ninfer> <name-substring> [more substrings...]
"""
import sys

ROOT = r"E:\ninfer\src\ninfer-4090w-ternary"
sys.path.insert(0, ROOT)

from tools.artifact import container  # noqa: E402


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: list_objects.py <artifact> <substr> [substr...]")
        return 2
    path = sys.argv[1]
    needles = [s.lower() for s in sys.argv[2:]]
    with container.Artifact.open(path) as art:
        print("identity:", art.identity)
        print("objects:", len(art.objects))
        hits = 0
        for o in art.objects:
            name = o.name
            low = name.lower()
            if not any(n in low for n in needles):
                continue
            hits += 1
            print(f"{name}  kind={o.kind} fmt={getattr(o, 'format', '-')} "
                  f"layout={getattr(o, 'layout', '-')} shape={getattr(o, 'shape', '-')} "
                  f"bytes={o.bytes}")
        print("hits:", hits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
