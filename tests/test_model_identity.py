"""A random-init control must be traceable end to end, and never collide.

The control records its identity as "<repo>#random-init-s<seed>". That "#"
has to survive four hops without being mistaken for the trained model and
without breaking any of them: the store's meta.json, the probe filename slug,
PROBE_RE, and the exporter's per-model drop guard. An earlier draft failed the
third -- PROBE_RE matches [A-Za-z0-9]+, so the slug with a "#" did not match,
the probe files were skipped, and the table would have published with an empty
probe column and no error.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from export_crosslang import PROBE_RE, model_slug  # noqa: E402
from extract_activations import random_init_id  # noqa: E402

TRAINED = ["Qwen/Qwen2.5-Coder-1.5B", "Qwen/Qwen2.5-1.5B",
           "deepseek-ai/deepseek-coder-1.3b-base", "bigcode/starcoder2-7b"]


def run() -> int:
    failures = 0

    def check(name, ok):
        nonlocal failures
        failures += not ok
        print(f"  {'OK  ' if ok else 'FAIL'} {name}")

    # Existing slugs must not move: they name stores and probe files already
    # on disk, and changing them orphans every one of them.
    check("trained slugs unchanged by the non-alphanumeric strip",
          model_slug("Qwen/Qwen2.5-Coder-1.5B") == "qwen25coder15b"
          and model_slug("Qwen/Qwen2.5-1.5B") == "qwen2515b"
          and model_slug("deepseek-ai/deepseek-coder-1.3b-base") == "deepseekcoder13bbase"
          and model_slug("bigcode/starcoder2-7b") == "starcoder27b")

    for mid in TRAINED:
        rid = random_init_id(mid, 0)
        check(f"{mid.split('/')[-1]}: random-init slug differs from trained",
              model_slug(rid) != model_slug(mid))
        fn = f"probe_iterator_python_to_php_{model_slug(rid)}.json"
        m = PROBE_RE.match(fn)
        check(f"{mid.split('/')[-1]}: probe filename parses",
              bool(m) and m.group(4) == model_slug(rid))

    check("seeds produce distinct identities",
          random_init_id("Qwen/Qwen2.5-1.5B", 0) != random_init_id("Qwen/Qwen2.5-1.5B", 7))
    check("slugs distinguish seeds too",
          model_slug(random_init_id("Qwen/Qwen2.5-1.5B", 0))
          != model_slug(random_init_id("Qwen/Qwen2.5-1.5B", 7)))
    check("the tag cannot be read as a HuggingFace repo id",
          "#" in random_init_id("Qwen/Qwen2.5-1.5B", 0))

    # Two different models must never collide after slugging.
    all_ids = TRAINED + [random_init_id(m, s) for m in TRAINED for s in (0, 7)]
    slugs = [model_slug(i) for i in all_ids]
    check(f"no collisions across {len(all_ids)} identities",
          len(set(slugs)) == len(slugs))

    print("\nALL PASS" if not failures else f"\n{failures} FAILURE(S)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(run())
