from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(
        description="Dispatch to basic_rag, self_consistency, self_refine, multi_agent_rag, or reflexion_rag.",
    )
    p.add_argument(
        "--mode",
        choices=["basic", "self_consistency", "self_refine", "multi_agent", "reflexion"],
        required=True,
        help="Which pipeline to run; remaining argv is forwarded unchanged.",
    )
    args, rest = p.parse_known_args(argv)

    if args.mode == "basic":
        from basic_rag import main as run

        return run(rest)
    if args.mode == "self_consistency":
        from self_consistency import main as run

        return run(rest)
    if args.mode == "self_refine":
        from self_refine import main as run

        return run(rest)
    if args.mode == "multi_agent":
        from multi_agent_rag import main as run

        return run(rest)
    if args.mode == "reflexion":
        from reflexion_rag import main as run

        return run(rest)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
