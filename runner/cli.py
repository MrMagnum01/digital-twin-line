"""CLI entry point for runner.pipeline.run().

Refuses the locked seed table unless BOTH the caller passes
`--table locked` AND the explicit `--i-have-clearance-to-run-locked-seeds`
flag - and even then, lock_guard.require_lock() and
generator.generate()'s own LockedSeedsError refusal still apply
underneath. Nothing in this milestone's tests, dashboard or monitoring
code passes that flag; it exists only so a future, separately authorised
locked run does not require editing this file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from runner.pipeline import run


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--table", choices=["dev", "locked"], default="dev")
    p.add_argument("--i-have-clearance-to-run-locked-seeds", action="store_true",
                   dest="allow_locked",
                   help="Required in addition to --table locked. Do not pass this without "
                        "a recorded freeze-clearance decision (see "
                        "vault:10-projects/freelance/launch/twin/astra-lock-3-signoff.md).")
    p.add_argument("--out", default=None, help="write the JSON report here (default: stdout)")
    args = p.parse_args(argv)

    if args.table == "locked" and not args.allow_locked:
        p.error("--table locked requires --i-have-clearance-to-run-locked-seeds")

    report = run(args.table, allow_locked=args.allow_locked)
    text = json.dumps(report, indent=1, sort_keys=True, default=str)
    if args.out:
        Path(args.out).write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
