"""CallDoctor command line.

    python -m calldoctor simulate --version v1          run every persona against prompt v1
    python -m calldoctor simulate --version v2 --persona skeptic
    python -m calldoctor compare v1 v2                  failure report: before vs after
    python -m calldoctor reanalyze                      re-run detectors on stored calls
    python -m calldoctor serve                          dashboard + webhooks on :8000
    python -m calldoctor reset                          delete all stored calls
"""

from __future__ import annotations

import argparse
import json
import os

from calldoctor import config, store


def cmd_simulate(args) -> None:
    from calldoctor.simulate import run_suite

    personas = [args.persona] if args.persona else None
    summary = run_suite(args.version, personas)
    print(f"\n{args.version}: {summary['failed']}/{summary['calls']} calls failed "
          f"({summary['failure_rate']}%), avg score {summary['avg_score']}")
    for cat, n in summary["categories"].items():
        print(f"  {n:>2} × {cat}")
    if summary["unverified_judge_claims"]:
        print(f"  ({summary['unverified_judge_claims']} judge claims discarded: quote not found in transcript)")


def cmd_compare(args) -> None:
    a, b = store.latest_suite(args.before), store.latest_suite(args.after)
    if not a or not b:
        raise SystemExit("Run `simulate --version` for both versions first.")
    sa, sb = a["summary"], b["summary"]
    print(f"\n{'':<28}{args.before:>10}{args.after:>10}")
    print(f"{'failed calls':<28}{sa['failed']:>7}/{sa['calls']:<2}{sb['failed']:>7}/{sb['calls']:<2}")
    print(f"{'failure rate':<28}{sa['failure_rate']:>9}%{sb['failure_rate']:>9}%")
    print(f"{'avg health score':<28}{sa['avg_score']:>10}{sb['avg_score']:>10}")
    print("\nfailures by category")
    for cat in sorted(set(sa["categories"]) | set(sb["categories"])):
        print(f"  {cat:<26}{sa['categories'].get(cat, 0):>10}{sb['categories'].get(cat, 0):>10}")


def cmd_reanalyze(args) -> None:
    from calldoctor.analyze import analyze
    from calldoctor.models import Call

    for row in store.list_calls(limit=5000):
        rec = store.get_call(row["id"])
        call = Call.from_dict(rec["call"])
        store.save_call(call, analyze(call, use_judge=not args.rules_only), suite_id=rec["suite_id"])
        print("re-analyzed", call.id)


def cmd_serve(args) -> None:
    import uvicorn

    uvicorn.run("calldoctor.app:app", host="0.0.0.0", port=int(os.environ.get("PORT", args.port)),
                reload=args.reload)


def main() -> None:
    p = argparse.ArgumentParser(prog="calldoctor")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("simulate")
    s.add_argument("--version", default=config.survey()["active_version"])
    s.add_argument("--persona", choices=list(config.personas()))
    s.set_defaults(fn=cmd_simulate)

    c = sub.add_parser("compare")
    c.add_argument("before")
    c.add_argument("after")
    c.set_defaults(fn=cmd_compare)

    r = sub.add_parser("reanalyze")
    r.add_argument("--rules-only", action="store_true")
    r.set_defaults(fn=cmd_reanalyze)

    v = sub.add_parser("serve")
    v.add_argument("--port", type=int, default=8000)
    v.add_argument("--reload", action="store_true")
    v.set_defaults(fn=cmd_serve)

    z = sub.add_parser("reset")
    z.set_defaults(fn=lambda a: (store.delete_all(), print("all calls deleted")))

    d = sub.add_parser("dump")
    d.add_argument("call_id")
    d.set_defaults(fn=lambda a: print(json.dumps(store.get_call(a.call_id), indent=2, ensure_ascii=False)))

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
