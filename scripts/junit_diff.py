"""Compare the outcomes of two junit xml files and print a markdown table of the tests that differ.

Usage: junit_diff.py base.xml new.xml [--predicted ids.txt]
Without --predicted the exit code is 0. With it, the exit code is 0 only if the set of changed
node ids equals the predicted set (one node id per line), else 1.
"""
import argparse
import sys
import xml.etree.ElementTree as ET


def read_outcomes(path):
    # node id -> passed / failed / error / skipped
    outcomes = {}
    for case in ET.parse(path).getroot().iter("testcase"):
        node = case.get("classname").replace(".", "/") + ".py::" + case.get("name")
        if case.find("failure") is not None:
            outcome = "failed"
        elif case.find("error") is not None:
            outcome = "error"
        elif case.find("skipped") is not None:
            outcome = "skipped"
        else:
            outcome = "passed"
        outcomes[node] = outcome
    return outcomes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base")
    parser.add_argument("new")
    parser.add_argument("--predicted")
    args = parser.parse_args()

    base = read_outcomes(args.base)
    new = read_outcomes(args.new)

    # tests whose outcome differs, including tests present in only one file
    changed = {}
    for node in sorted(set(base) | set(new)):
        before = base.get(node, "absent")
        after = new.get(node, "absent")
        if before != after:
            changed[node] = (before, after)

    print("| node id | before | after |")
    print("|---|---|---|")
    for node, (before, after) in changed.items():
        print(f"| {node} | {before} | {after} |")

    if args.predicted is None:
        return 0
    with open(args.predicted) as f:
        predicted = {ln.strip() for ln in f if ln.strip() and not ln.startswith("#")}
    unexpected = sorted(set(changed) - predicted)
    missing = sorted(predicted - set(changed))
    for node in unexpected:
        print(f"unexpected change: {node}", file=sys.stderr)
    for node in missing:
        print(f"predicted but unchanged: {node}", file=sys.stderr)
    return 1 if unexpected or missing else 0


if __name__ == "__main__":
    sys.exit(main())
