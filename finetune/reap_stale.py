"""Reap stale JOB_STATUS entries that say they're running but have no live
container behind them. Safe to run anytime — only flips jobs whose last
heartbeat is older than the threshold.

Usage:
    python reap_stale.py            # reap entries older than 30 minutes
    python reap_stale.py 0          # reap EVERY active entry (nuclear)
    python reap_stale.py 10         # reap entries older than 10 minutes
"""
import sys
import modal
from datetime import datetime, timedelta

ACTIVE = {"queued", "running", "cancelling"}

minutes = int(sys.argv[1]) if len(sys.argv) > 1 else 30
threshold = timedelta(minutes=minutes)
now = datetime.utcnow()

d = modal.Dict.from_name("lexforge-jobs", create_if_missing=True)

reaped = 0
kept = 0
for k, entry in list(d.items()):
    if not isinstance(entry, dict):
        continue
    status = entry.get("status")
    if status not in ACTIVE:
        kept += 1
        continue
    updated = (entry.get("updated_at") or "").rstrip("Z")
    try:
        t = datetime.fromisoformat(updated)
        age = now - t
    except Exception:
        age = threshold + timedelta(minutes=1)
    if age >= threshold:
        entry["status"] = "failed"
        entry["stage"] = "stale"
        entry["error"] = f"no heartbeat for {age.total_seconds()/60:.1f} min"
        entry["finished_at"] = now.isoformat(timespec="seconds") + "Z"
        d[k] = entry
        reaped += 1
        run = entry.get("run_name") or k
        print(f"reaped {run:60s}  (status={status}, last_seen={updated or '?'})")
    else:
        kept += 1

print(f"\ndone — reaped {reaped}, kept {kept}")
