"""Session-only analyst records. These never modify model outputs or labels."""

import pandas as pd

DECISIONS = ("Approve", "Escalate", "Block")
AUDIT_COLUMNS = [
    "dataset_sha256", "dataset_source", "provenance", "row_id", "risk_score",
    "model_result", "threshold", "decision", "notes", "timestamp_utc", "revision",
]


def latest_records(events, dataset_id):
    return {
        event["row_id"]: event
        for event in events if event["dataset_sha256"] == dataset_id
    }


def latest_decisions(events, dataset_id):
    return {key: event["decision"] for key, event in latest_records(events, dataset_id).items()}


def append_decision(events, analysis, position, threshold, decision, notes):
    if decision not in DECISIONS:
        raise ValueError("Choose Approve, Escalate, or Block.")
    if not 0 <= position < len(analysis.raw):
        raise ValueError("Select a valid transaction.")
    if len(notes) > 4000:
        raise ValueError("Notes must be 4,000 characters or fewer.")
    row_id = analysis.row_ids[position]
    previous = latest_records(events, analysis.dataset_id).get(row_id)
    event = {
        "dataset_sha256": analysis.dataset_id,
        "dataset_source": analysis.source,
        "provenance": analysis.provenance,
        "row_id": row_id,
        "risk_score": float(analysis.scores[position]),
        "model_result": "Flagged" if analysis.flags[position] else "Not flagged",
        "threshold": threshold,
        "decision": decision,
        "notes": notes.strip(),
        "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "revision": previous["revision"] + 1 if previous else 1,
    }
    return [*events, event]


def audit_frame(events):
    return pd.DataFrame(events, columns=AUDIT_COLUMNS)


def audit_export(events):
    return audit_frame(events).to_csv(index=False).encode("utf-8")
