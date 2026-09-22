"""
src/logging_util.py — Καταγραφή run για auditability.

Κάθε run παίρνει φάκελο results/runs/<timestamp>/ και μέσα:
  - config.json         : η ρύθμιση του πειράματος
  - stage_<name>.json   : per-instance αποτελέσματα κάθε σταδίου
  - gate_log.json       : οι αποφάσεις του gate (accept/reject + γιατί)  <- audit trail
  - audit.txt           : ανθρωποαναγνώσιμο log
  - summary.json        : τελικά αποτελέσματα (curve, D_test)
"""
from __future__ import annotations
import json, datetime
from pathlib import Path
import config as C


class RunLogger:
    def __init__(self, tag: str = "", run_id: str | None = None):
        if run_id:                       # resume σε υπάρχον run dir
            self.run_id = run_id
        else:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_id = f"{ts}{('_' + tag) if tag else ''}"
        self.dir = C.RUNS_DIR / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.audit_entries: list[dict] = []
        self.lines: list[str] = []
        self.save_json("config.json", {"config": C.summary(),
                                       "error_classes": C.ERROR_CLASSES,
                                       "gate": {"min_delta": C.MIN_DELTA_ACCURACY,
                                                "max_regressions": C.GATE_MAX_REGRESSIONS}})

    def save_json(self, name: str, obj) -> None:
        (self.dir / name).write_text(
            json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    def log_stage(self, name: str, results: list[dict]) -> None:
        slim = [{k: r.get(k) for k in
                 ("id", "source", "complexity", "gold_answer", "pred_answer",
                  "correct", "program_ok", "error_class", "thinking_level",
                  "pred_program", "gold_program")} for r in results]
        self.save_json(f"stage_{name}.json", slim)

    def line(self, s: str) -> None:
        self.lines.append(s)
        print(s)

    def audit(self, entry: dict) -> None:
        self.audit_entries.append(entry)
        self.save_json("gate_log.json", self.audit_entries)

    def finalize(self, summary: dict) -> None:
        self.save_json("summary.json", summary)
        (self.dir / "audit.txt").write_text("\n".join(self.lines) + "\n", encoding="utf-8")
