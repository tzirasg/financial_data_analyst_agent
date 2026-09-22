"""
run.py — Orchestrator (Financial Data Analyst Agent σε FinQA).

Ροή (validation-guarded gate + ranked fallback):
  1) Bounded self-refinement στο D_tune: baseline(low) -> L1(low) -> L2(medium) -> L3(high),
     με Monitor→Analyze→Plan→Gate→Log (K_max=3, early stopping). Κάθε patch γίνεται
     ACCEPT μόνο αν βελτιώνει το D_tune ΚΑΙ δεν κάνει regression στο ανεξάρτητο D_dev.
  2) Τελικό config -> ΜΙΑ deterministic αξιολόγηση στο D_test (50 held-out).
     Αξιολογούμε και το baseline config στο D_test για καθαρή σύγκριση before/after.
  3) Reports: accuracy curve, D_test before/after, μείωση ανά κλάση σφάλματος,
     governance/audit log. Αποθήκευση στο results/runs/<timestamp>/.

Χρήση:
  python run.py                 # πλήρες τρέξιμο (D_tune=20, D_dev=30, D_test=50)
  python run.py --limit 3       # γρήγορο smoke test (3 tune + 3 dev + 3 test)
  python run.py --mock          # end-to-end χωρίς API (έλεγχος αγωγού)
  python run.py --no-llm-plan   # κανόνες από templates αντί για reflection
  python run.py --judge llm     # LLM-as-judge για ταξινόμηση σφαλμάτων
"""
from __future__ import annotations
import argparse, json, hashlib, time
from collections import Counter
import config as C
from src.dataset import load_sample
from src.agent import answer_instance
from src.loop import run_refinement_guarded, ConfigState
from src.metrics import grade, aggregate
from src.error_taxonomy import classify_result
from src.logging_util import RunLogger


# ── answer functions ─────────────────────────────────────────────────
def make_answer_fn(model=None, logger=None):
    def fn(inst, level, rules, exemplars):
        last = None
        for attempt in range(1, C.INSTANCE_RETRIES + 2):
            try:
                return answer_instance(inst, level, rules=rules, exemplars=exemplars, model=model)
            except Exception as e:  # 503 storm κ.λπ.
                last = e
                if logger:
                    logger.line(f"    ! API fail {inst['id']} (try {attempt}/{C.INSTANCE_RETRIES+1}): {str(e)[:70]}")
                if attempt <= C.INSTANCE_RETRIES:
                    time.sleep(C.INSTANCE_COOLDOWN)
        # ολική αποτυχία -> api_error result (ΔΕΝ σπάει το run· το --resume το ξαναδοκιμάζει)
        if logger:
            logger.line(f"    !! GIVE UP {inst['id']} -> api_error")
        return {"id": inst["id"], "source": inst["source"], "complexity": inst["complexity"],
                "question": inst["question"], "gold_answer": inst["gold_answer"],
                "gold_program": inst["gold_program"], "pred_answer": None, "pred_program": None,
                "parsed_ok": False, "api_error": True, "error_text": str(last)[:200],
                "thinking_level": level,
                "usage": {"prompt_tokens": 0, "thought_tokens": 0, "output_tokens": 0, "attempts": 0}}
    return fn


def make_mock_fn():
    """Ντετερμινιστικός mock (χωρίς API) για έλεγχο του αγωγού end-to-end."""
    cls = lambda i: "scale_unit" if int(hashlib.md5(i["id"].encode()).hexdigest(), 16) % 2 == 0 else "sign_error"
    DIFF = {"scale_unit": 1, "sign_error": 2}; EFF = {"low": 0, "medium": 1, "high": 2}
    def fn(inst, level, rules, exemplars):
        ec = cls(inst); g = inst["gold_answer"]
        rb = 1 if any(r["error_class"] == ec for r in rules) else 0
        ok = (EFF[level] + rb) >= DIFF[ec]
        pa = g if ok else (g * 100 if ec == "scale_unit" else -g)
        return {"id": inst["id"], "source": inst["source"], "complexity": inst["complexity"],
                "question": inst["question"], "gold_answer": g, "gold_program": inst["gold_program"],
                "pred_answer": pa, "pred_program": inst["gold_program"], "parsed_ok": True,
                "thinking_level": level,
                "usage": {"prompt_tokens": 100, "thought_tokens": 10, "output_tokens": 10, "attempts": 1}}
    return fn


# ── D_test evaluation με per-instance checkpoint (resumable) ──────────
def evaluate_checkpointed(instances, effort, state, answer_fn, logger, name):
    path = logger.dir / f"{name}_progress.jsonl"
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line); done[r["id"]] = r
    n = len(instances)
    pending = [inst for inst in instances if inst["id"] not in done]
    import os, threading, concurrent.futures as cf
    W = int(os.getenv("MTA_WORKERS", "1"))
    lock = threading.Lock()
    counter = {"i": len(done)}
    with open(path, "a", encoding="utf-8") as f:
        def _one(inst):
            r = answer_fn(inst, effort, state.rules, state.exemplars)
            r = grade(r); r["error_class"] = classify_result(r, inst)
            with lock:
                counter["i"] += 1
                if not r.get("api_error"):
                    f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n"); f.flush()
                done[inst["id"]] = r
                logger.line(f"  [{name}] {counter['i']}/{n} {inst['id']} "
                            f"correct={r['correct']} ({r['error_class']})")
            return r
        if W <= 1:
            for inst in pending:
                _one(inst)
        else:
            with cf.ThreadPoolExecutor(max_workers=W) as ex:
                list(ex.map(_one, pending))
    results = [done[inst["id"]] for inst in instances]  # σταθερή σειρά = input order
    logger.log_stage(name, results)
    return results


def err_counts(results):
    return dict(Counter(r["error_class"] for r in results
                        if r.get("error_class") and r["error_class"] != C.CORRECT_LABEL))


# ── Report (markdown) ────────────────────────────────────────────────
def build_report(run_id, ref, curve, base_agg, final_agg, base_err, final_err,
                 final_effort, audit):
    L = [f"# Αποτελέσματα Run `{run_id}`", "",
         f"**Config:** {C.summary()}  ", f"**Τελικό effort:** {final_effort}  ",
         f"**Κανόνες/Exemplars που έμαθε:** {ref['n_rules']} / {ref['n_exemplars']}", "",
         "## 1. Accuracy curve στο D_tune (self-refinement)", "",
         "| Στάδιο | effort | exec_acc | top_error | patch | απόφαση | Δacc |",
         "|---|---|---|---|---|---|---|"]
    for c in curve:
        L.append(f"| {c.get('stage')} | {c.get('effort')} | {c.get('accuracy')} | "
                 f"{c.get('top_error','—')} | {c.get('patch','—')} | "
                 f"{'ACCEPT' if c.get('accepted') else ('REJECT' if c.get('accepted') is False else c.get('decision','—'))} | "
                 f"{c.get('delta','—')} |")
    L += ["", "## 2. D_test (50 held-out) — before/after", "",
          "| | baseline (low) | τελικό config |", "|---|---|---|",
          f"| execution accuracy | {base_agg['execution_accuracy']} | {final_agg['execution_accuracy']} |",
          f"| program accuracy | {base_agg['program_accuracy']} | {final_agg['program_accuracy']} |",
          f"| σωστά / n | {base_agg['n_correct']}/{base_agg['n']} | {final_agg['n_correct']}/{final_agg['n']} |",
          "", "## 3. Μείωση ανά κλάση σφάλματος (D_test)", "",
          "| Κλάση | baseline | τελικό |", "|---|---|---|"]
    for ec in C.ERROR_CLASSES:
        L.append(f"| {ec} | {base_err.get(ec,0)} | {final_err.get(ec,0)} |")
    L += ["", "## 4. Governance / Audit log", ""]
    if audit:
        L += ["| Loop | effort | top_error | patch | Δacc | fixed | regr | compliance | απόφαση |",
              "|---|---|---|---|---|---|---|---|---|"]
        for a in audit:
            L.append(f"| {a['loop']} | {a['effort']} | {a['top_error_class']} | {a['patch_rule_id']} | "
                     f"{a['delta_accuracy']:+.3f} | {a['n_fixed']} | {a['n_regressions']} | "
                     f"{'OK' if a['compliance_ok'] else 'BLOCKED:'+a['blocked_reason']} | {a['decision']} |")
    else:
        L.append("_(καμία loop απόφαση — early stop ή μηδέν σφάλματα)_")
    return "\n".join(L) + "\n"


# ── main ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="περιόρισε D_tune & D_test (smoke test)")
    ap.add_argument("--mock", action="store_true", help="χωρίς API (έλεγχος αγωγού)")
    ap.add_argument("--no-llm-plan", action="store_true", help="κανόνες από templates")
    ap.add_argument("--judge", choices=["heuristic", "llm"], default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--resume", default=None, help="run_id για συνέχεια υπάρχοντος run")
    args = ap.parse_args()
    if args.judge:
        C.JUDGE_MODE = args.judge

    logger = RunLogger(tag=args.tag or ("mock" if args.mock else ""), run_id=args.resume)
    d_tune = load_sample("d_tune"); d_test = load_sample("d_test")
    if args.limit:
        d_tune, d_test = d_tune[:args.limit], d_test[:args.limit]
    answer_fn = make_mock_fn() if args.mock else make_answer_fn(args.model, logger=logger)
    use_llm_plan = (not args.mock) and (not args.no_llm_plan)

    logger.line(f"=== RUN {logger.run_id} | {C.summary()} | mock={args.mock} "
                f"llm_plan={use_llm_plan} judge={C.JUDGE_MODE} ===")
    # 1) Self-refinement στο D_tune (validation-guarded gate + ranked fallback)
    d_dev = load_sample("d_dev")
    if args.limit:
        d_dev = d_dev[:args.limit]
    logger.line(f"D_tune={len(d_tune)}  D_dev={len(d_dev)}  D_test={len(d_test)}  [GUARDED]\n")
    ref = run_refinement_guarded(d_tune, d_dev, answer_fn, logger=logger, use_llm_plan=use_llm_plan)
    final_state = ref["final_state"]; final_effort = ref["final_effort"]
    logger.save_json("final_rules.json", final_state.rules)
    logger.save_json("final_exemplars.json", final_state.exemplars)
    logger.line(f"\n>>> Τελικό config: {ref['n_rules']} κανόνες, {ref['n_exemplars']} exemplars, "
                f"effort={final_effort}, D_tune acc={ref['final_tune_accuracy']:.3f}")

    # 2) D_test: baseline vs τελικό (single deterministic run το καθένα)
    logger.line("\n=== D_TEST: baseline (empty config, low) ===")
    base_test = evaluate_checkpointed(d_test, C.EFFORT_SCHEDULE[0], ConfigState(), answer_fn, logger, "dtest_baseline")
    logger.line("\n=== D_TEST: τελικό config ===")
    final_test = evaluate_checkpointed(d_test, final_effort, final_state, answer_fn, logger, "dtest_final")

    base_agg, final_agg = aggregate(base_test), aggregate(final_test)
    base_err, final_err = err_counts(base_test), err_counts(final_test)

    # 3) Reports
    summary = {
        "run_id": logger.run_id, "config": C.summary(), "final_effort": final_effort,
        "n_rules": ref["n_rules"], "n_exemplars": ref["n_exemplars"],
        "tune_curve": ref["curve"], "tune_final_accuracy": ref["final_tune_accuracy"],
        "dtest_baseline": base_agg, "dtest_final": final_agg,
        "dtest_error_reduction": {ec: {"baseline": base_err.get(ec, 0), "final": final_err.get(ec, 0)}
                                  for ec in C.ERROR_CLASSES},
        "audit": logger.audit_entries,
    }
    logger.finalize(summary)
    report = build_report(logger.run_id, ref, ref["curve"], base_agg, final_agg,
                          base_err, final_err, final_effort, logger.audit_entries)
    (logger.dir / "report.md").write_text(report, encoding="utf-8")

    logger.line("\n" + "=" * 60)
    logger.line(f"D_test execution accuracy:  baseline {base_agg['execution_accuracy']:.3f}"
                f"  ->  final {final_agg['execution_accuracy']:.3f}")
    logger.line(f"D_test program accuracy:    baseline {base_agg['program_accuracy']:.3f}"
                f"  ->  final {final_agg['program_accuracy']:.3f}")
    logger.line(f"Αποτελέσματα: {logger.dir}")
    logger.line(f"Report: {logger.dir/'report.md'}")


if __name__ == "__main__":
    main()
