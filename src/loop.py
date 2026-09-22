"""
src/loop.py — Ο βρόχος bounded self-refinement (Monitor→Analyze→Plan→Gate→Log).

Δομή (βλ. διάγραμμα):
  Baseline: thinking=low, άδεια registries -> αρχική καταγραφή σφαλμάτων (c0).
  Loop k (k=0,1,2), effort = low/medium/high:
    MONITOR : αξιολόγησε το τρέχον αποδεκτό config στο effort αυτού του loop (D_tune)
    ANALYZE : ταξινόμησε σφάλματα -> top-1 κλάση (+ έλεγχος early-stop)
    PLAN    : φτιάξε patch (κανόνας + διορθωτικό exemplar) για την top-1 κλάση
    GATE    : αξιολόγησε υποψήφιο (registries+patch) στο ίδιο effort· accept αν
              Δacc≥κατώφλι & compliance & no-regression -> αλλιώς μένει ως είχε
    LOG     : κατέγραψε την απόφαση (audit)
  Τελικό config = τελευταίο accepted -> ΜΙΑ deterministic αξιολόγηση στο D_test.

Το gate συγκρίνει candidate vs monitor στο ΙΔΙΟ effort => απομονώνει το κέρδος του
prompt patch (καθαρή απόδοση), ενώ το effort bump είναι ο δεύτερος (scheduled) μοχλός.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import config as C
from src.metrics import grade, aggregate
from src.error_taxonomy import classify_result
from src import gate as gate_mod


@dataclass
class ConfigState:
    rules: list = field(default_factory=list)
    exemplars: list = field(default_factory=list)

    def copy(self) -> "ConfigState":
        return ConfigState(rules=list(self.rules), exemplars=list(self.exemplars))


def evaluate_config(instances, thinking_level, state, answer_fn) -> list[dict]:
    """Τρέξε τον agent σε όλα τα instances, βαθμολόγησε + ταξινόμησε σφάλματα.

    Infra-only: αν MTA_WORKERS>1 τρέχει τις (ανεξάρτητες, temp=0) κλήσεις
    παράλληλα με σταθερή σειρά εξόδου -> ίδια ντετερμινιστικά αποτελέσματα, πιο γρήγορα.
    """
    import os, concurrent.futures as cf
    def _one(inst):
        r = answer_fn(inst, thinking_level, state.rules, state.exemplars)
        r = grade(r)
        r["error_class"] = classify_result(r, inst)
        return r
    W = int(os.getenv("MTA_WORKERS", "1"))
    if W <= 1:
        return [_one(i) for i in instances]
    with cf.ThreadPoolExecutor(max_workers=W) as ex:
        return list(ex.map(_one, instances))


# ══════════════════════════════════════════════════════════════════════
# Validation-guarded gate + ranked within-loop fallback
# ══════════════════════════════════════════════════════════════════════
def _err_dist(results):
    from collections import Counter
    return dict(Counter(r.get("error_class") for r in results
                        if r.get("error_class") not in (None, C.CORRECT_LABEL, "api_error")))


def run_refinement_guarded(d_tune, d_dev, answer_fn, logger=None, use_llm_plan=True) -> dict:
    """
    (a) Validation-guarded gate: ένα patch που περνά το tune-gate γίνεται ACCEPT
        μόνο αν ΔΕΝ ρίχνει την exec accuracy σε ανεξάρτητο D_dev (no-regression).
    (b) Ranked fallback: σε κάθε loop δοκιμάζονται οι κλάσεις με σειρά count↓·
        αν μια απορριφθεί (tune ή dev), δοκιμάζεται η ΕΠΟΜΕΝΗ κλάση — έτσι το
        scale_unit που έχανε το tie-break παίρνει τη σειρά του.
    """
    from src.planner import plan_patch_for_class, ranked_error_classes
    inst_by_id = {i["id"]: i for i in d_tune}
    state = ConfigState()
    curve = []

    def log(s):
        if logger: logger.line(s)
        else: print(s)

    # ── Baseline (low, άδειο) — tune + dev ──
    e0 = C.EFFORT_SCHEDULE[0]
    monitor = evaluate_config(d_tune, e0, state, answer_fn)
    if logger: logger.log_stage("baseline", monitor)
    acc = aggregate(monitor)["execution_accuracy"]
    dev0 = evaluate_config(d_dev, e0, state, answer_fn)
    if logger: logger.log_stage("dev_baseline", dev0)
    dev_acc0 = aggregate(dev0)["execution_accuracy"]
    log(f"[baseline] effort={e0} tune_acc={acc:.3f} dev_acc={dev_acc0:.3f} "
        f"errors={_err_dist(monitor)}")
    curve.append({"stage": "baseline", "effort": e0, "accuracy": acc,
                  "dev_accuracy": dev_acc0, "error_classes": _err_dist(monitor)})
    accepted_state, accepted_results, accepted_acc = state, monitor, acc
    final_effort = e0

    for k in range(C.K_MAX):
        effort = C.EFFORT_SCHEDULE[k]
        final_effort = effort
        # MONITOR (tune)
        if k == 0:
            mon = accepted_results
        else:
            mon = evaluate_config(d_tune, effort, accepted_state, answer_fn)
            if logger: logger.log_stage(f"L{k+1}_monitor", mon)
        mon_acc = aggregate(mon)["execution_accuracy"]
        # dev accuracy του ΤΡΕΧΟΝΤΟΣ accepted config στο τρέχον effort
        dev_prev = evaluate_config(d_dev, effort, accepted_state, answer_fn)
        dev_prev_acc = aggregate(dev_prev)["execution_accuracy"]

        ranked = ranked_error_classes(mon)
        log(f"[L{k+1}] MONITOR effort={effort} tune_acc={mon_acc:.3f} "
            f"dev_acc={dev_prev_acc:.3f} ranked={ranked}")
        if not ranked:
            log(f"[L{k+1}] EARLY STOP: δεν απομένουν σφάλματα.")
            curve.append({"stage": f"L{k+1}", "effort": effort, "accuracy": mon_acc,
                          "dev_accuracy": dev_prev_acc, "decision": "early_stop_no_errors"})
            break

        attempts = []
        chosen = None
        for rank_i, ec in enumerate(ranked):
            patch = plan_patch_for_class(k, mon, ec, inst_by_id, use_llm=use_llm_plan)
            if patch is None:
                continue
            cand = accepted_state.copy()
            cand.rules.append({"id": patch.rule_id, "error_class": patch.error_class,
                               "rule_text": patch.rule_text, "added_loop": patch.loop})
            cand.exemplars.append(patch.exemplar)
            cand_tune = evaluate_config(d_tune, effort, cand, answer_fn)
            cand_acc = aggregate(cand_tune)["execution_accuracy"]
            # tune-gate πρώτα (φθηνό: αποφεύγει dev eval σε patches που κόβονται στο tune)
            base_dec = gate_mod.evaluate(mon_acc, cand_acc, mon, cand_tune, patch)
            if not base_dec.accepted:
                log(f"[L{k+1}]   try#{rank_i+1} [{ec}] {patch.rule_id}: "
                    f"REJECT(tune) Δacc={base_dec.delta_accuracy:+.3f} "
                    f"regr={base_dec.n_regressions}")
                attempts.append({"rank": rank_i+1, "error_class": ec, "stage": "tune",
                                 "accepted": False, "tune_delta": base_dec.delta_accuracy,
                                 "n_regressions": base_dec.n_regressions,
                                 "rule_text": patch.rule_text})
                continue
            # tune πέρασε -> ξόδεψε dev eval (guard)
            cand_dev = evaluate_config(d_dev, effort, cand, answer_fn)
            cand_dev_acc = aggregate(cand_dev)["execution_accuracy"]
            dec = gate_mod.evaluate_with_dev(mon_acc, cand_acc, mon, cand_tune, patch,
                                             dev_prev_acc, cand_dev_acc)
            log(f"[L{k+1}]   try#{rank_i+1} [{ec}] {patch.rule_id}: {dec.reason}")
            attempts.append({"rank": rank_i+1, "error_class": ec,
                             "accepted": dec.accepted, "tune_delta": dec.delta_accuracy,
                             "n_fixed": dec.n_fixed, "n_regressions": dec.n_regressions,
                             "dev_prev": dec.dev_prev_acc, "dev_cand": dec.dev_cand_acc,
                             "dev_delta": dec.dev_delta, "rule_text": patch.rule_text})
            if dec.accepted:
                accepted_state = cand
                accepted_results = cand_tune
                accepted_acc = cand_acc
                if logger: logger.log_stage(f"L{k+1}_accepted", cand_tune)
                chosen = {"patch": patch, "dec": dec, "rank": rank_i+1}
                break

        if chosen is None:
            log(f"[L{k+1}] Καμία κλάση δεν πέρασε το guarded gate — καμία αλλαγή "
                f"({len(attempts)} attempts).")
            curve.append({"stage": f"L{k+1}", "effort": effort, "accuracy": accepted_acc,
                          "dev_accuracy": dev_prev_acc, "monitor_accuracy": mon_acc,
                          "top_error": ranked[0], "patch": "—", "accepted": False,
                          "delta": 0.0, "n_attempts": len(attempts), "attempts": attempts})
            if logger:
                logger.audit({
                    "loop": k+1, "effort": effort, "top_error_class": ranked[0],
                    "error_counts": _err_dist(mon), "patch_rule_id": "—", "rule_text": "",
                    "exemplar_id": "—", "source_failure_id": "",
                    "monitor_accuracy": mon_acc, "candidate_accuracy": mon_acc,
                    "delta_accuracy": 0.0, "n_fixed": 0, "n_regressions": 0,
                    "compliance_ok": True, "blocked_reason": "",
                    "dev_prev_acc": dev_prev_acc, "dev_cand_acc": dev_prev_acc,
                    "decision": "NO_ACCEPT", "n_attempts": len(attempts),
                    "attempts": attempts,
                    "reason": f"καμία από {len(attempts)} κλάσεις δεν πέρασε (tune+dev guard)",
                })
            continue

        p = chosen["patch"]; dec = chosen["dec"]
        log(f"[L{k+1}] ACCEPT [{p.error_class}] {p.rule_id} (rank #{chosen['rank']}/"
            f"{len(ranked)}) tune_acc->{accepted_acc:.3f} dev {dec.dev_prev_acc:.3f}->"
            f"{dec.dev_cand_acc:.3f}")
        curve.append({"stage": f"L{k+1}", "effort": effort, "accuracy": accepted_acc,
                      "dev_accuracy": dec.dev_cand_acc, "monitor_accuracy": mon_acc,
                      "top_error": p.error_class, "patch": p.rule_id, "accepted": True,
                      "delta": dec.delta_accuracy, "n_fixed": dec.n_fixed,
                      "n_regressions": dec.n_regressions, "chosen_rank": chosen["rank"],
                      "dev_delta": dec.dev_delta, "n_attempts": len(attempts),
                      "attempts": attempts})
        if logger:
            logger.audit({
                "loop": k+1, "effort": effort, "top_error_class": p.error_class,
                "error_counts": _err_dist(mon), "patch_rule_id": p.rule_id,
                "rule_text": p.rule_text, "exemplar_id": p.exemplar["id"],
                "source_failure_id": p.source_failure_id,
                "monitor_accuracy": mon_acc, "candidate_accuracy": accepted_acc,
                "delta_accuracy": dec.delta_accuracy, "n_fixed": dec.n_fixed,
                "n_regressions": dec.n_regressions, "compliance_ok": dec.compliance_ok,
                "blocked_reason": dec.blocked_reason,
                "dev_prev_acc": dec.dev_prev_acc, "dev_cand_acc": dec.dev_cand_acc,
                "dev_delta": dec.dev_delta, "chosen_rank": chosen["rank"],
                "decision": "ACCEPT", "n_attempts": len(attempts), "attempts": attempts,
                "reason": dec.reason,
            })

    return {
        "final_state": accepted_state,
        "final_effort": final_effort,
        "final_tune_accuracy": accepted_acc,
        "curve": curve,
        "n_rules": len(accepted_state.rules),
        "n_exemplars": len(accepted_state.exemplars),
    }
