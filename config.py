"""
config.py — Κλειδωμένες (locked) σταθερές του πειράματος.

Ό,τι ΔΕΝ είναι μέσα στο action space (temperature, αλλαγή μοντέλου, retrieval,
decomposition) παραμένει σταθερό ώστε κάθε κέρδος ακρίβειας να αποδίδεται καθαρά
στους δύο επιτρεπτούς μοχλούς: (α) prompt patches, (β) thinking_level.
"""
from __future__ import annotations
import os
from pathlib import Path

# ── Διαδρομές project ────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent

# Φόρτωσε αυτόματα το .env (GEMINI_API_KEY, MODEL_ID) από τον ρίζα του project
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"                # ακατέργαστα FinQA JSON
SAMPLE_DIR = DATA_DIR / "sample"          # τα στρωματοποιημένα δείγματα
PROMPTS_DIR = ROOT / "prompts"
RESULTS_DIR = ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"           # ανά-run logs (audit trail)

for _d in (DATA_DIR, RAW_DIR, SAMPLE_DIR, PROMPTS_DIR, RESULTS_DIR, RUNS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Μοντέλο (locked) ─────────────────────────────────────────────────
MODEL_ID = os.getenv("MODEL_ID", "gemini-3.6-flash")
# Fallbacks αν το 3.6 δεν είναι διαθέσιμο στο free tier (έχουν και 'minimal' level)
FALLBACK_MODELS = ["gemini-3.5-flash"]

# ── Ελεγχόμενη μεταβλητή (locked) ────────────────────────────────────
# thinking_level: επίσημη παράμετρος του Gemini API. Ένα επίπεδο ανά loop.
THINKING_LEVELS = ["low", "medium", "high"]

# Ό,τι μένει σταθερό εκτός action space:
TEMPERATURE = 0.0          # ντετερμινιστικά (single deterministic run)

# ── Βρόχος self-refinement (locked) ──────────────────────────────────
K_MAX = 3                  # ένα loop ανά επίπεδο effort (low/medium/high)
# Effort ανά loop index (0-based): Loop1->low, Loop2->medium, Loop3->high
EFFORT_SCHEDULE = {0: "low", 1: "medium", 2: "high"}

# ── Δείγμα δεδομένων (locked) ────────────────────────────────────────
SEED = 42                  # σταθερό seed για αναπαραγωγιμότητα
N_TEST = 50                # held-out D_test — μόνο τελική αξιολόγηση
N_TUNE = 20                # Dromos A: representative calibration-selected tune

# Στρωμάτωση D_test: πηγή × πολυπλοκότητα = 6 κελιά
#   πηγή:        {table, text}  (από πού αντλείται η απάντηση στο FinQA)
#   πολυπλοκότητα: {1_step, 2_step, 3plus_step}  (βήματα προγράμματος)
STRATA_SOURCE = ["table", "text"]
STRATA_COMPLEXITY = ["1_step", "2_step", "3plus_step"]

# Splits πηγής: D_test από FinQA test.json, D_tune από train.json (δομικά disjoint).
TEST_SPLIT = "test"
TUNE_SPLIT = "train"

# D_test = 50: όσο πιο ισοκατανεμημένα γίνεται στα 6 κελιά (base=8, +1 στα 2 πρώτα).
# (equal allocation -> καθαρή ανά-κελί ανάλυση σφαλμάτων στο reporting)

# D_tune = 10 «μέτριου» επιπέδου (ΟΧΙ τα δυσκολότερα): 2-step-centered ώστε ο
# agent να κάνει μαθήσιμα λάθη με χώρο βελτίωσης. Κατανομή ανά (πηγή, πολυπλοκότητα):
TUNE_ALLOCATION = {
    ("table", "1_step"): 1, ("text", "1_step"): 1,   # λίγα εύκολα
    ("table", "2_step"): 3, ("text", "2_step"): 3,   # πυρήνας: μεσαία
    ("table", "3plus_step"): 1, ("text", "3plus_step"): 1,  # λίγα απαιτητικά
}  # σύνολο = 10

# ── Μετρικές (locked) ────────────────────────────────────────────────
EXEC_TOLERANCE = 0.01      # execution accuracy: ανοχή 1%
# program accuracy: ακριβής αντιστοιχία δομής προγράμματος (βλ. metrics.py)

# ── Κλάσεις σφάλματος FinQA (locked) ─────────────────────────────────
ERROR_CLASSES = [
    "scale_unit",            # λάθος κλίμακα/μονάδα (π.χ. millions vs thousands)
    "wrong_cell",            # άντλησε λάθος κελί/τιμή από τον πίνακα/κείμενο
    "wrong_operation",       # λάθος αριθμητική πράξη
    "sign_error",            # λάθος πρόσημο
    "multi_step_order",      # λάθος σειρά βημάτων σε πολυβηματικό υπολογισμό
    "hallucinated_constant", # εισήγαγε αριθμό που δεν υπάρχει στα δεδομένα
    "format_error",          # σωστός υπολογισμός, λάθος μορφή/παρουσίαση απάντησης
]
CORRECT_LABEL = "correct"    # ετικέτα για σωστές απαντήσεις (όχι κλάση σφάλματος)

# Τρόπος ταξινόμησης σφαλμάτων:
#   "heuristic" -> ντετερμινιστικοί κανόνες (default, αναπαραγώγιμο, χωρίς επιπλέον API)
#   "llm"       -> LLM-as-judge (temperature=0) για τις σημασιολογικές κλάσεις
JUDGE_MODE = "heuristic"

# ── Governance Gate (locked λογική, ρυθμιζόμενα κατώφλια) ─────────────
# Αποδοχή patch αν: ΔAccuracy ≥ MIN_DELTA_ACCURACY ΚΑΙ compliance OK
#                   ΚΑΙ program-equivalence OK (δεν χαλάει ό,τι ήταν σωστό)
MIN_DELTA_ACCURACY = 0.0     # οριακό κέρδος στο D_tune για αποδοχή (>=0: no-regression)
# Early stopping: αν η οριακή βελτίωση < αυτό ή δεν βρίσκονται νέες κλάσεις σφάλματος
EARLY_STOP_DELTA = 0.0       # αν ΔAccuracy <= αυτό σε accepted patch -> σκέψου διακοπή
GATE_MAX_REGRESSIONS = 0     # program-equivalence: μέγιστα correct->wrong που ανέχεται το gate

# ── Validation-guarded gate (πείραμα a+b) ────────────────────────────
# Ένα patch που περνά το tune-gate γίνεται ACCEPT ΜΟΝΟ αν ΔΕΝ ρίχνει την
# exec accuracy σε ανεξάρτητο D_dev πάνω από DEV_MAX_ACC_DROP (strict: 0.0).
# Ranked fallback: αν απορριφθεί (tune ή dev), δοκιμάζεται η επόμενη κλάση.
DEV_MAX_ACC_DROP = 0.0

# ── Retry / ανθεκτικότητα κλήσεων API ────────────────────────────────
# Το gemini-3.6-flash επιστρέφει συχνά 503 (high demand) -> γενναιόδωρο backoff.
RETRYABLE_CODES = (429, 500, 503, 504)
MAX_RETRIES = 8            # πόσες επαναλήψεις πριν το παρατήσει (επίπεδο κλήσης)
RETRY_BASE_DELAY = 2.0     # exponential backoff base (sec): 2,4,8,16,...
RETRY_MAX_DELAY = 60.0     # πλαφόν ανά αναμονή (sec)
RETRY_JITTER = True        # τυχαίο jitter ώστε να μη συγχρονίζονται τα retries
INTER_CALL_DELAY = 1.0     # pacing: αναμονή (sec) ΜΕΤΑ από κάθε επιτυχή κλήση
PER_CALL_TIMEOUT = 240.0   # αν μια κλήση (με retries) ξεπεράσει αυτό -> σφάλμα
# Instance-level ανθεκτικότητα: αν εξαντληθεί το call budget, ξαναδοκίμασε το ΣΤΙΓΜΙΟΤΥΠΟ
# με cooldown (αφήνει τη «μπόρα» 503 να περάσει). Μετά -> api_error, ΧΩΡΙΣ να σπάσει το run.
INSTANCE_RETRIES = 4       # πόσες φορές ξαναδοκιμάζουμε ένα στιγμιότυπο
INSTANCE_COOLDOWN = 45.0   # αναμονή (sec) μεταξύ instance retries


def summary() -> str:
    """Σύντομη περίληψη της ρύθμισης για logging."""
    return (
        f"model={MODEL_ID} levels={THINKING_LEVELS} K_max={K_MAX} "
        f"seed={SEED} N_test={N_TEST} N_tune={N_TUNE} tol={EXEC_TOLERANCE}"
    )


if __name__ == "__main__":
    print(summary())
    print("ERROR_CLASSES:", ERROR_CLASSES)
