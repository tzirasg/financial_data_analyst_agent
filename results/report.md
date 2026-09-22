# Αποτελέσματα Run `20260901_151514_guarded_repr20`

**Config:** model=gemini-3.6-flash levels=['low', 'medium', 'high'] K_max=3 seed=42 N_test=50 N_tune=20 tol=0.01  
**Τελικό effort:** high  
**Κανόνες/Exemplars που έμαθε:** 3 / 3

## 1. Accuracy curve στο D_tune (self-refinement)

| Στάδιο | effort | exec_acc | top_error | patch | απόφαση | Δacc |
|---|---|---|---|---|---|---|
| baseline | low | 0.0 | — | — | — | — |
| L1 | low | 0.15 | scale_unit | R1 | ACCEPT | 0.15 |
| L2 | medium | 0.2 | sign_error | R2 | ACCEPT | 0.05 |
| L3 | high | 0.25 | hallucinated_constant | R3 | ACCEPT | 0.1 |

## 2. D_test (50 held-out) — before/after

| | baseline (low) | τελικό config |
|---|---|---|
| execution accuracy | 0.74 | 0.76 |
| program accuracy | 0.2 | 0.42 |
| σωστά / n | 37/50 | 38/50 |

## 3. Μείωση ανά κλάση σφάλματος (D_test)

| Κλάση | baseline | τελικό |
|---|---|---|
| scale_unit | 7 | 6 |
| wrong_cell | 1 | 0 |
| wrong_operation | 4 | 4 |
| sign_error | 0 | 0 |
| multi_step_order | 0 | 0 |
| hallucinated_constant | 1 | 2 |
| format_error | 0 | 0 |

## 4. Governance / Audit log

| Loop | effort | top_error | patch | Δacc | fixed | regr | compliance | απόφαση |
|---|---|---|---|---|---|---|---|---|
| 1 | low | scale_unit | R1 | +0.150 | 3 | 0 | OK | ACCEPT |
| 2 | medium | sign_error | R2 | +0.050 | 1 | 0 | OK | ACCEPT |
| 3 | high | hallucinated_constant | R3 | +0.100 | 2 | 0 | OK | ACCEPT |
