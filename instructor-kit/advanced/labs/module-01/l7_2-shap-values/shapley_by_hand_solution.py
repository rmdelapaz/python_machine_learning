# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 7.2 B (SOLUTION): Shapley values by hand for a four-feature fraud rule.

The lesson split a toy churn score among three features. Here the payments team's old hand-written rule
has four features, one of which does nothing, and an interaction (a new device is only suspicious at
night). Compute the shares by trying every arrival order, check them with shap's Exact explainer, and
see two properties for yourself: a feature that never changes the score gets 0, and the shares depend
on the reference payment you compare with.
"""
# %% Step 1: the rule, a reference payment and the payment to explain (done for you)
import itertools
import numpy as np


def fraud_rule(velocity, new_device, night, gift_card):
    score = 0.01
    if velocity >= 3:
        score += 0.20
    if new_device == 1 and night == 1:       # the interaction
        score += 0.30
    return score                              # gift_card is never used


names = ["velocity", "new_device", "night", "gift_card"]
reference = {"velocity": 0, "new_device": 0, "night": 0, "gift_card": 0}     # a typical payment
payment = {"velocity": 4, "new_device": 1, "night": 1, "gift_card": 1}       # the one to explain
print("reference score:", fraud_rule(**reference), "  payment score:", round(fraud_rule(**payment), 2))

# %% Step 2: Shapley values by trying every arrival order


def shapley(rule, x, ref):
    def value(present):
        return rule(**{n: x[n] if n in present else ref[n] for n in names})
    shares = dict.fromkeys(names, 0.0)
    orders = list(itertools.permutations(names))
    for order in orders:
        present = set()
        for name in order:
            before = value(present)
            present.add(name)
            shares[name] += value(present) - before
    return {n: round(s / len(orders), 3) for n, s in shares.items()}, len(orders)


shares, n_orders = shapley(fraud_rule, payment, reference)
print("orders tried:", n_orders)
print("Shapley values:", shares, " sum:", round(sum(shares.values()), 3))

# %% Step 3: the library agrees
import shap

rule_model = lambda rows: np.array([fraud_rule(*row) for row in rows])
exact = shap.explainers.Exact(rule_model, np.array([[reference[n] for n in names]]))
print("shap Exact:", exact(np.array([[payment[n] for n in names]])).values.round(3).tolist())

# %% Step 4: a different reference, a different story
night_owl = {"velocity": 0, "new_device": 0, "night": 1, "gift_card": 0}     # a typical NIGHT payment
owl_shares, _ = shapley(fraud_rule, payment, night_owl)
print("compared with a typical night payment:", owl_shares, " sum:", round(sum(owl_shares.values()), 3))
