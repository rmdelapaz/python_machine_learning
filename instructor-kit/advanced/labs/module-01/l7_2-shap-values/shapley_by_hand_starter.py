# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 7.2 B: Shapley values by hand for a four-feature fraud rule.

The lesson split a toy churn score among three features. Here the payments team's old hand-written rule
has four features, one of which does nothing, and an interaction (a new device is only suspicious at
night). Compute the shares by trying every arrival order, check them with shap's Exact explainer, and
see two properties for yourself: a feature that never changes the score gets 0, and the shares depend
on the reference payment you compare with. Work through TODO 1 to 4; the file runs as it is.
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
# TODO 1: write shapley(rule, x, ref) that returns ({name: share}, number_of_orders):
#         value(present) = the rule's score when the features in `present` take x's values and the others
#         ref's; for every order in itertools.permutations(names), credit each feature with how much the
#         score rises when it arrives; average over the orders and round to 3.
# TODO 2: print the shares for `payment` against `reference`, and their sum. Before you run it, predict:
#         how is the 0.30 night-and-new-device bonus split? What does gift_card get?
shapley = None
if shapley is None:
    print("TODO 1-2: Shapley values by hand (24 orders)")

# %% Step 3: the library agrees
# TODO 3: rule_model = lambda rows: np.array([fraud_rule(*row) for row in rows]);
#         shap.explainers.Exact(rule_model, <the reference as a 1-row array>) and call it on the payment.
print("TODO 3: shap's Exact explainer on the same payment")

# %% Step 4: a different reference, a different story
# TODO 4: explain the same payment against night_owl, a typical NIGHT payment, below. Which share changes,
#         and why does the total stay the same?
night_owl = {"velocity": 0, "new_device": 0, "night": 1, "gift_card": 0}
print("TODO 4: the same payment against a typical night payment")

print("\nExpected when done: 24 orders; velocity 0.2, new_device 0.15, night 0.15, gift_card 0.0, sum 0.5;")
print("shap Exact [[0.2, 0.15, 0.15, 0.0]]; against a night payment: velocity 0.2, new_device 0.3, night 0.0,")
print("gift_card 0.0, sum 0.5.")
