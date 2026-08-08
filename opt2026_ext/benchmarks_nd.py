"""
N-dimensional benchmark functions for the dimensionality sweep (Ext 2).

Ackley, Rastrigin, and Styblinski-Tang in see_core.py are already written
with .mean(1)/.sum(1) reductions over the last axis, which are the *same*
closed-form n-dimensional generalizations used throughout the optimization
literature (Ackley/Rastrigin/Styblinski-Tang all have standard n-dim forms
built exactly this way) -- so F_ackley/F_rastrigin/F_styblinski from
see_core.py are reused verbatim here, just called with X of shape (N, n)
for n != 2. Himmelblau and Levy are 2D-specific (no standard closed-form
n-dim analogue used in the literature) and are excluded from the
dimensionality sweep per the extension spec.
"""
FUNCS_ND = {'Ackley': F_ackley, 'Rastrigin': F_rastrigin, 'Styblinski': F_styblinski}
DOM_ND = {'Ackley': 5., 'Rastrigin': 5.12, 'Styblinski': 5.}  # same domain half-widths as the 2D baseline
DIMS = [2, 5, 10, 25, 50]
