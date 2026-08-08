# Table II: SEE at each optimizer's best LR, n=7 optimizers

## Himmelblau

| Optimizer | A (radius) | B (curvature) | C (eigen-disp) | D (loss-drop) |
|---|---|---|---|---|
| GD_fixed | 0.637 | 0.543 | 0.943 | 0.227 |
| Adam | 0.216 | 0.143 | 1.000 | 0.143 |
| RMSProp | 1.000 | 1.000 | 1.000 | 0.451 |
| AdaGrad | 0.210 | 0.199 | 1.000 | 0.199 |
| AdamW | 0.213 | 0.140 | 1.000 | 0.140 |
| SGD_Nesterov | 0.813 | 0.610 | 0.962 | 0.190 |
| Lion | 0.030 | 0.018 | 0.493 | 0.018 |

## Ackley

| Optimizer | A (radius) | B (curvature) | C (eigen-disp) | D (loss-drop) |
|---|---|---|---|---|
| GD_fixed | 0.305 | 0.254 | 0.758 | 0.251 |
| Adam | 0.000 | 0.342 | 1.000 | 0.333 |
| RMSProp | 1.000 | 0.286 | 1.000 | 0.286 |
| AdaGrad | 0.000 | 0.437 | 1.000 | 0.426 |
| AdamW | 0.001 | 0.338 | 1.000 | 0.329 |
| SGD_Nesterov | 0.509 | 0.258 | 0.855 | 0.201 |
| Lion | 0.000 | 0.155 | 0.266 | 0.155 |

## Rastrigin

| Optimizer | A (radius) | B (curvature) | C (eigen-disp) | D (loss-drop) |
|---|---|---|---|---|
| GD_fixed | 0.985 | 0.313 | 0.990 | 0.313 |
| Adam | 0.000 | 0.697 | 1.000 | 0.697 |
| RMSProp | 1.000 | 0.422 | 1.000 | 0.422 |
| AdaGrad | 0.000 | 0.697 | 1.000 | 0.697 |
| AdamW | 0.000 | 0.702 | 1.000 | 0.702 |
| SGD_Nesterov | 0.995 | 0.300 | 0.995 | 0.300 |
| Lion | 0.000 | 0.251 | 0.976 | 0.251 |

## Styblinski

| Optimizer | A (radius) | B (curvature) | C (eigen-disp) | D (loss-drop) |
|---|---|---|---|---|
| GD_fixed | 0.549 | 0.444 | 0.749 | 0.221 |
| Adam | 0.208 | 0.252 | 1.000 | 0.252 |
| RMSProp | 1.000 | 1.000 | 1.000 | 0.500 |
| AdaGrad | 0.200 | 0.267 | 1.000 | 0.267 |
| AdamW | 0.207 | 0.249 | 1.000 | 0.249 |
| SGD_Nesterov | 0.697 | 0.528 | 0.837 | 0.241 |
| Lion | 0.026 | 0.031 | 0.249 | 0.031 |

## Levy

| Optimizer | A (radius) | B (curvature) | C (eigen-disp) | D (loss-drop) |
|---|---|---|---|---|
| GD_fixed | 0.017 | 0.332 | 0.000 | 0.147 |
| Adam | 0.103 | 0.746 | 0.000 | 0.255 |
| RMSProp | 1.000 | 0.746 | 1.000 | 0.255 |
| AdaGrad | 0.013 | 0.746 | 0.000 | 0.255 |
| AdamW | 0.104 | 0.763 | 0.000 | 0.255 |
| SGD_Nesterov | 0.199 | 0.438 | 0.000 | 0.233 |
| Lion | 0.013 | 0.120 | 0.000 | 0.043 |

