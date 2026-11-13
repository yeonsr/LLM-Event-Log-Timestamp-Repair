# Strategies implemented by the synthesized programs

## Judge: gpt-5

Share of the 148 programs implementing each strategy, overall and by error category.

| Strategy | All | C1 | C2 | C3 |
|---|---|---|---|---|
| perm | 67% | 100% | 50% | 51% |
| beam | 45% | 53% | 40% | 41% |
| transprob | 80% | 100% | 86% | 55% |
| rightanch | 99% | 100% | 98% | 100% |
| r_ceiling | 91% | 96% | 90% | 88% |
| median | 24% | 16% | 30% | 27% |
| mean | 1% | 0% | 0% | 2% |
| lognorm | 14% | 12% | 14% | 14% |
| uniform | 27% | 27% | 20% | 35% |
| rescale | 80% | 100% | 50% | 92% |
| firstts | 67% | 100% | 50% | 51% |
| monotone | 99% | 100% | 100% | 98% |
| fallback | 100% | 100% | 100% | 100% |

## Judge: gpt-4o

Share of the 148 programs implementing each strategy, overall and by error category.

| Strategy | All | C1 | C2 | C3 |
|---|---|---|---|---|
| perm | 67% | 100% | 50% | 51% |
| beam | 40% | 41% | 38% | 41% |
| transprob | 83% | 100% | 90% | 59% |
| rightanch | 99% | 100% | 98% | 100% |
| r_ceiling | 95% | 98% | 94% | 94% |
| median | 84% | 78% | 90% | 86% |
| mean | 1% | 0% | 2% | 2% |
| lognorm | 35% | 43% | 26% | 37% |
| uniform | 20% | 27% | 14% | 18% |
| rescale | 79% | 100% | 50% | 88% |
| firstts | 57% | 88% | 44% | 41% |
| monotone | 97% | 100% | 98% | 94% |
| fallback | 97% | 94% | 98% | 100% |

## Agreement between the two judges

| strategy | gpt-4o | gpt-5 | agreement | gpt-4o_only | gpt-5_only |
|---|---|---|---|---|---|
| perm | 0.669 | 0.669 | 1.000 | 0 | 0 |
| beam | 0.399 | 0.446 | 0.953 | 0 | 7 |
| transprob | 0.831 | 0.804 | 0.959 | 5 | 1 |
| rightanch | 0.993 | 0.993 | 0.986 | 1 | 1 |
| r_ceiling | 0.953 | 0.912 | 0.892 | 11 | 5 |
| median | 0.845 | 0.243 | 0.385 | 90 | 1 |
| mean | 0.014 | 0.007 | 0.993 | 1 | 0 |
| lognorm | 0.351 | 0.135 | 0.770 | 33 | 1 |
| uniform | 0.196 | 0.270 | 0.750 | 13 | 24 |
| rescale | 0.791 | 0.804 | 0.973 | 1 | 3 |
| firstts | 0.574 | 0.669 | 0.905 | 0 | 14 |
| monotone | 0.973 | 0.993 | 0.980 | 0 | 3 |
| fallback | 0.973 | 1.000 | 0.973 | 0 | 4 |

## Two-sided anchor fitting (judge: gpt-5)

Asked separately, because it is the mechanism the comparison with MAXL turns on and the general strategy list is too coarse for it. MAXL obtains the gaps by minimising a per-transition likelihood subject to their sum equalling the span between the anchors, so a program counts here only when it both solves an objective and weights that objective per transition by the fitted parameters of that transition. Clipping, clamping, proportional rescaling, uniform division and looking up a duration statistic for the transition into the right anchor are all excluded by name: those are behaviours STAT has as well.

| Scope | Programs | Fit two-sided | Share |
|---|---|---|---|
| all | 148 | 0 | 0% |
| C1 | 49 | 0 | 0% |
| C2 | 50 | 0 | 0% |
| C3 | 49 | 0 | 0% |
| LLMU | 75 | 0 | 0% |
| LLMD | 73 | 0 | 0% |

No synthesized program implements it, on any category or either arm. The judge is not refusing by default: put a weaker question to the same model -- does the right anchor constrain the estimate at all, rescaling included -- and it returns a large majority, with the line of code to show for each. Under the mechanism that actually distinguishes MAXL it returns none. The programs read the right anchor and bound their output with it, which is what STAT does; they do not fit against it.
