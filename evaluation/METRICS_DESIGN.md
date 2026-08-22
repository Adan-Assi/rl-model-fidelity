# Evaluation Metrics — Design Rationale

This document records the reasoning behind the metric definitions used in
`evaluation/metrics.py`.

## The problem: reward structures differ across environments

The three environments in this project use meaningfully different reward
conventions, by design:

| Environment | Reward structure | Natural performance measure |
|---|---|---|
| FrozenLake | Terminal, `{0, 1}` | Win rate |
| Blackjack | Terminal, `{0, 1}` | Win rate |
| CartPole (discretized) | `+1.0` per step (Gymnasium convention), capped at `max_steps` | Average episode length |

A single shared metric formula across all three would either be meaningless
for some of them (e.g. "average reward" conflates a `[0, 2000]`-scale
quantity for CartPole with a `[0, 1]`-scale win rate for the others) or
would require inventing an artificial common unit. Two tiers are used
instead.

## Tier 1 — Raw performance (environment-specific)

Used for **within-environment learning curves** (reward/performance vs.
episode, per algorithm). Preserves each environment's natural semantics:

- **CartPole**: average episode length (steps survived, out of `max_steps`)
- **Blackjack / FrozenLake**: average reward, i.e. `win_rate`

This tier is not intended to be compared across environments (it isn't
even on a comparable scale to attempt that.)

## Tier 2 — Normalized performance (cross-comparison proxy)

Used for comparisons where a `[0, 1]`-bounded quantity is needed,
primarily **Pair 3's gap-vs-discretization-granularity analysis**
(model-based vs. model-free CartPole agents, swept across bin counts).

- **CartPole**: $ \; \text{normalized} = \frac{\text{avg\_episode\_length}}{\text{max\_steps}} $ 
- **Blackjack / FrozenLake**: `win_rate` (already in `[0, 1]`)

### Important caveat: this is a bounded proxy, not an optimality score

**Normalized performance is a `[0, 1]`-bounded performance *proxy*, not an
optimality-normalized score.** The two are easy to conflate because they
share a numeric range, but they answer different questions:

- CartPole's ceiling (`max_steps`) is a **true, achievable upper bound;**
  a perfect policy can reach `normalized = 1.0`.
- Blackjack's and slippery-FrozenLake's ceilings are **not 1.0**:
  - Blackjack has a structural house edge (the fixed `house_hit_threshold`
    dealer policy, plus draws counting as non-wins), so even an optimal
    gambler policy caps out below `1.0`.
  - Slippery FrozenLake (`is_slippery=True`, the project default) has
    stochastic transitions that can force failure regardless of the
    action taken, so even the exact-model-optimal policy (from Value
    Iteration) caps out below `1.0`.

Because of this, normalized performance is **valid for within-environment
comparisons**, e.g. comparing model-based vs. model-free performance on
CartPole across granularities, where both curves share the same ceiling, but should **not** be read as "how close to optimal" when comparing
*across* environments, and should not be used to build a single combined
plot/table claiming e.g. "FrozenLake's gap is larger than Blackjack's gap"
without further correction.

### Scope of current use

At the current project stage, normalized performance is used **only**
for Pair 3 (model-based vs. model-free on CartPole, across discretization
granularities), where both curves being compared share CartPole's true
`max_steps` ceiling. This keeps the metric's actual use inside the range
where it's valid.

### If cross-Pair comparison is added later

If a future summary figure/table directly compares gap magnitude across
all three Pairs (FrozenLake, Blackjack, CartPole) on one shared axis, the
simple proxy above is **not sufficient** and should be replaced with a
true optimality-normalized score, e.g. (Atari-literature style):

```
normalized = (perf - random_baseline) / (optimal_perf - random_baseline)
```

This requires, per environment:
- `random_baseline`: empirical performance of a uniform-random policy
- `optimal_perf`: either computed exactly (Value Iteration on FrozenLake's
  known transition model) or empirically estimated (e.g. best achievable
  win rate under Blackjack's fixed house policy)

This is deliberately **not implemented yet**, it isn't needed for the
project's current claims, and building it prematurely risks introducing
its own bugs (e.g. mismatched episode-length conventions between VI's
planning and the empirical runs) in service of a comparison that isn't
being made. Revisit only if a cross-Pair summary becomes an actual
deliverable.
