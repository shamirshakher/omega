"""
Timing test: does Omega learn cause -> effect, or just "together"?

Three fresh brains, each with an identical newborn (same seed). Each gets
one trial:

  forward   smell A ......[sugar]      smell comes first (A predicts sugar)
  backward  [sugar]...... smell A      sugar comes first (A comes after)
  unpaired  smell A    (long gap)  [sugar]   no timing link (control)

The fly learns forward pairings strongly. Backward pairings give weak or
even opposite learning. If Omega models cause and effect, only "forward"
should make A trigger approach.

Run:  .venv/bin/python experiments/timing_test.py
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from omega.core import Brain  # noqa: E402


def make():
    brain = Brain(2000, 100, groups=dict(smell_a=50, smell_b=50, approach=50), seed=1)
    return brain, np.random.default_rng(2)


def live(brain, rng, steps, smell=False, sugar=False):
    names = (["smell_a"] if smell else []) + (["approach"] if sugar else [])
    fired = 0
    for _ in range(steps):
        ext = None
        if names:
            ext = np.zeros(brain.N, np.float32)
            for n in names:
                g = brain.groups[n]
                ext[g] = (rng.random(g.size) < 0.1) * 1.2
        spk = brain.step(ext, dopamine=1.0 if sugar else 0.0)
        fired += int(spk[brain.groups["approach"]].sum())
    return fired


# Symmetric: each protocol has 75 ms of smell, 50 ms of sugar and 25 ms of overlap.
# Only the ORDER differs.
PROTOCOLS = {
    # smell 0-75, sugar 50-100  (smell first)
    "forward":  [(50, True, False), (25, True, True), (25, False, True)],
    # sugar 0-50, smell 25-100  (sugar first)
    "backward": [(25, False, True), (25, True, True), (50, True, False)],
    # smell 0-75, 1 s gap, sugar for 50 ms
    "unpaired": [(75, True, False), (1000, False, False), (50, False, True)],
}


def w(brain, src, dst):
    """Mean strength of the synapses from group src to group dst."""
    m = np.isin(brain.pre, brain.groups[src]) & np.isin(brain.post, brain.groups[dst])
    return brain.strength[m].mean()


def response(brain, rng):
    live(brain, rng, 300)
    a = live(brain, rng, 200, smell=True)
    none = live(brain, rng, 200)
    return a, none


print("approach spikes in 200 ms (A = smell A presented, none = no input)")
print("w(A->app) = mean strength of smell A -> approach synapses (newborn ~0.05)\n")
print(f"{'protocol':<10} {'newborn A/none':>15} {'after trial':>13} "
      f"{'after sleep':>13} {'w(A->app)':>10} {'w(app->app)':>12}")
for name, phases in PROTOCOLS.items():
    brain, rng = make()
    newborn = response(brain, rng)
    for steps, smell, sugar in phases:
        live(brain, rng, steps, smell, sugar)
    after = response(brain, rng)
    brain.sleep()
    slept = response(brain, rng)
    fmt = lambda r: f"{r[0]}/{r[1]}"
    print(f"{name:<10} {fmt(newborn):>15} {fmt(after):>13} {fmt(slept):>13} "
          f"{w(brain, 'smell_a', 'approach'):>10.3f} {w(brain, 'approach', 'approach'):>12.3f}")
