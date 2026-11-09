"""
First light: Baby Test v0, using the fruit fly's one-shot learning as the model.

A fly smells odour A, and sugar arrives shortly afterwards. After a single
pairing, odour A on its own should trigger "approach". Odour B (never paired)
should not.

The task is fly-like, but the brain is Omega's own design (omega/core.py),
not a fly simulation.

What we check:
  1. One-shot: does a single smell -> sugar pairing create a response to A?
  2. Specific: does B stay unaffected?
  3. Survives sleep: does the A memory survive downscaling and pruning?
  4. No forgetting: after B is learned too, does A still work?

Run:  .venv/bin/python experiments/first_light.py
"""

import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from omega.core import Brain  # noqa: E402

brain = Brain(
    n_neurons=2000,
    synapses_per_neuron=100,
    groups=dict(smell_a=50, smell_b=50, approach=50),  # body: 2 senses, 1 action
    seed=1,
)
rng = np.random.default_rng(2)
APPROACH = brain.groups["approach"]


def senses(names, rate=0.1, amp=1.2):
    """Input from the body: active sensory groups fire about once every 10 ms."""
    ext = np.zeros(brain.N, np.float32)
    for name in names:
        g = brain.groups[name]
        ext[g] = (rng.random(g.size) < rate) * amp
    return ext


def live(steps, smell=None, sugar=False):
    """Live for some steps. Sugar = innate reflex (drives approach) + dopamine.
    Returns how many times the approach neurons fired."""
    names = ([smell] if smell else []) + (["approach"] if sugar else [])
    fired = 0
    for _ in range(steps):
        spk = brain.step(senses(names) if names else None,
                         dopamine=1.0 if sugar else 0.0)
        fired += int(spk[APPROACH].sum())
    return fired


def pairing(smell):
    """One trial: smell for 100 ms, sugar arrives during the last 50 ms."""
    live(50, smell=smell)
    live(50, smell=smell, sugar=True)
    live(300)  # quiet afterwards


def test(label):
    live(300)  # let activity settle
    a, b, none = live(200, "smell_a"), live(200, "smell_b"), live(200)
    print(f"{label:<28} approach spikes:  A={a:5d}   B={b:5d}   nothing={none:5d}")
    return a, b


t0 = time.time()
print("Omega v0: first light\n", brain.stats(), "\n")

test("1. newborn")

pairing("smell_a")
test("2. after ONE A+sugar pairing")

print("   sleep:", brain.sleep())
test("3. after sleep")

pairing("smell_b")
print("   sleep:", brain.sleep())
test("4. after learning B + sleep")

print("\n", brain.stats())
elapsed = time.time() - t0
print(f"\n{brain.t} steps ({brain.t / 1000:.1f} s brain time) in {elapsed:.1f} s "
      f"-> {elapsed / brain.t * 1000:.2f} ms per step")
