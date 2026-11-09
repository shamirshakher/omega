"""
Omega core v0: neurons + synapses.

The smallest working version of the ideas in docs/brainstorm.md:

- Neurons and synapses are stored as flat arrays, one array per property.
  Synapse i is index i in every synapse array.
- Overgrowth: the brain starts with dense random connections.
- Timing = causality (STDP): pre fires before post -> positive tag;
  post fires before pre -> negative tag.
- Two-key lock (three-factor learning): a tag only becomes a real change
  in strength when a value signal (dopamine) arrives. Tagged synapses
  also gain stability, which protects them later.
- Sleep: scale all synapses down (stable ones are protected), prune weak or
  unused ones, let neurons with no connections die, and grow a few new,
  fluid synapses as a reserve for future learning.

One step = 1 ms of brain time.
"""

import numpy as np

DEFAULTS = dict(
    # neurons
    threshold=1.0,          # fire when the potential reaches this
    tau_membrane=20.0,      # how fast the potential leaks away (steps)
    refractory=2,           # steps a neuron can't fire after a spike
    noise_rate=0.002,       # chance per step of a random background kick
    noise_amp=1.2,          # size of that kick
    adapt=0.05,             # homeostasis: threshold rises this much per spike
    tau_adapt=200.0,        # ...and relaxes back to baseline over this many steps
    # synapses
    w_init=0.1,             # initial excitatory strengths: uniform(0, w_init)
    w_inhib=0.5,            # inhibitory strength (fixed in v0). Must beat
                            # excitation, or one input triggers a seizure
    w_max=1.0,              # maximum excitatory strength
    # timing (STDP) -> tags
    tau_trace=20.0,         # how long a spike is "remembered" for timing
    a_plus=1.0,             # tag gain when pre fired before post (causal)
    a_minus=0.6,            # tag loss when post fired before pre (anti-causal)
    tau_tag=200.0,          # how long a tag lasts (the capture window)
    # value signal (dopamine)
    learn_rate=0.01,        # tag -> strength conversion
    capture_rate=0.002,     # tag -> stability conversion (one trial = partly stable)
    # sleep
    sleep_downscale=0.2,    # fraction removed from fully fluid synapses
    consolidate=0.5,        # sleep: stability gained per sleep, x (strength / w_max)
    prune_below=0.01,       # weaker than this and not stable -> pruned
    stable_at=0.5,          # stability above this protects from pruning
    unused_limit=20_000,    # steps without use before a fluid synapse is pruned
    grow_fraction=0.01,     # new synapses grown per sleep (fraction of total)
    w_new=0.02,             # strength of newly grown synapses
)


class Brain:
    def __init__(self, n_neurons=2000, synapses_per_neuron=100,
                 inhibitory_fraction=0.2, groups=None, seed=None, **params):
        """
        groups: e.g. {"smell_a": 50, "approach": 50}. Named sensory and motor
        groups: the body's inputs and outputs. They are allocated first, are
        excitatory and are protected from dying.
        """
        self.p = {**DEFAULTS, **params}
        self.rng = np.random.default_rng(seed)
        self.N = N = n_neurons
        self.t = 0

        # --- neuron table -------------------------------------------------
        self.v = np.zeros(N, np.float32)                    # membrane potential
        self.threshold = np.full(N, self.p["threshold"], np.float32)
        self.refractory_until = np.zeros(N, np.int64)
        self.last_spike = np.full(N, -1, np.int64)
        self.trace = np.zeros(N, np.float32)                # recent spikes, decays
        self.alive = np.ones(N, bool)
        self.protected = np.zeros(N, bool)
        self.inhibitory = self.rng.random(N) < inhibitory_fraction
        self.incoming = np.zeros(N, np.float32)             # arrives next step
        self.spikes = np.zeros(N, bool)

        self.groups = {}
        start = 0
        for name, size in (groups or {}).items():
            idx = np.arange(start, start + size)
            self.groups[name] = idx
            self.protected[idx] = True
            self.inhibitory[idx] = False
            start += size
        assert start <= N, "groups are larger than the brain"

        # --- synapse table: overgrowth ---------------------------------------
        k = synapses_per_neuron
        pre = np.repeat(np.arange(N), k)
        post = self.rng.integers(0, N - 1, N * k)
        post += post >= pre                                  # no self-connections
        strength = np.where(
            self.inhibitory[pre],
            -self.p["w_inhib"],
            self.rng.uniform(0, self.p["w_init"], N * k),
        )
        self._set_synapses(
            pre, post, strength,
            stability=np.zeros(N * k),
            tag=np.zeros(N * k),
            last_used=np.zeros(N * k),
        )

        self._decay_m = np.float32(np.exp(-1 / self.p["tau_membrane"]))
        self._decay_trace = np.float32(np.exp(-1 / self.p["tau_trace"]))
        self._decay_tag = np.float32(np.exp(-1 / self.p["tau_tag"]))
        self._decay_adapt = np.float32(np.exp(-1 / self.p["tau_adapt"]))
        self.base_threshold = self.threshold.copy()

    # ------------------------------------------------------------------ storage
    def _set_synapses(self, pre, post, strength, stability, tag, last_used):
        """Store synapses sorted by pre-neuron, with CSR offsets for fast lookup."""
        order = np.argsort(pre, kind="stable")
        self.pre = pre[order].astype(np.uint32)
        self.post = post[order].astype(np.uint32)
        self.strength = strength[order].astype(np.float32)
        self.stability = stability[order].astype(np.float32)
        self.tag = tag[order].astype(np.float32)
        self.last_used = last_used[order].astype(np.int64)
        # synapses of neuron x are offsets[x] : offsets[x+1]
        self.offsets = np.zeros(self.N + 1, np.int64)
        np.cumsum(np.bincount(self.pre, minlength=self.N), out=self.offsets[1:])
        # only excitatory synapses learn in v0
        self.plastic = ~self.inhibitory[self.pre]
        if not hasattr(self, "budget"):
            # synaptic scaling: each neuron's total excitatory input at birth
            # is its budget for life (synapses compete for it)
            self.budget = np.bincount(self.post[self.plastic],
                                      self.strength[self.plastic], minlength=self.N)

    def _outgoing(self, neurons):
        """Indices of all synapses leaving the given neurons (vectorized)."""
        starts = self.offsets[neurons]
        lengths = self.offsets[neurons + 1] - starts
        total = int(lengths.sum())
        if total == 0:
            return np.empty(0, np.int64)
        shift = np.repeat(starts - (np.cumsum(lengths) - lengths), lengths)
        return shift + np.arange(total)

    # --------------------------------------------------------------------- life
    def step(self, external=None, dopamine=0.0):
        """Advance one step (1 ms). external: input current per neuron."""
        p = self.p
        self.t += 1

        # 1. potentials leak, receive last step's spikes, input and noise
        self.v *= self._decay_m
        self.v += self.incoming
        self.incoming[:] = 0
        if external is not None:
            self.v += external
        noise = self.rng.random(self.N) < p["noise_rate"]
        self.v[noise] += p["noise_amp"]

        # 2. who fires?
        spk = self.alive & (self.t >= self.refractory_until) & (self.v >= self.threshold)
        fired = np.flatnonzero(spk)
        self.v[fired] = 0.0
        # homeostasis: busy neurons get harder to fire, then relax back
        self.threshold += (self.base_threshold - self.threshold) * (1 - self._decay_adapt)
        self.threshold[fired] += p["adapt"]
        self.refractory_until[fired] = self.t + p["refractory"]
        self.last_spike[fired] = self.t

        if fired.size:
            # 3. transmit along outgoing synapses
            out = self._outgoing(fired)
            np.add.at(self.incoming, self.post[out], self.strength[out])
            self.last_used[out] = self.t

            # 4. timing -> tags (only learnable synapses)
            # anti-causal: pre fires now, post fired recently -> negative tag
            out_pl = out[self.plastic[out]]
            self.tag[out_pl] -= p["a_minus"] * self.trace[self.post[out_pl]]
            # causal: post fires now, pre fired recently -> positive tag
            into = np.flatnonzero(spk[self.post] & self.plastic)
            self.tag[into] += p["a_plus"] * self.trace[self.pre[into]]

        # 5. spike traces and tags fade
        self.trace *= self._decay_trace
        self.trace[fired] = 1.0     # capped: one spike can't dominate
        self.tag *= self._decay_tag

        # 6. value signal: the second key
        if dopamine:
            self.reward(dopamine)

        self.spikes = spk
        return spk

    def reward(self, amount):
        """Dopamine: turn tags into strength changes and stability (capture)."""
        p, m = self.p, self.plastic
        tag = self.tag[m]
        stab = self.stability[m]
        self.strength[m] = np.clip(
            self.strength[m] + p["learn_rate"] * amount * tag * (1 - stab),
            0, p["w_max"])
        self.stability[m] = np.minimum(
            1.0, stab + p["capture_rate"] * abs(amount) * np.abs(tag))

    def sleep(self):
        """Consolidate, downscale, scale, prune, let neurons die, grow a reserve,
        compact storage."""
        p, m = self.p, self.plastic
        before = self.pre.size

        # 1. downscale everything; stable synapses are protected
        self.strength[m] *= 1 - p["sleep_downscale"] * (1 - self.stability[m])

        # 1a. consolidation: strong synapses become more stable (big spines last)
        self.stability[m] = np.minimum(
            1.0, self.stability[m] + p["consolidate"] * self.strength[m] / p["w_max"])

        # 1b. synaptic scaling: if a neuron's inputs exceed its budget, shrink
        #     the fluid ones to fit. Stable synapses are protected.
        stable = m & (self.stability >= p["stable_at"])
        fluid_m = m & ~stable
        total = np.bincount(self.post[m], self.strength[m], minlength=self.N)
        stable_sum = np.bincount(self.post[stable], self.strength[stable], minlength=self.N)
        fluid_sum = total - stable_sum
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.clip((self.budget - stable_sum) / fluid_sum, 0, 1)
        scale = np.where(total > self.budget, scale, 1.0)
        self.strength[fluid_m] *= scale[self.post[fluid_m]]

        # 2. prune fluid synapses that are weak or unused
        fluid = self.stability < p["stable_at"]
        weak = self.strength < p["prune_below"]
        unused = (self.t - self.last_used) > p["unused_limit"]
        keep = ~(m & fluid & (weak | unused))

        # 3. neurons with no inputs or no outputs left die (unless protected)
        n_out = np.bincount(self.pre[keep], minlength=self.N)
        n_in = np.bincount(self.post[keep], minlength=self.N)
        dying = self.alive & ~self.protected & ((n_out == 0) | (n_in == 0))
        self.alive[dying] = False
        keep &= self.alive[self.pre] & self.alive[self.post]
        pruned = before - int(keep.sum())

        # 4. grow a few new, fluid synapses (reserve for future learning)
        living = np.flatnonzero(self.alive)
        exc = living[~self.inhibitory[living]]
        n_new = int(p["grow_fraction"] * keep.sum())
        new_pre = self.rng.choice(exc, n_new)
        new_post = self.rng.choice(living, n_new)
        ok = new_pre != new_post
        new_pre, new_post = new_pre[ok], new_post[ok]
        n_new = new_pre.size

        # 5. compact and rebuild; tags are cleared overnight
        self._set_synapses(
            np.concatenate([self.pre[keep], new_pre]),
            np.concatenate([self.post[keep], new_post]),
            np.concatenate([self.strength[keep], np.full(n_new, p["w_new"])]),
            stability=np.concatenate([self.stability[keep], np.zeros(n_new)]),
            tag=np.zeros(int(keep.sum()) + n_new),
            last_used=np.concatenate([self.last_used[keep], np.full(n_new, self.t)]),
        )
        return dict(pruned=pruned, grown=n_new, neurons_died=int(dying.sum()))

    # ------------------------------------------------------------------ helpers
    def stats(self):
        m = self.plastic
        nbytes = sum(a.nbytes for a in (self.pre, self.post, self.strength,
                                         self.stability, self.tag, self.last_used))
        return dict(
            t=self.t,
            neurons=int(self.alive.sum()),
            synapses=int(self.pre.size),
            mean_strength=float(self.strength[m].mean()),
            stable_synapses=int((self.stability >= self.p["stable_at"]).sum()),
            synapse_memory_mb=round(nbytes / 1e6, 1),
        )
