# Omega — Brainstorm

> **Omega — a model of cause and effect.**

## Goal

Build **Omega**, an AI that learns continuously, the way a child does: from a small stream of its own experience, in real time, without retraining on a huge dataset.

## The problem with today's LLMs

| Today's LLMs | What Omega should do |
|---|---|
| Learn once, in an expensive pretraining run | Learn all the time, during use |
| Need trillions of tokens | Learn from a small amount of data, the way a child does |
| Weights are frozen after deployment | Every interaction can update knowledge |
| Updates mean retraining or fine-tuning | Updates happen incrementally and locally |
| Learn by passively predicting text | Learn by acting and seeing what happens |
| Have no grounded model of cause and effect | Cause and effect *is* the model |

## Guiding principle: the Baby Test

> **If a human baby doesn't learn this way, the approach fails.**

Every idea, existing or new, is judged against how a baby actually learns:

1. **Starts from zero knowledge.** It has built-in reflexes and drives, but no pretrained knowledge and no dataset.
2. **Learns from its own experience.** It doesn't take in a huge corpus.
3. **Learns by acting.** It does something and observes the effect: *cause → effect*.
4. **Learns in real time.** There are no separate training and use phases.
5. **Learns from very few examples.** One hot stove is enough.
6. **Doesn't forget old knowledge when learning new things.** New knowledge builds on old knowledge.
7. **Is one integrated system.** There is no frozen brain with a lookup layer bolted on.

## Existing approaches: evaluated with the Baby Test

| # | Approach | How it works | Baby Test | Why |
|---|---|---|---|---|
| 1 | Classic continual learning (EWC, replay, growing nets) | Protects old weights or replays old data while training on new tasks | ❌ Fail | It patches a network trained on datasets. Babies don't learn in "tasks" or replay datasets. It reduces forgetting rather than solving it. |
| 2 | Test-time training / fast weights (TTT, Titans) | A pretrained model makes small weight updates while it runs | ❌ Fail | It depends on a model pretrained on mass data. The updates are temporary, on top of frozen knowledge. |
| 3 | External memory / RAG / skill libraries | A frozen model plus a lookup store | ❌ Fail | It is a second layer on a trained model. Nothing is learned internally. A baby doesn't "look up" a fact. |
| 4 | World models (JEPA, BabyZWM) | Learns by predicting the next frame or representation from recorded video | ❌ Fail | The learning is passive, offline and batch-trained, even on "child" video. Predicting what comes next is not understanding *why* it happens. |
| 5 | Developmental robotics / curiosity | An agent explores and is rewarded for novelty | ⚠️ Partial | The principle is right: babies act and explore. The implementations are wrong: they need millions of trial steps and backprop, and curiosity is usually defined as prediction error. |
| 6 | Brain-inspired (fast + slow memory, Hebbian, neuromodulation) | Copies mechanisms found in the brain | ⚠️ Partial | The mechanisms come from real brains, but in practice they are usually add-ons to standard deep networks rather than a system built from them. |

### Common flaws

- **They build on a pretrained model.** Almost every approach starts from a model already trained on vast data, then adds a layer such as RAG, memory or fast weights. **This won't work for Omega.** Omega has to start from zero.
- **Prediction is not the solution.** Predicting the next token or frame is correlation, not cause and effect. A model can predict what comes next without knowing why it happens.
- **They learn passively.** They observe data instead of acting on the world and seeing the result.
- **Training and use are separate phases.** A baby has no "training phase."

### Worth keeping (as ideas, not implementations)

- **Built-in structure, learned content.** Babies are born with reflexes, drives and learning machinery, not knowledge.
- **Acting and exploring** as the main way of learning (from approach 5).
- **Fast capture, slow consolidation**, like a baby sleeping on what happened that day (from approach 6).

## Omega approach

Omega has two parts. Part 2 is the core of the solution.

### Part 1: The DNA — ~800MB that builds a brain

The human genome is about 3.1 billion base pairs. At 2 bits per base pair, that comes to **~750–800MB**. This is all the "pre-existing data" a baby starts with. When it is "deployed", it builds a system with inputs (senses) and outputs (muscles), and that system builds intelligence through experience.

**Key insight: the genomic bottleneck.** The brain has about 86 billion neurons and roughly 100 trillion synapses. 800MB cannot hold a wiring diagram that size. So the DNA does not store knowledge or weights. It stores **rules that build and change a brain** (Zador, *A critique of pure learning*, 2019):

| What the DNA encodes | Neural structure it produces |
|---|---|
| Cell types | Neuron types: excitatory, inhibitory and modulatory, each with its own behaviour |
| Growth and wiring rules | Brain regions, layers, cortical columns, how neurons connect (repeated templates, not individual wires) |
| Learning rules | Synaptic plasticity: Hebbian ("fire together, wire together"), STDP (timing → cause and effect at the neuron level) |
| Drives and reflexes | Hunger, pain, pleasure, curiosity, grasping and sucking reflexes. These are the *purpose* signals |
| Modulators | Dopamine, serotonin and similar signals that set *when* and *how much* to learn |
| Developmental program | Grow too many connections, then prune what isn't used, with critical periods |

**Implication for Omega:** Omega's "DNA" is a compact program that **builds and rewires a network**. It is not a trained model and it has no weights to load. The intelligence comes from running that program inside a world.

### Part 2: How the 800MB was written — evolution (the solution)

Nobody designed the DNA. It was written by evolution over ~4 billion years, starting from the first self-replicating RNA molecule. Evolution had a single purpose:

> **Consume & Replicate.**

Nothing set intelligence or consciousness as a goal. They emerged as **by-products**, because organisms that modelled cause and effect better consumed and replicated better.

**So Omega's answer is not to design intelligence, but to recreate the process that produced it.**

```
OUTER LOOP: evolution (slow, across generations)  →  writes the DNA
   │   selection pressure: Consume & Replicate
   ▼
INNER LOOP: a lifetime (fast, real time)  →  the DNA builds a brain; the brain learns from experience
   │   this is where the Baby Test applies
   ▼
Intelligence  =  a by-product that improves survival and replication
```

- The **Baby Test** judges the *inner loop*, the individual organism.
- **Evolution** is the *outer loop*: it searches for a DNA whose babies pass the Baby Test.

### Related research for Part 2 (to study, not to copy)

| Work | What it shows | Relevance |
|---|---|---|
| **Computational Life** (Agüera y Arcas et al., Google, 2024) | Self-replicating programs arise *spontaneously* from random code soups, with no fitness function | The closest thing to "first RNA replicator" in code |
| **Tierra** (Ray, 1991), **Avida** | Digital organisms compete for CPU and memory, replicate and evolve parasites and complexity | "Consume & Replicate" as the only goal; complexity plateaus |
| **NEAT / HyperNEAT** (Stanley) | Evolves networks; HyperNEAT uses an *indirect* encoding (a compact rule → large network) | The DNA → brain idea in Part 1 |
| **Evolved plasticity** (Najarro & Risi, 2020) | Evolves *learning rules*, not weights; random networks then self-organize during a lifetime | DNA encodes learning rules, not knowledge |
| **AI-Generating Algorithms** (Clune, 2019), **POET** | Proposes evolving the architecture, the learning rules *and* the environments | The research agenda closest to Omega |
| **Lenia / artificial life** | Lifelike patterns emerge from simple local rules | Emergence from simple rules |

**Known gap:** every artificial-life system so far **plateaus**. None has achieved open-ended evolution that keeps producing new complexity. Solving open-endedness is the core research problem for Omega.

### Key challenges

1. **Compute.** Real evolution ran for ~4 billion years in parallel across a whole planet. Omega needs shortcuts, such as starting above the level of chemistry or evolving rules instead of weights.
2. **The environment matters as much as the DNA.** Intelligence only pays off in a world that is rich, changing and competitive. Who builds that world, and how complex must it be?
3. **Resource scarcity is required.** "Consume" only creates pressure if resources are limited.
4. **Open-endedness.** How do we keep evolution from settling on a "good enough" simple replicator?
5. **Starting level.** *Decided:* start from what we already know about the brain, not from the first RNA molecule. See Part 3.
6. **Measurement.** How do we detect that intelligence has emerged as a by-product?

### Part 3: The shortcut — start from what we know about humans

Evolution has already done ~4 billion years of work, and the result is the human brain. We know a lot about it, so we don't have to start from RNA.

**Analogy: directed evolution** (Frances Arnold, Nobel Prize 2018). Protein engineers don't evolve proteins from scratch. They start from a known protein that works and evolve it toward a goal. Omega does the same with a brain.

**Rule: our knowledge goes into the DNA, never into the brain.**

| Allowed ✅ | Not allowed ❌ |
|---|---|
| Use neuroscience to design the starting genome: neuron types, circuits, plasticity rules, drives | Load knowledge, facts, language or trained weights into the brain |
| Start from structure the way evolution left it | Start from a pretrained model (this fails the Baby Test) |

**What we know well, and what we don't:**

| Known: seed it | Unknown: evolve it |
|---|---|
| Neuron types and how they behave | The exact wiring rules of the human cortex |
| Cortical layers and repeated columns | How the cortex actually computes |
| Plasticity rules: Hebbian, STDP, dopamine-gated learning | How the parts combine into a mind |
| Major circuits: thalamus → cortex (perception), basal ganglia (choosing actions), hippocampus (episodes), cerebellum (motor control) | The right sizes, ratios and parameters |
| Drives, reflexes, and development by overgrowing then pruning | — |
| What a baby's world looks like: a body, objects, a caregiver | — |

The known parts become the **seed genome**. Evolution searches **only the unknown parameters**. The search space shrinks from "all possible life" to "variations of a brain that already works."

**Lesson from failure: structure alone isn't enough.** The Blue Brain and Human Brain projects spent over €1B simulating brain structure in detail, and no intelligence emerged. Their simulations had no body, no drives, no world and no development. Omega's seed must include all of these, not just the wiring.

### Principle: extract the pattern, don't copy the brain

> **We are not emulating or evolving a fly, a mouse or a human. We are building Omega, the last AI.**

Mapped brains are **study material**. From each one we extract the **pattern of learning**: *how* it learns, not *what* it is made of. Omega then recreates those patterns in its own way, with its own design and without biological baggage.

- ❌ Run a connectome, simulate neurons one for one, add learning to a fly model
- ✅ Ask what learning pattern the brain uses and why it works, then design Omega's own version

Simple brains show the core patterns most clearly. More complex brains show what gets added on top of them.

### Learning patterns extracted from the fly brain

The fly is a good first subject: its whole brain is mapped, and it learns cause → effect, often in one trial. Its learning centre (the **mushroom body**) shows these patterns:

**Adopted as core for Omega: patterns 2, 3, 4 and 5.** Pattern 1 must not be hard-coded. Sparse, separate codes should *emerge* from flexible neurons through competition and pruning (see Part 4).

| # | Pattern in the fly | Why it works | What it means for Omega (our own design) |
|---|---|---|---|
| 1 | **Expansion + sparse coding.** ~50 smell-input channels fan out to ~2,000 Kenyon cells, and only ~5% fire for any one smell | Each experience gets its own nearly non-overlapping code, so new learning doesn't overwrite old learning | ⚠️ Don't hard-code it. Let sparse codes *emerge* from overgrowth + pruning (Part 4) |
| 2 | **Three-factor learning.** A connection changes only when the input is active, the output is active *and* a value signal (dopamine) arrives | Learning is local and gated: no global training pass, no backprop | Learning is triggered by *meaningful events*, not by every input |
| 3 | **One-shot learning.** One smell + sugar pairing creates a lasting memory | Strong value signals produce big, immediate changes | Omega learns from single experiences. The size of the change scales with how much the event matters |
| 4 | **Timing = causality.** Smell *before* shock teaches "avoid". Smell *after* shock teaches "relief" (the opposite) | The order of events encodes what caused what | **Cause → effect at the most basic level:** Omega records *which came first*, not just what happened together |
| 5 | **Innate and learned pathways in parallel.** The lateral horn holds built-in reactions; the mushroom body holds learned ones | Learning adjusts reflexes rather than replacing them, so the fly is never "untrained" | Omega is born with working drives and reflexes, and learning modulates them |
| 6 | **Separate value channels.** Different dopamine neurons for reward and punishment, each acting on its own compartment | Different kinds of value are learned independently | Several value signals in parallel (food, danger, novelty), not a single reward number |
| 7 | **Several memory timescales.** Short-, mid- and long-term memories live in different compartments | Fast memories capture events; slow ones keep what lasts | Fast capture → slow consolidation (matches the brain-inspired insight) |
| 8 | **Active forgetting.** Dopamine also *erases* memories that are no longer useful | Keeps memory current and avoids clutter | Forgetting is a feature: Omega needs a deliberate forgetting mechanism |

### Reference: which brains are mapped (Nov 2026)

| Organism | Neurons | Status | What to study |
|---|---|---|---|
| Worm (*C. elegans*) | 302 | ✅ Complete | Reflexes, the simplest drives |
| Fly larva | ~3,000 | ✅ Complete (2023) | The minimal learning circuit |
| **Adult fruit fly** | **~139,000** | ✅ Complete (FlyWire, 2024) | **Core learning patterns (above)** |
| Fly full CNS | ~166,000 | ✅ Complete (Janelia, 2026) | How the brain connects to the body |
| Larval zebrafish | ~100,000 | 🟡 In progress | First vertebrate patterns |
| Mouse | ~70 million | 🔴 1 mm³ of cortex only | Cortex and hippocampus patterns, from neuroscience research |
| Human | ~86 billion | 🔴 Fragments | Language, reasoning, social learning, from neuroscience research |

## Part 4: First approach — overgrow, then prune

### How human neurons work (the essentials)

| Part | What it does |
|---|---|
| **Dendrites** | Receive signals from up to thousands of other neurons |
| **Cell body** | Adds up the incoming signals. When the total crosses a threshold, the neuron **fires a spike**, all or nothing |
| **Axon** | Carries the spike to other neurons |
| **Synapse** | The connection between two neurons. Each neuron has ~7,000 of them. Synapses are excitatory (push the next neuron to fire) or inhibitory (hold it back) |
| **Synaptic strength** | Changes with use. Fire together → stronger (LTP); out of sync → weaker (LTD) |
| **Spike timing (STDP)** | If A fires *just before* B, A→B gets stronger. If A fires *after* B, it gets weaker. **Order = causality** |
| **Neuromodulators** | Dopamine and similar signals that tell the brain "this mattered, learn now" |
| **Myelin** | Insulation that wraps paths used often. They become faster and more stable |

### How a baby's brain develops: overgrow → compete → prune → stabilize

1. **Overproduce.** The brain makes far more neurons and connections than it will keep. About half of the neurons made before birth die before birth. After birth, connections explode: a 2–3-year-old has about **50% more synapses than an adult**.
2. **Compete.** Neurons and synapses compete for limited resources (growth factors). Active ones get the resources; inactive ones don't.
3. **Prune: "use it or lose it."** Unused connections are removed, a process that continues through adolescence. Neurons that never become part of a working circuit die.
4. **Stabilize.** Frequently used paths become strong and myelinated: **stable patterns**. These are knowledge and skills.
5. **Keep some flexibility.** Adults keep a reserve of plasticity, and new synapses keep forming, so learning never stops.

**The brain isn't built up like a wall. It's carved like a statue:** start with a block (dense random connections) and let experience remove what isn't needed.

**Link to Part 2 (evolution):** this is **evolution inside the brain**. Connections compete for resources ("Consume"), and the useful ones survive and are reinforced. Neuroscience calls this *Neural Darwinism* (Edelman) and *selective stabilization* (Changeux). The same selection principle works at both scales: species over millions of years, and connections over a lifetime.

### Omega v0: the overgrow-and-prune design

| Stage | What Omega does |
|---|---|
| **Birth** | A large pool of flexible neurons with dense, random connections, plus innate drives and reflexes (pattern 5) |
| **Experience** | Signals flow through the network as Omega acts in its world |
| **Strengthen** | Connections strengthen when neurons fire in order (pattern 4: cause → effect), mostly when a value signal says "this mattered" (patterns 2 and 3) |
| **Prune** | Connections that are unused or useless weaken and are removed. Neurons that are never used die |
| **Stabilize** | Connections used often become stable and resistant to change. **This protects old knowledge without hard-coding it** |
| **Reserve** | Keep a pool of flexible connections, and keep growing new ones, so new learning is always possible |

**What should emerge rather than be designed:** sparse codes, separate memories, circuits and skills.

**Related work and how Omega differs:** the *Lottery Ticket Hypothesis* (2019) showed that a large random network contains small subnetworks that work well. But it prunes *after* offline training with backprop. Omega prunes *during* life, driven by experience, with local learning rules.

### Decision: Omega uses a neuron + synapse model

Since overgrowth and pruning happen mainly in **synapses**, Omega needs **synapses as their own objects**, each with its own state that can grow, change, stabilize and be removed. They cannot be numbers hidden inside a weight matrix. What we keep from biology is the *behaviour*, not the chemistry.

**Neuron (draft):** adds up its inputs → fires when it crosses a threshold → can die if it's never used.

**Synapse (draft):**

| Property | Meaning |
|---|---|
| `strength` | How strongly it drives the next neuron (positive = excitatory, negative = inhibitory) |
| `stability` | How protected it is from change (0 = fluid, 1 = locked) |
| `tag` | A temporary mark: "I was just active in something that may matter" |
| `last_used` / `age` | For "use it or lose it" pruning |

### What makes a connection stable? (answers from neuroscience)

| # | Brain mechanism | What happens | Omega rule |
|---|---|---|---|
| 1 | **Synaptic tagging and capture** (Frey & Morris, 1997) | An active synapse gets a temporary **tag**. If an important event (e.g. a dopamine signal) happens within a time window, proteins arrive and the change is made **long-term**. Otherwise it fades within hours | **Two keys to lock a connection:** local activity (the tag) + a global "this mattered" signal. Activity alone → temporary change that fades |
| 2 | **Spaced repetition** | Reactivating a memory over spaced intervals makes it far more stable than cramming | Each spaced reuse raises `stability` |
| 3 | **Sleep consolidation** (Tononi & Cirelli) | During sleep the brain replays the day's experiences and **scales down all synapses**. Strong ones survive; weak ones fall below threshold and are pruned | A periodic **sleep phase:** replay tagged experiences, scale everything down, prune what falls too low, stabilize what remains |
| 4 | **Spine size = stability** (Yang, Pan & Gan, 2009) | Large dendritic spines persist for months or a lifetime; small ones come and go within days | Strong, old synapses are hard to change; weak, new ones stay fluid |
| 5 | **Reconsolidation** (Nader et al., 2000) | Recalling a stable memory makes it **unstable again for a while**, especially if the outcome doesn't match. It can then be updated and restabilized | **The anti-freeze rule:** when a stable cause → effect link is used and the expected effect *doesn't happen*, it unlocks for updating |
| 6 | **Critical periods + perineuronal nets** | Early life is highly plastic. Then "nets" wrap mature circuits and lock them. Removing the nets (in animal experiments) reopens plasticity | Omega starts highly plastic and gradually locks core circuits. The locks are **releasable**, not permanent |
| 7 | **Metaplasticity + homeostasis** (BCM theory) | The threshold for change shifts with a neuron's history, and neurons keep their total activity within a range | Limits that prevent runaway strengthening or a silent network |
| 8 | **Competition for resources** (BDNF) | Active synapses capture limited growth factors; inactive ones starve and are removed | "Consume" at the synapse level: limited resources, so only useful synapses survive |

**Answer to "freeze or forget":**
- **Against forgetting:** the two-key lock (1), spaced reuse (2), sleep consolidation (3) and growing stability with strength and age (4).
- **Against freezing:** reopening links when the expected effect doesn't happen (5), releasable locks (6), and a constant reserve of new, fluid synapses.

### How neurons and synapses are stored in code

**The brain is a sparse graph.** Neurons are nodes and synapses are edges. Each neuron connects to only a tiny fraction of the others, and the number of synapses differs from neuron to neuron. So:

- ❌ **One object per synapse** (e.g. a Python object): easy to understand, but ~100+ bytes of overhead each and very slow at millions of synapses.
- ❌ **Dense N × N matrix:** 99.99% of it would be empty, and it doesn't fit growth and pruning.
- ✅ **A table of synapses stored as columns** (structure of arrays): one row per synapse, one array per property. Synapse `i` is index `i` across all the arrays. This is how spiking-network simulators (Brian2, NEST, GeNN) store synapses.

**Neuron table:**

| id | potential | threshold | last_spike | alive |
|---|---|---|---|---|
| 0 | 0.42 | 1.0 | t=1203 | ✅ |
| 1 | 0.05 | 1.0 | t=987 | ✅ |

**Synapse table:**

| i | pre (from) | post (to) | strength | stability | tag | last_used |
|---|---|---|---|---|---|---|
| 0 | 0 | 17 | +0.80 | 0.9 | 0.0 | t=1203 |
| 1 | 0 | 52 | −0.30 | 0.1 | 0.6 | t=1190 |
| 2 | 1 | 17 | +0.15 | 0.0 | 0.0 | t=400 |

Each column is one flat array (NumPy or GPU tensor):

```python
# neurons
potential  = np.zeros(N, dtype=np.float32)
threshold  = np.ones(N,  dtype=np.float32)
alive      = np.ones(N,  dtype=bool)

# synapses (one entry per synapse)
pre        = np.array([...], dtype=np.uint32)   # from-neuron
post       = np.array([...], dtype=np.uint32)   # to-neuron
strength   = np.array([...], dtype=np.float16)  # + excitatory / - inhibitory
stability  = np.array([...], dtype=np.float16)  # 0 = fluid, 1 = locked
tag        = np.array([...], dtype=np.float16)  # decays over time
last_used  = np.array([...], dtype=np.uint32)
```

**Key operations:**

| Operation | How |
|---|---|
| Neuron X fires → find its outgoing synapses | Keep synapses sorted by `pre`, plus an `offsets` array: neuron X's synapses are `offsets[X] : offsets[X+1]` (CSR format) |
| Grow a synapse | Write into a free slot (a free list) or append |
| Prune a synapse | Mark the slot dead and add it to the free list. Compact and re-sort periodically, e.g. during **sleep** |
| Learning updates | Vector operations on whole columns at once (fast on CPU or GPU) |

**Event-driven processing:** only ~1–5% of neurons fire at any moment, so the work per step is just "the neurons that fired × their synapses", not the whole network. This is why brains (and Omega) can be large and still cheap.

**Memory budget:** about **~18 bytes per synapse** (pre 4 + post 4 + strength 2 + stability 2 + tag 2 + last_used 4).

| Scale | Neurons | Synapses | Memory | Hardware |
|---|---|---|---|---|
| Prototype | 10,000 | 10 million | ~180 MB | Laptop |
| Fly-scale | 100,000 | 100 million | ~1.8 GB | Laptop |
| Large | 1 million | 1 billion | ~18 GB | Workstation / GPU |
| Human-scale | 86 billion | ~100 trillion | ~1.8 PB | Not feasible today |

Overgrowth means starting at roughly 1.5× the target number of synapses and letting pruning bring it down.

**Implementation path:** prototype in Python + NumPy (or PyTorch for GPU). Move the hot loop to Rust/C++/CUDA once the design is stable.

**Risks to solve:**
- **Too much stability** → Omega can't learn new things. Too little → it forgets. Getting the balance right is the stability–plasticity problem again, now handled by a mechanism instead of a fixed rule.
- **The timing of pruning** (critical periods). If it prunes too early, Omega loses abilities it will need later.
- **Compute.** Starting overgrown means starting large. We need to find the smallest "block" that still works.

## Part 5: Omega v0 — first code

**Code:** `omega/core.py` (the brain) and `experiments/first_light.py` (the first test).
**Brain:** 2,000 neurons (20% inhibitory) and 200,000 synapses (overgrown), stored as column arrays. It runs at ~1.5 ms per step on a 2015 dual-core i5.

**What's implemented:** spiking neurons, threshold adaptation (homeostasis), timing → tags (STDP), dopamine as the second key (tag → strength + stability), and sleep (consolidation, downscaling, synaptic scaling, pruning, neuron death, growth of new synapses).

**First test (a fly-style, one-shot task, Omega's own brain):** smell A + sugar, once. Then test whether A alone triggers "approach".

| Stage | A | B (control) | Nothing |
|---|---|---|---|
| Newborn | 13 | 21 | 12 |
| After **one** A + sugar pairing | 1501 | 587 | 211 |
| After sleep | **380** | 15 | 9 |
| After learning B + sleep | **840** | **481** | 23 |

(Approach spikes in 200 ms. The results hold across 4 random seeds.)

✅ One-shot learning ✅ Specific (B unaffected) ✅ Survives sleep ✅ Learning B doesn't erase A

### Lessons from building it (each one matched real brain biology)

1. **Weak inhibition → seizure.** The first run became epileptic: one smell set off self-sustaining activity across the whole network. The fix was strong inhibition, as in cortex, plus threshold adaptation.
2. **Neurons that fire together form loops.** The approach neurons wired to each other and kept firing with no input. **Synaptic scaling** (a fixed input budget per neuron) fixed it. It's competition for resources: "Consume" at the synapse level.
3. **Right after learning, the brain is over-excited; sleep cleans it up.** Sleep removed the noise and kept the memory.
4. **Without consolidation, new learning erased old learning.** Making strong synapses more stable during sleep ("big spines last") fixed it.
5. **"Use it or lose it" has a cost.** B's paths were partly pruned before B was ever learned. This is the critical-period effect, and it needs tuning.

### Timing test: cause → effect, not just "together" (`experiments/timing_test.py`)

Same newborn, one trial each. Each protocol has 75 ms of smell, 50 ms of sugar and 25 ms of overlap. **Only the order differs.**

| Protocol | A response after sleep (newborn ≈ 13) | Strength A → approach (newborn 0.05) |
|---|---|---|
| **Forward** (smell → sugar) | **288** | **0.26** |
| Backward (sugar → smell) | 19 | 0.07 |
| Unpaired (1 s apart) | 15 | 0.07 |

The results hold across 3 seeds. ✅ **Omega learns that A *causes* sugar only when A comes first.** The same events in the reverse order teach almost nothing. Like the fly, backward pairing gives weak or no learning.

**Lesson: a fair test matters.** The first version gave the backward trial twice as much sugar. The result looked like backward learning (4,000+ spikes), but it was really the approach neurons wiring into a loop. Once every protocol had the same sugar and overlap, the order effect was clean.

### Known issues / next

- **Approach → approach loops strengthen in every protocol** (0.05 → 0.16–0.31) because sugar makes those neurons fire together. Sleep keeps it in check for now, but it's a recurring risk.
- After learning B, A's response also rose (380 → 840). The approach group may be getting generally more excitable. Watch for this.
- Only one value channel (sugar). Backward pairing gives "nothing", not the fly's opposite ("relief") learning. That would need a second channel (danger/relief).
- Reconsolidation (unlock a link when its expected effect fails) is not implemented yet.
- Inhibitory synapses don't learn yet.

## Open questions

- Which brain parts and rules go into the seed genome, and which are left for evolution to find?
- Which learning patterns are core to Omega, and which are fly-specific details?
- What patterns do the mouse and human brains add on top of the fly's (cortex, hippocampus, language)?
- What are the right time windows: how long does a tag last, and how often does Omega sleep?
- Is the importance signal ("this mattered") innate (drives) only, or can it be learned?
- How dense is "overgrown", and how much pruning is right?
- What is the "world" (substrate and physics) that Omega's organisms live in?
- When and how do neurons and plasticity rules first appear: are they evolved, or seeded?
- How is a cause → effect link represented in an evolved brain?
- How do we test "learns like a child"? We need a Baby Test benchmark for evolved organisms.
