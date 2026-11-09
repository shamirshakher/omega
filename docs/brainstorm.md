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
5. **Starting level.** Do we start from random code (Computational Life), from simple cells, or from organisms that already have neurons?
6. **Measurement.** How do we detect that intelligence has emerged as a by-product?

## Open questions

- What is the minimum "chemistry" (instruction set) for the first self-replicator?
- What is the "world" (substrate and physics) that Omega's organisms live in?
- When and how do neurons and plasticity rules first appear: are they evolved, or seeded?
- How is a cause → effect link represented in an evolved brain?
- How do we test "learns like a child"? We need a Baby Test benchmark for evolved organisms.
