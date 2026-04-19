# A Point-Cloud Reconstruction Framework for Multi-Subdetector Experiments

**Project:** Point Transformer V3 + Object Condensation pipeline for
unified, grid-free multi-object reconstruction in heterogeneous
detectors.
**Status:** Working end-to-end prototype with two reference tasks
(shapes-on-canvas pseudo-data, toy left/right calorimeter) and a
documented extension path for arbitrary user data.

## 1. Background and motivation

Modern particle-physics detectors are **heterogeneous point clouds**.
A single event produces 10³–10⁵ hits distributed across silicon
trackers, electromagnetic and hadronic calorimeters, Cherenkov / TOF
systems, and muon chambers, each with its own native representation
(hit positions, energies, timing, channel identities). Reconstruction —
mapping this raw hit set back to the list of particles that produced
it, together with each particle's kinematics and identity — has
historically been solved with detector-specific, hand-tuned pipelines
(e.g., Kalman-filter tracking, calorimeter clustering with anti-kT-like
algorithms, particle-flow matching). These pipelines are **accurate
but brittle**: each subsystem has its own algorithm, each algorithm
has its own thresholds, and the stitching layer (particle flow)
propagates errors across subsystems.

A growing body of work (ATLAS, CMS, EIC, belle-II, and HEP in general)
asks a different question: can a single machine-learned pipeline
consume every subsystem's hits as one unified point cloud and produce
the reconstructed particle list directly? Two ingredients have
recently matured enough to make this tractable at scale:

- **Point Transformer V3** (PTv3, CVPR 2024) — a permutation-invariant,
  serialization-based attention backbone that scales linearly in the
  number of points and runs comfortably on 10⁵-point inputs. Prior
  point-cloud transformers (PointNet, DGCNN, PT v1/v2) either did not
  scale to detector-sized events or required dense voxelization that
  wastes the natural sparsity of detector data.
- **Object Condensation** (Kieseler, EPJ C 2020) — a training objective
  and inference procedure that turns per-hit outputs into a
  **variable-cardinality object list in a single forward pass**. Each
  hit predicts a condensation score β and a cluster coordinate `x` in
  a learned low-dimensional space; at inference, high-β hits become
  condensation points and claim nearby hits, producing a ragged list
  of objects with no anchors, no NMS, no upper bound on object count,
  and no threshold sweep per particle class.

Combining the two is natural — PTv3 provides permutation-invariant
per-hit features, OC turns those features into an object list — but
the combination has not been packaged as a **general, extensible
framework** that drops into arbitrary detector data. This project
fills that gap.

## 2. Methodology

### 2.1 Architecture

The model is a U-Net over serialized point attention followed by three
per-hit prediction heads:

```
raw hits (coord, feat, offset)
      │
      ▼  per-subdetector Linear + LN + GELU       lift features into a common width
unified (N, D) token stream
      │
      ▼  Voxelize + serialize (z- / Hilbert-order) bounded positive-quadrant grid
      ▼  PTv3 encoder (4 stages, SpConv pooling)  down-path; serialized attention per-patch
      ▼  PTv3 decoder (3 stages, SpConv unpool)   up-path; skip connections
per-hit features (N, D_out)
      │
      ├── β head   (MLP → sigmoid)         condensation likelihood per hit
      ├── x head   (MLP → R^cluster_dim)   cluster coordinate per hit
      └── payload head(s)                  per-hit class / regression targets
```

**Serialized attention** is the PTv3 innovation that makes this
practical. Rather than computing pairwise attention over `N` points
(quadratic), PTv3 orders points along a space-filling curve (Hilbert,
z-order) and applies windowed attention over sorted chunks. Four
distinct orderings are used within a block so every point attends to
multiple neighborhoods across depth; the union is empirically
equivalent to quadratic attention at a fraction of the compute.

**Object Condensation inference** follows the paper exactly:

1. Order all hits by their learned β.
2. Take the highest-β unassigned hit. If its β exceeds `t_β` (= 0.5 by
   default), it becomes a condensation point.
3. Every still-unassigned hit within Euclidean radius `t_d` (= 0.28 in
   `x`-space) joins that condensation point's cluster, regardless of
   the joiner's own β.
4. Repeat until no unassigned hit has β > `t_β`. Remaining hits are
   classified as noise.

This single-pass, grid-free decoder is what lets the same model
handle events with 3 particles and events with 300 without retuning.

### 2.2 Loss

The objective is a weighted sum of

* **OC attractive / repulsive / coward / noise** — the standard four
  terms of condensation_loss_tiger, which together shape the `x`-space
  so same-object hits pull toward a single condensation point and
  different-object hits repel.
* **Payload terms** — cross-entropy for per-object classification
  targets and MSE for per-object regression targets (energy, momentum,
  size, ...). Payload targets are frame-normalized so their MSE
  magnitudes are comparable to the OC terms without per-head weight
  tuning.

All terms are logged separately in TensorBoard under `train/loss/*`
and `val/loss/*`, so failure modes ("one term is 100× larger than the
others") are immediately visible instead of hidden inside the total.

### 2.3 Why this methodology

**Permutation invariance.** Detectors do not deliver hits in a canonical
order. A model that assumes one will learn spurious positional cues
from whatever ordering the readout happens to use. PTv3's serialized
attention is provably invariant to input permutation at the point
level — a hard guarantee, not a learned one.

**Variable cardinality.** Calorimeter showers may contain 3 cells or
300. Tracker tracks may have 5 hits or 80. Any reconstruction
architecture that fixes the number of output objects a priori must
either over-predict (and downstream NMS) or under-predict (and lose
objects). OC's condensation-point mechanism emits exactly as many
objects as the event contains, with no fixed cap.

**Single model, many subsystems.** Because PTv3 consumes a unified
token stream, a single backbone sees tracker, ECAL, HCAL, and other
hits simultaneously. The model can learn cross-subsystem correlations
(a track that points at an ECAL cluster is different from one that
doesn't) that are normally relegated to a separate particle-flow
stage. The present framework supports heterogeneous feature widths
through per-subdetector linear encoders that project into a shared
dimension, with a learned subdetector-type embedding preserving origin.

**Grid-free outputs.** No anchors, no NMS, no threshold sweeps per
class. The object list falls out of a deterministic single-pass
procedure controlled by two scalars (`t_β`, `t_d`), both of which have
well-understood effects and stable defaults from the original paper.

**Interpretable cluster space.** The learned `x` coordinates are
low-dimensional (default: 2) and directly plottable, so during
training the user watches both the condensation likelihood and the
emerging geometric structure of the clustering in the same TensorBoard
panel. In high-dimensional cases UMAP is available as a non-linear
reduction.

## 3. Implementation

### 3.1 Code structure

The codebase is intentionally small. Roughly 3,000 lines of Python
outside the two vendored submodules (PTv3 upstream, OC upstream).

| Module | Purpose | LOC |
|---|---:|---:|
| `src/tasks/` | `OCTask` abstract base + concrete `ShapesTask`; one subclass per data domain | ~650 |
| `src/models/` | PTv3 wrapper, OC heads | ~250 |
| `src/losses/` | Wrapper around condensation_loss_tiger + payload losses | ~120 |
| `src/inference/` | Paper-correct β-threshold + radius clustering | ~50 |
| `src/training/` | Minimal training loop with TB instrumentation | ~250 |
| `src/utils/` | TB logging, architecture report | ~300 |
| `src/augmentations/` | Per-event augmentation base + 3 reference impls | ~170 |
| `scripts/` | train, evaluate, export_predictions, export_onnx, generate_data | ~500 |
| `examples/calorimeter/` | Second reference task (toy detector) | ~400 |

### 3.2 Extensibility: the `OCTask` contract

The central abstraction is [`OCTask`](../src/tasks/base.py), a class
that bundles three responsibilities a user must implement for their
domain: data iteration, batch collation, and per-event visualization
(truth / prediction / OC-space panels). Once a user writes an
`OCTask` subclass for their format, the trainer, TB logger,
evaluator, and ONNX exporter all run unchanged.

The framework ships **two reference `OCTask` subclasses** specifically
to let new users learn by diff:

* **`ShapesTask`** — 2D "pixels on a canvas" pseudo-detector, with
  width/height regression and shape-class classification targets.
* **`CalorimeterTask`** (under `examples/`) — 3D toy calorimeter with
  two walls, per-particle energy regression, a natural side-by-side
  detector rendering.

A step-by-step onboarding tutorial ([`docs/custom_task.md`](custom_task.md))
walks a first-time user from "I have my own data" to "the model is
training and predictions are being written back to my HDF5 with full
provenance" using the calorimeter as its running example.

### 3.3 Operational surface

A complete experiment cycle on new data is four commands:

```
python <generator>                                         # → HDF5
python scripts/train.py --config <runcard>                 # → TensorBoard, checkpoints
python scripts/evaluate.py --config <runcard> --checkpoint <ckpt>  # → eval.json
python scripts/export_predictions.py --config <runcard> --checkpoint <ckpt>  # → HDF5 + provenance
```

The provenance export is notable: per-hit predictions are appended
into the source HDF5 under a named subgroup, and a sibling
`predictions_meta` group records the checkpoint path, entire runcard
YAML verbatim, UTC timestamp, git SHA, hostname, and inference
thresholds. Analyses that load predictions months after the fact can
answer "which model produced these numbers?" without reconstruction.

### 3.4 Augmentations

A minimal augmentation framework ([`src/augmentations/`](../src/augmentations/))
gives users a callable contract (one event in, one event out) with
three ready-to-use implementations (random rotation, per-hit and
per-shape feature jitter, random hit dropout) and a template for
writing task-specific augmentations. The contract enforces per-hit
tensor alignment (if N changes, it changes for every per-hit tensor)
so augmentations cannot silently corrupt the clustering targets.

### 3.5 Logging and diagnostics

TensorBoard instrumentation is comprehensive and opinionated. Every
run produces:

* **Scalars** — every loss component under `train/loss/*` and
  `val/loss/*`, plus `train/grad_norm` and `train/beta` histogram. The
  grouping is deliberate: "one loss term dominating" is the most common
  failure mode, and it is immediately visible.
* **Images → `viz/grid`** — a fixed 4×4 panel of val events, one cell
  per event, rendered by the task's `plot_truth` / `plot_pred` /
  `plot_oc` methods. Users watch the same events evolve across training
  steps, which makes convergence visually obvious.
* **Projector** — PCA / UMAP / t-SNE scatter of cluster coordinates
  labeled by truth object id.
* **Architecture report** — auto-generated Mermaid flowchart +
  per-module parameter counts, written to `<log_dir>/architecture/` on
  every run.
* **Text overview** — a markdown README explaining every panel is
  written into the TB Text tab on run start.

## 4. Outcomes and next steps

**What works now.** The pipeline trains end-to-end on both reference
tasks, produces correctly-formed OC outputs, exports heads to ONNX
for inference deployment, and writes timestamped predictions back
into user HDF5 files. The test suite exercises the data path
(generator → dataset → collate), the loss, and a small PTv3 forward
pass where CUDA is available.

**Near-term work.**

* **Real detector case study.** Integrate one real subsystem (e.g. the
  EIC ECAL barrel or the CMS HCAL barrel readout) to validate
  performance-parity with existing reconstruction at event scales
  where the latter is known-good.
* **Vectorized per-event inference.** The current `beta_threshold_cluster`
  is per-event in Python for clarity; batched GPU implementation is a
  straightforward kd-tree / scatter optimization.
* **Multi-subdetector token mixing.** The per-subdetector encoder + type
  embedding scaffolding is present but exercised only on single-subsystem
  toy data; a formal study of encoder-width ratios on a real mixed event
  is the natural first physics paper.
* **Active-learning-style event curation.** OC's per-hit β score is a
  natural uncertainty measure; flagging low-β events for human review
  or targeted simulation is a simple extension.

**Broader impact.** The framework is detector-agnostic and
domain-agnostic — any variable-cardinality object-from-point-cloud
problem (autonomous driving LiDAR, point-cloud segmentation, particle
physics, bio-imaging) can slot in a task class and reuse the rest.
The onboarding guide, runnable second example, and provenance-tracked
prediction export lower the cost of reproducing and extending results,
which we see as a prerequisite to the framework being useful outside
of a single lab.

## 5. References

1. Wu, X. *et al.* Point Transformer V3: Simpler, Faster, Stronger.
   *CVPR 2024*. [arXiv:2312.10035](https://arxiv.org/abs/2312.10035).
2. Kieseler, J. Object condensation: one-stage grid-free multi-object
   reconstruction in physics detectors, graph, and image data.
   *Eur. Phys. J. C* **80**, 886 (2020).
   [doi:10.1140/epjc/s10052-020-08461-2](https://doi.org/10.1140/epjc/s10052-020-08461-2).
3. McInnes, L., Healy, J., Melville, J. UMAP: Uniform Manifold
   Approximation and Projection for Dimension Reduction, 2018.
   [arXiv:1802.03426](https://arxiv.org/abs/1802.03426).
