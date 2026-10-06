# Partition manifest and cross-method identity verification

Reproducibility artifact (Sections 5.3 and 5.6 of the manuscript). It documents, with
cryptographic fingerprints, that the five methods of the original campaign (FedAvg, FedAvgM,
FedProx, FedHAD, FedNova) operate on **identical client partitions** for every experimental
scenario, which is what licenses the paired, per-seed comparisons reported in the paper.

Two later sets of runs are covered differently:

- **SCAFFOLD** was added after this manifest was built. It uses the same partitioning code, and
  the per-client sample counts recorded in its reports coincide with those of FedNova in all
  390 of its runs (Section 5.6).
- **The control batteries** (component analysis, step-preserving permutation, tuned controls,
  uniform learning rate, global gradient) record the SHA-256 of every client partition in the
  `fingerprint.json` of each run (`partition_hashes`), and their runners verify them. The
  equal-size label-skew partition of the global-gradient diagnostic is a different construction
  and is documented with that battery.

No training and no federated simulation is executed. The script,
`analysis/data_partition_manifest/make_partition_manifest.py`, only imports the partitioning
routines from the per-method scripts in `algoritmos/` and regenerates the partitions; it writes
to this folder. A full run takes several hours, almost all of it in the FEMNIST scenarios.

## Running

```bash
# from the repository root, using the project environment
./env_flwr_pt/bin/python analysis/data_partition_manifest/make_partition_manifest.py

# fast smoke test on a reduced grid
./env_flwr_pt/bin/python analysis/data_partition_manifest/make_partition_manifest.py --quick

# also fingerprint the initial model weights of each method
./env_flwr_pt/bin/python analysis/data_partition_manifest/make_partition_manifest.py --check-init

# skip the FEMNIST scenarios (they dominate the runtime)
./env_flwr_pt/bin/python analysis/data_partition_manifest/make_partition_manifest.py --skip-femnist
```

The script exits with a **non-zero status if any scenario fails any check**, so it is
usable as a regression test in CI, not only as a table generator.

## Grid

| Axis | Values |
|---|---|
| Datasets (Dirichlet) | MNIST, FashionMNIST, CIFAR10 |
| α | 1.0, 0.5, 0.1, 0.01 |
| Clients | 3, 5, 10 |
| Seeds | 42–71 (30 seeds) |
| FEMNIST (natural, per writer) | 10 clients × seeds 42–71 (30) |
| Methods | FedAvg, FedAvgM, FedProx, FedHAD, FedNova |

The FEMNIST configuration mirrors the campaign's `test8_femnist` block exactly
(10 client writers, 30 seeds). This matters: a manifest that documents a
configuration the campaign never runs is worse than no manifest at all. The
manifest's FEMNIST loading path was verified against the campaign's own path
(`FL_DATASET=FEMNIST`) and reproduces it byte for byte — identical per-client
index fingerprints and identical train/validation sizes.

## Outputs (`artifacts/partitions/`)

| File | Content |
|---|---|
| `partition_manifest.csv` | One row per (scenario, method, client): `writer_id`, `n_k`, `n_train`/`n_val`, class coverage, `H_k`, `role`, `excluded_reason`, and SHA-256 of both the client's index list and its label-count vector. This is the raw manifest. |
| `partition_verification.csv` | One row per scenario: whether all methods agree on the index fingerprint and on the label-count matrix, plus both controls. |
| `partition_verification.tex` | `booktabs` table (condensed by dataset and α). Requires `\usepackage{booktabs}`. |
| `femnist_heldout_manifest.csv` | One row per (scenario, held-out writer): the writers forming the centralised FEMNIST test set, with `n_k`, class coverage and `H_k`. Documents the FEMNIST evaluation protocol. |
| `model_init_check.csv` | Only with `--check-init`: SHA-256 of the initial `state_dict` per (architecture, seed, method) for seeds 42, 43 and 44. |

### Column semantics worth knowing

- `partition_granularity` — the Dirichlet rows fingerprint the **full client partition**
  (the 90/10 train/validation split happens downstream, in `load_data`), whereas the
  FEMNIST loader already returns the split, so those rows fingerprint the **training
  split**. The column makes this explicit rather than leaving it implicit.
- `role` — `client` in the main manifest; `heldout_test` in the FEMNIST held-out file.
  The held-out writers are disjoint from the client writers and are recorded once per
  scenario (they are method-independent; their agreement across methods is verified).
- `excluded_reason` — the FEMNIST loader keeps a client slot even when a writer has no
  samples (it attaches empty loaders rather than dropping the writer). Such a row is
  emitted with `n_k = 0` and an explicit reason instead of being silently omitted, so
  the manifest never hides a client.
- `n_train` / `n_val` — the per-writer train/validation assignment (FEMNIST only; empty
  for Dirichlet rows, where the split is downstream).

## What is being verified, and why it holds

The identity is **by construction, not by coincidence**: `dirichlet_split_noniid`
calls `np.random.seed(seed)` as its first executable statement, which makes it a pure
function of `(labels, n_clients, alpha, seed)`. It is therefore immune to how much of
the global random stream each method consumed beforehand — model initialisation,
sampling order, or scheduler warm-up cannot perturb it.

Two controls are reported so the result is not vacuous:

1. **Negative control** — distinct seeds must produce distinct partitions. A degenerate
   partitioner that ignored the seed would otherwise "pass" the identity check trivially.
2. **RNG stress test** — roughly 1.5M draws are consumed from `numpy`, `torch` and
   `random`, and every global generator is reseeded to 999 immediately before
   partitioning. The fingerprint must be unchanged.

The verification has itself been fault-injection tested: perturbing one method's
partitioner (`np.random.seed(seed)` → `np.random.seed(seed + 1)`) makes every affected
scenario fail and the script exit non-zero.

## Notes

- FEMNIST uses the dataset's natural per-writer partitioning, so α does not apply. The
  writer assignment is fixed per setup, but the 90/10 train/validation split is
  seed-dependent, which is why the negative control is still meaningful there.
- `H_k` is the normalised class-imbalance score used throughout the paper: the
  population standard deviation of the per-class counts divided by their mean, mapped
  to [0, 1] by the upper bound √(K−1). It is computed here over each client's full
  partition.

- **FEMNIST `H_k`: this manifest and the campaign's telemetry differ slightly, by
  design.** The campaign computes `H_k` by iterating the client's `DataLoader`, which
  is built with `drop_last=True`, so it sees `floor(n/32) x 32` samples rather than all
  `n`; and because the loader also shuffles, *which* samples fall in the discarded tail
  depends on the global RNG state at iteration time. The logged FEMNIST `H_k` therefore
  carries roughly +/-0.002 of sampling noise. Measured on three writers: a writer whose
  training split is an exact multiple of the batch size agrees to 1.3e-09, while writers
  losing 5 and 15 samples to `drop_last` differ by 4.3e-03 and 2.8e-03. This manifest
  reports `H_k` over the **complete** training split, which is deterministic and is the
  quantity a partition manifest should describe. The Dirichlet datasets are unaffected:
  there the campaign reads `targets` directly over all client indices.
