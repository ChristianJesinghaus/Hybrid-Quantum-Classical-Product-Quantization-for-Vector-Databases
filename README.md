# Hybrid Quantum Product Quantization k‑NN (PQ‑kNN)

*A prototype that re‑formulates **Product Quantization (PQ)** and its **k‑NN** lookup within a quantum‑information setting.*

**What’s inside**

- **Classical baseline**: PQ‑kNN with Euclidean sub‑distances and scikit‑learn K‑Means (for comparison).
- **Hybrid‑quantum variant**: replaces Euclidean sub‑distances with **log‑fidelity** where the fidelity is estimated via the **SWAP test** (Qiskit Aer), and trains sub‑codebooks with a safeguarded **Quantum K‑Means** routine.
- **Thesis**: The approach and experiments are documented in the bachelor’s thesis PDF.
- **Result highlight (Digits 64‑D)**: The hybrid model approaches the classical accuracy (~95% at `M = 300`).

---

## Quick start

### 1) Environment

~~~bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
~~~

**Dependencies:** `numpy`, `scikit-learn`, `qiskit`, `qiskit-aer`, `pandas`, `matplotlib`, `seaborn`

---

### 2) Prepare dataset

~~~bash
python create_digits_npz.py
~~~

Generates `example_data.npz` from scikit‑learn’s Digits dataset (64‑D, ~1800 samples).

---

### 3) Configure

Edit `config.txt` (simple `key=value` format). Example:

~~~txt
# data split & file
train_size = 300
test_size  = 60
data_file  = "example_data.npz"
normalize_data = true

# PQ / classifier
n = 8
c = 10
k = 9
k_clusters = 10

# algorithmic path
algorithm = "quantum"          # "classical" or "quantum"
distance_metric = "log_fidelity"   # "log_fidelity" | "one_minus_fidelity" | "swap_test" | "1-f" | "lf" | "logf" | "omf"

# quantum controls
quantum_shots = 7000
max_iter_qk = 100
~~~

See `config.txt` for more options.

---

### 4) Train & evaluate

~~~bash
python hybrid_quantum_example.py
~~~

- **Classical path**: per‑partition K‑Means (scikit‑learn).
- **Quantum path**: per‑partition **Quantum K‑Means** with **log‑fidelity** distances (SWAP test via Qiskit Aer).
- **Outputs**: model, config, results, confusion matrices, and histories saved under `experiments/`.

---

### 5) Test saved models interactively

~~~bash
python test_saved_model.py
~~~

Reloads a saved model, lets you adjust `k` (and distance metric for quantum), and shows the confusion matrix & accuracy.

---

### 6) Confusion matrix for classical runs

~~~bash
python confusion_classical.py --model_dir experiments/models/<your_model_dir>
~~~

Generates a **normalized** confusion matrix PNG.

---

## Components

- **Classical PQ‑kNN** — `ProductQuantizationKNN`  
  Parallelized partitions, scikit‑learn K‑Means, inertia histories.
- **Quantum distance** — `QuantumDistanceCalculator`  
  SWAP‑test **fidelity**, **log‑fidelity**, and **one‑minus‑fidelity**.
- **Quantum K‑Means** — `QuantumKMeans`  
  Rayleigh–Ritz eigenvector update + Riemannian fallback.
- **Hybrid PQ‑kNN** — `QuantumProductQuantizationKNN`  
  Quantum distance for compression & prediction, exports partition histories.
- **Simulators / stubs** — `QuantumSimulator`, `QRAMSimulator`  
  Resource estimates and a QRAM mock.
- **Config loader** — `ConfigLoader`  
  Key–value parsing with defaults.
- **Persistence** — `ModelPersistence`  
  Save/load/export model, configs, results, histories, summaries.

---

## Key results

- **Accuracy**: The quantum pipeline underperforms on small datasets but converges to classical levels at larger `M` (both ~95% at `M = 300` on Digits 64‑D).  
- **Iterations**: Quantum K‑Means generally needs more iterations (occasional spikes; can be mitigated by tolerance tuning).  
- **Cluster quality**: Per‑point losses remain stable with dataset size.  
- **Runtime**: Quantum path is **much slower** due to simulated SWAP tests.

---

## Repository structure

~~~text
.
├─ Bachelorarbeit_Informatik-8.pdf
├─ CITATION.cff
├─ LICENSE
├─ config.txt
├─ requirements.txt
├─ create_digits_npz.py
├─ hybrid_quantum_example.py
├─ test_saved_model.py
├─ confusion_classical.py
├─ product_quantization/
│  ├─ __init__.py
│  ├─ PQKNN.py
│  ├─ quantum_distance.py
│  ├─ quantum_kmeans.py
│  ├─ quantum_pqknn.py
│  ├─ quantum_simulator.py
│  ├─ txt_config_loader.py
│  ├─ model_persistence.py
│  ├─ experiment_utils.py
│  ├─ normalize.py
│  ├─ util.py
│  └─ visualization.py
└─ experiments/           # created at runtime
~~~

---

## Troubleshooting

- **Qiskit/Aer missing** → `pip install qiskit qiskit-aer`  
- **Quantum runs too slow** → reduce `quantum_shots`, lower `max_iter_qk`, or set `algorithm="classical"`  
- **Metric errors** → only one of  
  `"log_fidelity" | "one_minus_fidelity" | "swap_test" | "1-f" | "lf" | "logf" | "omf"` is allowed  
- **Partition mismatch** → the last partition **takes the remainder** by design

---

## Citation

If you use this repository or its results, please cite:

> **Christian Jesinghaus**, *Product Quantization Techniques for Accelerating Vector Database Operations on a Hybrid Quantum Computing Machine*. Bachelor’s Thesis, Leibniz University Hannover, Institute for Systems Engineering, **September 2025**.

A machine‑readable `CITATION.cff` is included.

---

## License

- **Own code**: custom license `LicenseRef-BA-Citation` (see `LICENSE`). Academic use **requires citing** the thesis above.  
- **Third‑party**: `product_quantization/PQKNN.py` adapted from **Jeroen Van Der Donckt** under the **MIT License**.  
  License text is preserved at `THIRD_PARTY_LICENSES/classical_PQKNN-MIT.txt`.

---

## What to add (placeholders)

- `docs/architecture.png` — high‑level system diagram.  
- `experiments/reports/accuracy_vs_train_size.png` — accuracy vs. training size.  
- `experiments/reports/confusion_matrix.png` — example confusion matrix (e.g., `M=300`).

---

## Acknowledgments

- Classical PQ‑kNN baseline by **Jeroen Van Der Donckt** (MIT).  
- **Qiskit Aer** used for SWAP‑test simulation.  
- Research context and evaluation are detailed in the thesis (`Bachelorarbeit_Informatik-8.pdf`).
