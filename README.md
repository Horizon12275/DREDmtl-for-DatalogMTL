## Introduction

This repo is for the paper ["Incremental Maintenance for DatalogMTL Materialisations (AAAI-26)"](https://arxiv.org/abs/2511.12169). This repo contains the DRED_MTL implementation, datasets, programs, and scripts needed to reproduce all incremental-reasoning experiments on LUBMt, iTemporal, and the meteorological benchmark.

We implemented DRED<sub>MTL</sub> algorithm and evaluated its performance against the naive baseline in incremental maintenance tasks across three benchmarks: (1) LUBM<sub>t</sub> (Wang et al. 2022), a temporal extension of LUBM (Guo et al. 2005); (2) iTemporal (Bellomarini et al. 2022); and (3) a meteorological dataset (Wang et al. 2022). The experiment is based on the MeTeoR framework (Wałega et al. 2023). The three test programs of the benchmarks have 85, 11, and 4 rules, respectively.

This data appendix contains 3 datasets (LUBM<sub>t</sub> 10^6, iTemporal 10^5 and a meteorological small dataset) and 3 corresponding programs, as well as our DRED_mtl implementation, the adapted MeTeoR DatalogMTL reasoner, and the source code for running our experiments. Due to size limit, the meteorological benchmark was compressed. All code and datasets have been made publicly available since the paper was accepted.

## Structure

The structure of the data appdendix is as follows (some folders are initially empty):

```
data_appendix/
├── code/
│   ├── DRED_mtl.py
│   ├── preparation.py
│   ├── time_exp.py
│   └── count_exp.py
├── data/
│   ├── original_e/
│   └── delta_e/
├── programs/
├── canonical_representations/      (initially empty)
├── results/                        (initially empty)
├── README.pdf
└── requirements.txt
```

**code/** contains the implementation of DRED<sub>MTL</sub> algorithm and source code for running the experiments, including the preparation and the actual experiments.

**data/** contains the original and delta datasets (E<sup>±</sup> mentioned in the article) for LUBM<sub>t</sub>, iTemporal, and the meteorological benchmark. The original datasets are in the **original_e/** folder, and the delta datasets are in the **delta_e/** folder.

**programs/** contains the DatalogMTL programs of LUBM<sub>t</sub>, iTemporal, and the meteorological benchmark.

**canonical_representations/** is initially empty and will be populated with the canonical representations generated during the preparation step.

**README.pdf** provides an overview of the data appendix, including instructions for setup and running experiments.

**requirements.txt** lists the Python packages required to run the experiments.

### The 'code' folder for running experiments

The code in the **code/** folder is used to run the experiments. It includes:

- **DRED_mtl.py**: The implementation of the DRED<sub>MTL</sub> algorithm and related functions.
- **preparation.py**: A script to generate and save canonical representations for experiment preparation.
- **time_exp.py**: A script to run the time experiments.
- **count_exp.py**: A script to run the count experiments.

In our experiments, we first use the **preparation.py** script to load the original datasets from the **data/original_e/** folder, generate the canonical representations and their periods infomation, and save them in the **canonical_representations/** folder as .pkl format (to avoid redundant computations in subsequent experiments). Then, we run the time experiments using the **time_exp.py** script, which will load the pre-computed canonical representations into memory first and perform the incremental deletion and insertion maintenance tests. The time we record for DRED<sub>MTL</sub> is the time taken to judge whether the program is cyclic or not, and the time taken to update the canonical representation after it has been loaded into memory. The time for the baseline includes the dataset operation (like E ∪ E<sup>+</sup> or E \ E<sup>-</sup>) and the time taken to recompute the canonical representation with the updated dataset from scratch. The results of the time experiments are saved in the **results/time_exp/** folder as .csv files. Similarly, we run the count experiments using the **count_exp.py** script, which is almost the same as the time experiments, but instead of measuring the time, it counts the number of facts in the canonical representation after the deletion or insertion updates. The counting process is performed for both DRED<sub>MTL</sub> and the baseline, and the results are saved in the **results/count_exp/** folder as .csv files.

## Setup

Example OS version: Fedora Linux 40

```bash
conda create -n DREDmtl python=3.7
conda activate DREDmtl
pip install -r requirements.txt
```

### MeTeoR install

The code is based on the MeTeoR framework (Wałega et al. 2023). Please refer to the MeTeoR GitHub repository (https://github.com/Horizon12275/MeTeoR) for detailed installation instructions and usage guidelines. Below is a brief summary of the installation steps:

```bash
cd ./MeTeoR
conda install -c conda-forge pytest-runner
pip install -e .
```

## Run Experiments

### Preparation

Before running the experiments, we need to prepare the datasets and generate the canonical representations. The **preparation.py** script will load the original datasets from the **data/original_e/** folder, generate the canonical representations, and save them in the **canonical_representations/** folder. The script can be run as follows:

```bash

# LUBM 10^6 dataset
python ./code/preparation.py --dataset=LUBM
# iTemporal 10^5 dataset
python ./code/preparation.py --dataset=iTemporal
# weather small dataset
python ./code/preparation.py --dataset=Weather

```

### Deletion & Insertion Test

After preparation, follow the steps below to reproduce the results of our experiments on the LUBM, iTemporal and meteorological benchmark (due to appendix size limit, we provided a smaller meteorological dataset as an example). The results will be printed to the console and saved to the corresponding folders:

```bash

# For Small Deletion & Insertion
python ./code/time_exp.py --dataset=LUBM --type=fixed --scale=6
python ./code/time_exp.py --dataset=iTemporal --type=fixed --scale=5
python ./code/time_exp.py --dataset=Weather --type=fixed 

# For Large Deletion & Insertion
python ./code/time_exp.py --dataset=LUBM --type=percentage  --scale=6
python ./code/time_exp.py --dataset=iTemporal --type=percentage --scale=5
python ./code/time_exp.py --dataset=Weather --type=percentage

```

For the count experiments, which count the number of facts in the canonical representation after deletion or insertion updates, you can run the following commands:

```bash

# For Small Deletion & Insertion
python ./code/count_exp.py --dataset=LUBM --type=fixed --scale=6
python ./code/count_exp.py --dataset=iTemporal --type=fixed --scale=5
python ./code/count_exp.py --dataset=Weather --type=fixed

# For Large Deletion & Insertion
python ./code/count_exp.py --dataset=LUBM --type=percentage  --scale=6
python ./code/count_exp.py --dataset=iTemporal --type=percentage --scale=5
python ./code/count_exp.py --dataset=Weather --type=percentage

```

### Example code

To integrate DRED<sub>MTL</sub> into your own project, you can directly import the `DRED_mtl.py` module from the **code/** folder. Below is an example of how to use DRED<sub>MTL</sub> for incremental maintenance:

```python
from meteor_reasoner.utils.loader import load_dataset, load_program
from meteor_reasoner.canonical.canonical_representation import CanonicalRepresentation
from meteor_reasoner.canonical.utils import find_periods
from meteor_reasoner.materialization.DRED_mtl import DRED   
from meteor_reasoner.materialization.coalesce import coalescing_d

# -----------------------------
# 1. Load original dataset & rules
# -----------------------------
E = load_dataset("data/example_dataset.txt")
rules = load_program(open("programs/example_rules.txt").readlines())
coalescing_d(E)

# -----------------------------
# 2. Load delta datasets (E_minus / E_plus)
# -----------------------------
E_minus = load_dataset("data/example_minus.txt")   # facts to delete
E_plus  = load_dataset("data/example_plus.txt")    # facts to insert
coalescing_d(E_minus)
coalescing_d(E_plus)

# Only keep meaningful deltas
E_minus = dataset_difference(dataset_intersection(E, E_minus), E_plus)
E_plus  = dataset_difference(E_plus, E)

# -----------------------------
# 3. Build Canonical Representation (CR)
# -----------------------------
CR = CanonicalRepresentation(E.copy(), rules)
CR.initilization()   # build index, compute base interval, etc.

# -----------------------------
# 4. Precompute periodic information
# -----------------------------
I, common, varrho_left, left_period, left_len,
    varrho_right, right_period, right_len = find_periods(CR)

# -----------------------------
# 5. Run DRED-MTL incremental maintenance
# -----------------------------
new_E, new_CR,
 I_varrho_left, I_varrho_right,
 I_left_period, I_right_period,
 I_left_len, I_right_len = DRED(
    E, CR, E_minus, E_plus,
    varrho_left, left_period, left_len,
    varrho_right, right_period, right_len,
    do_count=False,
    is_acyclic=False,
)

print("Updated materialisation computed!")
```

### Notes

To run an experiment process with a high priority and only one kernel to avoid the interference of context switching and other processes on the same machine, we can use the following command:

```bash
# Example
sudo taskset -c 0 nice -n -20 python ./code/preparation.py --dataset=lubm
```