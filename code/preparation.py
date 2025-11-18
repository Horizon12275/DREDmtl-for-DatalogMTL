from meteor_reasoner.materialization.coalesce import *
from meteor_reasoner.materialization.index_build import *
from meteor_reasoner.utils.loader import *
from meteor_reasoner.canonical.utils import find_periods
from meteor_reasoner.canonical.canonical_representation import CanonicalRepresentation
from meteor_reasoner.canonical.utils import fact_entailment
import argparse
import time
import dill
import psutil
from meteor_reasoner.utils.operate_dataset import count_facts

# To Generate Canonical Representation and save it, need to input the dataset and the program
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, type=str, help="Input the dataset name")

# Check the input
args = parser.parse_args()
print("Current dataset is:", args.dataset)
dataset_name = args.dataset

if dataset_name == "LUBM":
    datapath = "./data/original_e/lubm_10^6.txt"
    rulepath = "./programs/lubm.txt"
    CRpath = "./canonical_representations/lubm_10^6.pkl"
elif dataset_name == "iTemporal":
    datapath = "./data/original_e/iTemporal_10^5"
    rulepath = "./programs/iTemporal.txt"
    CRpath = "./canonical_representations/iTemporal_10^5.pkl"
elif dataset_name == "Weather":
    datapath = "./data/original_e/weather_dataset.txt"
    rulepath = "./programs/weather.txt"
    CRpath = "./canonical_representations/weather_dataset.pkl"

# Load the program
with open(rulepath) as file:
    rules = file.readlines()
    program = load_program(rules)

# Load the dataset
D = load_dataset(datapath)

# Start the canonical timer
print("\nBuilding the Canonical Representation...")
start_canonical_build_time = time.time()

# Find the Canonical Representation
CR = CanonicalRepresentation(D, program)
CR.initilization()
origin_facts_num = count_facts(CR.D)
D1, common, varrho_left, left_period, left_len, varrho_right, right_period, right_len = find_periods(CR)

# Stop the canonical timer
end_canonical_build_time = time.time()
canonical_build_time = end_canonical_build_time - start_canonical_build_time
canonical_build_time = canonical_build_time
print("Execution_canonical_build_time =", canonical_build_time, "seconds")

# Create the Folders
import os
if not os.path.exists('./canonical_representations'):
    os.makedirs('./canonical_representations')

# Save the Canonical Representation
CR_path = CRpath
with open(CR_path, 'wb') as f:
    dill.dump((D1, common, varrho_left, left_period, left_len, varrho_right, right_period, right_len), f)
print("Canonical Representation saved to " + CR_path)
CR_time_path = CR_path.replace('./canonical_representations/', './canonical_representations/time/')

# Ensure the time directory exists
if not os.path.exists(os.path.dirname(CR_time_path)):
    os.makedirs(os.path.dirname(CR_time_path))

CR_time_path = CR_time_path.rstrip('.pkl') +'_time.txt'
with open(CR_time_path, 'w') as f:
    f.write("Canonical Representation build time is(with init):"+str(canonical_build_time))
    f.write(f"\nFact Number Before: {origin_facts_num}")
    f.write(f"\nFact Number After: {count_facts(CR.D)}")