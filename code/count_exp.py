import random
import time
from pathlib import Path
import argparse
from meteor_reasoner.materialization.coalesce import *
from meteor_reasoner.materialization.index_build import *
from meteor_reasoner.utils.loader import *
from meteor_reasoner.canonical.utils import find_periods, fact_entailment
from meteor_reasoner.canonical.canonical_representation import CanonicalRepresentation
from meteor_reasoner.utils.parser import parse_str_fact
from meteor_reasoner.utils.operate_dataset import *
from meteor_reasoner.utils.operate_program import *
from meteor_reasoner.classes.atom import Atom
from meteor_reasoner.materialization.t_operator import *
from meteor_reasoner.materialization.naive_join import *
from meteor_reasoner.materialization.materialize import *
from meteor_reasoner.utils.entail_check import entail
from DRED_mtl import DRED, has_cycle
import copy
import os
from collections import defaultdict
import dill
import csv
import gc
import datetime

parser = argparse.ArgumentParser(description="Run single experiment for Eeq- counting.")
parser.add_argument("--dataset", type=str, required=True, help="Path to the original dataset.")
parser.add_argument("--type", type=str, required=True, choices=["fixed", "percentage"],
                    help="Type of deletion: 'fixed' for a fixed number of facts, 'percentage' for a percentage of the dataset.")
parser.add_argument("--scale", type=int, default=0, help="Scale of the dataset (e.g., 4 for 10^4, default: 4)")

args = parser.parse_args()
dataset_name = args.dataset
type = args.type
scale = args.scale

fixed_deletion = "100"
percentage_deletion = "0.1"
if type == "fixed":
    fixed_file_sub = type + "_" + fixed_deletion
elif type == "percentage":
    fixed_file_sub = type + "_" + percentage_deletion

if dataset_name == "LUBM":
    E_path = f"./data/original_e/lubm_10^{scale}.txt"
    program_path = "./programs/lubm.txt"
    E_minus_path = f"./data/delta_e/LUBM/lubm_10^{scale}_{fixed_file_sub}.txt"
    E_plus_path = f"./data/delta_e/LUBM/lubm_10^{scale}_{fixed_file_sub}.txt"
    cr_pickle_path = f"./canonical_representations/lubm_10^{scale}.pkl"
elif dataset_name == "iTemporal":
    E_path = f"./data/original_e/iTemporal_10^{scale}"
    program_path = "./programs/iTemporal.txt"
    E_minus_path = f"./data/delta_e/iTemporal/iTemporal_10^{scale}_{fixed_file_sub}.txt"
    E_plus_path = f"./data/delta_e/iTemporal/iTemporal_10^{scale}_{fixed_file_sub}.txt"
    cr_pickle_path = f"./canonical_representations/iTemporal_10^{scale}.pkl"
elif dataset_name == "Weather":
    E_path = "./data/original_e/weather_dataset.txt"
    program_path = "./programs/weather.txt"
    E_minus_path = f"./data/delta_e/weather/weather_dataset_{fixed_file_sub}.txt"
    E_plus_path = f"./data/delta_e/weather/weather_dataset_{fixed_file_sub}.txt"
    cr_pickle_path = "./canonical_representations/weather_dataset.pkl"

# Load the original dataset
E_0 = load_dataset(E_path)
coalescing_d(E_0)

# Load the program
with open(program_path) as file:
    rules = file.readlines()
    program = load_program(rules)

# Load the E_minus dataset
E_minus_0 = load_dataset(E_minus_path)
E_plus_0 = defaultdict(lambda: defaultdict(list))

# Load the E_plus dataset
E_minus_1 = defaultdict(lambda: defaultdict(list))
E_plus_1 = load_dataset(E_plus_path)

# Load the canonical representation from pickle
print(f"Loading canonical representation from {cr_pickle_path}")
with open(cr_pickle_path, 'rb') as f:
    I_0, common_0, varrho_left_0, left_period_0, left_len_0, varrho_right_0, right_period_0, right_len_0 = dill.load(f)
tmp_load = copy.deepcopy(E_0)
E_1_pre = copy.deepcopy(E_0)
CR_0 = CanonicalRepresentation(tmp_load, program)
CR_0.initilization()
CR_0.Program = copy.deepcopy(program)
CR_0.D = copy.deepcopy(I_0)

##############################
# Phase 1: DRED with E_minus #
############################ #

print(f"\n##### Phase 1: DRED with E_minus #####")

###
# DRED approach
###
print(f"DRED approach starting")

is_cyclic = has_cycle(program)
is_acyclic = not is_cyclic  # DRED requires acyclic programs
print(f"Program is {'acyclic' if is_acyclic else 'cyclic'}")
E_1, CR_1, varrho_left_1, varrho_right_1, left_period_1, right_period_1, left_len_1, right_len_1 = DRED(
    E_0, CR_0, E_minus_0, E_plus_0, varrho_left_0, left_period_0, left_len_0, varrho_right_0, right_period_0, right_len_0, do_count=True, is_acyclic=is_acyclic)


CR_1_facts_num = count_facts(CR_1.D)

###
# Naive approach
###
print(f"Naive approach starting")

coalescing_d(E_minus_0)

for predicate in E_minus_0:
    if predicate in E_1_pre:
        for entity in E_minus_0[predicate]:
            if entity in E_1_pre[predicate]:
                diff_intervals = Interval.diff_list_incre_opt(E_1_pre[predicate][entity], E_minus_0[predicate][entity])
                if diff_intervals:  
                    E_1_pre[predicate][entity] = diff_intervals
                else: 
                    del E_1_pre[predicate][entity]
        
        if not E_1_pre[predicate]:
            del E_1_pre[predicate]

# Compute expected result directly
E_1_test = dataset_union(E_1_pre, E_plus_0)
coalescing_d(E_1_test)
CR_1_test = CanonicalRepresentation(E_1_test, program)
CR_1_test.initilization()
I_1_test, common_1_test, varrho_left_1_test, left_period_1_test, left_len_1_test, varrho_right_1_test, right_period_1_test, right_len_1_test = find_periods(CR_1_test)

CR_1_test_facts_num = count_facts(CR_1_test.D)
print("CR_1 test facts num:", CR_1_test_facts_num)

##############################
# Phase 2: DRED with E_plus  #
##############################

print(f"\n##### Phase 2: DRED with E_plus #####")

E_2_pre = copy.deepcopy(E_1)

###
# DRED approach with E_plus
###
print(f"DRED approach with E_plus starting")

is_cyclic = has_cycle(program)
is_acyclic = not is_cyclic  # DRED requires acyclic programs
print(f"Program is {'acyclic' if is_acyclic else 'cyclic'}")
E_2, CR_2, varrho_left_2, varrho_right_2, left_period_2, right_period_2, left_len_2, right_len_2 = DRED(
    E_1, CR_1, E_minus_1, E_plus_1, varrho_left_1, left_period_1, left_len_1, varrho_right_1, right_period_1, right_len_1, do_count=True, is_acyclic=is_acyclic)     

CR_2_facts_num = count_facts(CR_2.D)

###
# Naive approach with E_plus
###
print(f"Naive approach with E_plus starting")

# Load the computed canonical representation from file
CR_2_test_file_path = cr_pickle_path
if os.path.exists(CR_2_test_file_path):
    with open(CR_2_test_file_path, 'rb') as f:
        I_2_test, common_2_test, varrho_left_2_test, left_period_2_test, left_len_2_test, varrho_right_2_test, right_period_2_test, right_len_2_test = dill.load(f)
tmp_D = defaultdict(lambda: defaultdict(list))
tmp_D["A"][tuple([Term("mike")])] = [Interval(3, 4, False, False), Interval(6, 10, True, True)]
CR_2_test = CanonicalRepresentation(tmp_D, program)
CR_2_test.initilization()
CR_2_test.D = copy.deepcopy(I_2_test)

CR_2_test_facts_num = count_facts(CR_2_test.D)
print("CR_2 facts num:", CR_2_facts_num)

# Print results only facts num with no time
print(f"\nResults:")
print(f"CR_1 facts num: {CR_1_facts_num}, CR_1_test facts num: {CR_1_test_facts_num}")
print(f"CR_2 facts num: {CR_2_facts_num}, CR_2_test facts num: {CR_2_test_facts_num}")

# Save the facts num results to a CSV file
output_dir = Path("./results/count_exp")
output_dir.mkdir(parents=True, exist_ok=True)
datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = output_dir / f"results_{dataset_name}_{type}_{datetime_str}.csv"
with output_file.open("a", newline='') as csvfile:
    writer = csv.writer(csvfile)
    # Write header if file is empty
    if output_file.stat().st_size == 0:
        writer.writerow(["Dataset", "Type", "Scale", "CR_1_facts_num", "CR_1_test_facts_num", "CR_2_facts_num", "CR_2_test_facts_num"])
    # Write the results
    writer.writerow([dataset_name, type, scale, CR_1_facts_num, CR_1_test_facts_num, CR_2_facts_num, CR_2_test_facts_num])