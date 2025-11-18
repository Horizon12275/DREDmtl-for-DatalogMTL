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
    E_minus_path = f"./data/delta_e/lubm/lubm_10^{scale}_{fixed_file_sub}.txt"
    E_plus_path = f"./data/delta_e/lubm/lubm_10^{scale}_{fixed_file_sub}.txt"
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
# Time DRED approach
###
print(f"DRED approach starting")
start_dred_1 = time.time()

is_cyclic = has_cycle(program)
is_acyclic = not is_cyclic  # DRED requires acyclic programs
print(f"Program is {'acyclic' if is_acyclic else 'cyclic'}")
E_1, CR_1, varrho_left_1, varrho_right_1, left_period_1, right_period_1, left_len_1, right_len_1 = DRED(
    E_0, CR_0, E_minus_0, E_plus_0, varrho_left_0, left_period_0, left_len_0, varrho_right_0, right_period_0, right_len_0, do_count=False, is_acyclic=is_acyclic)

end_dred_1 = time.time()
dred_time_1 = end_dred_1 - start_dred_1
print(f"DRED time: {dred_time_1:.2f} seconds")

###
# Time Naive approach
###

E_2_pre = copy.deepcopy(E_1)
print(f"Naive approach starting")
start_time_test_1 = time.time()

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

E_1_test = dataset_union(E_1_pre, E_plus_0)
coalescing_d(E_1_test)
CR_1_test = CanonicalRepresentation(E_1_test, program)
CR_1_test.initilization()
I_1_test, common_1_test, varrho_left_1_test, left_period_1_test, left_len_1_test, varrho_right_1_test, right_period_1_test, right_len_1_test = find_periods(CR_1_test)

end_time_test_1 = time.time()
naive_time_1 = end_time_test_1 - start_time_test_1
print(f"Naive approach execution time: {naive_time_1:.4f} seconds")

###
# Check correctness
###

# Compare results
diff_A = dataset_difference(CR_1.D, CR_1_test.D)
diff_B = dataset_difference(CR_1_test.D, CR_1.D)

# Print test results
print("\n########## Test Results: ##########")
print(f"- DRED execution time: {dred_time_1:.4f} seconds")
print(f"- Naive execution time: {naive_time_1:.4f} seconds")

if dataset_is_empty(diff_A) and dataset_is_empty(diff_B):
    print("- RESULT: PASSED (results match exactly)")
else:
    print("- RESULT: Differences found, verifying entailment..")
    
    # Verify that all differences have equivalent entailment
    all_match = True
    mismatch_details = []
    
    if diff_A is not None:
        for predicate in diff_A:
            for entity in diff_A[predicate]:
                for interval in diff_A[predicate][entity]:
                    F = Atom(predicate, entity, interval)
                    result_DRed = fact_entailment(
                        CR_1.D, F, CR_1.base_interval, 
                        left_period_1, left_len_1, 
                        right_period_1, right_len_1
                    )
                    result_se = fact_entailment(
                        CR_1_test.D, F, CR_1_test.base_interval, 
                        left_period_1_test, left_len_1_test,
                        right_period_1_test, right_len_1_test
                    )
                    if result_DRed != result_se:
                        all_match = False
    
    if diff_B is not None:
        for predicate in diff_B:
            for entity in diff_B[predicate]:
                for interval in diff_B[predicate][entity]:
                    F = Atom(predicate, entity, interval)
                    result_DRed = fact_entailment(
                        CR_1.D, F, CR_1.base_interval, 
                        left_period_1, left_len_1, 
                        right_period_1, right_len_1
                    )
                    result_se = fact_entailment(
                        CR_1_test.D, F, CR_1_test.base_interval, 
                        left_period_1_test, left_len_1_test,
                        right_period_1_test, right_len_1_test
                    )
                    if result_DRed != result_se:
                        all_match = False
    
    if all_match:
        print("- RESULT: PASSED (results are semantically equivalent)")
    else:
        print("- RESULT: FAILED (semantic differences found)")

##############################
# Phase 2: DRED with E_plus  #
##############################

print(f"\n##### Phase 2: DRED with E_plus #####")

###
# Time DRED approach with E_plus
###

print(f"DRED approach with E_plus starting")
start_dred_2 = time.time()

is_cyclic = has_cycle(program)
is_acyclic = not is_cyclic  # DRED requires acyclic programs
print(f"Program is {'acyclic' if is_acyclic else 'cyclic'}")
E_2, CR_2, varrho_left_2, varrho_right_2, left_period_2, right_period_2, left_len_2, right_len_2 = DRED(
    E_1, CR_1, E_minus_1, E_plus_1, varrho_left_1, left_period_1, left_len_1, varrho_right_1, right_period_1, right_len_1, do_count=False, is_acyclic=is_acyclic)     

end_dred_2 = time.time()
dred_time_2 = end_dred_2 - start_dred_2
print(f"DRED with E_plus time: {dred_time_2:.2f} seconds")

###
# Time Naive approach with E_plus
###

print(f"Naive approach with E_plus starting")
start_time_test_2 = time.time()

# Compute expected result directly
E_2_test = dataset_union(E_2_pre, E_plus_1)
coalescing_d(E_2_test)
CR_2_test = CanonicalRepresentation(E_2_test, program)
CR_2_test.initilization()
I_2_test, common_2_test, varrho_left_2_test, left_period_2_test, left_len_2_test, varrho_right_2_test, right_period_2_test, right_len_2_test = find_periods(CR_2_test)     

end_time_test_2 = time.time()
naive_time_2 = end_time_test_2 - start_time_test_2
print(f"Naive approach with E_plus execution time: {naive_time_2:.4f} seconds")

###
# Check correctness with E_plus
###

# Compare results
diff_A_2 = dataset_difference(CR_2.D, CR_2_test.D)
diff_B_2 = dataset_difference(CR_2_test.D, CR_2.D)

# Print test results
print("\n########## Test Results with E_plus: ##########")
print(f"- DRED with E_plus execution time: {dred_time_2:.4f} seconds")
print(f"- Naive with E_plus execution time: {naive_time_2:.4f} seconds")    
if dataset_is_empty(diff_A_2) and dataset_is_empty(diff_B_2):
    print("- RESULT: PASSED (results match exactly)")
else:
    print("- RESULT: Differences found, verifying entailment with E_plus..")
    
    # Verify that all differences have equivalent entailment
    all_match_2 = True
    mismatch_details_2 = []
    
    if diff_A_2 is not None:
        for predicate in diff_A_2:
            for entity in diff_A_2[predicate]:
                for interval in diff_A_2[predicate][entity]:
                    F = Atom(predicate, entity, interval)
                    result_DRed = fact_entailment(
                        CR_2.D, F, CR_2.base_interval, 
                        left_period_2, left_len_2, 
                        right_period_2, right_len_2
                    )
                    result_se = fact_entailment(
                        CR_2_test.D, F, CR_2_test.base_interval, 
                        left_period_2_test, left_len_2_test,
                        right_period_2_test, right_len_2_test
                    )
                    if result_DRed != result_se:
                        all_match_2 = False
    
    if diff_B_2 is not None:
        for predicate in diff_B_2:
            for entity in diff_B_2[predicate]:
                for interval in diff_B_2[predicate][entity]:
                    F = Atom(predicate, entity, interval)
                    result_DRed = fact_entailment(
                        CR_2.D, F, CR_2.base_interval, 
                        left_period_2, left_len_2, 
                        right_period_2, right_len_2
                    )
                    result_se = fact_entailment(
                        CR_2_test.D, F, CR_2_test.base_interval, 
                        left_period_2_test, left_len_2_test,
                        right_period_2_test, right_len_2_test
                    )
                    if result_DRed != result_se:
                        all_match_2 = False
    
    if all_match_2:
        print("- RESULT: PASSED (results are semantically equivalent with E_plus)")
    else:
        print("- RESULT: FAILED (semantic differences found with E_plus)")

# Print final results
print("\n########## Final Results: ##########")
print(f"- DRED with E_minus execution time: {dred_time_1:.4f} seconds")
print(f"- DRED with E_plus execution time: {dred_time_2:.4f} seconds")
print(f"- Naive with E_minus execution time: {naive_time_1:.4f} seconds")
print(f"- Naive with E_plus execution time  : {naive_time_2:.4f} seconds")

# Save results to CSV
output_dir = Path("./results/time_exp")
output_dir.mkdir(parents=True, exist_ok=True)
datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = output_dir / f"results_{dataset_name}_{type}_{datetime_str}.csv"
if type == "fixed":
    output_file = output_dir / f"results_{dataset_name}_{type}_{fixed_deletion}_{datetime_str}.csv"
elif type == "percentage":
    output_file = output_dir / f"results_{dataset_name}_{type}_{percentage_deletion}_{datetime_str}.csv"
with output_file.open('w', newline='') as csvfile:
    fieldnames = ['Dataset', 'Type', 'DRED_E_minus_Time', 'DRED_E_plus_Time', 
                  'Naive_E_minus_Time', 'Naive_E_plus_Time']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    writer.writerow({
        'Dataset': dataset_name,
        'Type': type,
        'DRED_E_minus_Time': dred_time_1,
        'DRED_E_plus_Time': dred_time_2,
        'Naive_E_minus_Time': naive_time_1,
        'Naive_E_plus_Time': naive_time_2
    })