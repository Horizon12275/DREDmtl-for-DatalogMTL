from meteor_reasoner.materialization.coalesce import *
from meteor_reasoner.materialization.index_build import *
from meteor_reasoner.utils.loader import *
from meteor_reasoner.canonical.utils import find_periods, fact_entailment
from meteor_reasoner.canonical.canonical_representation import CanonicalRepresentation
from meteor_reasoner.utils.operate_dataset import *
from meteor_reasoner.classes.interval import Interval
from meteor_reasoner.materialization.t_operator import *
from meteor_reasoner.materialization.naive_join import *
from meteor_reasoner.materialization.materialize import *
from meteor_reasoner.utils.ruler_interval import *
from meteor_reasoner.canonical.class_common_fragment import CommonFragment
from meteor_reasoner.canonical.utils import find_right_period, find_left_period, find_right_period_incre, \
    find_left_period_incre
from meteor_reasoner.classes.atom import Atom
import copy
import math
from line_profiler import *


def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def lcm(a, b):
    return abs(a * b) // gcd(a, b)


def remove_extra_fact(D, varrho_l, varrho_r):
    for predicate in list(D.keys()):  # Use list to avoid runtime modification issues
        for entity in list(D[predicate].keys()):
            intervals = D[predicate][entity]

            # Remove from head (left side)
            if varrho_l is not None and intervals:  # Added check for intervals
                while intervals and intervals[0].right_value <= varrho_l.left_value and Interval.intersection(
                        intervals[0], varrho_l) is None:
                    intervals.pop(0)
                if intervals and intervals[0].left_value < varrho_l.left_value:
                    intervals[0] = Interval(varrho_l.left_value, intervals[0].right_value,
                                            varrho_l.left_open, intervals[0].right_open)

            # Remove from tail (right side)
            if varrho_r is not None and intervals:  # Added check for intervals
                while intervals and intervals[-1].left_value >= varrho_r.right_value and Interval.intersection(
                        intervals[-1], varrho_r) is None:
                    intervals.pop()
                if intervals and intervals[-1].right_value > varrho_r.right_value:
                    intervals[-1] = Interval(intervals[-1].left_value, varrho_r.right_value,
                                             intervals[-1].left_open, varrho_r.right_open)

            # Update back
            if intervals:
                D[predicate][entity] = intervals
            else:
                del D[predicate][entity]
                if not D[predicate]:  # If no entities left for this predicate
                    del D[predicate]


def get_dataset_interval(D):
    left_value = Decimal("+inf")
    right_value = Decimal("-inf")
    left_open = True
    right_open = True
    for predicate in list(D.keys()):
        for entity in list(D[predicate].keys()):
            intervals = D[predicate][entity]
            if len(intervals) == 0: continue
            if left_value is None:
                left_value = intervals[0].left_value
                left_open = intervals[0].left_open
            else:
                if intervals[0].left_value < left_value:
                    left_value = intervals[0].left_value
                    left_open = intervals[0].left_open
                elif intervals[0].left_value == left_value:
                    if intervals[0].left_open == False:
                        left_open = False
            if right_value is None:
                right_value = intervals[-1].right_value
                right_open = intervals[-1].right_open
            else:
                if intervals[-1].right_value > right_value:
                    right_value = intervals[-1].right_value
                    right_open = intervals[-1].right_open
                elif intervals[-1].right_value == right_value:
                    if intervals[-1].right_open == False:
                        right_open = False
    return Interval(left_value, right_value, left_open, right_open)


def period(CR, varrho_l, varrho_r, D, Delta_D):
    left_period, left_len = defaultdict(list), 0
    right_period, right_len = defaultdict(list), 0
    common_fragment = CommonFragment(CR.base_interval)
    common_fragment.common = Interval(Decimal("-inf"), Decimal("+inf"), True, True)

    terminate_flag = False

    for head_predicate in Delta_D:
        for head_entity, T in Delta_D[head_predicate].items():
            diff_delta = copy.deepcopy(T)

            for cr_interval in diff_delta:
                if Interval.intersection(cr_interval, common_fragment.base_interval):
                    # it denotes that now |\varrho_max != Dnext |\varrho_max
                    # diff_delta 中的区间和 common_fragment 存在交集
                    common_fragment.cr_flag = False
                    common_fragment.common = None
                    terminate_flag = True
                    break
                else:
                    # 若没有交集，则更新 common_fragment 的左右端点
                    if cr_interval.right_value <= common_fragment.base_interval.left_value:
                        if cr_interval.right_value >= common_fragment.common.left_value:
                            common_fragment.common.left_value = cr_interval.right_value
                            common_fragment.common.left_open = not cr_interval.right_open
                    elif cr_interval.left_value >= common_fragment.base_interval.right_value:
                        if cr_interval.left_value <= common_fragment.common.right_value:
                            common_fragment.common.right_value = cr_interval.left_value
                            common_fragment.common.right_open = not cr_interval.left_open
                    else:
                        print(str(cr_interval))
                        print(str(common_fragment.common))
                        raise ValueError("Error Happen")
            if terminate_flag:
                break
        if terminate_flag:
            break
    if varrho_l is not None:
        varrho_left_range_right_end = varrho_l.right_value + CR.w
    else:
        varrho_left_range_right_end = CR.min_x

    if varrho_r is not None:
        varrho_right_range_left_end = varrho_r.left_value - CR.w
    else:
        varrho_right_range_left_end = CR.max_x
    if (common_fragment.common is None or varrho_left_range_right_end - common_fragment.common.left_value <= CR.w
            or common_fragment.common.right_value - varrho_right_range_left_end <= CR.w):
        return None, None, None, None, None, None
    varrho_left_range = Interval(common_fragment.common.left_value, varrho_left_range_right_end,
                                 True, False)
    varrho_right_range = Interval(varrho_right_range_left_end, common_fragment.common.right_value, False,
                                  True)

    # print("varrho_left_range:", varrho_left_range)
    # print("varrho_right_range:", varrho_right_range)
    if varrho_left_range.left_value in [Decimal("-inf")] and varrho_right_range.right_value in [Decimal("+inf")]:
        return None, None, None, None, None, None

    # 公共片段不为空，则函数通过调用 find_left_period 和 find_right_period 找到左、右两端的周期性区间。
    if varrho_left_range.left_value in [Decimal("-inf")]:
        varrho_right, varrho_right_dict = find_right_period_incre(D, varrho_right_range, CR, varrho_r)
        if varrho_right is not None:
            right_len = varrho_right.right_value - varrho_right.left_value
            for key, values in varrho_right_dict.items():
                for value in values:
                    right_period[value].append(key)
            for key, value in right_period.items():
                right_period[key] = coalescing(value)

            return None, None, None, varrho_right, right_period, right_len  # 有右周期无左周期
    else:
        varrho_left, varrho_left_dict = find_left_period_incre(D, varrho_left_range, CR, varrho_l)
        if varrho_left is not None:
            if varrho_right_range.right_value in [Decimal("+inf")]:
                left_len = varrho_left.right_value - varrho_left.left_value
                for key, values in varrho_left_dict.items():
                    for value in values:
                        left_period[value].append(key)
                for key, value in left_period.items():
                    left_period[key] = coalescing(value)
                return varrho_left, left_period, left_len, None, None, None  # 有左周期无右周期

            else:
                varrho_right, varrho_right_dict = find_right_period_incre(D, varrho_right_range, CR, varrho_r)
                if varrho_right is not None:
                    left_len = varrho_left.right_value - varrho_left.left_value
                    for key, values in varrho_left_dict.items():
                        for value in values:
                            left_period[value].append(key)
                    for key, value in left_period.items():
                        left_period[key] = coalescing(value)

                    right_len = varrho_right.right_value - varrho_right.left_value
                    for key, values in varrho_right_dict.items():
                        for value in values:
                            right_period[value].append(key)
                    for key, value in right_period.items():
                        right_period[key] = coalescing(value)
                    return varrho_left, left_period, left_len, varrho_right, right_period, right_len  # 左右均有周期
    return None, None, None, None, None, None


def change_period(D, varrho, origin_period_len, new_period_len, direction):
    times = int(new_period_len / origin_period_len)
    if times == 1:
        return D, varrho
    if direction == "left":
        for predicate in D:
            for entity in D[predicate]:
                intervals = D[predicate][entity]

                intersect_intervals = Interval.list_intersection([varrho], intervals)
                print(intersect_intervals)
                for i in range(1, times):
                    added_intervals = []
                    for interval in intersect_intervals:
                        added_intervals.append(Interval(interval.left_value - i * origin_period_len,
                                                        interval.right_value - i * origin_period_len,
                                                        interval.left_open, interval.right_open))
                    intervals += added_intervals
        return D, Interval(varrho.right_value - times * origin_period_len, varrho.right_value, varrho.left_open,
                           varrho.right_open)
    elif direction == "right":
        for predicate in D:
            for entity in D[predicate]:
                intervals = D[predicate][entity]

                intersect_intervals = Interval.list_intersection([varrho], intervals)
                print(intersect_intervals)
                for i in range(1, times):
                    added_intervals = []
                    for interval in intersect_intervals:
                        added_intervals.append(Interval(interval.left_value + i * origin_period_len,
                                                        interval.right_value + i * origin_period_len,
                                                        interval.left_open, interval.right_open))
                    intervals += added_intervals
        return D, Interval(varrho.left_value, varrho.left_value + times * origin_period_len, varrho.left_open,
                           varrho.right_open)
    return None


def panning(D, varrho, target, direction):
    period_len = varrho.right_value - varrho.left_value

    if direction == "left":
        times = math.ceil((varrho.left_value - target) / period_len)
        if times <= 0: return D, varrho
        for predicate in D:
            for entity in D[predicate]:
                intervals = D[predicate][entity]

                intersect_intervals = Interval.list_intersection([varrho], intervals)
                for i in range(1, times + 1):
                    added_intervals = []
                    for interval in intersect_intervals:
                        added_intervals.append(Interval(interval.left_value - i * period_len,
                                                        interval.right_value - i * period_len,
                                                        interval.left_open, interval.right_open))
                    intervals += added_intervals
        return D, Interval(varrho.left_value - period_len * times, varrho.right_value - period_len * times,
                           varrho.left_open, varrho.right_open)
    elif direction == "right":
        times = math.ceil((target - varrho.right_value) / period_len)
        if times <= 0: return D, varrho
        for predicate in D:
            for entity in D[predicate]:
                intervals = D[predicate][entity]

                intersect_intervals = Interval.list_intersection([varrho], intervals)
                for i in range(1, times + 1):
                    added_intervals = []
                    for interval in intersect_intervals:
                        added_intervals.append(Interval(interval.left_value + i * period_len,
                                                        interval.right_value + i * period_len,
                                                        interval.left_open, interval.right_open))
                    intervals += added_intervals
        return D, Interval(varrho.left_value + period_len * times, varrho.right_value + period_len * times,
                           varrho.left_open, varrho.right_open)
    return None


def get_farthest_time(D, direction):
    res = None

    for predicate in D:
        for entity in D[predicate]:
            intervals = D[predicate][entity]
            if not intervals:  # Skip if intervals list is empty
                continue

            if direction == "left":
                if res is None:
                    res = intervals[0].left_value
                else:
                    res = min(res, intervals[0].left_value)

            elif direction == "right":  # Changed to elif for better logic
                if res is None:
                    res = intervals[-1].right_value
                else:
                    res = max(res, intervals[-1].right_value)

    return res

def has_cycle(program):
    """
    Check if the given program contains cyclic dependencies (recursive predicates).
    
    Args:
        program: A list of Rule objects representing the logic program.
        
    Returns:
        bool: True if the program contains at least one cyclic dependency, False otherwise.
    """
    # Get all predicates in the program
    predicates = set()
    for rule in program:
        predicates.add(rule.head.get_predicate())
        for literal in rule.body:
            if isinstance(literal, BinaryLiteral):
                predicates.add(literal.left_literal.get_predicate())
                predicates.add(literal.right_literal.get_predicate())
            else:
                predicates.add(literal.get_predicate())
    
    # Find non-recursive predicates
    CF = CycleFinder(program=program)
    non_recursive_predicates = CF.get_non_recursive_predicates()
    
    # If any predicate is not in non_recursive_predicates, it means it's part of a cycle
    return len(predicates) - len(non_recursive_predicates) > 0

def dataset_difference_periodical(D1, D2, D1_varrho_left, D1_varrho_right, D2_varrho_left, D2_varrho_right):
    res_varrho_left, res_varrho_right = None, None

    # Handle left varrho
    if D2_varrho_left is not None:
        if D1_varrho_left is None:
            # If D1 has no left varrho but D2 does, we'll just use D2's
            res_varrho_left = D2_varrho_left
        else:
            D1_left_len = D1_varrho_left.right_value - D1_varrho_left.left_value
            D2_left_len = D2_varrho_left.right_value - D2_varrho_left.left_value
            res_left_len = lcm(D1_left_len, D2_left_len)
            D1, D1_varrho_left = change_period(D1, D1_varrho_left, D1_left_len, res_left_len, 'left')
            D2, D2_varrho_left = change_period(D2, D2_varrho_left, D2_left_len, res_left_len, 'left')
            if D1_varrho_left.left_value < D2_varrho_left.left_value:
                D2, D2_varrho_left = panning(D2, D2_varrho_left, D1_varrho_left.left_value, 'left')
                res_varrho_left = D1_varrho_left
            elif D1_varrho_left.left_value > D2_varrho_left.left_value:
                D1, D1_varrho_left = panning(D1, D1_varrho_left, D2_varrho_left.left_value, 'left')
                res_varrho_left = D2_varrho_left
            else:
                res_varrho_left = D1_varrho_left
    else:
        res_varrho_left = D1_varrho_left

    # Handle right varrho
    if D2_varrho_right is not None:
        if D1_varrho_right is None:
            # If D1 has no right varrho but D2 does, we'll just use D2's
            res_varrho_right = D2_varrho_right
        else:
            D1_right_len = D1_varrho_right.right_value - D1_varrho_right.left_value
            D2_right_len = D2_varrho_right.right_value - D2_varrho_right.left_value
            res_right_len = lcm(D1_right_len, D2_right_len)
            D1, D1_varrho_right = change_period(D1, D1_varrho_right, D1_right_len, res_right_len, 'right')
            D2, D2_varrho_right = change_period(D2, D2_varrho_right, D2_right_len, res_right_len, 'right')
            if D1_varrho_right.right_value < D2_varrho_right.right_value:
                D1, D1_varrho_right = panning(D1, D1_varrho_right, D2_varrho_right.right_value, 'right')
                res_varrho_right = D2_varrho_right
            elif D1_varrho_right.right_value > D2_varrho_right.right_value:
                D2, D2_varrho_right = panning(D2, D2_varrho_right, D1_varrho_right.right_value, 'right')
                res_varrho_right = D1_varrho_right
            else:
                res_varrho_right = D2_varrho_right
    else:
        res_varrho_right = D1_varrho_right

    coalescing_d(D1)
    coalescing_d(D2)

    res = dataset_difference(D1, D2)
    remove_extra_fact(res, res_varrho_left, res_varrho_right)
    return res, res_varrho_left, res_varrho_right


def dataset_union_periodical(D1, D2, D1_varrho_left, D1_varrho_right, D2_varrho_left, D2_varrho_right):
    res = None
    res_varrho_left, res_varrho_right = None, None

    if D1_varrho_left is not None:
        D1_left_len = D1_varrho_left.right_value - D1_varrho_left.left_value
        if D2_varrho_left is not None:
            D2_left_len = D2_varrho_left.right_value - D2_varrho_left.left_value
            res_left_len = lcm(D1_left_len, D2_left_len)

            D1, D1_varrho_left = change_period(D1, D1_varrho_left, D1_left_len, res_left_len, 'left')
            D2, D2_varrho_left = change_period(D2, D2_varrho_left, D2_left_len, res_left_len, 'left')

            if D1_varrho_left.left_value < D2_varrho_left.left_value:
                D2, D2_varrho_left = panning(D2, D2_varrho_left, D1_varrho_left.left_value, 'left')
                res_varrho_left = D1_varrho_left
            elif D1_varrho_left.left_value > D2_varrho_left.left_value:
                D1, D1_varrho_left = panning(D1, D1_varrho_left, D2_varrho_left.left_value, 'left')
                res_varrho_left = D2_varrho_left
            else:
                res_varrho_left = D2_varrho_left
        else:
            target = get_farthest_time(D2, 'left')
            D1, D1_varrho_left = panning(D1, D1_varrho_left, target - D1_left_len, 'left')
            res_varrho_left = D1_varrho_left
    else:
        if D2_varrho_left is not None:
            D2_left_len = D2_varrho_left.right_value - D2_varrho_left.left_value
            target = get_farthest_time(D1, 'left')
            D2, D2_varrho_left = panning(D2, D2_varrho_left, target - D2_left_len, 'left')
            res_varrho_left = D2_varrho_left
        else:
            res_varrho_left = None

    if D1_varrho_right is not None:
        D1_right_len = D1_varrho_right.right_value - D1_varrho_right.left_value
        if D2_varrho_right is not None:
            D2_right_len = D2_varrho_right.right_value - D2_varrho_right.left_value
            res_right_len = lcm(D1_right_len, D2_right_len)

            D1, D1_varrho_right = change_period(D1, D1_varrho_right, D1_right_len, res_right_len, 'right')
            D2, D2_varrho_right = change_period(D2, D2_varrho_right, D2_right_len, res_right_len, 'right')

            if D1_varrho_right.right_value < D2_varrho_right.right_value:
                D1, D1_varrho_right = panning(D1, D1_varrho_right, D2_varrho_right.right_value, 'right')
                res_varrho_right = D2_varrho_right
            elif D1_varrho_right.right_value > D2_varrho_right.right_value:
                D2, D2_varrho_right = panning(D2, D2_varrho_right, D1_varrho_right.right_value, 'right')
                res_varrho_right = D1_varrho_right
            else:
                res_varrho_right = D2_varrho_right
        else:
            target = get_farthest_time(D2, 'right')
            if target is None:
                target = decimal.Decimal("0")
            D1, D1_varrho_right = panning(D1, D1_varrho_right, target + D1_right_len, 'right')
            res_varrho_right = D1_varrho_right
    else:
        if D2_varrho_right is not None:
            D2_right_len = D2_varrho_right.right_value - D2_varrho_right.left_value
            target = get_farthest_time(D1, 'right')
            D2, D2_varrho_right = panning(D2, D2_varrho_right, target + D2_right_len, 'right')
            res_varrho_right = D2_varrho_right
        else:
            res_varrho_right = None

    coalescing_d(D1)
    coalescing_d(D2)
    res = dataset_union(D1, D2)
    remove_extra_fact(res, res_varrho_left, res_varrho_right)
    return res, res_varrho_left, res_varrho_right


def remove_empty(D):
    to_delete_predicates = []
    for tmp_predicate in list(D.keys()):
        to_delete_entities = []
        for tmp_entity in list(D[tmp_predicate].keys()):
            if not D[tmp_predicate][tmp_entity]:
                to_delete_entities.append(tmp_entity)
        # 删除空的实体
        for tmp_entity in to_delete_entities:
            del D[tmp_predicate][tmp_entity]
        # 如果 predicate 下已经没有实体了，也删除它
        if not D[tmp_predicate]:
            to_delete_predicates.append(tmp_predicate)
    # 删除空的 predicate
    for tmp_predicate in to_delete_predicates:
        del D[tmp_predicate]


def build_period(D, varrho_left, varrho_right):
    left_period = defaultdict(list)
    right_period = defaultdict(list)
    if varrho_left is not None:
        for predicate in D:
            for entity in D[predicate]:
                intervals = D[predicate][entity]
                left_period[str(Atom(predicate, entity))] += Interval.list_intersection(intervals, [varrho_left])
    if varrho_right is not None:
        for predicate in D:
            for entity in D[predicate]:
                intervals = D[predicate][entity]
                right_period[str(Atom(predicate, entity))] += Interval.list_intersection(intervals, [varrho_right])
    return left_period, right_period


def reverse_flattened_period(period_dict):
    D = defaultdict(lambda: defaultdict(list))
    for atom_str, intervals in period_dict.items():
        # 假设 atom_str 的格式为 "predicate(entity)"
        predicate, rest = atom_str.split('(', 1)
        entity = tuple([Term(item) for item in rest[:-1].split(",")])
        D[predicate][entity] = intervals
    return D


def unfold(D, D_varrho_left, D_varrho_right, left_end, right_end):
    if D_varrho_left is not None and left_end not in [Decimal("-inf"), Decimal("+inf")]:
        D, D_varrho_left = panning(D, D_varrho_left, left_end, 'left')
    if D_varrho_right is not None and right_end not in [Decimal("-inf"), Decimal("+inf")]:
        D, D_varrho_right = panning(D, D_varrho_right, right_end, 'right')
    return D


# DRED(Π,E,I,E−,E+)
@profile
def DRED(E, CR, E_minus, E_plus, varrho_left, left_period, left_len, varrho_right, right_period, right_len,
         do_count=False, is_acyclic=False):
    """
    Conservative refactor of DRED following the paper structure:
    1) Over-deletion (compute D)
    2) One-step rederivation (compute R)
    3) Insertion (compute A)

    This implementation reuses existing helpers and adds defensive checks when
    touching nested dictionaries.
    """
    D = defaultdict(lambda: defaultdict(list))
    A = defaultdict(lambda: defaultdict(list))
    R = defaultdict(lambda: defaultdict(list))

    I = copy.deepcopy(CR.D)
    I_varrho_left = varrho_left
    I_varrho_right = varrho_right

    coalescing_d(E_minus)
    coalescing_d(E_plus)

    E_minus_real = dataset_intersection(E, E_minus)
    E_minus_real = dataset_difference(E_minus_real, E_plus)

    E_plus_real = dataset_difference(E_plus, E)
    # ------------------
    # 1) Over-deletion
    # ------------------
    N_D = copy.deepcopy(E_minus_real)
    delete_varrho_left = delete_left_period = delete_left_len = None
    delete_varrho_right = delete_right_period = delete_right_len = None
    while True:
        delta_D = dataset_difference(N_D, D)
        if dataset_is_empty(delta_D):
            delete_varrho_left, delete_left_period, delete_left_len, delete_varrho_right, delete_right_period, delete_right_len = None, None, None, None, None, None
            break
        delete_varrho_left, delete_left_period, delete_left_len, delete_varrho_right, delete_right_period, delete_right_len = period(
            CR, I_varrho_left, I_varrho_right, D, delta_D)
        if delete_varrho_left is not None or delete_varrho_right is not None:
            break
        interval_D = get_dataset_interval(D)
        unfolded_I = unfold(I, I_varrho_left, I_varrho_right, interval_D.left_value - CR.w,
                            interval_D.right_value + CR.w)
        unfolded_I_minus_D = dataset_difference(unfolded_I, D)
        N_D = incre_seminaive_immediate_consequence_operator(CR.Program, unfolded_I_minus_D,
                                                             build_index(unfolded_I_minus_D), delta_D)
        coalescing_d(N_D)
        # print("Over-deletion iteration completed.")
        # print_dataset(N_D)
        D = dataset_union(D, delta_D)

    I, I_varrho_left, I_varrho_right = dataset_difference_periodical(I, D, I_varrho_left, I_varrho_right,
                                                                     delete_varrho_left, delete_varrho_right)
    # next 2 line should be delete after \Pi is corrected
    cur_E = dataset_difference(E, E_minus_real)
    I=dataset_union(I, cur_E)

    coalescing_d(I)

    # update CR.baseinterval... here
    CR.points, CR.min_x, CR.max_x = get_dataset_points_x(dataset_union(E, E_plus), min_x_flag=True)
    CR.base_interval = Interval(CR.min_x, CR.max_x, False, False)

    print("Over-deletion completed.")

    # -------------------------
    # 2) Rederivation
    # -------------------------
    rede_index = 20
    interval_I = get_dataset_interval(I)
    if I_varrho_left is not None:
        t_L_init = I_varrho_left.left_value - rede_index * max(CR.w, left_len)
    else:
        t_L_init = interval_I.left_value
    if I_varrho_right is not None:
        t_R_init = I_varrho_right.right_value + rede_index * max(CR.w, right_len)
    else:
        t_R_init = interval_I.right_value
    unfolded_I_init = unfold(I, I_varrho_left, I_varrho_right, t_L_init, t_R_init)
    N_T_pi = naive_immediate_consequence_operator(CR.Program, unfolded_I_init,
                                                    build_index(unfolded_I_init))
    N_R = defaultdict(lambda: defaultdict(list))
    k = 1
    rederive_varrho_left = rederive_left_period = rederive_left_len = None
    rederive_varrho_right = rederive_right_period = rederive_right_len = None
    while True:
        # interval_I = get_dataset_interval(I)
        if I_varrho_left is not None:
            t_L = I_varrho_left.left_value - k * max(CR.w, left_len)
        else:
            t_L = interval_I.left_value
        if I_varrho_right is not None:
            t_R = I_varrho_right.right_value + k * max(CR.w, right_len)
        else:
            t_R = interval_I.right_value
        # unfolded_I = unfold(I, I_varrho_left, I_varrho_right, t_L, t_R)
        unfolded_D_between_tL_tR = unfold(D, delete_varrho_left, delete_varrho_right, t_L, t_R)
        # N_T_pi = naive_immediate_consequence_operator(CR.Program, unfolded_I,
                                                    #   build_index(unfolded_I))
        coalescing_d(N_T_pi)
        T_pi = dataset_difference(N_T_pi, unfolded_I_init)
        N_R = dataset_union(N_R, dataset_intersection(unfolded_D_between_tL_tR, T_pi))
        delta_R = dataset_difference(N_R, R)
        if dataset_is_empty(delta_R):
            rederive_varrho_left, rederive_left_period, rederive_left_len, rederive_varrho_right, rederive_right_period, rederive_right_len = None, None, None, None, None, None
            break
        rederive_varrho_left, rederive_left_period, rederive_left_len, rederive_varrho_right, rederive_right_period, rederive_right_len = period(
            CR, I_varrho_left, I_varrho_right, R, delta_R)
        if rederive_varrho_left is not None or rederive_varrho_right is not None:
            break
        R = dataset_union(R, delta_R)
        coalescing_d(R)
        k += 1
        interval_R = get_dataset_interval(R)
        unfolded_I = unfold(I, I_varrho_left, I_varrho_right, interval_R.left_value - CR.w,
                            interval_R.left_value + CR.w)
        unfolded_I_union_R = dataset_union(unfolded_I, R)
        N_R = incre_seminaive_immediate_consequence_operator(CR.Program, unfolded_I_union_R,
                                                             build_index(unfolded_I_union_R), delta_R)
        coalescing_d(N_R)
        # print(f"Rederivation iteration {k} completed.")
        # print_dataset(N_R)

    coalescing_d(R)
    I, I_varrho_left, I_varrho_right = dataset_union_periodical(I, R, I_varrho_left, I_varrho_right,
                                                                rederive_varrho_left,
                                                                rederive_varrho_right)
    coalescing_d(I)

    print("Rederivation completed.")

    # ------------------
    # 3) Insertion
    # ------------------
    N_A = copy.deepcopy(E_plus_real)
    E_updated = dataset_union(dataset_difference(E, E_minus_real), E_plus_real)
    insert_varrho_left = insert_left_period = insert_left_len = None
    insert_varrho_right = insert_right_period = insert_right_len = None
    while True:
        interval_A = get_dataset_interval(A)
        unfolded_I = unfold(I, I_varrho_left, I_varrho_right, interval_A.left_value - CR.w,
                            interval_A.right_value + CR.w)
        unfolded_I_union_A = dataset_union(unfolded_I, A)
        delta_A = dataset_difference(N_A, unfolded_I_union_A)
        if dataset_is_empty(delta_A):
            insert_varrho_left, insert_left_period, insert_left_len, insert_varrho_right, insert_right_period, insert_right_len = None, None, None, None, None, None
            break
        insert_varrho_left, insert_left_period, insert_left_len, insert_varrho_right, insert_right_period, insert_right_len = period(
            CR, I_varrho_left, I_varrho_right, A, delta_A)
        if insert_varrho_left is not None or insert_varrho_right is not None:
            break
        A = dataset_union(A, delta_A)
        unfolded_I_union_A = dataset_union(unfolded_I, A)
        N_A = incre_seminaive_immediate_consequence_operator(CR.Program, unfolded_I_union_A,
                                                             build_index(unfolded_I_union_A), delta_A)
        coalescing_d(N_A)

    coalescing_d(A)
    I, I_varrho_left, I_varrho_right = dataset_union_periodical(I, A, I_varrho_left, I_varrho_right, insert_varrho_left,
                                                                insert_varrho_right)

    coalescing_d(I)

    print("Insertion completed.")

    I_left_period, I_right_period = build_period(CR.D, I_varrho_left, I_varrho_right)

    I_left_len = I_varrho_left.right_value - I_varrho_left.left_value if I_varrho_left is not None else None
    I_right_len = I_varrho_right.right_value - I_varrho_right.left_value if I_varrho_right is not None else None

    CR.D = copy.deepcopy(I)

    return E_updated, CR, I_varrho_left, I_varrho_right, I_left_period, I_right_period, I_left_len, I_right_len


if __name__ == "__main__":

    islubm = True

    lubm_scala = 2

    test_program = f"Period_Change_1"
    # test_program=f"Period_Change_2"
    # test_program=f"Catastrophic_Delete"

    datapath = f"../data/added_test/" + test_program + "/dataset.txt"
    # datapath = f"./data/eg2022.txt"

    if islubm:
        datapath = f"../data/regression/lubm_10^{lubm_scala}.txt"
    E = load_dataset(datapath)
    coalescing_d(E)

    rulepath = f"../programs/PeriodTest/" + test_program + ".txt"
    # rulepath = f"../programs/eg2022.txt"

    if islubm:
        rulepath = f"../programs/lubm.txt"
    with open(rulepath) as file:
        rules = file.readlines()
        program = load_program(rules)

    datapath_minus = f"../data/added_test/" + test_program + "/dataset-.txt"
    # datapath_minus = f"../data/eg2022-.txt"

    if islubm:
        datapath_minus = f"../data/regression/lubm_10^{lubm_scala}-.txt"
    E_minus = load_dataset(datapath_minus)

    datapath_plus = f"../data/added_test/" + test_program + "/dataset+.txt"
    # datapath_plus = f"../data/eg2022+.txt"

    if islubm:
        datapath_plus = f"../data/regression/lubm_10^{lubm_scala}+.txt"
    E_plus = load_dataset(datapath_plus)

    # E_plus = {}
    # print("\nThe orginal E is:")
    # #print_dataset(E)
    # print("\nThe orginal I is:")
    # #print_dataset(I)
    # print("\nThe orginal E_minus is:")
    # print_dataset(E_minus)
    # print("\nThe orginal E_plus is:")
    # print_dataset(E_plus)

    E_minus = dataset_difference(dataset_intersection(E, E_minus), E_plus)
    E_plus = dataset_difference(E_plus, E)

    CR = CanonicalRepresentation(copy.deepcopy(E), program)
    CR.initilization()
    I, common, varrho_left, left_period, left_len, varrho_right, right_period, right_len = find_periods(CR)

    start_time = time.time()

    new_E, CR, I_varrho_left, I_varrho_right, I_left_period, I_right_period, I_left_len, I_right_len = DRED(CR, E,
                                                                                                               E_minus,
                                                                                                               E_plus,
                                                                                                               varrho_left,
                                                                                                               left_period,
                                                                                                               left_len,
                                                                                                               varrho_right,
                                                                                                               right_period,
                                                                                                               right_len)

    end_time = time.time()

    new_I=CR.D

    coalescing_d(new_E)
    coalescing_d(new_I)

    # print("\nThe new E is:")
    # print_dataset(new_E)
    # print("\nThe new I is:")
    # print_dataset(new_I)
    print("\nThe new varrho left:")
    print(varrho_left)
    print("\nThe new varrho right:")
    print(varrho_right)

    ## directly compute the result of new_E
    E_test = dataset_union(dataset_difference(E, E_minus), E_plus)

    CR_test = CanonicalRepresentation(E_test, program)
    CR_test.initilization()

    start_time_test = time.time()

    I_test, common_test, varrho_left_test, left_period_test, left_len_test, varrho_right_test, right_period_test, right_len_test = find_periods(
        CR_test)

    end_time_test = time.time()

    print("\nThe new varrho left test:")
    print(varrho_left_test)
    print("\nThe new varrho right test:")
    print(varrho_right_test)

    print(f"DRed time:{end_time - start_time}")
    print(f"seminaive time:{end_time_test - start_time_test}")

    diff_A = dataset_difference(new_I, I_test)
    diff_B = dataset_difference(I_test, new_I)

    # print("\nThe new_I is:")
    # print_dataset(new_I)
    # print("\nThe I_test is:")
    # print_dataset(I_test)

    print("\nThe difference(incre - naive) is:")
    print_dataset(diff_A)
    print("\nThe difference(naive - incre) is:")
    print_dataset(diff_B)

    if diff_A is not None:
        for predicate in diff_A:
            for entity in diff_A[predicate]:
                for interval in diff_A[predicate][entity]:
                    F = Atom(predicate, entity, interval)
                    result_DRed = fact_entailment(new_I, F, CR.base_interval, I_left_period, I_left_len, I_right_period,
                                                  I_right_len)
                    result_se = fact_entailment(I_test, F, CR_test.base_interval, left_period_test, left_len_test,
                                                right_period_test, right_len_test)
                    if result_DRed != result_se:
                        print("\nThe result is not the same!")
                        print(F)
                        print(result_DRed)
                        print(result_se)
                    # assert result_DRed == result_se

    if diff_B is not None:
        for predicate in diff_B:
            for entity in diff_B[predicate]:
                for interval in diff_B[predicate][entity]:
                    F = Atom(predicate, entity, interval)
                    result_DRed = fact_entailment(new_I, F, CR.base_interval, I_left_period, I_left_len, I_right_period,
                                                  I_right_len)
                    result_se = fact_entailment(I_test, F, CR_test.base_interval, left_period_test, left_len_test,
                                                right_period_test, right_len_test)
                    if result_DRed != result_se:
                        print("\nThe result is not the same!")
                        print(F)
                        print(result_DRed)
                        print(result_se)
                    # assert result_DRed == result_se

    # print("\nThe datasets are the same?")
    # print(dataset_Same(new_I, I_test))
