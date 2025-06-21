"""
Convert molecular formula into vectors.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
from more_itertools import split_when, pairwise
from itertools import chain
from collections import Counter


def nest_brackets(tokens, i=0):
    l = []
    while i < len(tokens):
        if tokens[i] == ')':
            return i, l
        elif tokens[i] == '(':
            i, subl = nest_brackets(tokens, i + 1)
            l.append(subl)
        else:
            l.append(tokens[i])
        i += 1
    return i, l


def parse_compound(s):
    tokens = [''.join(t) for t in
              split_when(s, lambda a, b: b.isupper() or b in '()' or (b.isdigit() and not a.isdigit()))]
    tokens = [(int(t) if t.isdigit() else t) for t in tokens]
    i, l = nest_brackets(tokens)
    assert (i == len(tokens))  # crash if unmatched ')'
    return l


def count_elems(parsed_compound):
    c = Counter()
    for a, b in pairwise(chain(parsed_compound, (1,))):
        if not isinstance(a, int):
            subcounter = count_elems(a) if isinstance(a, list) else {a: 1}
            n = b if isinstance(b, int) else 1
            for elem, k in subcounter.items():
                c[elem] += k * n
    return c

import torch
def generate_formula(formula, Remained=None):
    if Remained is None:
        Remained = ["C", "H", "N", "O", "S", "F", "P", "Cl", "Br", "I"]
    elements = count_elems(parse_compound(formula))
    # Remained = ['C', 'O', 'N', 'P', 'S', 'F', 'Cl', 'Br', 'I', '&']
    formula_vec = []
    for element in Remained:
        formula_vec.append(elements[element])
    formula_vec = torch.tensor(formula_vec)
    formula_vec = formula_vec.unsqueeze(0)
    return formula_vec



if __name__ == '__main__':
    pass
    # Convert formula vecter for single formula
    Allowed_elements = ["C", "H", "N", "O", "S", "F", "P", "Cl", "Br", "I"]
    Formula = "NONa"
    elements = count_elems(parse_compound(Formula)).keys()
    print(any(item not in Allowed_elements for item in elements))
