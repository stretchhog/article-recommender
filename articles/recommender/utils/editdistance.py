import numpy as np


def levenshtein(source, target, sub_cost=2):
    if len(source) < len(target):
        return levenshtein(target, source)
    if len(target) == 0:
        return len(source)

    source = np.array(tuple(source))
    target = np.array(tuple(target))

    previous_row = np.arange(target.size + 1)
    for s in source:
        # Insertion (target grows longer than source):
        current_row = previous_row + 1

        # Substitution or matching:
        current_row[1:] = np.minimum(
                current_row[1:],
                np.add(previous_row[:-1], (target != s)*sub_cost))

        # Deletion (target grows shorter than source):
        current_row[1:] = np.minimum(
                current_row[1:],
                current_row[0:-1] + 1)

        previous_row = current_row

    return previous_row[-1]


def damerau_levenshtein(source, target, sub_cost=2, trans_cost=1):
    if len(source) < len(target):
        return levenshtein(target, source)
    if len(target) == 0:
        return len(source)

    source = np.array(tuple(source))
    target = np.array(tuple(target))

    previous_row = np.arange(target.size + 1)
    i = 0
    for s in source:
        transposition_row = previous_row
        previous_s = s

        # Insertion (target grows longer than source):
        current_row = previous_row + 1

        # Substitution or matching:
        current_row[1:] = np.minimum(
                current_row[1:],
                np.add(previous_row[:-1], (target != s)*sub_cost))

        # Deletion (target grows shorter than source):
        current_row[1:] = np.minimum(
                current_row[1:],
                current_row[0:-1] + 1)

        # Transposition
        if i > 1:
            current_row[2:] = np.minimum(
                    current_row[2:],
                    np.add(transposition_row[:-2], (target[1:] != previous_s)*trans_cost))

        previous_row = current_row
        i = i + 1

    return previous_row[-1]

damerau_levenshtein("enter", "etner")
