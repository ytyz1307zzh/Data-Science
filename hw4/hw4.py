"""
Homework 4: implement Apriori from scratch
"""

# -------------- Prof. Jiang's code ---------------
TDB = [['a', 'b'],
       ['b', 'c', 'd'],
       ['a', 'c', 'd', 'e'],
       ['a', 'd', 'e'],
       ['a', 'b', 'c'],
       ['a', 'b', 'c', 'd'],
       ['a'],
       ['a', 'b', 'c'],
       ['a', 'b', 'd'],
       ['b', 'c', 'e']]
min_sup = 2
# -------------- Prof. Jiang's code ---------------


def same_prefix(x: str, y: str) -> bool:
    """
    Check whether x and y have the same (k-1)-prefix
    """
    assert len(x) == len(y)
    n = len(x)
    return x[:n-1] == y[:n-1]


def get_freq(x: str) -> int:
    """
    Calculate the frequency of itemset x in TDB
    """
    freq = 0

    for trans in TDB:
        if all(x_i in trans for x_i in x):
            freq += 1

    return freq


# 1-itemset
k = 1
cand_set = list(set([x for li in TDB for x in li]))
freq_set = []

for cand in cand_set:
    # check the frequency of 1-itemsets
    cand_freq = get_freq(cand)
    if cand_freq >= min_sup:
        freq_set.append(cand)
        print(f"Itemset {cand}: frequency {cand_freq}")

print(f"{k}-frequent set: ", freq_set)

while len(freq_set) > 0:
    # generate candidate (k+1)-frequent set from k-frequent set
    cand_set, new_freq_set = [], []
    freq_set = sorted(freq_set)  # sort the itemset in ascending order

    i = 0  # iteration pointer
    while i != len(freq_set):
        j = i + 1  # another iteration pointer
        while j != len(freq_set):
            if same_prefix(freq_set[i], freq_set[j]):
                # append the last character from j to i to form a new candidate itemset
                cand_set.append(freq_set[i] + freq_set[j][k-1])
                j += 1
            else:
                break  # since the list is sorted, the remaining ones cannot have the same prefix with i
        i += 1
    print(f"{k + 1}-candidate set: ", cand_set)

    for cand in cand_set:
        # pruning: check the frequency of each candidate
        cand_freq = get_freq(cand)
        if cand_freq >= min_sup:
            new_freq_set.append(cand)
            print(f"Itemset {cand}: frequency {cand_freq}")

    freq_set = new_freq_set  # (k+1)-frequent set
    k += 1
    print(f"{k}-frequent set: ", freq_set)

