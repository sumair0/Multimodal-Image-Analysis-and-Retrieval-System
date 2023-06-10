#!/usr/bin/env python
# coding: utf-8
import getopt
import math
import sys
import numpy as np
import pandas as pd
import networkx as nx


# run ASCOS++ on the graph
def ascospp(A):
    """
    apply naive ASCOS++ calculation (algorithm 2) in section 4

    input:
    A - a square symmertric adjacency matrix

    parameter:
    c - relative importance or discounted parameter
    c controls the relative importance between the direct neighbors and indirect neighbors

    output:
    S - ASCOS++ similarity score matrix
    """
    c = 0.5
    dim = math.sqrt(np.size(A))
    a = int(dim)
    s = np.zeros([a, a])
    con = 0  # convergent matrix: not changing much between the iterations
    while True:  # recursive
        for i in range(1, a):
            s_r = 1  # initial node is type1 - type1, whose ASCOS++ score is 1
            for j in range(1, a):
                if i == j:
                    s[i][j] = 1
                else:
                    w_i = A[i].sum()  # sum of the values in row i, consider to be w_i*
                    w_ij = A[i, j]  # weight of the edge from i to j
                    s_ij = s[j][i]
                    s_r += (w_ij / w_i) * (1 - math.exp(- w_ij)) * s_ij
                    s_r = c * s_r

                    s[i][j] = s_r
                    con = s_r / (s[i].sum())

                    print('Converged:', con)
        if con < 0.001:
            break  # the matrix changed almost nothing after adding the new s_ij, became convergent
    return s


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "t:m:n:")  # Grabbing all of the available flags
    except getopt.GetoptError:  # Throws an error for not having a correct flag, or a flag in the wrong position
        print('Usage: p2task8.py -t <csv file>, -n <val> -m <val>')
        sys.exit(2)  # Exit the program
    for opt, arg in opts:  # For every flag and arg
        if opt in ("-t"):  # If flag is -t
            typetype = arg  # Store the typetype matrix
        if opt in ("-n"):  # If flag is -n
            n = int(arg)  # Store n
        if opt in ("-m"):  # If flag is -m
            m = int(arg)  # Store m

    # n = int(n)  # the number of types to be considered
    # m = int(m)  # the number of the most significant types needed

    data = pd.read_csv(typetype, header=None)
    type_set = pd.read_csv(typetype, index_col=0, nrows=0).columns.tolist()  # get only the header of the input file

    # input_matrix = '/Users/apple/Desktop/type-type_similarity_matrix.csv'
    # type_set = ["cc", "con", "detail", "emboss", "jitter", "neg", "noise1", "noise2", "original",
    #            "poster", "rot", "smooth"]

    # keep top n most similar types from the input matrix
    # keep the n-biggest scores, and replace other scores with 0
    T = data.to_numpy()
    for v, i in enumerate(T):
        j = sorted(i, reverse=True)[:n]
        for u, k in enumerate(i):
            if k not in j:
                i[u] = 0
        T[v] = i

    df = pd.DataFrame.from_records(T)

    # turn the filtered matrix into a similarity graph
    df_adj = pd.DataFrame(df.to_numpy(), index=type_set, columns=type_set)
    G = nx.from_pandas_adjacency(df_adj)
    data = G.edges(data=True)
    print("data", data)
    print("df_adj", df_adj)

    # call ASCOS++ function on the graph G
    test_ascospp = ascospp(df_adj.to_numpy())
    print(test_ascospp)

    # find m significant types from the ASCOS++ score matrix for above score matrix
    # keep m-biggest scores and replace others with 0
    T = test_ascospp
    for v, i in enumerate(T):
        j = sorted(i, reverse=True)[:m]
        for u, k in enumerate(i):
            if k not in j:
                i[u] = 0
        T[v] = i
    test_result = pd.DataFrame.from_records(T)
    test_result.index = type_set
    test_result.columns = type_set
    print(test_result)

    # get row-column pairs for non-zero and non-one scores
    print([(test_result.index[x], test_result.columns[y])
           for x, y in np.argwhere(test_result > 0 and test_result != 1)])


if __name__ == "__main__":  # Setup for main
    main(sys.argv[1:])  # Call main with all args
