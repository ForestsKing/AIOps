import numpy as np
from tabulate import tabulate


def prCal(scoreList, prk, rightOne):
    if rightOne in scoreList[:prk]:
        return 1
    else:
        return 0


def pr_stat(scoreList, rightOne, k=5):
    topk_list = range(1, k + 1)
    prkS = [0] * len(topk_list)
    for j, k in enumerate(topk_list):
        prkS[j] += prCal(scoreList, k, rightOne)
    return prkS


def print_prk_acc(prkS):
    headers = ['PR@{}'.format(i + 1) for i in range(len(prkS))] + ['PR@Avg']
    data = prkS + [np.mean(prkS)]
    print(tabulate([data], headers=headers, floatfmt="#06.4f"))
