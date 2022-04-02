import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score


def best_threshold(test_scores, test_label, start=0.0001, end=100, search_step=10000, method='F1'):
    best_score = 0.0
    best_threshold = 0.0

    for i in range(search_step):
        threshold = start + i * ((end - start) / search_step)
        test_pred = (test_scores > threshold).astype(np.int)
        if method == 'F1':
            score = f1_score(test_label, test_pred)
        elif method == 'P':
            score = precision_score(test_label, test_pred)
        elif method == 'R':
            score = recall_score(test_label, test_pred)
        else:
            print('method=F1 or P or R')
            score = f1_score(test_label, test_pred)

        if score > best_score:
            best_threshold = threshold
            best_score = score

    return best_threshold
