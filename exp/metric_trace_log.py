import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

from utils.evalmethods import best_threshold


class MetricTraceLogExp:
    def __init__(self, log_rate=0.25, verbose=True):
        self.log_rate = log_rate
        self.verbose = verbose

        valid_metric_trace = pd.read_csv('./result/result/metric_trace_result_valid.csv')
        valid_log = pd.read_csv('./result/result/log_result_valid.csv')
        test_metric_trace = pd.read_csv('./result/result/metric_trace_result.csv')
        test_log = pd.read_csv('./result/result/log_result.csv')

        valid_len = min(len(valid_metric_trace), len(valid_log))
        self.valid_metric_trace_score = valid_metric_trace['Score_Global'].values[-valid_len:]
        self.valid_log_score = valid_log['score'].values[-valid_len:]
        self.valid_label = valid_log['label'].values[-valid_len:]
        self.valid_time = valid_log['timestamp'].values[-valid_len:]
        self.valid_score = np.sqrt((self.valid_metric_trace_score ** 2 + self.log_rate * self.valid_log_score ** 2))

        test_len = min(len(test_metric_trace), len(test_log))
        self.test_metric_trace_score = test_metric_trace['Score_Global'].values[-test_len:]
        self.test_log_score = test_log['score'].values[-test_len:]
        self.test_label = test_log['label'].values[-test_len:]
        self.test_time = test_log['timestamp'].values[-test_len:]
        self.test_score = np.sqrt((self.test_metric_trace_score ** 2 + self.log_rate * self.test_log_score ** 2))

        if not os.path.exists('./result/result/'):
            os.makedirs('./result/result/')

        self.thresholdpath = './checkpoint/metric_trace_log_threshold.pkl'
        self.validresultpath = './result/result/metric_trace_log_result_valid.csv'
        self.resultpath = './result/result/metric_trace_log_result.csv'

    def update_threshold(self):
        result = pd.DataFrame()
        result['timestamp'] = self.valid_time
        result['metric_trace_score'] = self.valid_metric_trace_score
        result['log_score'] = self.valid_log_score
        result['log_rate'] = self.log_rate
        result['score'] = self.valid_score
        result['label'] = self.valid_label

        threshold = best_threshold(self.valid_score, self.valid_label)
        if self.verbose:
            print('Threshold is {0:.6f}'.format(threshold))

        valid_pred = (self.valid_score > threshold).astype(np.int)

        result["pred"] = valid_pred
        result["threshold"] = threshold

        result.to_csv(self.validresultpath, index=False)
        print(
            "Valid || precision: {0:.6f} recall: {1:.6f} f1: {2:.6f}".format(
                precision_score(self.valid_label, valid_pred),
                recall_score(self.valid_label, valid_pred),
                f1_score(self.valid_label, valid_pred)))
        joblib.dump(threshold, self.thresholdpath)

    def detection(self):
        result = pd.DataFrame()
        result['timestamp'] = self.test_time
        result['metric_trace_score'] = self.test_metric_trace_score
        result['log_score'] = self.test_log_score
        result['log_rate'] = self.log_rate
        result['score'] = self.test_score
        result['label'] = self.test_label

        threshold = joblib.load(self.thresholdpath)

        test_pred = (self.test_score > threshold).astype(np.int)

        result["pred"] = test_pred
        result["threshold"] = threshold
        result.to_csv(self.resultpath, index=False)

        print(
            "Valid || precision: {0:.6f} recall: {1:.6f} f1: {2:.6f}".format(
                precision_score(self.test_label, test_pred),
                recall_score(self.test_label, test_pred),
                f1_score(self.test_label, test_pred)))
