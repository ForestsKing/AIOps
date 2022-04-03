import warnings

import pandas as pd

from exp.log import LogExp
from utils.setseed import set_seed

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    set_seed(42)
    # root cause
    # exp = RootCauseExp(input_path='./dataset/processed/root_cause/')
    # exp.location()
    # exp.evaluation()

    # log
    train = pd.read_csv('./dataset/processed/train/logs/logs.csv')
    test = pd.read_csv('./dataset/processed/test/logs/logs.csv')
    label = pd.read_csv('./dataset/processed/test/label.csv')
    nEvent = len(pd.read_csv('./dataset/processed/tmp/log.log_templates.csv'))
    exp = LogExp(nEvent)
    exp.fit(train)
    exp.update_threshold(test[test['timestamp'] < label['timestamp'].values[180]], label.iloc[:180])
    exp.detection(test[test['timestamp'] >= label['timestamp'].values[180]], label.iloc[180:])

'''
    # matric
    train = pd.read_csv('dataset/processed/train/metrics/metrics_clean.csv')
    test = pd.read_csv('dataset/processed/test/metrics/metrics_clean.csv')
    label = pd.read_csv('dataset/processed/test/label.csv')

    feature = pd.read_excel('./dataset/processed/tmp/info.xlsx')
    feature = feature[feature['use'] == 1]['metric'].values.tolist()

    exp = MetricExp(feature)
    exp.fit(train)
    exp.update_threshold(test.iloc[:180], label.iloc[:180])
    exp.detection(test.iloc[180:], label.iloc[180:])

    # trace
    train = pd.read_csv('dataset/processed/train/traces/traces_clean.csv')
    test = pd.read_csv('dataset/processed/test/traces/traces_clean.csv')
    label = pd.read_csv('dataset/processed/test/label.csv')

    feature = ['carts -> carts',
               'orders -> orders',
               'orders -> payment',
               'orders -> user',
               'payment -> payment',
               'root -> carts',
               'root -> orders',
               'root -> user',
               'user -> user']

    exp = TraceExp(feature)
    exp.fit(train)
    exp.update_threshold(test.iloc[:180], label.iloc[:180])
    exp.detection(test.iloc[180:], label.iloc[180:])

    # matric & trace
    train_metric = pd.read_csv('dataset/processed/train/metrics/metrics_clean.csv')
    test_metric = pd.read_csv('dataset/processed/test/metrics/metrics_clean.csv')
    train_trace = pd.read_csv('dataset/processed/train/traces/traces_clean.csv')
    test_trace = pd.read_csv('dataset/processed/test/traces/traces_clean.csv')

    train = pd.merge(train_metric, train_trace, on='timestamp')
    test = pd.merge(test_metric, test_trace, on='timestamp')

    label = pd.read_csv('dataset/processed/test/label.csv')

    feature = pd.read_excel('./dataset/processed/tmp/info.xlsx')
    feature = feature[feature['use'] == 1]['metric'].values.tolist()

    feature.extend(['carts -> carts',
                    'orders -> orders',
                    'orders -> payment',
                    'orders -> user',
                    'payment -> payment',
                    'root -> carts',
                    'root -> orders',
                    'root -> user',
                    'user -> user'])

    exp = MetricTraceExp(feature)
    exp.fit(train)
    exp.update_threshold(test.iloc[:180], label.iloc[:180])
    exp.detection(test.iloc[180:], label.iloc[180:])
'''
