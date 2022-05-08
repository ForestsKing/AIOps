import pandas as pd
import warnings

from exp.log import LogExp
from exp.metric import MetricExp
from exp.metric_trace import MetricTraceExp
from exp.metric_trace_log import MetricTraceLogExp
from exp.root_cause import RootCauseExp
from exp.trace import TraceExp
from utils.setseed import set_seed

warnings.filterwarnings("ignore")

feature_metric = ['catalogue_latency', 'cpu_usage_carts', 'cpu_usage_carts_db', 'cpu_usage_catalogue',
                  'cpu_usage_catalogue_db', 'cpu_usage_ip6', 'cpu_usage_ip7', 'cpu_usage_orders',
                  'cpu_usage_orders_db', 'cpu_usage_payment', 'cpu_usage_rabbitmq', 'cpu_usage_session_db',
                  'cpu_usage_user', 'cpu_usage_user_db', 'frontend_latency', 'mem_usage_carts',
                  'mem_usage_catalogue', 'mem_usage_front_end', 'mem_usage_ip6', 'mem_usage_ip7',
                  'mem_usage_orders', 'mem_usage_payment', 'mem_usage_user', 'payment_latency',
                  'shipping_latency', 'user_latency']

feature_trace = ['carts -> carts', 'orders -> orders', 'orders -> payment', 'orders -> user', 'payment -> payment',
                 'root -> carts', 'root -> orders', 'root -> user', 'user -> user']

TRAIN = False
if __name__ == '__main__':
    set_seed(42)

    # metric
    print('====================Metric====================')
    train = pd.read_csv('dataset/processed/train/metrics/metrics_clean.csv')
    test = pd.read_csv('dataset/processed/test/metrics/metrics_clean.csv')
    label = pd.read_csv('dataset/processed/test/label.csv')

    exp = MetricExp(feature_metric)
    if TRAIN:
        exp.fit(train)
        exp.update_threshold(test.iloc[:180], label.iloc[:180])
    exp.detection(test.iloc[180:], label.iloc[180:])
    print('====================End====================', '\n\n\n\n\n')

    # trace
    print('====================Trace====================')
    train = pd.read_csv('dataset/processed/train/traces/traces_clean.csv')
    test = pd.read_csv('dataset/processed/test/traces/traces_clean.csv')
    label = pd.read_csv('dataset/processed/test/label.csv')

    exp = TraceExp(feature_trace)
    if TRAIN:
        exp.fit(train)
        exp.update_threshold(test.iloc[:180], label.iloc[:180])
    exp.detection(test.iloc[180:], label.iloc[180:])
    print('====================End====================', '\n\n\n\n\n')

    # log
    print('====================Log====================')
    train = pd.read_csv('./dataset/processed/train/logs/logs.csv')
    test = pd.read_csv('./dataset/processed/test/logs/logs.csv')
    label = pd.read_csv('./dataset/processed/test/label.csv')
    nEvent = len(pd.read_csv('./dataset/processed/tmp/log.log_templates.csv'))
    exp = LogExp(nEvent)
    if TRAIN:
        exp.fit(train)
        exp.update_threshold(test[test['timestamp'] < label['timestamp'].values[180]], label.iloc[:180])
    exp.detection(test[test['timestamp'] >= label['timestamp'].values[180]], label.iloc[180:])
    print('====================End====================', '\n\n\n\n\n')

    # metric & trace
    print('====================Metric & Trace====================')
    train_metric = pd.read_csv('dataset/processed/train/metrics/metrics_clean.csv')
    test_metric = pd.read_csv('dataset/processed/test/metrics/metrics_clean.csv')
    train_trace = pd.read_csv('dataset/processed/train/traces/traces_clean.csv')
    test_trace = pd.read_csv('dataset/processed/test/traces/traces_clean.csv')

    train = pd.merge(train_metric, train_trace, on='timestamp')
    test = pd.merge(test_metric, test_trace, on='timestamp')

    label = pd.read_csv('dataset/processed/test/label.csv')

    feature = feature_metric + feature_trace

    exp = MetricTraceExp(feature)
    if TRAIN:
        exp.fit(train)
        exp.update_threshold(test.iloc[:180], label.iloc[:180])
    exp.detection(test.iloc[180:], label.iloc[180:])
    print('====================End====================', '\n\n\n\n\n')

    # metric trace log
    print('====================Metric & Trace & Log====================')
    exp = MetricTraceLogExp()
    if TRAIN:
        exp.update_threshold()
    exp.detection()
    print('====================End====================', '\n\n\n\n\n')

    # root cause
    print('====================Root Cause====================')
    exp = RootCauseExp(input_path='./dataset/processed/root_cause/')
    exp.location()
    exp.evaluation()
    print('====================End====================')
