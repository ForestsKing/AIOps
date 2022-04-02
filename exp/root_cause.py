import os
import warnings
from functools import reduce

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from model.Squeeze.model.param import Param
from model.Squeeze.model.squeeze import Squeeze
from model.Squeeze.utils.attribute_combination import AttributeCombination as AC
from model.Squeeze.utils.compare import compare

warnings.filterwarnings("ignore")


class RootCauseExp:
    def __init__(self, input_path, num_workers=10):
        self.input_path = input_path
        self.num_workers = num_workers

        if not os.path.exists('./result/result/'):
            os.makedirs('./result/result/')
        self.result_path = './result/result/root_cause_result.csv'

    def executor(self, timestamp):
        df = pd.read_csv(self.input_path + timestamp + '.csv')
        model = Squeeze(data=df, param=Param())
        model.run()

        try:
            root_cause = AC.batch_to_string(
                frozenset(reduce(lambda x, y: x.union(y), model.root_cause, set())))
        except IndexError:
            root_cause = ""

        return [timestamp, root_cause]

    def location(self):
        filenames = sorted(os.listdir(self.input_path))[:-1]
        try:
            filenames.remove('.ipynb_checkpoints')
        except:
            pass
        timestamps = list(map(lambda x: x.split('.')[0], filenames))

        root_cause = Parallel(n_jobs=self.num_workers, backend="multiprocessing")(
            delayed(self.executor)(timestamp)
            for timestamp in tqdm(timestamps))
        root_cause = np.array(root_cause).T

        result = pd.DataFrame()
        result['timestamp'] = root_cause[0]
        result['predict'] = root_cause[1]

        label = pd.read_csv(self.input_path + 'injection_info.csv')
        result = pd.merge(label, result, on='timestamp', how='left')
        result.rename(columns={'root_cause': 'label'}, inplace=True)
        result['label'] = result['label'].apply(lambda x: 'device=' + x.split('_usage_')[0] + '&node=' + x.split('_usage_')[1])
        result[['timestamp', 'label', 'predict']].to_csv(self.result_path, index=False)

    def evaluation(self):
        df = pd.read_csv(self.result_path)

        df['predict'].fillna("", inplace=True)
        df['FN'] = df.apply(lambda x: compare(x['label'], x['predict'], columns=['device', 'node'])[0], axis=1)
        df['TP'] = df.apply(lambda x: compare(x['label'], x['predict'], columns=['device', 'node'])[1], axis=1)
        df['FP'] = df.apply(lambda x: compare(x['label'], x['predict'], columns=['device', 'node'])[2], axis=1)

        df[['timestamp', 'label', 'predict', 'FN', 'TP', 'FP']].to_csv(self.result_path,index=False)

        f1 = 2 * np.sum(df['TP']) / (2 * np.sum(df['TP']) + np.sum(df['FP']) + np.sum(df['FN']))
        precision = np.sum(df['TP']) / (np.sum(df['TP']) + np.sum(df['FP']))
        recall = np.sum(df['TP']) / (np.sum(df['TP']) + np.sum(df['FN']))

        print("f1: %.4f" % f1)
        print("precision: %.4f" % precision)
        print("recall: %.4f" % recall)
