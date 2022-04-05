import os
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

from model.DyCause.model.dycause import DyCause
from model.DyCause.utils.evaluation_function import pr_stat, my_acc, print_prk_acc

warnings.filterwarnings("ignore")


class RootCauseExp:
    def __init__(self, feature):
        self.feature = feature
        self.result = None
        self.ranked_nodes = None

        if not os.path.exists('./result/result/'):
            os.makedirs('./result/result/')

        self.out_path = './result/result/root_cause_result.csv'

    def location(self, df, error_times):
        data = df[self.feature].values
        self.result = pd.DataFrame()
        self.result['timestamp'] = error_times[:1]

        self.ranked_nodes = []
        for error_time in tqdm(error_times):
            error_time = df['timestamp'].values.tolist().index(error_time)
            dycause = DyCause(feature=self.feature)
            ranked_node = dycause.location(data=data, error_time=error_time)[:5]
            self.ranked_nodes.append(ranked_node)
            break
        self.result['ranked_nodes'] = list(map(lambda x: ' '.join(str(s[0]) for s in x), self.ranked_nodes))

    def evaluation(self, true_root_cause):
        prkS_list, acc_list = [], []
        self.result['label'] = true_root_cause[:1]
        for i in range(len(self.result)):
            prkS = pr_stat(self.ranked_nodes[i], true_root_cause.reshape(-1, 1)[i])
            acc = my_acc(self.ranked_nodes[i], true_root_cause.reshape(-1, 1)[i], len(self.feature))
            prkS_list.append(prkS)
            acc_list.append(acc)

        self.result['PR@1'] = np.array(prkS_list).T[0]
        self.result['PR@2'] = np.array(prkS_list).T[1]
        self.result['PR@3'] = np.array(prkS_list).T[2]
        self.result['PR@4'] = np.array(prkS_list).T[3]
        self.result['PR@5'] = np.array(prkS_list).T[4]
        self.result['Acc'] = acc_list

        self.result.to_csv(self.out_path, index=False)

        prkS = np.mean(np.array(prkS_list), axis=0).tolist()
        acc = np.mean(np.array(acc_list))
        print_prk_acc(prkS, acc)
