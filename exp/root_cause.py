import os
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

from model.Squeeze.model.param import Param
from model.Squeeze.model.squeeze import Squeeze
from model.Squeeze.utils.attribute_combination import AttributeCombination as AC
from model.Squeeze.utils.evaluation_function import pr_stat, print_prk_acc

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
            root_cause = AC.batch_to_string(model.root_cause[0])
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

        root_cause = []
        for timestamp in tqdm(timestamps):
            root_cause.append(self.executor(timestamp))

        root_cause = np.array(root_cause).T

        self.result = pd.DataFrame()
        self.result['timestamp'] = root_cause[0]
        self.result['predict'] = root_cause[1]

        label = pd.read_csv(self.input_path + 'injection_info.csv')
        self.result = pd.merge(label, self.result, on='timestamp', how='left')
        self.result.rename(columns={'root_cause': 'label'}, inplace=True)
        self.result['label'] = self.result['label'].apply(
            lambda x: 'device=' + x.split('_usage_')[0] + '&node=' + x.split('_usage_')[1])

    def evaluation(self):
        prkS_list = []
        for i in range(len(self.result)):
            prkS = pr_stat(self.result['predict'].apply(lambda x: x.split(';')).values[i],
                           self.result['label'].values[i])
            prkS_list.append(prkS)
        prkS = np.mean(np.array(prkS_list), axis=0).tolist()
        print_prk_acc(prkS)

        prkS_list = np.array(prkS_list).T
        self.result['PR@1'] = prkS_list[0]
        self.result['PR@2'] = prkS_list[1]
        self.result['PR@3'] = prkS_list[2]
        self.result['PR@4'] = prkS_list[3]
        self.result['PR@5'] = prkS_list[4]
        self.result[['timestamp', 'label', 'predict', 'PR@1', 'PR@2', 'PR@3', 'PR@4', 'PR@5']].to_csv(self.result_path,
                                                                                                      index=False)
