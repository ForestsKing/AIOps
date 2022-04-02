import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from torch import optim, nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.DeepLog.data.dataset import MyDataset
from model.DeepLog.model.deeplog import DeepLog
from model.DeepLog.utils.plot import plot_loss
from utils.earlystoping import EarlyStopping


class LogExp:
    def __init__(self, nEvent, error_rate=0.01, g_rate=0.1, epochs=100, batch_size=16, patience=7, lr=0.001, w=5,
                 verbose=True):
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.lr = lr
        self.w = w
        self.nEvent = nEvent
        self.g = int(nEvent * g_rate)
        self.error_rate = error_rate

        self.verbose = verbose

        if not os.path.exists('./checkpoint/model/'):
            os.makedirs('./checkpoint/model/')
        if not os.path.exists('./result/img/'):
            os.makedirs('./result/img/')
        if not os.path.exists('./result/result/'):
            os.makedirs('./result/result/')

        self.modelpath = './checkpoint/log_model.pkl'
        self.imgpath = './result/img/log_loss.png'
        self.resultpath = './result/result/log_result.csv'

        self._get_model()

    def _get_model(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DeepLog(input_size=self.nEvent, hidden_size=512).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda epoch: 0.75 ** ((epoch - 1) // 2))
        self.early_stopping = EarlyStopping(patience=self.patience, verbose=self.verbose, path=self.modelpath)

        self.criterion = nn.CrossEntropyLoss()

    def _process_one_batch(self, batch_x, batch_y):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.to(self.device)

        out = self.model(batch_x)
        loss = self.criterion(out, batch_y)

        return out, loss

    def fit(self, df):
        lossdict = {'train': [], 'valid': []}

        data = df['EventId'].values
        traindata = data[:int(0.7 * len(data))]
        validdata = data[int(0.7 * len(data)):]

        dataset_train = MyDataset(traindata, w=self.w, num_class=self.nEvent)
        dataset_valid = MyDataset(validdata, w=self.w, num_class=self.nEvent)
        dataloader_train = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True)
        dataloader_valid = DataLoader(dataset_valid, batch_size=self.batch_size, shuffle=True)

        # init loss
        self.model.eval()
        train_loss = []
        for (batch_x, batch_y) in tqdm(dataloader_train):
            _, loss = self._process_one_batch(batch_x, batch_y)
            train_loss.append(loss.item())

        valid_loss = []
        for (batch_x, batch_y) in dataloader_valid:
            _, loss = self._process_one_batch(batch_x, batch_y)
            valid_loss.append(loss.item())

        train_loss = np.sqrt(np.average(np.array(train_loss) ** 2))
        valid_loss = np.sqrt(np.average(np.array(valid_loss) ** 2))

        if self.verbose:
            print(
                "Init || Total Loss| Train: {0:.6f} Vali: {1:.6f}".format(train_loss, valid_loss))

        for e in range(self.epochs):
            self.model.train()
            train_loss = []
            for (batch_x, batch_y) in tqdm(dataloader_train):
                self.optimizer.zero_grad()
                _, loss = self._process_one_batch(batch_x, batch_y)
                train_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()

            self.model.eval()
            valid_loss = []
            for (batch_x, batch_y) in dataloader_valid:
                _, loss = self._process_one_batch(batch_x, batch_y)
                valid_loss.append(loss.item())

            train_loss = np.sqrt(np.average(np.array(train_loss) ** 2))
            valid_loss = np.sqrt(np.average(np.array(valid_loss) ** 2))

            lossdict['train'].append(train_loss)
            lossdict['valid'].append(valid_loss)

            if self.verbose:
                print(
                    "Epoch: {0} || Total Loss| Train: {1:.6f} Vali: {2:.6f}".format(e + 1, train_loss, valid_loss))

            self.early_stopping(valid_loss, self.model)
            if self.early_stopping.early_stop:
                break
            self.scheduler.step()
        self.model.load_state_dict(torch.load(self.modelpath))

        plot_loss(lossdict["train"], lossdict["valid"], self.imgpath)

    def detection(self, df, label):
        self.model.load_state_dict(torch.load(self.modelpath))

        data = df['EventId'].values
        dataset = MyDataset(data, w=self.w, num_class=self.nEvent)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        true, pred = [], []
        result = pd.DataFrame()
        for (batch_X, batch_y) in tqdm(dataloader):
            out, _ = self._process_one_batch(batch_X, batch_y)
            true.extend(batch_y.detach().cpu().numpy())
            pred.extend(out.detach().cpu().numpy())

        result['True'] = true
        result['Pred'] = pred
        result['Pred'] = result.apply(lambda x: 1 - int(x['True'] in x['Pred'].argsort()[-self.g:][::-1]), axis=1)
        result['timestamp'] = df['timestamp'].values[self.w:]
        result = result[['timestamp', 'Pred']].groupby('timestamp').apply(
            lambda x: np.sum(x) > len(x) * self.error_rate).reset_index()
        result['Pred'] = result['Pred'].astype(np.int)
        label = pd.merge(label, result, on='timestamp', how='left')
        label[['timestamp', 'label', 'Pred']].to_csv(self.resultpath, index=False)
        actual_label = label['label'].fillna(0).values
        test_pred = label['Pred'].fillna(0).values

        print("Test || precision: {0:.6f} recall: {1:.6f} f1: {2:.6f}".format(precision_score(actual_label, test_pred),
                                                                              recall_score(actual_label, test_pred),
                                                                              f1_score(actual_label, test_pred)))
