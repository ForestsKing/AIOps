import os

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.MTAD_GAT.data.dataset import MyDataset
from model.MTAD_GAT.model.loss import JointLoss
from model.MTAD_GAT.model.mtad_gat import MTAD_GAT
from model.MTAD_GAT.utils.plot import plot_loss
from utils.earlystoping import EarlyStopping
from utils.evalmethods import best_threshold


class MetricExp:
    def __init__(self, feature, epochs=100, batch_size=16, patience=7, lr=0.001, w=30, gamma=0.1, r=0.5, verbose=True):
        self.feature = feature
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.w = w
        self.gamma = gamma
        self.lr = lr
        self.verbose = verbose
        self.r = r

        if not os.path.exists('./checkpoint/'):
            os.makedirs('./checkpoint/')
        if not os.path.exists('./result/img/'):
            os.makedirs('./result/img/')
        if not os.path.exists('./result/result/'):
            os.makedirs('./result/result/')

        self.modelpath = './checkpoint/metric_model.pkl'
        self.thresholdpath = './checkpoint/metric_threshold.pkl'
        self.trainimgpath = './result/img/metric_train_loss.png'
        self.validimgpath = './result/img/metric_valid_loss.png'
        self.validresultpath = './result/result/metric_result_valid.csv'
        self.resultpath = './result/result/metric_result.csv'

        self._get_model()

    def _get_model(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MTAD_GAT(n_features=len(self.feature), seq_len=self.w).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda epoch: 0.75 ** ((epoch - 1) // 2))
        self.early_stopping = EarlyStopping(patience=self.patience, verbose=self.verbose, path=self.modelpath)

        self.criterion = JointLoss(self.r)

    def _process_one_batch(self, batch_x, batch_y):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)

        reconstruct, forecast = self.model(batch_x)
        forecast_loss, reconstruct_loss, loss = self.criterion(batch_x, batch_y, reconstruct, forecast)

        return forecast_loss, reconstruct_loss, loss

    def _get_score(self, data, dataloader):
        self.model.eval()
        forecasts, reconstructs = [], []
        for (batch_x, batch_y) in tqdm(dataloader):
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)

            _, forecast = self.model(batch_x)

            recon_x = torch.cat((batch_x[:, 1:, :], batch_y), dim=1)
            reconstruct, _ = self.model(recon_x)

            forecasts.append(forecast.detach().cpu().numpy())
            reconstructs.append(reconstruct.detach().cpu().numpy()[:, -1, :])

        forecasts = np.concatenate(forecasts, axis=0).squeeze()
        reconstructs = np.concatenate(reconstructs, axis=0)
        actuals = data[self.w:]

        df = pd.DataFrame()
        scores = np.zeros_like(actuals)
        for i in range(actuals.shape[1]):
            df["For_" + self.feature[i]] = forecasts[:, i]
            df["Rec_" + self.feature[i]] = reconstructs[:, i]
            df["Act_" + self.feature[i]] = actuals[:, i]

            score = self.gamma * np.sqrt((forecasts[:, i] - actuals[:, i]) ** 2) + (1 - self.gamma) * np.sqrt(
                (reconstructs[:, i] - actuals[:, i]) ** 2)
            scores[:, i] = score
            df["Score_" + self.feature[i]] = score

        scores = np.mean(scores, axis=1)
        df['Score_Global'] = scores

        return df

    def fit(self, df):
        lossdict = {'train': {'forecast': [], 'reconstruct': [], 'total': []},
                    'valid': {'forecast': [], 'reconstruct': [], 'total': []}}

        data = df[self.feature].values
        traindata = data[:int(0.7 * len(data))]
        validdata = data[int(0.7 * len(data)):]

        dataset_train = MyDataset(traindata, w=self.w)
        dataset_valid = MyDataset(validdata, w=self.w)
        dataloader_train = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True)
        dataloader_valid = DataLoader(dataset_valid, batch_size=self.batch_size, shuffle=True)

        # init loss
        self.model.eval()
        train_forecast_loss, train_reconstruct_loss, train_loss = [], [], []
        for (batch_x, batch_y) in tqdm(dataloader_train):
            forecast_loss, reconstruct_loss, loss = self._process_one_batch(batch_x, batch_y)
            train_forecast_loss.append(forecast_loss.item())
            train_reconstruct_loss.append(reconstruct_loss.item())
            train_loss.append(loss.item())

        valid_forecast_loss, valid_reconstruct_loss, valid_loss = [], [], []
        for (batch_x, batch_y) in dataloader_valid:
            forecast_loss, reconstruct_loss, loss = self._process_one_batch(batch_x, batch_y)
            valid_forecast_loss.append(forecast_loss.item())
            valid_reconstruct_loss.append(reconstruct_loss.item())
            valid_loss.append(loss.item())

        train_forecast_loss = np.sqrt(np.average(np.array(train_forecast_loss) ** 2))
        valid_forecast_loss = np.sqrt(np.average(np.array(valid_forecast_loss) ** 2))
        train_reconstruct_loss = np.sqrt(np.average(np.array(train_reconstruct_loss) ** 2))
        valid_reconstruct_loss = np.sqrt(np.average(np.array(valid_reconstruct_loss) ** 2))
        train_loss = np.sqrt(np.average(np.array(train_loss) ** 2))
        valid_loss = np.sqrt(np.average(np.array(valid_loss) ** 2))

        if self.verbose:
            print(
                "Init || Total Loss| Train: {0:.6f} Vali: {1:.6f} || Forecast Loss| Train:{2:.6f} Valid: {3:.6f} || "
                "Reconstruct Loss| Train: {4:.6f} Valid: {5:.6f}".format(train_loss, valid_loss, train_forecast_loss,
                                                                         valid_forecast_loss, train_reconstruct_loss,
                                                                         valid_reconstruct_loss))

        for e in range(self.epochs):
            self.model.train()
            train_forecast_loss, train_reconstruct_loss, train_loss = [], [], []
            for (batch_x, batch_y) in tqdm(dataloader_train):
                self.optimizer.zero_grad()
                forecast_loss, reconstruct_loss, loss = self._process_one_batch(batch_x, batch_y)
                train_forecast_loss.append(forecast_loss.item())
                train_reconstruct_loss.append(reconstruct_loss.item())
                train_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()

            self.model.eval()
            valid_forecast_loss, valid_reconstruct_loss, valid_loss = [], [], []
            for (batch_x, batch_y) in dataloader_valid:
                forecast_loss, reconstruct_loss, loss = self._process_one_batch(batch_x, batch_y)
                valid_forecast_loss.append(forecast_loss.item())
                valid_reconstruct_loss.append(reconstruct_loss.item())
                valid_loss.append(loss.item())

            train_forecast_loss = np.sqrt(np.average(np.array(train_forecast_loss) ** 2))
            valid_forecast_loss = np.sqrt(np.average(np.array(valid_forecast_loss) ** 2))
            train_reconstruct_loss = np.sqrt(np.average(np.array(train_reconstruct_loss) ** 2))
            valid_reconstruct_loss = np.sqrt(np.average(np.array(valid_reconstruct_loss) ** 2))
            train_loss = np.sqrt(np.average(np.array(train_loss) ** 2))
            valid_loss = np.sqrt(np.average(np.array(valid_loss) ** 2))

            lossdict['train']['forecast'].append(train_forecast_loss)
            lossdict['train']['reconstruct'].append(train_reconstruct_loss)
            lossdict['train']['total'].append(train_loss)
            lossdict['valid']['forecast'].append(valid_forecast_loss)
            lossdict['valid']['reconstruct'].append(valid_reconstruct_loss)
            lossdict['valid']['total'].append(valid_loss)

            if self.verbose:
                print(
                    "Epoch: {0} || Total Loss| Train: {1:.6f} Vali: {2:.6f} || Forecast Loss| Train:{3:.6f} Valid"
                    ": {4:.6f} || Reconstruct Loss| Train: {5:.6f} Valid: {6:.6f}".format(e + 1, train_loss, valid_loss,
                                                                                          train_forecast_loss,
                                                                                          valid_forecast_loss,
                                                                                          train_reconstruct_loss,
                                                                                          valid_reconstruct_loss))

            self.early_stopping(valid_loss, self.model)
            if self.early_stopping.early_stop:
                break
            self.scheduler.step()
        self.model.load_state_dict(torch.load(self.modelpath))

        plot_loss(lossdict["train"]["forecast"], lossdict["train"]["reconstruct"], lossdict["train"]["total"],
                  self.trainimgpath)
        plot_loss(lossdict["valid"]["forecast"], lossdict["valid"]["reconstruct"], lossdict["valid"]["total"],
                  self.validimgpath)

    def update_threshold(self, df, label):
        self.model.load_state_dict(torch.load(self.modelpath))

        data = df[self.feature].values
        dataset = MyDataset(data, w=self.w)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        actual_label = label['label'].values[self.w:]
        validresult = self._get_score(data, dataloader)
        valid_score = validresult["Score_Global"].values

        threshold = best_threshold(valid_score, actual_label)
        if self.verbose:
            print('Threshold is {0:.6f}'.format(threshold))

        valid_pred = (valid_score > threshold).astype(np.int)

        validresult["Pred_Global"] = valid_pred
        validresult["Label_Global"] = actual_label
        validresult["Threshold_Global"] = threshold

        validresult.to_csv(self.validresultpath, index=False)
        print(
            "Valid || precision: {0:.6f} recall: {1:.6f} f1: {2:.6f}".format(precision_score(actual_label, valid_pred),
                                                                             recall_score(actual_label, valid_pred),
                                                                             f1_score(actual_label, valid_pred)))
        joblib.dump(threshold, self.thresholdpath)

    def detection(self, df, label):
        self.model.load_state_dict(torch.load(self.modelpath))
        threshold = joblib.load(self.thresholdpath)

        data = df[self.feature].values
        dataset = MyDataset(data, w=self.w)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        actual_label = label['label'].values[self.w:]
        testresult = self._get_score(data, dataloader)
        test_score = testresult["Score_Global"].values
        test_pred = (test_score > threshold).astype(np.int)

        testresult["Pred_Global"] = test_pred
        testresult["Label_Global"] = actual_label
        testresult["Threshold_Global"] = threshold

        testresult.to_csv(self.resultpath, index=False)
        print("Test || precision: {0:.6f} recall: {1:.6f} f1: {2:.6f}".format(precision_score(actual_label, test_pred),
                                                                              recall_score(actual_label, test_pred),
                                                                              f1_score(actual_label, test_pred)))
