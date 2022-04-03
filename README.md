| 训练集开始时间有突变               |      |      |      |
| ---------------------------------- | ---- | ---- | ---- |
| 测试集均值比训练集大               |      |      |      |
| 时间太短，部分指标的周期性无法体现 |      |      |      |
| 有的异常会引起一段时间的阶越变化   |      |      |      |



ssh://wcs@10.112.79.143:22/home/wcs/miniconda3/envs/py39/bin/python -u /home/wcs/demo/AIOps/main.py
100%|██████████████████████████████████████████| 62/62 [00:00<00:00, 174.63it/s]
Init || Total Loss| Train: 1.001636 Vali: 1.018296 || Forecast Loss| Train:1.005413 Valid: 1.023013 || Reconstruct Loss| Train: 0.998445 Valid: 1.013959
100%|███████████████████████████████████████████| 62/62 [00:01<00:00, 58.01it/s]
Epoch: 1 || Total Loss| Train: 0.884319 Vali: 0.930441 || Forecast Loss| Train:0.854478 Valid: 0.860052 || Reconstruct Loss| Train: 0.914959 Valid: 1.001419
Validation loss decreased (inf --> 0.930441).  Saving model ...
100%|███████████████████████████████████████████| 62/62 [00:01<00:00, 58.28it/s]
Epoch: 2 || Total Loss| Train: 0.814599 Vali: 0.896837 || Forecast Loss| Train:0.782634 Valid: 0.805721 || Reconstruct Loss| Train: 0.847350 Valid: 0.988277
Validation loss decreased (0.930441 --> 0.896837).  Saving model ...
100%|███████████████████████████████████████████| 62/62 [00:01<00:00, 58.37it/s]
Epoch: 3 || Total Loss| Train: 0.791788 Vali: 0.856239 || Forecast Loss| Train:0.753901 Valid: 0.784939 || Reconstruct Loss| Train: 0.830725 Valid: 0.928104
Validation loss decreased (0.896837 --> 0.856239).  Saving model ...
100%|███████████████████████████████████████████| 62/62 [00:01<00:00, 58.55it/s]
Epoch: 4 || Total Loss| Train: 0.775411 Vali: 0.859375 || Forecast Loss| Train:0.737668 Valid: 0.775639 || Reconstruct Loss| Train: 0.814038 Valid: 0.944084
EarlyStopping counter: 1 out of 7
100%|███████████████████████████████████████████| 62/62 [00:01<00:00, 58.58it/s]
Epoch: 5 || Total Loss| Train: 0.763884 Vali: 0.856650 || Forecast Loss| Train:0.723269 Valid: 0.769899 || Reconstruct Loss| Train: 0.805296 Valid: 0.944037
EarlyStopping counter: 2 out of 7
100%|███████████████████████████████████████████| 62/62 [00:01<00:00, 58.60it/s]
Epoch: 6 || Total Loss| Train: 0.754427 Vali: 0.857584 || Forecast Loss| Train:0.710599 Valid: 0.759256 || Reconstruct Loss| Train: 0.798692 Valid: 0.956759
EarlyStopping counter: 3 out of 7
100%|███████████████████████████████████████████| 62/62 [00:01<00:00, 58.63it/s]
Epoch: 7 || Total Loss| Train: 0.746187 Vali: 0.850831 || Forecast Loss| Train:0.698631 Valid: 0.756741 || Reconstruct Loss| Train: 0.794803 Valid: 0.945548
Validation loss decreased (0.856239 --> 0.850831).  Saving model ...
100%|███████████████████████████████████████████| 62/62 [00:01<00:00, 58.61it/s]
Epoch: 8 || Total Loss| Train: 0.741405 Vali: 0.852219 || Forecast Loss| Train:0.692437 Valid: 0.760845 || Reconstruct Loss| Train: 0.791119 Valid: 0.944237
EarlyStopping counter: 1 out of 7
100%|███████████████████████████████████████████| 62/62 [00:01<00:00, 58.59it/s]
Epoch: 9 || Total Loss| Train: 0.734381 Vali: 0.845009 || Forecast Loss| Train:0.684662 Valid: 0.755587 || Reconstruct Loss| Train: 0.784850 Valid: 0.934849
Validation loss decreased (0.850831 --> 0.845009).  Saving model ...
100%|███████████████████████████████████████████| 62/62 [00:01<00:00, 58.32it/s]
Epoch: 10 || Total Loss| Train: 0.732254 Vali: 0.851391 || Forecast Loss| Train:0.681354 Valid: 0.755392 || Reconstruct Loss| Train: 0.783711 Valid: 0.947827
EarlyStopping counter: 1 out of 7
100%|███████████████████████████████████████████| 62/62 [00:01<00:00, 58.34it/s]
Epoch: 11 || Total Loss| Train: 0.729849 Vali: 0.847246 || Forecast Loss| Train:0.678240 Valid: 0.751116 || Reconstruct Loss| Train: 0.782411 Valid: 0.943996
EarlyStopping counter: 2 out of 7
100%|███████████████████████████████████████████| 62/62 [00:01<00:00, 58.42it/s]
Epoch: 12 || Total Loss| Train: 0.724756 Vali: 0.847755 || Forecast Loss| Train:0.672004 Valid: 0.749751 || Reconstruct Loss| Train: 0.778329 Valid: 0.946488
EarlyStopping counter: 3 out of 7
100%|███████████████████████████████████████████| 62/62 [00:01<00:00, 58.29it/s]
Epoch: 13 || Total Loss| Train: 0.721406 Vali: 0.846654 || Forecast Loss| Train:0.667146 Valid: 0.751295 || Reconstruct Loss| Train: 0.776727 Valid: 0.942606
EarlyStopping counter: 4 out of 7
100%|███████████████████████████████████████████| 62/62 [00:01<00:00, 58.39it/s]
Epoch: 14 || Total Loss| Train: 0.717762 Vali: 0.851300 || Forecast Loss| Train:0.663263 Valid: 0.756579 || Reconstruct Loss| Train: 0.773123 Valid: 0.946956
EarlyStopping counter: 5 out of 7
100%|███████████████████████████████████████████| 62/62 [00:01<00:00, 58.76it/s]
Epoch: 15 || Total Loss| Train: 0.717756 Vali: 0.850685 || Forecast Loss| Train:0.664130 Valid: 0.751941 || Reconstruct Loss| Train: 0.772077 Valid: 0.949974
EarlyStopping counter: 6 out of 7
100%|███████████████████████████████████████████| 62/62 [00:01<00:00, 58.53it/s]
Epoch: 16 || Total Loss| Train: 0.717191 Vali: 0.851740 || Forecast Loss| Train:0.659909 Valid: 0.754338 || Reconstruct Loss| Train: 0.775251 Valid: 0.949546
EarlyStopping counter: 7 out of 7
100%|███████████████████████████████████████████| 10/10 [00:00<00:00, 97.22it/s]
Threshold is 1.670098
Valid || precision: 0.727273 recall: 0.800000 f1: 0.761905
100%|███████████████████████████████████████████| 32/32 [00:00<00:00, 96.23it/s]
Test || precision: 0.868421 recall: 0.776471 f1: 0.819876
100%|██████████████████████████████████████████| 62/62 [00:00<00:00, 195.55it/s]
Init || Total Loss| Train: 0.967512 Vali: 1.069971 || Forecast Loss| Train:0.967781 Valid: 1.065497 || Reconstruct Loss| Train: 0.969917 Valid: 1.076482
100%|███████████████████████████████████████████| 62/62 [00:01<00:00, 59.17it/s]
Epoch: 1 || Total Loss| Train: 0.959530 Vali: 1.080927 || Forecast Loss| Train:0.957440 Valid: 1.077266 || Reconstruct Loss| Train: 0.965473 Valid: 1.086679
Validation loss decreased (inf --> 1.080927).  Saving model ...
100%|███████████████████████████████████████████| 62/62 [00:01<00:00, 59.43it/s]
Epoch: 2 || Total Loss| Train: 0.951670 Vali: 1.062561 || Forecast Loss| Train:0.955685 Valid: 1.064047 || Reconstruct Loss| Train: 0.949941 Valid: 1.062585
Validation loss decreased (1.080927 --> 1.062561).  Saving model ...
100%|███████████████████████████████████████████| 62/62 [00:01<00:00, 59.53it/s]
Epoch: 3 || Total Loss| Train: 0.941675 Vali: 1.057874 || Forecast Loss| Train:0.944103 Valid: 1.057821 || Reconstruct Loss| Train: 0.942049 Valid: 1.060374
Validation loss decreased (1.062561 --> 1.057874).  Saving model ...
100%|███████████████████████████████████████████| 62/62 [00:01<00:00, 58.95it/s]
Epoch: 4 || Total Loss| Train: 0.943211 Vali: 1.051909 || Forecast Loss| Train:0.956458 Valid: 1.049239 || Reconstruct Loss| Train: 0.935295 Valid: 1.056084
Validation loss decreased (1.057874 --> 1.051909).  Saving model ...
100%|███████████████████████████████████████████| 62/62 [00:01<00:00, 58.83it/s]
Epoch: 5 || Total Loss| Train: 0.936330 Vali: 1.052177 || Forecast Loss| Train:0.941223 Valid: 1.051412 || Reconstruct Loss| Train: 0.933155 Valid: 1.054167
EarlyStopping counter: 1 out of 7
100%|███████████████████████████████████████████| 62/62 [00:01<00:00, 58.87it/s]
Epoch: 6 || Total Loss| Train: 0.937444 Vali: 1.043146 || Forecast Loss| Train:0.942818 Valid: 1.042642 || Reconstruct Loss| Train: 0.934506 Valid: 1.048044
Validation loss decreased (1.051909 --> 1.043146).  Saving model ...
100%|███████████████████████████████████████████| 62/62 [00:01<00:00, 58.90it/s]
Epoch: 7 || Total Loss| Train: 0.933113 Vali: 1.044992 || Forecast Loss| Train:0.937878 Valid: 1.044875 || Reconstruct Loss| Train: 0.930556 Valid: 1.048250
EarlyStopping counter: 1 out of 7
100%|███████████████████████████████████████████| 62/62 [00:01<00:00, 58.96it/s]
Epoch: 8 || Total Loss| Train: 0.932075 Vali: 1.056297 || Forecast Loss| Train:0.936631 Valid: 1.072344 || Reconstruct Loss| Train: 0.929616 Valid: 1.045474
EarlyStopping counter: 2 out of 7
100%|███████████████████████████████████████████| 62/62 [00:01<00:00, 59.38it/s]
Epoch: 9 || Total Loss| Train: 0.929041 Vali: 1.055973 || Forecast Loss| Train:0.934695 Valid: 1.068007 || Reconstruct Loss| Train: 0.925951 Valid: 1.047607
EarlyStopping counter: 3 out of 7
100%|███████████████████████████████████████████| 62/62 [00:01<00:00, 59.23it/s]
Epoch: 10 || Total Loss| Train: 0.927220 Vali: 1.052362 || Forecast Loss| Train:0.931318 Valid: 1.059368 || Reconstruct Loss| Train: 0.926309 Valid: 1.046736
EarlyStopping counter: 4 out of 7
100%|███████████████████████████████████████████| 62/62 [00:01<00:00, 58.93it/s]
Epoch: 11 || Total Loss| Train: 0.928510 Vali: 1.042050 || Forecast Loss| Train:0.933170 Valid: 1.042856 || Reconstruct Loss| Train: 0.926383 Valid: 1.044873
Validation loss decreased (1.043146 --> 1.042050).  Saving model ...
100%|███████████████████████████████████████████| 62/62 [00:01<00:00, 58.80it/s]
Epoch: 12 || Total Loss| Train: 0.924148 Vali: 1.052971 || Forecast Loss| Train:0.930421 Valid: 1.061522 || Reconstruct Loss| Train: 0.921030 Valid: 1.046991
EarlyStopping counter: 1 out of 7
100%|███████████████████████████████████████████| 62/62 [00:01<00:00, 58.86it/s]
Epoch: 13 || Total Loss| Train: 0.928170 Vali: 1.047302 || Forecast Loss| Train:0.933645 Valid: 1.048814 || Reconstruct Loss| Train: 0.925340 Valid: 1.047833
EarlyStopping counter: 2 out of 7
100%|███████████████████████████████████████████| 62/62 [00:01<00:00, 58.71it/s]
Epoch: 14 || Total Loss| Train: 0.926019 Vali: 1.048120 || Forecast Loss| Train:0.931530 Valid: 1.053854 || Reconstruct Loss| Train: 0.922446 Valid: 1.044417
EarlyStopping counter: 3 out of 7
100%|███████████████████████████████████████████| 62/62 [00:01<00:00, 59.32it/s]
Epoch: 15 || Total Loss| Train: 0.920369 Vali: 1.052642 || Forecast Loss| Train:0.925973 Valid: 1.060545 || Reconstruct Loss| Train: 0.917969 Valid: 1.046787
EarlyStopping counter: 4 out of 7
100%|███████████████████████████████████████████| 62/62 [00:01<00:00, 41.80it/s]
Epoch: 16 || Total Loss| Train: 0.921947 Vali: 1.047028 || Forecast Loss| Train:0.926755 Valid: 1.051771 || Reconstruct Loss| Train: 0.920523 Valid: 1.045653
EarlyStopping counter: 5 out of 7
100%|███████████████████████████████████████████| 62/62 [00:03<00:00, 18.93it/s]
Epoch: 17 || Total Loss| Train: 0.925680 Vali: 1.051084 || Forecast Loss| Train:0.935628 Valid: 1.059969 || Reconstruct Loss| Train: 0.919097 Valid: 1.043717
EarlyStopping counter: 6 out of 7
100%|███████████████████████████████████████████| 62/62 [00:03<00:00, 18.96it/s]
Epoch: 18 || Total Loss| Train: 0.925755 Vali: 1.053447 || Forecast Loss| Train:0.938210 Valid: 1.068052 || Reconstruct Loss| Train: 0.916750 Valid: 1.041080
EarlyStopping counter: 7 out of 7
100%|███████████████████████████████████████████| 10/10 [00:00<00:00, 32.46it/s]
Threshold is 0.250100
Valid || precision: 0.256410 recall: 1.000000 f1: 0.408163
100%|███████████████████████████████████████████| 32/32 [00:01<00:00, 31.92it/s]
Test || precision: 0.174538 recall: 1.000000 f1: 0.297203
100%|███████████████████████████████████████████| 62/62 [00:01<00:00, 61.66it/s]
Init || Total Loss| Train: 0.998664 Vali: 1.023156 || Forecast Loss| Train:1.001311 Valid: 1.026748 || Reconstruct Loss| Train: 0.996437 Valid: 1.019828
100%|███████████████████████████████████████████| 62/62 [00:03<00:00, 18.58it/s]
Epoch: 1 || Total Loss| Train: 0.908518 Vali: 0.939639 || Forecast Loss| Train:0.883045 Valid: 0.894894 || Reconstruct Loss| Train: 0.934539 Valid: 0.985036
Validation loss decreased (inf --> 0.939639).  Saving model ...
100%|███████████████████████████████████████████| 62/62 [00:03<00:00, 18.61it/s]
Epoch: 2 || Total Loss| Train: 0.847023 Vali: 0.951917 || Forecast Loss| Train:0.818414 Valid: 0.876935 || Reconstruct Loss| Train: 0.876633 Valid: 1.027733
EarlyStopping counter: 1 out of 7
100%|███████████████████████████████████████████| 62/62 [00:03<00:00, 18.62it/s]
Epoch: 3 || Total Loss| Train: 0.828943 Vali: 0.927076 || Forecast Loss| Train:0.799905 Valid: 0.872202 || Reconstruct Loss| Train: 0.858664 Valid: 0.982592
Validation loss decreased (0.939639 --> 0.927076).  Saving model ...
100%|███████████████████████████████████████████| 62/62 [00:03<00:00, 18.60it/s]
Epoch: 4 || Total Loss| Train: 0.815489 Vali: 0.926473 || Forecast Loss| Train:0.781411 Valid: 0.861820 || Reconstruct Loss| Train: 0.850130 Valid: 0.991548
Validation loss decreased (0.927076 --> 0.926473).  Saving model ...
100%|███████████████████████████████████████████| 62/62 [00:03<00:00, 18.54it/s]
Epoch: 5 || Total Loss| Train: 0.805591 Vali: 0.924564 || Forecast Loss| Train:0.769125 Valid: 0.858213 || Reconstruct Loss| Train: 0.842621 Valid: 0.991416
Validation loss decreased (0.926473 --> 0.924564).  Saving model ...
100%|███████████████████████████████████████████| 62/62 [00:03<00:00, 18.60it/s]
Epoch: 6 || Total Loss| Train: 0.799943 Vali: 0.915202 || Forecast Loss| Train:0.763010 Valid: 0.851379 || Reconstruct Loss| Train: 0.837834 Valid: 0.979905
Validation loss decreased (0.924564 --> 0.915202).  Saving model ...
100%|███████████████████████████████████████████| 62/62 [00:03<00:00, 18.63it/s]
Epoch: 7 || Total Loss| Train: 0.795111 Vali: 0.910423 || Forecast Loss| Train:0.758830 Valid: 0.848574 || Reconstruct Loss| Train: 0.832959 Valid: 0.972883
Validation loss decreased (0.915202 --> 0.910423).  Saving model ...
100%|███████████████████████████████████████████| 62/62 [00:03<00:00, 18.57it/s]
Epoch: 8 || Total Loss| Train: 0.786493 Vali: 0.909333 || Forecast Loss| Train:0.744433 Valid: 0.849520 || Reconstruct Loss| Train: 0.829397 Valid: 0.969678
Validation loss decreased (0.910423 --> 0.909333).  Saving model ...
100%|███████████████████████████████████████████| 62/62 [00:03<00:00, 18.61it/s]
Epoch: 9 || Total Loss| Train: 0.780753 Vali: 0.910493 || Forecast Loss| Train:0.735368 Valid: 0.848260 || Reconstruct Loss| Train: 0.826499 Valid: 0.973407
EarlyStopping counter: 1 out of 7
100%|███████████████████████████████████████████| 62/62 [00:03<00:00, 18.54it/s]
Epoch: 10 || Total Loss| Train: 0.775792 Vali: 0.905007 || Forecast Loss| Train:0.728942 Valid: 0.845289 || Reconstruct Loss| Train: 0.823424 Valid: 0.965164
Validation loss decreased (0.909333 --> 0.905007).  Saving model ...
100%|███████████████████████████████████████████| 62/62 [00:03<00:00, 18.59it/s]
Epoch: 11 || Total Loss| Train: 0.772763 Vali: 0.908627 || Forecast Loss| Train:0.723967 Valid: 0.853354 || Reconstruct Loss| Train: 0.822268 Valid: 0.964708
EarlyStopping counter: 1 out of 7
100%|███████████████████████████████████████████| 62/62 [00:03<00:00, 18.73it/s]
Epoch: 12 || Total Loss| Train: 0.768557 Vali: 0.903564 || Forecast Loss| Train:0.717600 Valid: 0.847638 || Reconstruct Loss| Train: 0.820180 Valid: 0.960114
Validation loss decreased (0.905007 --> 0.903564).  Saving model ...
100%|███████████████████████████████████████████| 62/62 [00:03<00:00, 18.68it/s]
Epoch: 13 || Total Loss| Train: 0.768386 Vali: 0.905812 || Forecast Loss| Train:0.718618 Valid: 0.845172 || Reconstruct Loss| Train: 0.818723 Valid: 0.966883
EarlyStopping counter: 1 out of 7
100%|███████████████████████████████████████████| 62/62 [00:03<00:00, 18.70it/s]
Epoch: 14 || Total Loss| Train: 0.764718 Vali: 0.906466 || Forecast Loss| Train:0.712617 Valid: 0.847427 || Reconstruct Loss| Train: 0.817335 Valid: 0.966036
EarlyStopping counter: 2 out of 7
100%|███████████████████████████████████████████| 62/62 [00:03<00:00, 18.65it/s]
Epoch: 15 || Total Loss| Train: 0.765726 Vali: 0.902621 || Forecast Loss| Train:0.715785 Valid: 0.845768 || Reconstruct Loss| Train: 0.816680 Valid: 0.959912
Validation loss decreased (0.903564 --> 0.902621).  Saving model ...
100%|███████████████████████████████████████████| 62/62 [00:03<00:00, 18.70it/s]
Epoch: 16 || Total Loss| Train: 0.761836 Vali: 0.902179 || Forecast Loss| Train:0.707932 Valid: 0.844574 || Reconstruct Loss| Train: 0.816341 Valid: 0.960336
Validation loss decreased (0.902621 --> 0.902179).  Saving model ...
100%|███████████████████████████████████████████| 62/62 [00:03<00:00, 18.68it/s]
Epoch: 17 || Total Loss| Train: 0.760546 Vali: 0.905813 || Forecast Loss| Train:0.706831 Valid: 0.850619 || Reconstruct Loss| Train: 0.814873 Valid: 0.961413
EarlyStopping counter: 1 out of 7
100%|███████████████████████████████████████████| 62/62 [00:03<00:00, 18.59it/s]
Epoch: 18 || Total Loss| Train: 0.760683 Vali: 0.902701 || Forecast Loss| Train:0.707729 Valid: 0.844690 || Reconstruct Loss| Train: 0.814501 Valid: 0.961339
EarlyStopping counter: 2 out of 7
100%|███████████████████████████████████████████| 62/62 [00:03<00:00, 18.71it/s]
Epoch: 19 || Total Loss| Train: 0.755162 Vali: 0.902540 || Forecast Loss| Train:0.698339 Valid: 0.845244 || Reconstruct Loss| Train: 0.812851 Valid: 0.960565
EarlyStopping counter: 3 out of 7
100%|███████████████████████████████████████████| 62/62 [00:03<00:00, 18.70it/s]
Epoch: 20 || Total Loss| Train: 0.755628 Vali: 0.902079 || Forecast Loss| Train:0.697122 Valid: 0.844536 || Reconstruct Loss| Train: 0.815024 Valid: 0.960496
Validation loss decreased (0.902179 --> 0.902079).  Saving model ...
100%|███████████████████████████████████████████| 62/62 [00:03<00:00, 18.73it/s]
Epoch: 21 || Total Loss| Train: 0.755458 Vali: 0.904082 || Forecast Loss| Train:0.698738 Valid: 0.846868 || Reconstruct Loss| Train: 0.812706 Valid: 0.961855
EarlyStopping counter: 1 out of 7
100%|███████████████████████████████████████████| 62/62 [00:03<00:00, 18.78it/s]
Epoch: 22 || Total Loss| Train: 0.753753 Vali: 0.902119 || Forecast Loss| Train:0.695491 Valid: 0.844741 || Reconstruct Loss| Train: 0.812767 Valid: 0.960163
EarlyStopping counter: 2 out of 7
100%|███████████████████████████████████████████| 62/62 [00:03<00:00, 18.77it/s]
Epoch: 23 || Total Loss| Train: 0.754505 Vali: 0.903561 || Forecast Loss| Train:0.696985 Valid: 0.845718 || Reconstruct Loss| Train: 0.812565 Valid: 0.962157
EarlyStopping counter: 3 out of 7
100%|███████████████████████████████████████████| 62/62 [00:03<00:00, 18.68it/s]
Epoch: 24 || Total Loss| Train: 0.751933 Vali: 0.906187 || Forecast Loss| Train:0.693507 Valid: 0.852786 || Reconstruct Loss| Train: 0.810826 Valid: 0.960264
EarlyStopping counter: 4 out of 7
100%|███████████████████████████████████████████| 62/62 [00:03<00:00, 18.55it/s]
Epoch: 25 || Total Loss| Train: 0.752622 Vali: 0.903791 || Forecast Loss| Train:0.693335 Valid: 0.848979 || Reconstruct Loss| Train: 0.812468 Valid: 0.958946
EarlyStopping counter: 5 out of 7
100%|███████████████████████████████████████████| 62/62 [00:03<00:00, 18.63it/s]
Epoch: 26 || Total Loss| Train: 0.752537 Vali: 0.902871 || Forecast Loss| Train:0.693357 Valid: 0.845619 || Reconstruct Loss| Train: 0.812482 Valid: 0.960798
EarlyStopping counter: 6 out of 7
100%|███████████████████████████████████████████| 62/62 [00:03<00:00, 18.68it/s]
Epoch: 27 || Total Loss| Train: 0.750693 Vali: 0.904669 || Forecast Loss| Train:0.691195 Valid: 0.845822 || Reconstruct Loss| Train: 0.810786 Valid: 0.964154
EarlyStopping counter: 7 out of 7
100%|███████████████████████████████████████████| 10/10 [00:00<00:00, 32.08it/s]
Threshold is 1.100099
Valid || precision: 0.689655 recall: 1.000000 f1: 0.816327
100%|███████████████████████████████████████████| 32/32 [00:01<00:00, 31.48it/s]
Test || precision: 0.787879 recall: 0.917647 f1: 0.847826

进程已结束,退出代码0


ssh://wcs@10.112.79.143:22/home/wcs/miniconda3/envs/py39/bin/python -u /home/wcs/demo/AIOps/main.py
100%|███████████████████████████████████████████| 10/10 [00:00<00:00, 28.35it/s]
Threshold is 1.590098
Valid || precision: 0.727273 recall: 0.800000 f1: 0.761905
100%|███████████████████████████████████████████| 32/32 [00:01<00:00, 31.86it/s]
Test || precision: 0.857143 recall: 0.776471 f1: 0.814815
100%|███████████████████████████████████████████| 10/10 [00:00<00:00, 32.70it/s]
Threshold is 0.410100
Valid || precision: 0.269841 recall: 0.850000 f1: 0.409639
100%|███████████████████████████████████████████| 32/32 [00:00<00:00, 32.35it/s]
Test || precision: 0.195704 recall: 0.964706 f1: 0.325397
100%|███████████████████████████████████████████| 10/10 [00:00<00:00, 32.16it/s]
Threshold is 1.100099
Valid || precision: 0.689655 recall: 1.000000 f1: 0.816327
100%|███████████████████████████████████████████| 32/32 [00:01<00:00, 31.85it/s]
Test || precision: 0.787879 recall: 0.917647 f1: 0.847826

进程已结束,退出代码0




ssh://wcs@10.112.79.143:22/home/wcs/miniconda3/envs/py39/bin/python -u /home/wcs/demo/AIOps/main.py
100%|████████████████████████████████████████| 368/368 [00:01<00:00, 245.12it/s]
Threshold is 0.346300
Valid || precision: 0.256410 recall: 1.000000 f1: 0.408163
100%|██████████████████████████████████████| 7823/7823 [00:31<00:00, 248.18it/s]
Valid || precision: 0.177358 recall: 0.989474 f1: 0.300800