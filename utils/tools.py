def fillna_mean(train, test):
    for column in list(train.columns[train.isnull().sum() > 0]):
        mean_val = train[column].mean()
        train[column].fillna(mean_val, inplace=True)

    for column in list(test.columns[test.isnull().sum() > 0]):
        mean_val = train[column].mean()
        test[column].fillna(mean_val, inplace=True)
    return train, test
