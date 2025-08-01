import os
import pickle
import torch
import numpy as np
import pandas as pd
from config.config import PROJECT_ROOT

from torch.utils.data import Dataset
from sklearn.impute import KNNImputer


class TrainSet(Dataset):
    def __init__(self, X, Y, prev_X, prev_Y):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()
        self.prev_X = torch.from_numpy(prev_X).float()
        self.prev_Y = torch.from_numpy(prev_Y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.prev_X[idx], self.prev_Y[idx]


class Dataset_Osaka_15min():
    def __init__(self, root_path, data_path, cache_dir):
        self.root_path = PROJECT_ROOT / root_path
        self.data_path = data_path
        self.cache_dir = PROJECT_ROOT / cache_dir

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_name = f"{self.data_path.split('.')[0]}.pkl"
        self.cache_file = self.cache_dir / cache_name

        self.__read_data__()

    def __read_data__(self):

        if self.__load_from_cache():
            print(f"Loaded cached data from {self.cache_file}")
            return

        print(f"Processing raw data (no cache found at {self.cache_file})")
        file_path = os.path.join(PROJECT_ROOT, self.root_path, self.data_path)
        df_raw = pd.read_csv(file_path)

        df = df_raw[['Time', 'in2temp', 'outtemp', 'Power']].copy()
        df.rename(columns={'Time': 'time', 'in2temp': 'intemp', 'outtemp': 'outtemp', 'Power': 'power'}, inplace=True)
        df['time'] = pd.to_datetime(df['time'])

        df = df.set_index('time')
        df = df.sort_index()

        df = df.resample('1h').mean()

        nan_columns = df.columns[df.isna().all()].tolist()
        if nan_columns:
            df[nan_columns] = df[nan_columns].fillna(0)

        imputer = KNNImputer(n_neighbors=2)
        df_imputed = imputer.fit_transform(df)
        df = pd.DataFrame(df_imputed, columns=df.columns, index=df.index)

        self.df = df

        self.train_X, self.train_Y, self.prev_X, self.prev_Y = self.__create_X_Y()

        self.__save_to_cache()

    def __create_X_Y(self, seq_len=24, pred_len=1):
        X = []
        Y = []
        prev_X = []
        prev_Y = []
        for i in range(len(self.df) - seq_len - pred_len + 1):
            X.append(self.df.iloc[i:i + seq_len].values)
            Y.append(self.df['intemp'].iloc[i + seq_len:i + seq_len + pred_len].values)
            if i == 0:
                prev_X.append(np.zeros_like(self.df.iloc[i:i + seq_len].values))
                prev_Y.append(np.zeros_like(Y[-1]))
            else:
                prev_X.append(self.df.iloc[i - 1:i - 1 + seq_len].values)
                prev_Y.append(self.df['intemp'].iloc[i - 1 + seq_len:i - 1 + seq_len + pred_len].values)

        return np.array(X), np.array(Y), np.array(prev_X), np.array(prev_Y)

    def __load_from_cache(self):
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.df = cached_data['df']
                    self.train_X = cached_data['train_X']
                    self.train_Y = cached_data['train_Y']
                    self.prev_X = cached_data.get('prev_X', None)
                    self.prev_Y = cached_data.get('prev_Y', None)
                    return True
            except:
                print(f"Warning: Failed to load cache from {self.cache_file}")
                return False
        return False

    def __save_to_cache(self):
        cache_data = {
            'df': self.df,
            'train_X': self.train_X,
            'train_Y': self.train_Y,
            'prev_X': self.prev_X,
            'prev_Y': self.prev_Y
        }

        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"Saved processed data to cache at {self.cache_file}")
        except Exception as e:
            print(f"Warning: Failed to save cache to {self.cache_file}: {str(e)}")
