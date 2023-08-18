import numpy as np
import pandas as pd

class SynData50Sources:
  def __init__(self):
    data = np.load('datasets/Synthetic_Data/data-50sources.npz', allow_pickle=True)
    self.data_lst = data['data_lst']
    self.n_replicates = data['n_replicates'].item()
    self.n_sources = data['n_sources'].item()
    self.source_size = data['Ts'].item()
    self.train_size = 100
    self.test_size = 50
    self.val_size = 50
    
  def get_train_test_val(self, m_sources=1):
    for i in range(self.n_replicates):
      data = self.data_lst[i]
      n_data_points = m_sources*self.source_size
      n_data_points_test = self.n_sources*self.source_size
      t, y, y_cf = data[:, 0][:, np.newaxis], data[:, 1][:, np.newaxis], data[:, 2][:, np.newaxis]
      mu_0, mu_1, x = data[:, 3][:, np.newaxis], data[:, 4][:, np.newaxis], data[:, 5:-10]
      R = data[:, -10:] # missing indicators
      x_orgi = x.copy()
      x[:,:10][R==0]=np.nan # there are 10 missing features

      idx_train = np.concatenate([list(range(i, i+self.train_size))
                                  for i in np.arange(0, n_data_points, self.source_size)])
      idx_test = np.concatenate([list(range(i+self.train_size, i+self.train_size+self.test_size))
                                  for i in np.arange(0, n_data_points_test, self.source_size)])
      idx_val = np.concatenate([list(range(i+self.train_size+self.test_size,i+self.train_size+self.test_size+self.val_size))
                                  for i in np.arange(0, n_data_points, self.source_size)])

      train = (x[idx_train], t[idx_train], y[idx_train]), (y_cf[idx_train], mu_0[idx_train], mu_1[idx_train], x_orgi[idx_train])
      test = (x[idx_test], t[idx_test], y[idx_test]), (y_cf[idx_test], mu_0[idx_test], mu_1[idx_test], x_orgi[idx_test])
      val = (x[idx_val], t[idx_val], y[idx_val]), (y_cf[idx_val], mu_0[idx_val], mu_1[idx_val], x_orgi[idx_val])
      yield train, test, val
    
    
class IHDP:
  def __init__(self):
    self.n_replicates = 10
    self.n_sources = 6
    self.source_size = 124
    self.train_size = 80
    self.test_size = 24
    self.val_size = 20
    
  def get_train_test_val(self, m_sources=1):
    for i in range(self.n_replicates):
      data = pd.read_csv('datasets/IHDP/csv/ihdp_missing_npci_{}.csv'.format(i+1),header=None).values[:self.source_size*self.n_sources,:]

      n_data_points = m_sources*self.source_size
      n_data_points_test = self.n_sources*self.source_size
      t, y, y_cf = data[:, 0][:, np.newaxis], data[:, 1][:, np.newaxis], data[:, 2][:, np.newaxis]
      mu_0, mu_1, x = data[:, 3][:, np.newaxis], data[:, 4][:, np.newaxis], data[:, 5:-4]
      R = data[:, -4:] # missing indicators
      x_orgi = x.copy()
      x[:,:4][R==0]=np.nan # there are 4 missing features

      idx_train = np.concatenate([list(range(i, i+self.train_size))
                                  for i in np.arange(0, n_data_points, self.source_size)])
      idx_test = np.concatenate([list(range(i+self.train_size, i+self.train_size+self.test_size))
                                  for i in np.arange(0, n_data_points_test, self.source_size)])
      idx_val = np.concatenate([list(range(i+self.train_size+self.test_size,i+self.train_size+self.test_size+self.val_size))
                                  for i in np.arange(0, n_data_points, self.source_size)])

      train = (x[idx_train], t[idx_train], y[idx_train]), (y_cf[idx_train], mu_0[idx_train], mu_1[idx_train], x_orgi[idx_train])
      test = (x[idx_test], t[idx_test], y[idx_test]), (y_cf[idx_test], mu_0[idx_test], mu_1[idx_test], x_orgi[idx_test])
      val = (x[idx_val], t[idx_val], y[idx_val]), (y_cf[idx_val], mu_0[idx_val], mu_1[idx_val], x_orgi[idx_val])
      yield train, test, val
