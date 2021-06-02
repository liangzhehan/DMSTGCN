import pickle
import numpy as np
import os
import torch


class DataLoader(object):
    def __init__(self, xs, ys, batch_size, begin=0, days=288, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys
        self.ind = np.arange(begin, begin + self.size)
        self.days = days

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.ind = self.ind[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                i_i = self.ind[start_ind: end_ind, ...] % self.days
                # xi_i = np.tile(np.arange(x_i.shape[1]), [x_i.shape[0], x_i.shape[2], 1, 1]).transpose(
                #     [0, 3, 1, 2]) + self.ind[start_ind: end_ind, ...].reshape([-1, 1, 1, 1])
                # x_i = np.concatenate([x_i, xi_i % self.days / self.days, np.eye(7)[xi_i // self.days % 7].squeeze(-2)],
                #                      axis=-1)
                yield (x_i, y_i, i_i)
                self.current_ind += 1

        return _wrapper()


class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def load_dataset(dataset_dir, batch_size, valid_batch_size=None, test_batch_size=None, days=288, sequence=12,
                 in_seq=12):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x'][:, -in_seq:, :, 0:2]  # B T N F speed flow
        data['y_' + category] = cat_data['y'][:, :sequence, :, 0:1]
        if category == "train":
            data['scaler'] = StandardScaler(mean=cat_data['x'][..., 0].mean(), std=cat_data['x'][..., 0].std())
    for si in range(0, data['x_' + category].shape[-1]):
        scaler_tmp = StandardScaler(mean=data['x_train'][..., si].mean(), std=data['x_train'][..., si].std())
        for category in ['train', 'val', 'test']:
            data['x_' + category][..., si] = scaler_tmp.transform(data['x_' + category][..., si])
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size, days=days, begin=0)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], valid_batch_size, days=days,
                                    begin=data['x_train'].shape[0])
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size, days=days,
                                     begin=data['x_train'].shape[0] + data['x_val'].shape[0])
    return data


def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    return mae, mape, rmse
