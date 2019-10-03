from torch.utils.data import Dataset


class Simple_BB_Dataset(Dataset):
    def __init__(self, features, labels, dtp_features):
        """
        Args:
            filename (string): npy file name with data
            root_dir (string): Path to directory with the npy file.
        """
        self.features = features
        self.labels = labels
        self.dtp_features = dtp_features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        '''
        Returns:
            sample (dict): Containing:
                motion_features (np.array):  Motion features of dim 2048
                label: bounding box label
        '''
        features = self.features[idx]
        labels = self.labels[idx]
        dtp_features = self.dtp_features[idx]

        sample = {'features': features, 'labels': labels, 'dtp_features': dtp_features}
        return sample
