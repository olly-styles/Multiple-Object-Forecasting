from torch.utils.data import Dataset


class Simple_BB_Dataset(Dataset):
    def __init__(self, boxes, labels, dtp_features):
        """
        Args:
            boxes (np.array): file with bounding boxes with velocities
            labels (np.array): label file
            dtp_features (np.array): file name with pre-computed features from dtp

        """
        self.boxes = boxes
        self.labels = labels
        self.dtp_features = dtp_features

    def __len__(self):
        return len(self.boxes)

    def __getitem__(self, idx):
        '''
        Returns:
            sample (dict): Containing:
                features (np.array): bounding boxes with velocities
                dtp_features (np.array):  dtp_features of dim 2048
                label: bounding box label
        '''
        boxes = self.boxes[idx]
        labels = self.labels[idx]
        dtp_features = self.dtp_features[idx]

        sample = {'features': boxes, 'labels': labels, 'dtp_features': dtp_features}
        return sample
