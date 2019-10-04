import models
import datasets
import utils
import trainer
import torch.optim as optim
import numpy as np
import torch
import pandas as pd
import argparse
import copy

parser = argparse.ArgumentParser()
parser.add_argument('-model_save_path', help='Path to save the encoder and decoder models')
parser.add_argument('-data_path', help='Path to bounding box statistics and precomputed DTP features')

args = parser.parse_args()

batch_size = 1024
learning_rate = 1e-3
weight_decay = 0
num_workers = 8
num_epochs = 20
layers_enc = 1
layers_dec = 1
dropout_p = 0
num_hidden = 512
normalize = True
device = torch.device("cuda")
model_save_path = args.model_save_path
data_path = args.data_path

for detector in ['yolo', 'mask-rcnn']:
    for fold in [1, 2, 3]:

        print(detector + ' fold ' + str(fold))

        encoder = models.EncoderRNN(device, num_hidden, layers_enc)
        encoder = encoder.to(device)
        encoder = encoder.float()
        decoder = models.DecoderRNN(device, num_hidden, dropout_p, layers_dec)
        decoder = decoder.to(device)
        decoder = decoder.float()

        path = data_path + detector + '_features/fold' + str(fold) + '/'

        print('Loading data')

        try:
            train_boxes = np.load(path + 'fold_' + str(fold) + '_train_dtp_box_statistics.npy')
            val_boxes = np.load(path + 'fold_' + str(fold) + '_val_dtp_box_statistics.npy')
            test_boxes = np.load(path + 'fold_' + str(fold) + '_test_dtp_box_statistics.npy')

            train_labels = np.load(path + 'fold_' + str(fold) + '_train_dtp_targets.npy')
            val_labels = np.load(path + 'fold_' + str(fold) + '_val_dtp_targets.npy')
            test_labels = np.load(path + 'fold_' + str(fold) + '_test_dtp_targets.npy')

            train_dtp_features = np.load(path + 'fold_' + str(fold) + '_train_dtp_features.npy')
            val_dtp_features = np.load(path + 'fold_' + str(fold) + '_val_dtp_features.npy')
            test_dtp_features = np.load(path + 'fold_' + str(fold) + '_test_dtp_features.npy')

        except Exception:
            print('Failed to load data from ' + str(data_path))
            exit()

        # Normalize boxes
        for i in range(8):
            val_boxes[:, i, ] = (val_boxes[:, i, ] - train_boxes[:, i, ].mean()) / \
                train_boxes[:, i, ].std()
            test_boxes[:, i, ] = (test_boxes[:, i, ] - train_boxes[:, i, ].mean()) / \
                train_boxes[:, i, ].std()
            train_boxes[:, i, ] = (train_boxes[:, i, ] - train_boxes[:,
                                                                     i, ].mean()) / train_boxes[:, i, ].std()

        loss_function = torch.nn.SmoothL1Loss()

        train_set = datasets.Simple_BB_Dataset(
            train_boxes, train_labels, train_dtp_features)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                   shuffle=True, num_workers=num_workers)

        val_set = datasets.Simple_BB_Dataset(
            val_boxes, val_labels, val_dtp_features)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                                 shuffle=False, num_workers=num_workers)
        test_set = datasets.Simple_BB_Dataset(
            test_boxes, test_labels, test_dtp_features)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                                  shuffle=False, num_workers=num_workers)

        optimizer_encoder = optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
        optimizer_decoder = optim.Adam(decoder.parameters(), lr=learning_rate, weight_decay=weight_decay)

        best_ade = np.inf
        for epoch in range(num_epochs):
            print('----------- EPOCH ' + str(epoch) + ' -----------')
            print('Training...')
            trainer.train_seqseq(encoder, decoder, device, train_loader, optimizer_encoder, optimizer_decoder,
                                 epoch, loss_function, learning_rate)
            print('Validating...')
            val_predictions, val_targets, val_ade, val_fde = trainer.test_seqseq(
                encoder, decoder, device, val_loader, loss_function, return_predictions=True)
            if epoch == 4:
                optimizer_encoder = optim.Adam(encoder.parameters(), lr=1e-4, weight_decay=weight_decay)
                optimizer_decoder = optim.Adam(decoder.parameters(), lr=1e-4, weight_decay=weight_decay)
            if epoch == 9:
                optimizer_encoder = optim.Adam(encoder.parameters(), lr=5e-5, weight_decay=weight_decay)
                optimizer_decoder = optim.Adam(decoder.parameters(), lr=5e-5, weight_decay=weight_decay)
            if epoch == 14:
                optimizer_encoder = optim.Adam(encoder.parameters(), lr=2.5e-5, weight_decay=weight_decay)
                optimizer_decoder = optim.Adam(decoder.parameters(), lr=2.5e-5, weight_decay=weight_decay)
            if val_ade < best_ade:
                best_encoder, best_decoder = copy.deepcopy(encoder), copy.deepcopy(decoder)
                best_ade = val_ade
                best_fde = val_fde
            print('Best validation ADE: ', np.round(best_ade, 1))
            print('Best validation FDE: ', np.round(best_fde, 1))

        print('Saving model weights to ', model_save_path)
        torch.save(encoder.state_dict(), model_save_path + '/encoder_' + detector + str(fold) + '_gru.weights')
        torch.save(decoder.state_dict(), model_save_path + '/decoder_' + detector + str(fold) + '_gru.weights')

        print('Testing...')
        encoder.eval()
        decoder.eval()
        predictions, targets, ade, fde = trainer.test_seqseq(
            encoder, decoder, device, test_loader, loss_function, return_predictions=True, phase='Test')

        print('Getting IOU metrics')

        # Predictions are reletive to constant velocity. To compute AIOU / FIOU, we need the constant velocity predictions.
        test_cv_preds = pd.read_csv('./outputs/constant_velocity/test_' + detector + '_fold_' + str(fold) + '.csv')
        results_df = pd.DataFrame()
        results_df['vid'] = test_cv_preds['vid'].copy()
        results_df['filename'] = test_cv_preds['filename'].copy()
        results_df['frame_num'] = test_cv_preds['frame_num'].copy()

        # First 3 columns are file info. Remaining columns are bounding boxes.
        test_cv_preds = test_cv_preds.iloc[:, 3:].values.reshape(len(test_cv_preds), -1, 4)
        predictions = predictions.reshape(-1, 240, order='A')
        predictions = predictions.reshape(-1, 4, 60)

        predictions = utils.xywh_to_x1y1x2y2(predictions)
        predictions = np.swapaxes(predictions, 1, 2)

        predictions = np.around(predictions).astype(int)

        predictions = test_cv_preds - predictions

        gt_df = pd.read_csv('./outputs/ground_truth/test_' + detector + '_fold_' + str(fold) + '.csv')
        gt_boxes = gt_df.iloc[:, 3:].values.reshape(len(gt_df), -1, 4)
        aiou = utils.calc_aiou(gt_boxes, predictions)
        fiou = utils.calc_fiou(gt_boxes, predictions)
        print('AIOU: ', round(aiou * 100, 1))
        print('FIOU: ', round(fiou * 100, 1))
