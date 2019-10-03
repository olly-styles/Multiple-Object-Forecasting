import models
import datasets
import utils
import trainer
import numpy as np
import torch
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-model_path', help='Path to encoder and decoder models')
parser.add_argument('-data_path', help='Path to bounding box statistics and precomputed DTP features')

args = parser.parse_args()

batch_size = 1024
num_workers = 8
layers_enc = 1
layers_dec = 1
dropout_p = 0
num_hidden = 512

device = torch.device("cuda")
model_path = args.model_path
data_path = args.data_path

for detector in ['yolo', 'mask-rcnn']:
    for fold in [1, 2, 3]:

        print(detector + ' fold ' + str(fold))

        print('loading model')

        encoder = models.EncoderRNN(device, num_hidden, layers_enc)
        encoder = encoder.to(device)
        encoder = encoder.float()
        decoder = models.DecoderRNN(device, num_hidden, dropout_p, layers_dec)
        decoder = decoder.to(device)
        decoder = decoder.float()

        try:
            encoder_path = model_path + '/encoder_' + detector + str(fold) + '_gru.weights'
            decoder_path = model_path + '/decoder_' + detector + str(fold) + '_gru.weights'
            encoder.load_state_dict(torch.load(encoder_path))
            decoder.load_state_dict(torch.load(decoder_path))
        except Exception:
            print('Failed to load model from ' + str(model_path))
            exit()

        encoder.eval()
        decoder.eval()

        path = data_path + detector + '_features/fold' + str(fold) + '/'

        print('Loading data')

        try:
            train_boxes = np.load(path + 'fold_' + str(fold) + '_train_dtp_box_statistics.npy')
            test_boxes = np.load(path + 'fold_' + str(fold) + '_test_dtp_box_statistics.npy')
            test_labels = np.load(path + 'fold_' + str(fold) + '_test_dtp_targets.npy')
            test_dtp_features = np.load(path + 'fold_' + str(fold) + '_test_dtp_features.npy')

        except Exception:
            print('Failed to load data from ' + str(data_path))
            exit()

        # Normalize boxes
        for i in range(8):
            test_boxes[:, i, ] = (test_boxes[:, i, ] - train_boxes[:, i, ].mean()) / \
                train_boxes[:, i, ].std()
            train_boxes[:, i, ] = (train_boxes[:, i, ] - train_boxes[:,
                                                                     i, ].mean()) / train_boxes[:, i, ].std()

        loss_function = torch.nn.SmoothL1Loss()

        test_set = datasets.Simple_BB_Dataset(
            test_boxes, test_labels, test_dtp_features)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                                  shuffle=False, num_workers=num_workers)

        print('Getting predictions')

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

        print('Saving predictions')

        for i in range(1, 61):
            results_df['x1_' + str(i)] = predictions[:, i - 1, 0]
            results_df['y1_' + str(i)] = predictions[:, i - 1, 1]
            results_df['x2_' + str(i)] = predictions[:, i - 1, 2]
            results_df['y2_' + str(i)] = predictions[:, i - 1, 3]

        results_df.to_csv(
            './outputs/sted/test_' + detector + '_fold_' + str(fold) + '.csv', index=False)
