import numpy as np
import metrics
import torch
import torch.nn as nn
from tqdm import tqdm


def train_seqseq(encoder, decoder, device, train_loader, encoder_optimizer, decoder_optimizer, epoch, loss_function, learning_rate, timestep_norm=False):
    encoder.train()
    decoder.train()
    total_loss = 0
    ades = []
    fdes = []
    for batch_idx, data in enumerate(tqdm(train_loader)):
        # if batch_idx == 5:
        #     break
        features, labels, dtp_features = data['features'].to(
            device), data['labels'].to(device), data['dtp_features'].to(device)

        features = features.float()
        labels = labels.float()
        dtp_features = dtp_features.float()

        context = encoder(features)

        output = decoder(context, dtp_features, val=False)

        loss = loss_function(output, labels)

        ades.append(list(metrics.calc_ade(output.cpu().detach().numpy(), labels.cpu().detach().numpy(), return_mean=False)))
        fdes.append(list(metrics.calc_fde(output.cpu().detach().numpy(),
                                          labels.cpu().detach().numpy(), 60, return_mean=False)))

        # Backward and optimize
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

        # Clip gradients
        # nn.metrics.clip_grad_norm(encoder.parameters(), 1)
        # for p in encoder.parameters():
        #     p.data.add_(-learning_rate, p.grad.data)
        nn.utils.clip_grad_norm(decoder.parameters(), 1)
        for p in decoder.parameters():
            p.data.add_(-learning_rate, p.grad.data)

        total_loss += loss
    # Flatten lists
    ades = [item for sublist in ades for item in sublist]
    fdes = [item for sublist in fdes for item in sublist]

    print('Train ADE: ', np.round(np.mean(ades), 1))
    print('Train FDE: ', np.round(np.mean(fdes), 1))
    print('Train loss: ', total_loss.cpu().detach().numpy())


def test_seqseq(encoder, decoder, device, test_loader, loss_function, iou=False, timestep_norm=False, calc_iou=False, return_predictions=False, phase='Val'):
    '''
    Evaluates STED. AIOU / FIOU are not computed.
    args:
        model: STED model as defined in models
        device: GPU or CPU
        test_loader: Dataloader to produce stacks of optical flow images
        loss_function: eg. ADE
    returns:
        ADE and FDE at intervals of 5,10,15 frames into the future
        Outputs and targets 15 frames into the future
    '''
    encoder.eval()
    decoder.eval()
    ades = []
    fdes = []
    outputs = np.array([])
    targets = np.array([])
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(test_loader)):
            # if batch_idx == 5:
            #     break
            features, labels, dtp_features = data['features'].to(device), data['labels'].to(
                device), data['dtp_features'].to(device)
            features = features.float()
            labels = labels.float()
            dtp_features = dtp_features.float()
            context = encoder(features, val=True)
            output = decoder(context, dtp_features, val=True)
            ades.append(list(metrics.calc_ade(output.cpu().numpy(),
                                              labels.cpu().numpy(), return_mean=False)))
            fdes.append(list(metrics.calc_fde(output.cpu().numpy(),
                                              labels.cpu().numpy(), 60, return_mean=False)))
            if return_predictions:
                outputs = np.append(outputs, output.cpu().numpy())
                targets = np.append(targets, labels.cpu().numpy())

    #print('Val loss: ', test_loss.cpu().detach().numpy())

    # Flatten lists
    ades = [item for sublist in ades for item in sublist]
    fdes = [item for sublist in fdes for item in sublist]

    print(phase + ' ADE: ' + str(np.round(np.mean(ades), 1)))
    print(phase + ' FDE: ' + str(np.round(np.mean(fdes), 1)))

    return outputs, targets, np.mean(ades), np.mean(fdes)


def test_return_features(model, device, test_loader, loss_function, iou=False, print_frequency=10):
    '''
    Evaluates DTP
    args:
        model: DTP as defined in the FlowStream class
        device: GPU or CPU
        test_loader: Dataloader to produce stacks of optical flow images
        loss_function: eg. ADE
    returns:
        ADE and FDE at intervals of 5,10,15 frames into the future
        Outputs and targets 15 frames into the future
    '''
    model.eval()
    test_loss = 0
    all_features = np.array([])
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            if print_frequency:
                if batch_idx % print_frequency == 0:
                    print('Batch ', batch_idx, ' of ', len(test_loader))
            features, labels = data['features'].to(device), data['labels'].to(device)
            features = features.float()
            labels = labels.float()

            output, hidden = model(features, test=True)
            all_features = np.append(all_features, hidden[0].detach().cpu().numpy())
            output = output[:, :, 24:]
            labels = labels[:, :, 24:]
            # output = output[:,29:]
            # labels = labels[:,29:]

            loss = loss_function(output, labels)
            # print(all_features.shape)

            test_loss += loss
    print('Val loss: ', test_loss.cpu().detach().numpy())

    return test_loss.cpu().detach().numpy(), all_features
