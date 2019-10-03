import numpy as np


def get_iou(bboxes1, bboxes2):
    """
    Adapted from https://gist.github.com/zacharybell/8d9b1b25749fe6494511f843361bb167
    Calculates the intersection-over-union of two bounding boxes.
    Args:
        bbox1 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
        bbox2 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
    Returns:
        int: intersection-over-onion of bbox1, bbox2
    """
    ious = []
    for bbox1, bbox2 in zip(bboxes1, bboxes2):
        bbox1 = [float(x) for x in bbox1]
        bbox2 = [float(x) for x in bbox2]
        (x0_1, y0_1, x1_1, y1_1) = bbox1
        (x0_2, y0_2, x1_2, y1_2) = bbox2
        # get the overlap rectangle
        overlap_x0 = max(x0_1, x0_2)
        overlap_y0 = max(y0_1, y0_2)
        overlap_x1 = min(x1_1, x1_2)
        overlap_y1 = min(y1_1, y1_2)
        # check if there is an overlap
        if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
            ious.append(0)
            continue
        # if yes, calculate the ratio of the overlap to each ROI size and the unified size
        size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
        size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
        size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
        size_union = size_1 + size_2 - size_intersection
        iou = size_intersection / size_union
        ious.append(iou)
    return ious


def cxcywh_to_x1y1x2y2(boxes):
    old_boxes = boxes.copy()
    boxes[:, 0, :] = old_boxes[:, 0, :] - (old_boxes[:, 2, :] / 2)
    boxes[:, 2, :] = old_boxes[:, 0, :] + (old_boxes[:, 2, :] / 2)
    boxes[:, 1, :] = old_boxes[:, 1, :] - (old_boxes[:, 3, :] / 2)
    boxes[:, 3, :] = old_boxes[:, 1, :] + (old_boxes[:, 3, :] / 2)
    return boxes


def calc_fiou(outputs, targets, cv_preds, return_mean=True):
    '''
    Calculates the final IOU between outputs and
    targets
    Args:
        outputs: np array. 3D array format trajectory x timestep x x1y1x2y2
        targets: np array. 3D array format trajectory x timestep x x1y1x2y2
    Returns:
        1D array: Final IOU between outputs and targets
    '''
    outputs_copy = outputs.copy()
    targets_copy = targets.copy()
    cv_preds_copy = cv_preds.copy()

    outputs_copy = outputs_copy.reshape(-1, 240, order='A')
    outputs_copy = outputs_copy.reshape(-1, 4, 60)

    targets_copy = targets_copy.reshape(-1, 240, order='A')
    targets_copy = targets_copy.reshape(-1, 4, 60)

    cv_preds_copy = cv_preds_copy.reshape(-1, 240, order='A')
    cv_preds_copy = cv_preds_copy.reshape(-1, 4, 60, order='F')
    targets_copy = cxcywh_to_x1y1x2y2(targets_copy)
    outputs_copy = cxcywh_to_x1y1x2y2(outputs_copy)

    outputs_copy = (cv_preds_copy - outputs_copy)
    targets_copy = (cv_preds_copy - targets_copy)

    targets_copy = np.around(targets_copy).astype(int)
    outputs_copy = np.around(outputs_copy).astype(int)

    final_outputs = outputs_copy[:, :, -1]
    final_targets = targets_copy[:, :, -1]

    fiou = get_iou(final_outputs, final_targets)

    if return_mean:
        return np.mean(fiou)
    else:
        return fiou


def calc_aiou(outputs, targets, cv_preds, return_mean=True):
    '''
    Calculates the average IOU between outputs and
    targets
    Args:
        outputs: np array. 3D array format trajectory x timestep x x1y1x2y2
        targets: np array. 3D array format trajectory x timestep x x1y1x2y2
    Returns:
        1D array: Average IOU between outputs and targets
    '''
    outputs_copy = outputs.copy()
    targets_copy = targets.copy()
    cv_preds_copy = cv_preds.copy()

    outputs_copy = outputs_copy.reshape(-1, 240, order='A')
    outputs_copy = outputs_copy.reshape(-1, 4, 60)

    targets_copy = targets_copy.reshape(-1, 240, order='A')
    targets_copy = targets_copy.reshape(-1, 4, 60)

    cv_preds_copy = cv_preds_copy.reshape(-1, 240, order='A')
    cv_preds_copy = cv_preds_copy.reshape(-1, 4, 60, order='F')

    targets_copy = cxcywh_to_x1y1x2y2(np.around(targets_copy).astype(int))
    outputs_copy = cxcywh_to_x1y1x2y2(np.around(outputs_copy).astype(int))

    #print(targets_copy[0])

    outputs_copy = (cv_preds_copy - outputs_copy)
    targets_copy = (cv_preds_copy - targets_copy)

    # targets_copy = np.around(targets_copy).astype(int)
    # outputs_copy = np.around(outputs_copy).astype(int)

    # outputs_copy = outputs_copy.astype(int)
    # targets_copy = targets_copy.astype(int)

    # np.save('tar.npy', targets_copy)
    # np.save('pred.npy', outputs_copy)

    ious = np.zeros((targets_copy.shape[0], targets_copy.shape[2]))
    for t in range(targets_copy.shape[2]):
        t_outputs = outputs_copy[:, :, t]
        t_targets = targets_copy[:, :, t]
        t_iou = get_iou(t_outputs, t_targets)
        ious[:, t] = t_iou

    if return_mean:
        return np.mean(np.mean(ious, axis=1))
    else:
        return np.mean(ious, axis=1)


def calc_fde(outputs, targets, n, return_mean=True):
    '''
    Calculates the final displacement error (L2 distance) between outputs and
    targets (final output and final target)
    Args:
        outputs: np array. 1D array formated [x,x,x,x... y,y,y,y...]
        targets: np array. 1D array formated [x,x,x,x... y,y,y,y...]
        n: Number of predictions
    Returns:
        Final displacement error at n timesteps between outputs and targets
    '''

    # Reshape to [[x,y],[x,y],...)
    outputs = outputs.reshape(-1, n * 4, order='A')
    outputs = outputs.reshape(-1, 4, n)
    outputs = outputs[:, 0:2, :]

    # Reshape to [[x,y],[x,y],...)
    targets = targets.reshape(-1, n * 4, order='A')
    targets = targets.reshape(-1, 4, n)
    targets = targets[:, 0:2, :]

    # Get the final prediction
    outputs = outputs[:, :, -1]
    targets = targets[:, :, -1]

    # L2 Distance
    diff = (outputs - targets) * (outputs - targets)

    if return_mean:
        return np.mean(np.sqrt(np.sum(diff, axis=1)))
    else:
        return np.sqrt(np.sum(diff, axis=1))


def calc_ade(outputs, targets, return_mean=True):
    '''
    Calculates the average displacement error (L2 distance) between outputs and
    targets
    Args:
        outputs: np array. 1D array formated [x,x,x,x... y,y,y,y...]
        targets: np array. 1D array formated [x,x,x,x... y,y,y,y...]
    Returns:
        Final displacement error at n timesteps between outputs and targets
    '''
    # Reshape to [[x,y],[x,y],...)
    outputs = outputs.reshape(-1, 240, order='A')
    outputs = outputs.reshape(-1, 4, 60)
    # Just the centroids
    outputs = outputs[:, 0:2, :]
    # Reshape to [[x,y],[x,y],...)
    targets = targets.reshape(-1, 240, order='A')
    targets = targets.reshape(-1, 4, 60)
    targets = targets[:, 0:2, :]
    # Get the final prediction
    # outputs = outputs[:,:,-1]
    # targets = targets[:,:,-1]
    # L2 Distance

    out_mid_xs = outputs[:, 0, :]
    out_mid_ys = outputs[:, 1, :]
    tar_mid_xs = targets[:, 0, :]
    tar_mid_ys = targets[:, 1, :]

    diff = ((out_mid_xs - tar_mid_xs) * (out_mid_xs - tar_mid_xs)) + \
        ((out_mid_ys - tar_mid_ys) * (out_mid_ys - tar_mid_ys))

    if return_mean:
        return np.mean(np.sqrt(np.mean(diff, axis=1)))
    else:
        return np.sqrt(np.mean(diff, axis=1))
