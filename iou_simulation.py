"""
all boxes follow the format [x, y, w, h]
rotation is not considered for simplicity
"""

import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import time
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'Rotated_IoU'))
from oriented_iou_loss import cal_giou, cal_diou, cal_iou
from typing import Tuple

#TARGER = np.array([10, 10, 1, 1], dtype=np.float32)     # use one target for simplicity
ASPECT_RATIO = np.array([4., 3., 2., 1., 1/2, 1/3, 1/4], dtype=np.float32)
SCALE = np.array([1/2, 2/3, 3/4, 1, 4/3, 3/2, 2], dtype=np.float32)
NUM_HEADING = 12
HEADING = np.arange(NUM_HEADING) * 2 * np.pi / NUM_HEADING
NUM_POINTS = 5000
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
EPOCHS = 2500
LR = 0.05


def main():
    # prepare data
    target = generate_gt()
    boxes = generate_boxes()
    vis_centers(boxes, tile="centers before training")
    boxes = torch.from_numpy(boxes).float().to(DEVICE)
    N = boxes.size(0)
    B = target.size(0)
    boxes = torch.unsqueeze(0).repeat(B,1,1)
    target = torch.from_numpy(target).float().to(DEVICE)
    target = target.unsqueeze(1).repeat(1, N, 1)
    print(target.size())
    print(boxes.size())

    # convert normal tensor to learnable parameter
    boxes = torch.nn.Parameter(boxes)     
    # use built-in optimizer instead of updating parameters manually 
    optimizer = torch.optim.Adam([boxes], lr = LR)  
    # the scheduler is empirically tuned
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [100, 200, 300], gamma = 0.5)

    begin = time.time()
    for i in range(EPOCHS):
        # clear gradient
        optimizer.zero_grad()
        iou,_,_,_ = cal_iou(boxes, target)
        mean_iou = torch.mean(iou)
        # use average giou as loss function
        mean_iou.backward()
        optimizer.step()
        if i%100 == 0:
            print("[{:04d}/{:04d}] mean diou loss: {:6f}. mean iou: {:6f}".format(
                i, EPOCHS, mean_iou.detach().item()
            ))
        # update learning rate
        lr_scheduler.step()

    print("training time: {:.02f} s".format(time.time() - begin))
    vis_centers(boxes.detach().cpu().numpy(), tile="centers after training")


def generate_gt():
    gt_w = np.sqrt(ASPECT_RATIO)
    gt_h = 1/gt_w
    gt_w = np.expand_dims(gt_w, axis=1)
    gt_h = np.expand_dims(gt_h, axis=1)
    gt_w = np.repeat(gt_w, NUM_HEADING, axis=1)     # (7, NUM_HEADING)
    gt_h = np.repeat(gt_h, NUM_HEADING, axis=1)     # (7, NUM_HEADING)

    gt_heading = np.expand_dims(HEADING, axis=0)
    gt_heading = np.repeat(gt_heading, 7, axis=0)   # (7, NUM_HEADING)

    gt_x = np.full_like(gt_w, 10)
    gt_y = np.full_like(gt_w,10)

    target = np.stack([gt_x, gt_y, gt_w, gt_h, gt_heading], axis=2)
    target = np.reshape(target, (-1,5))
    return target

def generate_boxes() -> np.ndarray:
    # convert aspect ratio to w and h
    w : np.ndarray = np.sqrt(ASPECT_RATIO)
    h : np.ndarray = 1/w

    w = np.expand_dims(w, axis = 0)
    h = np.expand_dims(h, axis = 0)
    scale = np.expand_dims(SCALE, axis = 1)
    w_scaled = scale * w        # (7, 7)
    h_scaled = scale * h        # (7, 7)
    w_scaled = np.reshape(w_scaled, (1, -1))  # (1, 49)
    h_scaled = np.reshape(h_scaled, (1, -1))  # (1, 49)
    w_scaled = np.repeat(w_scaled, NUM_POINTS, axis = 0)    # (NUM_POINTS, 49)
    h_scaled = np.repeat(h_scaled, NUM_POINTS, axis = 0)    # (NUM_POINTS, 49)

    # sample points in circle
    r = np.random.rand(NUM_POINTS, 1) * 10
    theta = np.random.rand(NUM_POINTS, 1) * 2 * np.pi
    x = r * np.cos(theta) + 10
    y = r * np.sin(theta) + 10
    x = np.repeat(x, 49, axis = 1)
    y = np.repeat(y, 49, axis = 1)

    boxes = np.stack([x, y, w_scaled, h_scaled], axis=2)
    boxes = np.reshape(boxes, (-1, 4))

    # BBox with rotation
    # heading angle 0-2pi, split into NUM_HEADING piece
    heading = np.expand_dims(HEADING, axis= 1)          # (NUM_HEADING,1)
    heading = np.expand_dims(heading, axis= 0)          # (1,NUM_HEADING,1)
    heading = np.repeat(heading, boxes.shape[0], axis=0) # (N,NUM_HEADING,1)

    boxes = np.expand_dims(boxes, axis= 1)
    boxes = np.repeat(boxes, NUM_HEADING, axis=1)       # (N,NUM_HEADING,4)
    boxes = np.concatenate((boxes, heading), axis=2)           # (N,NUM_HEADING,5)
    boxes = np.reshape(boxes, (-1, 5))
    print("generated boxes: ", boxes.shape)
    return boxes


# def box2corners(boxes:torch.Tensor) -> torch.Tensor:
#     """
#     1---0
#     |   |
#     2---3
#
#     Args:
#         boxes (torch.Tensor): [N, 4]
#
#     Returns:
#         torch.Tensor: [N, 4, 2]
#     """
#     signx = torch.FloatTensor([0.5, -0.5, -0.5, 0.5]).to(boxes.device).unsqueeze(0)     # (1, 4)
#     signy = torch.FloatTensor([0.5, 0.5, -0.5, -0.5]).to(boxes.device).unsqueeze(0)     # (1, 4)
#     w = boxes[:, 2:3]
#     h = boxes[:, 3:4]
#     center = boxes[..., 0:2]
#     center = center.unsqueeze(1).repeat(1, 4, 1)        # (N, 4, 2)
#     offsetx = signx * w     # (N, 4)
#     offsety = signy * h     # (N, 4)
#     offset = torch.stack([offsetx, offsety], dim=2)     # (N, 4, 2)
#     return offset + center
#
#
# def calculate_diou_loss(boxes:torch.Tensor, targets:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#     c_b = box2corners(boxes)
#     c_t = box2corners(targets)
#
#     # calculcate iou
#     area_b = boxes[:, 2] * boxes[:, 3]
#     area_t = targets[:, 2] * boxes[:, 3]
#     interx = torch.min(c_b[:, 3, 0], c_t[:, 3, 0]) - torch.max(c_b[:, 1, 0], c_t[:, 1, 0])
#     interx = torch.clamp_min(interx, min = 0.)
#     intery = torch.min(c_b[:, 1, 1], c_t[:, 1, 1]) - torch.max(c_b[:, 3, 1], c_t[:, 3, 1])
#     intery = torch.clamp_min(intery, min = 0.)
#     intersection = interx * intery
#     union = area_b + area_t - intersection
#     iou = intersection / union
#
#     # calculate distance
#     dist = boxes[:, :2] - targets[:, :2]
#     dist2 = torch.sum(dist * dist, dim = -1)
#     enclosex = torch.max(c_b[:, 3, 0], c_t[:, 3, 0]) - torch.min(c_b[:, 1, 0], c_t[:, 1, 0])
#     enclosey = torch.max(c_b[:, 1, 1], c_t[:, 1, 1]) - torch.min(c_b[:, 3, 1], c_t[:, 3, 1])
#     c2 = enclosex**2 + enclosey**2
#
#     diou_loss = 1 - iou + dist2 / c2
#     return diou_loss, iou


def vis_centers(boxes:np.ndarray, tile:str) -> None:
    """visualize centers of boxes

    Args:
        boxes (np.ndarray): (N, 4)
        tile (str): tile for the figure
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    # sample 5000 points. too many points makes matplotlib slow
    index = np.random.choice(len(x), 5000)
    plt.figure()
    plt.scatter(x[index], y[index])  
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    plt.title(tile)
    plt.show()


if __name__ == "__main__":
    main()

