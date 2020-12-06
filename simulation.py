# AUTHOR: Si Ni
# DATE: 10:52
# FILE NEME: simulation.py.py
# TOOL: PyCharm



import torch
import numpy as np
import sys
import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'Rotated_IoU'))
from oriented_iou_loss import cal_giou, cal_diou, cal_iou

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 5000 points in circle with radius 3, central point (10,10)
r = 3 * torch.rand((5000,1)).to(device)        # (5000,1)
theta = 2 * np.pi * torch.rand((5000,1)).to(device)    # (5000,1)
x = 10 + r * torch.cos(theta)     # (5000,1)
y = 10 + r * torch.sin(theta)      # (5000,1)


# 7 aspect ratios(AR)
c1 = np.sqrt(3)
c2 = np.sqrt(2)
A = torch.Tensor([[0.5, 2.0], [c1/3, c1],[c2/2, c2], [1,1], [c2,c2/2], [c1,c1/3], [2.0, 0.5]]).to(device)  # (7,2)
# 7 scale of anchor box
S = torch.Tensor([0.5, 0.67, 0.75, 1, 1.33, 1.5, 2.0]).to(device)  # (7)
# heading angles (12)
H = np.pi/12 * torch.range(0,11).to(device)

# gt
gt_xy = torch.Tensor([[10,10]]).unsqueeze(0).repeat(7,12,1)    # (1,2)--> (7,12,2)
gt_wh = A.unsqueeze(1).repeat(1,12,1)       # (7,2)--->(7,12,2)
gt_heading = H.unsqueeze(0).unsqueeze(-1).repeat(7,1,1)  # (12)--->(7,12,1)
gt = torch.cat((gt_xy, gt_wh, gt_heading), dim = -1)   # (7,12,5)

# anchor box
pred_xy = torch.cat((x,y), -1).unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1,7,7,12,1)    # (5000,2) ---> (5000,7,7,12,2)
pred_wh = (A.unsqueeze(1).repeat(1,7,1) * S.unsqueeze(0).unsqueeze(-1).repeat(7,1,2)).unsqueeze(0).unsqueeze(-2).repeat(5000,1,1,12,1)     # (7,7,2)---> (5000,7,7,12,2)
pred_heading =  H.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(5000,7,7,1,1)  # (12)---> (5000,7,7,12,1)
pred = torch.cat((pred_xy, pred_wh, pred_heading), dim=-1)    # (5000,7,7,12,5)

pred = pred.expand(7,12,5000,7,7,12,5).requires_grad_()             # (7,12,5000,7,7,12,5)
gt = gt.unsqueeze(-2).unsqueeze(-2).unsqueeze(-2).unsqueeze(-2).repeat(1,1,5000,7,7,12,1)  #(7,12,5000,7,7,12,5)
# can view to (7*12, 5000, 7*7, 12, 5)

lr_decay_steps = [80, 160, 180]

def regression(gt, pred, iou_type:str = 'iou' ):
    lr = 0.1
    loss = torch.zeros(pred.size())
    for it in range(200):
        if it in lr_decay_steps:
            lr *= 0.1
        iou = cal_iou(pred, gt)
        if iou_type == 'iou':
            iou.backward(torch.ones(iou.size()))         # 可以这么写吗？
        elif iou_type == 'giou':
            giou = cal_giou(pred,gt)
            giou.backward(torch.ones(giou.size()))
        elif iou_type == 'diou':
            diou = cal_diou(pred,gt)
            diou.backward(torch.ones(diou.size()))
        else:
            ValueError("unknown iou type")
        pred += lr * (2. - iou) * pred.grad
        offsets = torch.abs(pred - gt)  # (...,5)
        # save offests

        loss += offsets
    return loss

if __name__=='__main__':
    loss = regression(gt, pred, 'iou')
    iou_loss = torch.sum(loss, -1)

    loss = regression(gt, pred, 'giou')
    giou_loss = torch.sum(loss, -1)

    loss = regression(gt, pred, 'diou')
    diou_loss = torch.sum(loss, -1)
