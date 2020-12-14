# AUTHOR: 
# DATE: 22:25
# FILE NEME: test.py.py
# TOOL: PyCharm

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib as mpl

def box2lines(box:torch.Tensor)-> torch.Tensor:
    """convert box coordinate to corners

    Args:
        box (torch.Tensor): (B, N, 5) with x, y, w, h, alpha

    Returns:
        torch.Tensor: (B, N, 4, 2) corners
    """
    B = box.size()[0]
    x = box[..., 0:1]
    y = box[..., 1:2]
    w = box[..., 2:3]
    h = box[..., 3:4]
    alpha = box[..., 4:5] # (B, N, 1)
    x4 = torch.FloatTensor([0.5, -0.5, -0.5, 0.5]).unsqueeze(0).unsqueeze(0).to(box.device) # (1,1,4)
    x4 = x4 * w     # (B, N, 4)
    y4 = torch.FloatTensor([0.5, 0.5, -0.5, -0.5]).unsqueeze(0).unsqueeze(0).to(box.device)
    y4 = y4 * h     # (B, N, 4)
    corners = torch.stack([x4, y4], dim=-1)     # (B, N, 4, 2)
    sin = torch.sin(alpha)
    cos = torch.cos(alpha)
    row1 = torch.cat([cos, sin], dim=-1)
    row2 = torch.cat([-sin, cos], dim=-1)       # (B, N, 2)
    rot_T = torch.stack([row1, row2], dim=-2)   # (B, N, 2, 2)
    rotated = torch.bmm(corners.view([-1,4,2]), rot_T.view([-1,2,2]))
    rotated = rotated.view([B,-1,4,2])          # (B*N, 4, 2) -> (B, N, 4, 2)
    rotated[..., 0] += x
    rotated[..., 1] += y
    line = torch.cat([rotated, rotated[:, :, [1, 2, 3, 0], :]], dim=3)  # (B,N,4,4)
    return line.squeeze()



if __name__=='__main__':
    result = torch.load('boxes cpu.txt')
    box = result['box']
    target = result['target']
    lines_gt = box2lines(target)
    plt.figure()
    x = lines_gt[:, [0, 2]].view(-1)
    y = lines_gt[:, [1, 3]].view(-1)
    plt.plot(x, y, 'r')
    for i in range(70):
        line_box = box2lines(box[i*10, ...])
        x = line_box[:, [0, 2]]
        y = line_box[:, [1, 3]]
        plt.plot(x, y, 'g')
        plt.pause(1)
    plt.show()