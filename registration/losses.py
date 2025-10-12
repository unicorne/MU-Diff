# losses.py
import torch
import torch.nn.functional as F

def ncc_loss(x, y, win=9, eps=1e-5):
    # x,y: (B,1,H,W)
    pad = win//2
    filt = torch.ones(1,1,win,win, device=x.device)
    x2, y2, xy = x*x, y*y, x*y
    sum_x  = F.conv2d(x,  filt, padding=pad)
    sum_y  = F.conv2d(y,  filt, padding=pad)
    sum_x2 = F.conv2d(x2, filt, padding=pad)
    sum_y2 = F.conv2d(y2, filt, padding=pad)
    sum_xy = F.conv2d(xy, filt, padding=pad)
    N = win*win
    u_x = sum_x / N
    u_y = sum_y / N
    cross = sum_xy - u_y*sum_x - u_x*sum_y + u_x*u_y*N
    var_x = sum_x2 - 2*u_x*sum_x + u_x*u_x*N
    var_y = sum_y2 - 2*u_y*sum_y + u_y*u_y*N
    ncc = cross * cross / (var_x * var_y + eps)
    return 1 - ncc.mean()

def grad_smooth_loss(v):
    dy = (v[:,:,1:,:] - v[:,:,:-1,:]).abs().mean()
    dx = (v[:,:,:,1:] - v[:,:,:,:-1]).abs().mean()
    return (dx + dy)

def dice_loss(probs_or_mask, target_mask, eps=1e-6):
    # inputs are expected in {0,1} for masks; if logits/probs, binarize externally
    num = 2.0 * (probs_or_mask * target_mask).sum(dim=[1,2,3])
    den = (probs_or_mask + target_mask).sum(dim=[1,2,3]) + eps
    return 1 - (num/den).mean()
