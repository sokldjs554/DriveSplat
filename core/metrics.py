"""
Depth Estimation & Novel View Synthesis 평가 지표
★ Custom implementation
"""

import numpy as np
import torch


def compute_depth_metrics(pred, gt,
                           min_depth=1e-3, max_depth=80.0):
    """
    KITTI Eigen split 기준 depth 평가 지표.

    Args:
        pred, gt: (H, W) depth maps (numpy or torch)
    Returns:
        dict: abs_rel, sq_rel, rmse, rmse_log, d1, d2, d3
    """
    if isinstance(pred, np.ndarray):
        pred = torch.tensor(pred)
    if isinstance(gt, np.ndarray):
        gt   = torch.tensor(gt)

    mask = (gt > min_depth) & (gt < max_depth)
    pred = pred[mask]
    gt   = gt[mask]

    # δ 기반 지표
    thresh = torch.max(gt / pred, pred / gt)
    d1 = (thresh < 1.25     ).float().mean()
    d2 = (thresh < 1.25 ** 2).float().mean()
    d3 = (thresh < 1.25 ** 3).float().mean()

    # 오차 기반 지표
    abs_rel  = ((gt - pred).abs() / gt).mean()
    sq_rel   = (((gt - pred) ** 2) / gt).mean()
    rmse     = torch.sqrt(((gt - pred) ** 2).mean())
    rmse_log = torch.sqrt(
        (torch.log(gt) - torch.log(pred)).pow(2).mean()
    )

    return {
        'abs_rel' : abs_rel.item(),
        'sq_rel'  : sq_rel.item(),
        'rmse'    : rmse.item(),
        'rmse_log': rmse_log.item(),
        'd1'      : d1.item(),
        'd2'      : d2.item(),
        'd3'      : d3.item(),
    }


def compute_psnr(pred, gt):
    """
    PSNR (Peak Signal-to-Noise Ratio) 계산.

    Args:
        pred, gt: (H, W, C) or (H, W) float tensors [0, 1]
    Returns:
        psnr (float, dB)
    """
    if isinstance(pred, np.ndarray):
        pred = torch.tensor(pred)
    if isinstance(gt, np.ndarray):
        gt   = torch.tensor(gt)

    mse  = ((pred - gt) ** 2).mean()
    psnr = -10 * torch.log10(mse + 1e-8)
    return psnr.item()


def compute_ssim(pred, gt, window_size=11):
    """SSIM (torchmetrics 래퍼)"""
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0)
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        gt   = gt.unsqueeze(0)
    return ssim_fn(pred, gt).item()
