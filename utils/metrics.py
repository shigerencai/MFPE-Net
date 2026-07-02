from typing import Dict, Iterable, List

import numpy as np
import torch
import torch.nn.functional as F


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def _gaussian_window(window_size = 11, sigma = 1.5, device=None, dtype=None):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = torch.outer(g, g)
    return window.view(1, 1, window_size, window_size)


def _ssim_alpha_style(pred, gt, value_scale, window_size = 11, sigma = 1.5):
    if isinstance(pred, torch.Tensor):
        pred_t = pred.detach().float()
    else:
        pred_t = torch.from_numpy(np.asarray(pred)).float()
    if isinstance(gt, torch.Tensor):
        gt_t = gt.detach().float().to(device=pred_t.device)
    else:
        gt_t = torch.from_numpy(np.asarray(gt)).float().to(device=pred_t.device)

    pred_t = pred_t.clamp(0.0, 1.0) * float(value_scale)
    gt_t = gt_t.clamp(0.0, 1.0) * float(value_scale)

    if pred_t.ndim == 2:
        pred_t = pred_t[None, None]
        gt_t = gt_t[None, None]
    elif pred_t.ndim == 3:
        pred_t = pred_t.reshape(-1, 1, pred_t.shape[-2], pred_t.shape[-1])
        gt_t = gt_t.reshape(-1, 1, gt_t.shape[-2], gt_t.shape[-1])
    elif pred_t.ndim >= 4:
        pred_t = pred_t.reshape(-1, 1, pred_t.shape[-2], pred_t.shape[-1])
        gt_t = gt_t.reshape(-1, 1, gt_t.shape[-2], gt_t.shape[-1])
    else:
        return 0.0

    if pred_t.shape[-2] < window_size or pred_t.shape[-1] < window_size:
        return 0.0

    data_range = float(value_scale)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    window = _gaussian_window(window_size, sigma, device=pred_t.device, dtype=pred_t.dtype)

    mu1 = F.conv2d(pred_t, window, padding=0)
    mu2 = F.conv2d(gt_t, window, padding=0)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred_t * pred_t, window, padding=0) - mu1_sq
    sigma2_sq = F.conv2d(gt_t * gt_t, window, padding=0) - mu2_sq
    sigma12 = F.conv2d(pred_t * gt_t, window, padding=0) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return float(ssim_map.mean().detach().cpu())


class Evaluator:
    def __init__(
        self,
        thresholds = (20, 30, 35, 40),
        value_scale = 80.0,
        compute_ssim = False,
    ):
        self.thresholds = list(thresholds)
        self.value_scale = float(value_scale)
        self.compute_ssim = bool(compute_ssim)

        self.hits = {th: 0 for th in self.thresholds}
        self.misses = {th: 0 for th in self.thresholds}
        self.falsealarms = {th: 0 for th in self.thresholds}
        self.correctnegs = {th: 0 for th in self.thresholds}
        self.mse_norm_sum = 0.0
        self.mae_norm_sum = 0.0
        self.mse_paper_sum = 0.0
        self.mae_paper_sum = 0.0
        self.ssim_sum = 0.0
        self.n_ssim_items = 0
        self.n_pixels = 0
        self.n_batches = 0

    def update(self, pred, gt):
        if self.compute_ssim:
            ssim = _ssim_alpha_style(pred, gt, self.value_scale)
        else:
            ssim = None

        pred = np.clip(_to_numpy(pred), 0.0, 1.0)
        gt = np.clip(_to_numpy(gt), 0.0, 1.0)
        assert pred.shape == gt.shape, f"pred shape {pred.shape} != gt shape {gt.shape}"

        diff_norm = pred - gt
        self.mse_norm_sum += float(np.sum(diff_norm ** 2))
        self.mae_norm_sum += float(np.sum(np.abs(diff_norm)))

        diff_paper = diff_norm * self.value_scale
        self.mse_paper_sum += float(np.sum(diff_paper ** 2))
        self.mae_paper_sum += float(np.sum(np.abs(diff_paper)))

        self.n_pixels += int(np.prod(gt.shape))
        self.n_batches += 1

        if self.compute_ssim and ssim is not None:
            n_ssim_items = int(np.prod(gt.shape[:-2])) if gt.ndim >= 2 else 1
            self.ssim_sum += float(ssim) * n_ssim_items
            self.n_ssim_items += n_ssim_items

        pred_v = pred * self.value_scale
        gt_v = gt * self.value_scale
        for th in self.thresholds:
            o = gt_v >= th
            s = pred_v >= th
            self.hits[th] += int(np.logical_and(o, s).sum())
            self.misses[th] += int(np.logical_and(o, ~s).sum())
            self.falsealarms[th] += int(np.logical_and(~o, s).sum())
            self.correctnegs[th] += int(np.logical_and(~o, ~s).sum())

    @staticmethod
    def _safe_div(a, b):
        return float(a / b) if b != 0 else 0.0

    def _compute_base(self):
        res: Dict[str, float] = {}
        csi_values: List[float] = []
        hss_values: List[float] = []
        for th in self.thresholds:
            h = self.hits[th]
            m = self.misses[th]
            f = self.falsealarms[th]
            cn = self.correctnegs[th]
            csi = self._safe_div(h, h + m + f)
            pod = self._safe_div(h, h + m)
            far = self._safe_div(f, h + f)
            hss = self._safe_div(2 * (h * cn - m * f), (h + m) * (m + cn) + (h + f) * (f + cn))
            res[f"CSI{th}"] = csi
            res[f"POD{th}"] = pod
            res[f"FAR{th}"] = far
            res[f"HSS{th}"] = hss
            csi_values.append(csi)
            hss_values.append(hss)

        res["CSI-M"] = float(np.mean(csi_values)) if csi_values else 0.0
        res["HSS-M"] = float(np.mean(hss_values)) if hss_values else 0.0

        mse_norm = self._safe_div(self.mse_norm_sum, self.n_pixels)
        mae_norm = self._safe_div(self.mae_norm_sum, self.n_pixels)
        mse_paper = self._safe_div(self.mse_paper_sum, self.n_pixels)
        mae_paper = self._safe_div(self.mae_paper_sum, self.n_pixels)

        res["MSE_norm"] = mse_norm
        res["MAE_norm"] = mae_norm
        res["RMSE_norm"] = float(np.sqrt(mse_norm))
        res["MSE_paper"] = mse_paper
        res["MAE_paper"] = mae_paper
        res["RMSE_paper"] = float(np.sqrt(mse_paper))

        if self.compute_ssim:
            res["SSIM"] = self._safe_div(self.ssim_sum, self.n_ssim_items)
        return res

    def compute_train_metrics(self):
        res = self._compute_base()
        res["MSE"] = res["MSE_norm"]
        res["MAE"] = res["MAE_norm"]
        res["RMSE"] = res["RMSE_norm"]
        res.pop("MSE_paper", None)
        res.pop("MAE_paper", None)
        res.pop("RMSE_paper", None)
        return res

    def compute_paper_metrics(self):
        res = self._compute_base()
        res["MSE"] = res["MSE_paper"]
        res["MAE"] = res["MAE_paper"]
        res["RMSE"] = res["RMSE_paper"]
        return res

    def compute(self):
        return self.compute_train_metrics()


def format_metrics(metrics):
    parts = []
    if "CSI-M" in metrics:
        parts.append(f"CSI-M={metrics['CSI-M']:.5f}")

    csi_keys = []
    for k in metrics.keys():
        if k.startswith("CSI") and k != "CSI-M":
            suffix = k[3:]
            if suffix.replace(".", "", 1).isdigit():
                csi_keys.append(k)
    csi_keys.sort(key=lambda x: float(x[3:]))
    for k in csi_keys:
        parts.append(f"{k}={metrics[k]:.5f}")
    for k in [
        "HSS-M",
        "SSIM",
        "MSE",
        "MAE",
        "RMSE",
        "MSE_norm",
        "MAE_norm",
        "RMSE_norm",
        "MSE_paper",
        "MAE_paper",
        "RMSE_paper",
    ]:
        if k in metrics:
            parts.append(f"{k}={metrics[k]:.5f}")
    return " | ".join(parts)
