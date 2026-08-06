"""Lung-field ROI cropping -- new preprocessing lever, not present in the
baseline notebook nor the user's SSRN paper. Directly targets the shortcut
the baseline's own Grad-CAM diagnosed (model attending to image borders/
resolution/modality artifacts rather than lungs) and the independent finding
that CXR models can identify country-of-origin at ~86% accuracy from
background/device cues alone -- cropping to the lung bounding box removes
most of that non-anatomical signal before the classifier ever sees it.

Uses torchxrayvision's pretrained CXR lung-segmentation model (PSPNet on
NIH/RSNA-derived masks) -- no training data or fine-tuning needed, it's a
frozen off-the-shelf segmenter, run once as a preprocessing pass over the
whole dataset (not part of the training loop).
"""
import numpy as np
import torch


class LungCropper:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu", margin_frac: float = 0.05):
        import torchxrayvision as xrv
        self.model = xrv.baseline_models.chestx_det.PSPNet().to(device).eval()
        self.device = device
        self.margin_frac = margin_frac

    @torch.no_grad()
    def segment(self, arr: np.ndarray) -> np.ndarray:
        """arr: (H, W) float32, any range. Returns a (H, W) bool mask in
        arr's original resolution/coordinate space (the model runs at 512x512
        internally and this upsamples the prediction back)."""
        import torchxrayvision as xrv
        h, w = arr.shape
        # xrv.datasets.normalize is deprecated in favor of xrv.utils.normalize
        # (same signature/behavior, scales to roughly [-1024, 1024]) -- use
        # the non-deprecated path since we're already pinned to a recent
        # enough torchxrayvision (1.3.4) to have it.
        x = xrv.utils.normalize(arr, 255)
        x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).float().to(self.device)
        x = torch.nn.functional.interpolate(x, size=(512, 512), mode="bilinear", align_corners=False)
        pred = self.model(x)  # (1, n_classes, 512, 512); lung classes per xrv's chestx_det label set
        lung_channels = [i for i, name in enumerate(self.model.targets)
                          if "lung" in name.lower()]
        mask = pred[0, lung_channels].sigmoid().amax(dim=0, keepdim=True).unsqueeze(0)  # (1,1,512,512)
        mask = torch.nn.functional.interpolate(mask, size=(h, w), mode="bilinear", align_corners=False)
        return (mask[0, 0] > 0.5).cpu().numpy()

    def box_from_mask(self, mask: np.ndarray) -> tuple[int, int, int, int]:
        """Bounding box (y0, y1, x0, x1) with margin, clamped to bounds. Falls
        back to the full frame if the mask is empty (e.g. a corrupted/blank
        scan) -- crop is a helper, never a hard failure point."""
        h, w = mask.shape
        ys, xs = np.where(mask)
        if len(ys) == 0:
            return 0, h, 0, w
        y0, y1 = float(ys.min()), float(ys.max())
        x0, x1 = float(xs.min()), float(xs.max())
        my, mx = (y1 - y0) * self.margin_frac, (x1 - x0) * self.margin_frac
        y0, y1 = max(0, int(y0 - my)), min(h, int(y1 + my))
        x0, x1 = max(0, int(x0 - mx)), min(w, int(x1 + mx))
        return y0, y1, x0, x1

    def crop(self, arr: np.ndarray) -> np.ndarray:
        mask = self.segment(arr)
        y0, y1, x0, x1 = self.box_from_mask(mask)
        return arr[y0:y1, x0:x1]
