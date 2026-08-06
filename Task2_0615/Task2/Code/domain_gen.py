"""Domain-generalization techniques for Task 2.

Base recipe per the user's own validated prior work (Domain Generalization of
Tuberculosis Detection in Chest X-Rays through MixStyle and Multi-Level
Augmentation, SSRN 5644210): DenseNet-121 + MixStyle (feature-level) +
multi-level augmentation (image-level), 66%->89% / 71%->78% accuracy over the
un-augmented baseline on held-out Shenzhen/Pakistan TB cohorts.

Extended with three further techniques from a second research pass, each
targeting a different weakness of the base recipe:

  - RandConv (Xu et al., ICLR 2021): a single randomly-initialized conv
    layer, resampled fresh every call, blended with the original image.
    Approximately shape-preserving (spatial structure survives a linear
    filter) but scrambles local texture -- creates an entirely NEW synthetic
    "acquisition look" each call, strictly more diverse than MixStyle (which
    can only recombine style statistics already present in the current
    mini-batch -- if a batch happens to be style-homogeneous, MixStyle has
    little to mix).
  - Spectral amplitude perturbation: CXR scanner/contrast differences are
    largely a frequency-domain (amplitude-spectrum) effect, while anatomical
    structure lives in the phase spectrum. Perturbing amplitude while
    leaving phase exactly unchanged synthesizes plausible new "looks"
    without ever touching the anatomy -- a literal match to what actually
    differs between scanners/sites, not just a generic image transform.
  - Domain-aware (cross-domain) mixup: published to beat ERM + 6 other DG
    baselines specifically on CXR thoracic-disease diagnosis across unseen
    domains (Wang & Xia's DELCOM). Deliberately pairs samples from
    DIFFERENT Modality_DICOM values (rather than vanilla mixup's uniform
    random pairing) so the synthesized virtual samples interpolate across
    the actual domain axis this dataset has, not an arbitrary one. Needs
    per-sample domain labels at the batch level, so it's implemented in
    train_task2.py's training loop (not here) -- domain_gen.py exposes the
    mixing function; dataset.py exposes the domain label per sample.

MixStyle is wired into model.py's densenet121 path. build_augmentation
(including the two new per-image transforms) is wired into dataset.py for
all three backbones. Domain-aware mixup is wired into train_task2.py.
"""
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


class MixStyle(nn.Module):
    """Domain Generalization with MixStyle (Zhou et al., ICLR 2021).

    Mixes per-instance channel-wise feature statistics (mean/std) within a
    mini-batch, with a random pairing and a Beta-sampled mixing coefficient.
    Applied only during training (identity at eval time); expects a 4D
    (B, C, H, W) feature map, matching an intermediate CNN activation.
    """

    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps

    def forward(self, x):
        if not self.training or random.random() > self.p:
            return x
        B = x.size(0)
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        x_norm = (x - mu) / sig

        lam = self.beta.sample((B, 1, 1, 1)).to(x.device)
        perm = torch.randperm(B, device=x.device)
        mu_mix = lam * mu + (1 - lam) * mu[perm]
        sig_mix = lam * sig + (1 - lam) * sig[perm]
        return x_norm * sig_mix + mu_mix


class RandConv(nn.Module):
    """Robust and Generalizable Visual Representation Learning via Random
    Convolutions (Xu et al., ICLR 2021). A fresh, randomly-initialized
    (never trained) conv layer is sampled every forward call and applied to
    the image, then blended back with the original via a random alpha --
    the blend avoids the semantic/shape loss the original paper notes for
    larger kernel sizes used alone. No gradient needed for the sampled
    weights (they're thrown away after one use), so this runs under
    no_grad internally regardless of the surrounding training context.
    """

    def __init__(self, p: float = 0.3, kernel_sizes=(1, 3, 5, 7)):
        super().__init__()
        self.p = p
        self.kernel_sizes = kernel_sizes

    def forward(self, x):
        if random.random() > self.p:
            return x
        c = x.shape[0]
        k = random.choice(self.kernel_sizes)
        with torch.no_grad():
            weight = torch.randn(c, c, k, k, device=x.device, dtype=x.dtype) / (c * k * k) ** 0.5
            out = F.conv2d(x.unsqueeze(0), weight, padding=k // 2).squeeze(0)
        alpha = random.random()
        return alpha * out + (1 - alpha) * x


class SpectralAmplitudePerturb(nn.Module):
    """Perturbs the Fourier amplitude spectrum with smooth multiplicative
    noise while leaving the phase spectrum exactly unchanged. Phase encodes
    an image's structural/shape content; amplitude encodes its contrast/
    texture "look" -- the axis that actually differs across scanners and
    sites. Single-image, no reference/paired domain needed."""

    def __init__(self, strength: float = 0.3, p: float = 0.3):
        super().__init__()
        self.strength = strength
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        fft = torch.fft.fft2(x)
        amp, phase = fft.abs(), fft.angle()
        noise = 1.0 + self.strength * (torch.rand_like(amp) * 2 - 1)
        amp = amp * noise
        out = torch.fft.ifft2(torch.polar(amp, phase)).real
        return out.to(x.dtype)


def domain_aware_mixup(x, y, domains, alpha: float = 0.2):
    """Cross-domain mixup (Wang & Xia's DELCOM, adapted): pairs each sample
    preferentially with a partner from a DIFFERENT domain in the same batch
    (falls back to a random same-batch partner if none exists -- e.g. a
    batch that happens to be single-domain -- so this never crashes, it
    just degrades to vanilla mixup for that batch). `domains` is any
    per-sample hashable label (Modality_DICOM strings work directly, no
    need to pre-encode to ints). `y` can be a soft target already (this
    project's binary labels are 0./1. floats, so the mixed target is a
    valid soft BCE target with no loss-function changes needed).
    """
    if alpha <= 0:
        return x, y
    B = x.size(0)
    lam = float(np.random.beta(alpha, alpha))
    domains = np.asarray(domains)
    perm = np.empty(B, dtype=np.int64)
    for i in range(B):
        candidates = np.where(domains != domains[i])[0]
        perm[i] = np.random.choice(candidates) if len(candidates) > 0 else np.random.choice(B)
    perm_t = torch.from_numpy(perm).to(x.device)
    x_mix = lam * x + (1 - lam) * x[perm_t]
    y_mix = lam * y + (1 - lam) * y[perm_t]
    return x_mix, y_mix


def build_augmentation(img_size: int, strength: str = "strong"):
    """Multi-level (image-level) augmentation pipeline. `strength` selects how
    many/how aggressive the stacked transforms are -- "strong" is the DG
    recipe (matches the "multi-level" framing: several augmentation types
    stacked, not just one), "light" is closer to the baseline notebook's
    single flip for a quick ablation comparison, "none" disables entirely.

    Operates on a numpy (3, H, W) float32 array already produced by
    common.preprocess_image (percentile-clip/CLAHE/[bone-suppress]/resize/
    3-channel/mean-std already applied) -- these augmentations work directly
    on the tensor after preprocessing, so they compose with per-backbone
    preprocessing without needing PIL round-trips.
    """
    if strength == "none":
        return nn.Identity()

    tfs = [transforms.RandomHorizontalFlip(p=0.5)]
    if strength in ("light", "strong"):
        tfs.append(transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5))
    if strength == "strong":
        tfs += [
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))], p=0.3),
            transforms.RandomApply([_ResolutionJitter(img_size)], p=0.3),
            transforms.RandomAffine(degrees=7, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            RandConv(p=0.3),
            SpectralAmplitudePerturb(strength=0.3, p=0.3),
        ]
    return transforms.Compose(tfs)


class _ResolutionJitter(nn.Module):
    """Downsample then upsample back to img_size -- simulates the low-
    resolution external scans the notebook/organizers explicitly warn about
    (multi-country, multi-modality external validation)."""

    def __init__(self, img_size: int, min_scale: float = 0.4, max_scale: float = 0.8):
        super().__init__()
        self.img_size = img_size
        self.min_scale = min_scale
        self.max_scale = max_scale

    def forward(self, x):
        scale = random.uniform(self.min_scale, self.max_scale)
        small = max(8, int(self.img_size * scale))
        down = transforms.functional.resize(x, [small, small], antialias=True)
        return transforms.functional.resize(down, [self.img_size, self.img_size], antialias=True)
