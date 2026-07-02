from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange


def _valid_groups(channels, max_groups = 8):
    groups = min(max_groups, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return groups



class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8, kernel_size = 3,
                 padding_mode = "zeros", groupnorm = True, dilation = 1):
        super().__init__()
        padding = (kernel_size // 2) * dilation
        groups = _valid_groups(dim_out, groups)
        self.proj = nn.Conv2d(dim, dim_out, kernel_size=kernel_size, padding=padding,
                              dilation=dilation, padding_mode=padding_mode)
        self.norm = nn.GroupNorm(groups, dim_out) if groupnorm else nn.BatchNorm2d(dim_out)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.proj(x)))


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, groups = 8, kernel_size = 3,
                 padding_mode = "zeros", dilation = 1):
        super().__init__()
        self.block1 = Block(dim, dim_out, groups=groups, kernel_size=kernel_size,
                            padding_mode=padding_mode, dilation=dilation)
        self.block2 = Block(dim_out, dim_out, groups=groups, kernel_size=kernel_size,
                            padding_mode=padding_mode, dilation=dilation)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        return self.block2(self.block1(x)) + self.res_conv(x)


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction = 4):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.net(x)


class MultiScaleBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dw3 = nn.Conv2d(channels, channels, 3, padding=1)
        self.dw5 = nn.Conv2d(channels, channels, 5, padding=2)
        self.dwd = nn.Conv2d(channels, channels, 3, padding=2, dilation=2)
        self.mix = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1),
            nn.GroupNorm(_valid_groups(channels), channels),
            nn.SiLU(inplace=True),
            ChannelAttention(channels),
            nn.Conv2d(channels, channels, 1),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        y = torch.cat([self.dw3(x), self.dw5(x), self.dwd(x)], dim=1)
        return self.act(x + self.mix(y))



class IntensityEvolutionBranch(nn.Module):
    def __init__(self, pre_seq_length, aft_seq_length, dim,
                 hidden_dim, n_layers = 3):
        super().__init__()
        self.pre_seq_length = pre_seq_length
        self.aft_seq_length = aft_seq_length
        self.dim = dim
        self.stem = nn.Sequential(
            Block(pre_seq_length * dim, hidden_dim, kernel_size=3),
            MultiScaleBlock(hidden_dim),
        )
        self.blocks = nn.Sequential(*[MultiScaleBlock(hidden_dim) for _ in range(max(n_layers, 1))])
        self.out = nn.Conv2d(hidden_dim, aft_seq_length * dim, kernel_size=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    @staticmethod
    def _safe_logit(x):
        x = x.clamp(1e-4, 1.0 - 1e-4)
        return torch.log(x / (1.0 - x))

    def forward(self, x):
        b, t, c, h, w = x.shape
        x_flat = rearrange(x, "b t c h w -> b (t c) h w")
        feat = self.blocks(self.stem(x_flat))
        delta = self.out(feat)
        delta = rearrange(delta, "b (t c) h w -> b t c h w", t=self.aft_seq_length)
        last_logit = self._safe_logit(x[:, -1:]).expand(-1, self.aft_seq_length, -1, -1, -1)
        return last_logit + delta


class PositionEvolutionBranch(nn.Module):
    def __init__(self, input_shape, pre_seq_length, aft_seq_length,
                 input_dim, hidden_dim):
        super().__init__()
        h, w = input_shape
        self.pre_seq_length = pre_seq_length
        self.aft_seq_length = aft_seq_length
        self.input_dim = input_dim

        phase_in_ch = input_dim * pre_seq_length + 2
        spatial_in_ch = input_dim * pre_seq_length
        self.phase_feature_encoder = nn.Sequential(
            Block(phase_in_ch, hidden_dim, kernel_size=1),
            ResnetBlock(hidden_dim, hidden_dim, kernel_size=1),
        )
        self.spatial_motion_encoder = nn.Sequential(
            Block(spatial_in_ch, hidden_dim, kernel_size=3),
            MultiScaleBlock(hidden_dim),
        )
        self.phase_offset_estimator = nn.Sequential(
            ResnetBlock(hidden_dim * 2, hidden_dim, kernel_size=3, padding_mode="circular"),
            MultiScaleBlock(hidden_dim),
            nn.Conv2d(hidden_dim, aft_seq_length * input_dim, kernel_size=1),
        )

        nn.init.zeros_(self.phase_offset_estimator[-1].weight)
        nn.init.zeros_(self.phase_offset_estimator[-1].bias)

        try:
            u, v = torch.meshgrid(torch.fft.fftfreq(h), torch.fft.rfftfreq(w), indexing="ij")
        except TypeError:
            u, v = torch.meshgrid(torch.fft.fftfreq(h), torch.fft.rfftfreq(w))
        self.register_buffer("uv", torch.stack((u, v), dim=0), persistent=False)

    @staticmethod
    def normalize_phase(x):
        return x / torch.pi

    def forward(self, x):

        b, _, _, h, w = x.shape
        x_fft = torch.fft.rfft2(x, s=(h, w))
        input_amplitude_spectrum, x_phas = torch.abs(x_fft), torch.angle(x_fft)
        phase_flat = rearrange(self.normalize_phase(x_phas), "b t c h w -> b (t c) h w")
        phase_input = torch.cat([phase_flat, self.uv.unsqueeze(0).repeat(b, 1, 1, 1)], dim=1)
        spatial_input = rearrange(x, "b t c h w -> b (t c) h w")
        phase_feat = self.phase_feature_encoder(phase_input)
        spatial_feat = self.spatial_motion_encoder(spatial_input)
        spatial_feat = F.interpolate(spatial_feat, size=phase_feat.shape[-2:], mode="bilinear", align_corners=False)
        offset = self.phase_offset_estimator(torch.cat([phase_feat, spatial_feat], dim=1))
        offset = rearrange(offset, "b (t c) h w -> b t c h w", t=self.aft_seq_length)
        predicted_phase = x_phas[:, -1:].expand(-1, self.aft_seq_length, -1, -1, -1) + torch.tanh(offset) * torch.pi
        last_amp = input_amplitude_spectrum[:, -1:].expand(-1, self.aft_seq_length, -1, -1, -1)
        xt_fft = torch.polar(last_amp, predicted_phase)
        xt = torch.fft.irfft2(xt_fft, s=(h, w))
        return xt, predicted_phase, input_amplitude_spectrum



class EvoDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels = None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_valid_groups(mid_channels), mid_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_valid_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.shortcut(x)


class EvoDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), EvoDoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.net(x)


class EvoUp(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = EvoDoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = EvoDoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class PhysicalEvolutionEstimator(nn.Module):
    def __init__(self, in_channels, out_steps, base_c = 16, bilinear = True):
        super().__init__()
        factor = 2 if bilinear else 1
        self.out_steps = out_steps
        self.inc = EvoDoubleConv(in_channels, base_c)
        self.down1 = EvoDown(base_c, base_c * 2)
        self.down2 = EvoDown(base_c * 2, base_c * 4)
        self.down3 = EvoDown(base_c * 4, base_c * 8)
        self.down4 = EvoDown(base_c * 8, base_c * 16 // factor)
        bottleneck_c = base_c * 16 // factor

        def make_decoder(out_channels):
            return nn.ModuleDict({
                "up1": EvoUp(base_c * 16, base_c * 8 // factor, bilinear),
                "up2": EvoUp(base_c * 8, base_c * 4 // factor, bilinear),
                "up3": EvoUp(base_c * 4, base_c * 2 // factor, bilinear),
                "up4": EvoUp(base_c * 2, base_c, bilinear),
                "out": nn.Conv2d(base_c, out_channels, kernel_size=1),
            })

        self.intensity_residual_head = make_decoder(out_steps)
        self.local_motion_residual_head = make_decoder(out_steps * 2)
        self.coarse_motion_head = nn.Sequential(
            nn.Conv2d(bottleneck_c, bottleneck_c, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(bottleneck_c, out_steps * 2, 1),
        )
        self.intensity_residual_gate = nn.Parameter(torch.zeros(1, out_steps, 1, 1), requires_grad=True)
        self.local_scale = nn.Parameter(torch.tensor(0.25), requires_grad=True)

        for head in [self.intensity_residual_head["out"], self.local_motion_residual_head["out"], self.coarse_motion_head[-1]]:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def _decode(self, dec, x5, x4, x3, x2, x1):
        x = dec["up1"](x5, x4)
        x = dec["up2"](x, x3)
        x = dec["up3"](x, x2)
        x = dec["up4"](x, x1)
        return dec["out"](x)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        h, w = x.shape[-2:]
        intensity = self._decode(self.intensity_residual_head, x5, x4, x3, x2, x1) * self.intensity_residual_gate
        local_motion = self._decode(self.local_motion_residual_head, x5, x4, x3, x2, x1)
        coarse_motion = self.coarse_motion_head(x5)
        coarse_motion = F.interpolate(coarse_motion, size=(h, w), mode="bilinear", align_corners=True)
        motion = coarse_motion + torch.tanh(self.local_scale) * local_motion
        return intensity, motion, coarse_motion, local_motion


class PhysicalEvolutionBranch(nn.Module):
    def __init__(self, pre_seq_length, aft_seq_length, input_dim = 1,
                 base_c = 16, max_motion = 8.0, residual_scale = 0.25):
        super().__init__()
        if input_dim != 1:
            raise ValueError("This branch currently assumes one radar channel. Set input_dim=1 for CIKM.")
        self.pre_seq_length = pre_seq_length
        self.aft_seq_length = aft_seq_length
        self.max_motion = max_motion
        self.residual_scale = residual_scale
        self.physical_evolution_estimator = PhysicalEvolutionEstimator(pre_seq_length * input_dim, aft_seq_length, base_c=base_c)
        self.register_buffer("_grid", torch.empty(0), persistent=False)
        self._grid_hw = None

    def _make_grid(self, b, h, w, device, dtype):
        if self._grid.numel() == 0 or self._grid_hw != (h, w) or self._grid.device != device or self._grid.dtype != dtype:
            yy, xx = torch.meshgrid(
                torch.arange(h, device=device, dtype=dtype),
                torch.arange(w, device=device, dtype=dtype),
                indexing="ij",
            )
            grid = torch.stack([xx, yy], dim=0).unsqueeze(0)
            self._grid = grid
            self._grid_hw = (h, w)
        return self._grid.repeat(b, 1, 1, 1)

    @staticmethod
    def warp(x, flow, grid,
             mode = "bilinear", padding_mode = "border"):
        b, c, h, w = x.shape
        vgrid = grid + flow
        vx = 2.0 * vgrid[:, 0] / max(w - 1, 1) - 1.0
        vy = 2.0 * vgrid[:, 1] / max(h - 1, 1) - 1.0
        vgrid = torch.stack([vx, vy], dim=-1)
        return F.grid_sample(x, vgrid, mode=mode, padding_mode=padding_mode, align_corners=True)

    def forward(self, x):
        b, t, c, h, w = x.shape
        x_flat = rearrange(x, "b t c h w -> b (t c) h w")
        intensity, motion, _, _ = self.physical_evolution_estimator(x_flat)
        intensity = torch.tanh(intensity) * self.residual_scale
        motion = torch.tanh(motion) * self.max_motion
        intensity = intensity.view(b, self.aft_seq_length, 1, h, w)
        motion = motion.view(b, self.aft_seq_length, 2, h, w)
        last = x[:, -1]
        grid = self._make_grid(b, h, w, x.device, x.dtype)
        series = []
        for i in range(self.aft_seq_length):
            advected = self.warp(last, motion[:, i], grid, mode="bilinear", padding_mode="border")
            last = (advected + intensity[:, i]).clamp(0.0, 1.0)
            series.append(last.unsqueeze(1))
        return torch.cat(series, dim=1)


class ConfidenceGuidedFusionModule(nn.Module):
    def __init__(self, input_shape, spec_num, input_dim,
                 hidden_dim, aft_seq_length, use_phy = True):
        super().__init__()
        h, w = input_shape
        self.aft_seq_length = aft_seq_length
        self.use_phy = use_phy
        self.input_dim = input_dim
        self.n_sources = 4 if use_phy else 3
        spec_mask = torch.zeros(h, w // 2 + 1)
        spec_mask[..., :spec_num, :spec_num] = 1.0
        spec_mask[..., -spec_num:, :spec_num] = 1.0
        self.register_buffer("spec_mask", spec_mask, persistent=False)
        in_ch = self.n_sources * input_dim
        self.confidence_estimator = nn.Sequential(
            ResnetBlock(in_ch, hidden_dim),
            ResnetBlock(hidden_dim, hidden_dim),
            nn.Conv2d(hidden_dim, self.n_sources, 1),
        )
        self.residual_fusion_branch = nn.Sequential(
            ResnetBlock(in_ch + input_dim, hidden_dim),
            ResnetBlock(hidden_dim, hidden_dim),
            nn.Conv2d(hidden_dim, input_dim, 1),
        )
        nn.init.zeros_(self.confidence_estimator[-1].weight)
        nn.init.zeros_(self.confidence_estimator[-1].bias)
        nn.init.zeros_(self.residual_fusion_branch[-1].weight)
        nn.init.zeros_(self.residual_fusion_branch[-1].bias)

    def forward(self, y_int, y_pos, predicted_phase,
                y_phy = None):
        b, t, c, h, w = y_int.shape
        y_int_fft = torch.fft.rfft2(y_int, s=(h, w))
        intensity_amplitude_spectrum = torch.abs(y_int_fft)
        y_freq_fft = torch.polar(intensity_amplitude_spectrum * self.spec_mask, predicted_phase)
        y_freq = torch.fft.irfft2(y_freq_fft, s=(h, w))
        sources = [y_int, y_pos, y_freq]
        if self.use_phy:
            if y_phy is None:
                raise ValueError("y_phy is required when use_phy=True")
            sources.append(y_phy)
        fused = torch.cat(sources, dim=2)
        fused_2d = rearrange(fused, "b t c h w -> (b t) c h w")
        logits = self.confidence_estimator(fused_2d)
        weights = torch.softmax(logits, dim=1)
        src = torch.stack([rearrange(s, "b t c h w -> (b t) c h w") for s in sources], dim=1)
        weighted = (weights.unsqueeze(2) * src).sum(dim=1)
        delta_y_fuse = torch.tanh(self.residual_fusion_branch(torch.cat([weighted, fused_2d], dim=1))) * 0.2
        y_coarse = weighted + delta_y_fuse
        return rearrange(y_coarse, "(b t) c h w -> b t c h w", b=b, t=t)


class StrongEchoLeadTimeResidualRefinementModule(nn.Module):
    def __init__(
        self,
        aft_seq_length,
        input_dim = 1,
        hidden_dim = 32,
        n_blocks = 3,
        threshold = 35.0,
        value_scale = 80.0,
        gate_sharpness = 1.0,
        max_delta = 0.15,
        time_gate_start = 0.20,
        time_gate_end = 1.00,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.threshold = float(threshold)
        self.value_scale = float(value_scale)
        self.gate_sharpness = float(gate_sharpness)
        self.max_delta = float(max_delta)
        in_ch = input_dim * 5 + 2
        layers = [Block(in_ch, hidden_dim, groups=8, kernel_size=3)]
        for _ in range(max(int(n_blocks), 1)):
            layers.append(ResnetBlock(hidden_dim, hidden_dim, groups=8, kernel_size=3))
        layers.append(nn.Conv2d(hidden_dim, input_dim, kernel_size=1))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        lead = torch.linspace(float(time_gate_start), float(time_gate_end), steps=aft_seq_length).view(1, aft_seq_length, 1, 1, 1)
        self.register_buffer("lead_gate", lead, persistent=False)

    def _soft_mask(self, x):
        return torch.sigmoid(self.gate_sharpness * (x * self.value_scale - self.threshold))

    def forward(self, coarse, y_int, y_pos,
                y_phy, last_frame):
        b, t, c, h, w = coarse.shape
        if y_phy is None:
            y_phy = torch.zeros_like(coarse)
        last_rep = last_frame.unsqueeze(1).expand(-1, t, -1, -1, -1)
        coarse_clip = coarse.clamp(0.0, 1.0)
        y_phy_clip = y_phy.clamp(0.0, 1.0)
        last_clip = last_rep.clamp(0.0, 1.0)
        gate = torch.maximum(torch.maximum(self._soft_mask(coarse_clip), self._soft_mask(y_phy_clip)), self._soft_mask(last_clip))
        lead_gate = self.lead_gate[:, :t].to(dtype=coarse.dtype, device=coarse.device)
        gate = gate * lead_gate
        mask_feature = gate.mean(dim=2, keepdim=True)
        time_feature = lead_gate.expand(b, -1, 1, h, w)
        feat = torch.cat([
            coarse_clip,
            y_int.clamp(0.0, 1.0),
            y_pos.clamp(0.0, 1.0),
            y_phy_clip,
            last_clip,
            mask_feature,
            time_feature,
        ], dim=2)
        feat = rearrange(feat, "b t c h w -> (b t) c h w")
        gate_2d = rearrange(gate, "b t c h w -> (b t) c h w")
        delta = torch.tanh(self.net(feat)) * self.max_delta
        refined = rearrange(coarse, "b t c h w -> (b t) c h w") + gate_2d * delta
        refined = rearrange(refined, "(b t) c h w -> b t c h w", b=b, t=t)
        return refined.clamp(0.0, 1.0)

    def high_echo_loss(self, pred, target):
        target_mask = self._soft_mask(target.clamp(0.0, 1.0))
        lead = self.lead_gate[:, :target.shape[1]].to(dtype=target.dtype, device=target.device)
        target_mask = target_mask * lead
        err = (pred - target).pow(2)
        return (err * target_mask).sum() / (target_mask.sum() + 1e-6)


class MFPENet(nn.Module):
    def __init__(
        self,
        pre_seq_length = 5,
        aft_seq_length = 10,
        input_shape = (128, 128),
        input_dim = 1,
        hidden_dim = 128,
        n_layers = 4,
        spec_num = 20,
        pha_weight = 0.01,
        anet_weight = 0.1,
        amp_weight = 0.01,
        phy_weight = 0.1,
        aweight_stop_steps = 5000,
        use_phy = True,
        phy_base_c = 48,
        max_motion = 8.0,
        residual_scale = 0.25,
        use_high_refiner = True,
        refiner_hidden_dim = 96,
        refiner_blocks = 4,
        refiner_threshold = 35.0,
        refiner_value_scale = 80.0,
        refiner_gate_sharpness = 1.0,
        refiner_max_delta = 0.15,
        refiner_weight = 0.05,
    ):
        super().__init__()
        self.intensity_evolution_branch = IntensityEvolutionBranch(pre_seq_length, aft_seq_length, input_dim, hidden_dim, n_layers=n_layers)
        self.position_evolution_branch = PositionEvolutionBranch(input_shape, pre_seq_length, aft_seq_length, input_dim, hidden_dim)
        self.use_phy = use_phy
        self.physical_evolution_branch = PhysicalEvolutionBranch(pre_seq_length, aft_seq_length, input_dim,
                                                        base_c=phy_base_c, max_motion=max_motion,
                                                        residual_scale=residual_scale) if use_phy else None
        self.confidence_guided_fusion_module = ConfidenceGuidedFusionModule(input_shape, spec_num, input_dim, hidden_dim,
                                                           aft_seq_length, use_phy=use_phy)
        self.use_high_refiner = bool(use_high_refiner)
        self.refiner_weight = float(refiner_weight)
        self.strong_echo_lead_time_residual_refinement_module = StrongEchoLeadTimeResidualRefinementModule(
            aft_seq_length=aft_seq_length,
            input_dim=input_dim,
            hidden_dim=refiner_hidden_dim,
            n_blocks=refiner_blocks,
            threshold=refiner_threshold,
            value_scale=refiner_value_scale,
            gate_sharpness=refiner_gate_sharpness,
            max_delta=refiner_max_delta,
        ) if self.use_high_refiner else None

        self.input_shape = input_shape
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.spec_num = spec_num
        self.pha_weight = pha_weight
        self.anet_weight = anet_weight
        self.amp_weight = amp_weight
        self.phy_weight = phy_weight
        self.pre_seq_length = pre_seq_length
        self.aft_seq_length = aft_seq_length
        self.aweight_stop_steps = max(int(aweight_stop_steps), 1)
        self.criterion = nn.MSELoss()
        self.register_buffer("global_step_tensor", torch.zeros((), dtype=torch.long), persistent=True)
        h, w = input_shape
        spec_mask = torch.zeros(h, w // 2 + 1)
        spec_mask[..., :spec_num, :spec_num] = 1.0
        spec_mask[..., -spec_num:, :spec_num] = 1.0
        self.register_buffer("spec_mask", spec_mask, persistent=False)

    @property
    def global_step(self):
        return int(self.global_step_tensor.item())

    def set_global_step(self, step):
        self.global_step_tensor.fill_(int(step))

    def forward(self, x):
        y_int = torch.sigmoid(self.intensity_evolution_branch(x))
        y_pos, predicted_phase, input_amplitude_spectrum = self.position_evolution_branch(x)
        y_phy = self.physical_evolution_branch(x) if self.use_phy else None
        y_coarse = self.confidence_guided_fusion_module(y_int, y_pos, predicted_phase, y_phy)
        y_final = y_coarse
        if self.use_high_refiner and self.strong_echo_lead_time_residual_refinement_module is not None:
            y_final = self.strong_echo_lead_time_residual_refinement_module(y_coarse, y_int, y_pos, y_phy, x[:, -1])
        return y_final, y_pos, y_int, predicted_phase, input_amplitude_spectrum, y_phy

    def predict(self, frames_in, frames_gt = None,
                compute_loss = False):
        y_final, y_pos, y_int, predicted_phase, input_amplitude_spectrum, y_phy = self(frames_in)
        pred = y_final.clamp(0.0, 1.0)
        if not compute_loss:
            return pred, None
        if frames_gt is None:
            raise ValueError("frames_gt must be provided when compute_loss=True")

        b = frames_in.shape[0]
        loss = self.criterion(pred, frames_gt)
        frames_fft = torch.fft.rfft2(frames_gt, s=frames_gt.shape[-2:])
        target_phase = torch.angle(frames_fft)
        target_amplitude = torch.abs(frames_fft)
        phase_loss = (1 - torch.cos(target_phase * self.spec_mask - predicted_phase * self.spec_mask)).sum()
        phase_loss = phase_loss / (self.spec_mask.sum() * b * self.aft_seq_length * self.input_dim + 1e-6)

        current_amp_weight = self.amp_weight * max(0.0, 1.0 - self.global_step / self.aweight_stop_steps)
        y_int_fft = torch.fft.rfft2(y_int, s=y_int.shape[-2:])
        amplitude_loss = self.criterion(torch.abs(y_int_fft), target_amplitude)
        intensity_loss = self.criterion(y_int, frames_gt)
        total = loss + self.pha_weight * phase_loss + current_amp_weight * amplitude_loss + self.anet_weight * intensity_loss
        loss_dict: Dict[str, torch.Tensor] = {
            "total_loss": total,
            "out_loss": loss.detach(),
            "phase_loss": (self.pha_weight * phase_loss).detach(),
            "ampli_loss": (current_amp_weight * amplitude_loss).detach(),
            "intensity_loss": (self.anet_weight * intensity_loss).detach(),
        }
        if self.use_phy and y_phy is not None:
            physical_loss = self.criterion(y_phy, frames_gt)
            total = total + self.phy_weight * physical_loss
            loss_dict["total_loss"] = total
            loss_dict["physical_loss"] = (self.phy_weight * physical_loss).detach()
        if self.use_high_refiner and self.strong_echo_lead_time_residual_refinement_module is not None and self.refiner_weight > 0:
            refinement_loss = self.strong_echo_lead_time_residual_refinement_module.high_echo_loss(pred, frames_gt)
            total = total + self.refiner_weight * refinement_loss
            loss_dict["total_loss"] = total
            loss_dict["refinement_loss"] = (self.refiner_weight * refinement_loss).detach()
        return pred, loss_dict


def get_model(**kwargs):
    return MFPENet(**kwargs)
