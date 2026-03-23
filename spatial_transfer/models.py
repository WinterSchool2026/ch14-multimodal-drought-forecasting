"""Model architectures for multi-modal NDVI forecasting.

All models accept:
    context_s2:   (B, C, 4, H, W)   4 Sentinel-2 bands
    context_era5: (B, C, 6)          ERA5 scalars
    target_era5:  (B, F, 6)          future ERA5 (known forcing)
    dem:          (B, 1, H, W)       static DEM

Per-timestep input = 4 S2 + 6 ERA5 + 1 DEM = 11 channels.
"""

import torch
import torch.nn as nn


def _build_frame(s2_t, era5_t, dem, H, W):
    B = era5_t.shape[0]
    era5_spatial = era5_t.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
    parts = []
    if s2_t is not None:
        parts.append(s2_t)
    parts.append(era5_spatial)
    parts.append(dem)
    return torch.cat(parts, dim=1)


class PixelMLP(nn.Module):
    def __init__(self, context_length=10, n_s2=4, n_era5=6, forecast_horizon=20, hidden=256):
        super().__init__()
        input_dim = context_length * (n_s2 + 1) + forecast_horizon * n_era5
        self.forecast_horizon = forecast_horizon
        self.context_length = context_length
        self.n_s2 = n_s2
        self.n_era5 = n_era5

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, forecast_horizon),
        )

    def forward(self, context_s2, context_era5, target_era5, dem, **kwargs):
        B, C, n_s2, H, W = context_s2.shape
        F = self.forecast_horizon
        s2_px = context_s2.permute(0, 3, 4, 1, 2).reshape(B * H * W, C * n_s2)
        dem_px = dem.squeeze(1).permute(0, 1, 2).reshape(B * H * W, 1).expand(-1, C).reshape(B * H * W, C)
        era5_flat = target_era5.reshape(B, -1).unsqueeze(1).unsqueeze(1).expand(-1, H, W, -1)
        era5_flat = era5_flat.reshape(B * H * W, -1)
        x = torch.cat([s2_px, dem_px, era5_flat], dim=-1)
        pred = self.net(x)
        return pred.reshape(B, H, W, F).permute(0, 3, 1, 2)


class PixelLSTM(nn.Module):
    def __init__(self, n_s2=4, n_era5=6, hidden=64, num_layers=2, forecast_horizon=20):
        super().__init__()
        self.hidden = hidden
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        self.lstm = nn.LSTM(n_s2 + 1 + n_era5, hidden, num_layers, batch_first=True)
        self.lstm_fwd = nn.LSTM(1 + n_era5 + 1, hidden, num_layers, batch_first=True)
        self.decoder = nn.Linear(hidden, 1)

    def forward(self, context_s2, context_era5, target_era5, dem, **kwargs):
        B, C, n_s2, H, W = context_s2.shape
        F = self.forecast_horizon
        s2_px = context_s2.permute(0, 3, 4, 1, 2).reshape(B * H * W, C, n_s2)
        dem_px = dem.squeeze(1).permute(0, 1, 2).reshape(B * H * W, 1, 1).expand(-1, C, -1)
        ctx_era5 = context_era5.unsqueeze(1).unsqueeze(1).expand(-1, H, W, -1, -1)
        ctx_era5 = ctx_era5.reshape(B * H * W, C, -1)
        ctx_input = torch.cat([s2_px, dem_px, ctx_era5], dim=-1)
        _, (h, c) = self.lstm(ctx_input)

        tgt_era5 = target_era5.unsqueeze(1).unsqueeze(1).expand(-1, H, W, -1, -1)
        tgt_era5 = tgt_era5.reshape(B * H * W, F, -1)
        dem_fwd = dem.squeeze(1).permute(0, 1, 2).reshape(B * H * W, 1, 1).expand(-1, F, -1)
        prev_anom = torch.zeros(B * H * W, 1, 1, device=context_s2.device)

        predictions = []
        for t in range(F):
            step_input = torch.cat([prev_anom, tgt_era5[:, t:t+1, :], dem_fwd[:, t:t+1, :]], dim=-1)
            out, (h, c) = self.lstm_fwd(step_input, (h, c))
            pred_t = self.decoder(out)
            predictions.append(pred_t.squeeze(-1))
            prev_anom = pred_t

        pred = torch.cat(predictions, dim=1)
        return pred.reshape(B, H, W, F).permute(0, 3, 1, 2)


class MiniUNet(nn.Module):
    """3-level U-Net with autoregressive rollout."""

    def __init__(self, in_channels=8, out_channels=1, base=32):
        super().__init__()
        self.enc1 = self._block(in_channels, base)
        self.enc2 = self._block(base, base * 2)
        self.enc3 = self._block(base * 2, base * 4)
        self.bottleneck = self._block(base * 4, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = self._block(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = self._block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = self._block(base * 2, base)
        self.final = nn.Conv2d(base, out_channels, 1)
        self.pool = nn.MaxPool2d(2)
        self.ctx_encoder = nn.Sequential(
            nn.Conv2d(11, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 1, 1),
        )

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.GELU(),
        )

    def forward_single(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)

    def forward(self, context_s2, context_era5, target_era5, dem, **kwargs):
        B, C, n_s2, H, W = context_s2.shape
        F = target_era5.shape[1]
        last_s2 = context_s2[:, -1]
        last_era5 = context_era5[:, -1].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        ctx_frame = torch.cat([last_s2, last_era5, dem], dim=1)
        current = self.ctx_encoder(ctx_frame)

        predictions = []
        for t in range(F):
            era5_t = target_era5[:, t, :].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
            x = torch.cat([current, era5_t, dem], dim=1)
            pred_t = self.forward_single(x)
            predictions.append(pred_t.squeeze(1))
            current = pred_t
        return torch.stack(predictions, dim=1)


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(
            in_channels + hidden_channels, 4 * hidden_channels,
            kernel_size, padding=kernel_size // 2,
        )

    def forward(self, x, state=None):
        B, _, H, W = x.shape
        if state is None or state[0] is None:
            h = torch.zeros(B, self.hidden_channels, H, W, device=x.device, dtype=x.dtype)
            c = torch.zeros(B, self.hidden_channels, H, W, device=x.device, dtype=x.dtype)
        else:
            h, c = state
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = gates.chunk(4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTMForecaster(nn.Module):
    def __init__(self, ctx_channels=11, fwd_channels=8, hidden_channels=64, num_layers=2, kernel_size=3):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels

        self.ctx_cells = nn.ModuleList()
        for i in range(num_layers):
            in_ch = ctx_channels if i == 0 else hidden_channels
            self.ctx_cells.append(ConvLSTMCell(in_ch, hidden_channels, kernel_size))

        self.fwd_cells = nn.ModuleList()
        for i in range(num_layers):
            in_ch = fwd_channels if i == 0 else hidden_channels
            self.fwd_cells.append(ConvLSTMCell(in_ch, hidden_channels, kernel_size))

        self.decoder = nn.Conv2d(hidden_channels, 1, 1)

    def _run_step(self, x, states, cells):
        for i, cell in enumerate(cells):
            h, c = cell(x, states[i])
            states[i] = (h, c)
            x = h
        return self.decoder(h), states

    def forward(self, context_s2, context_era5, target_era5, dem, **kwargs):
        B, C, n_s2, H, W = context_s2.shape
        F = target_era5.shape[1]
        states = [(None, None) for _ in range(self.num_layers)]

        for t in range(C):
            s2_t = context_s2[:, t]
            era5_t = context_era5[:, t, :].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
            x_t = torch.cat([s2_t, era5_t, dem], dim=1)
            _, states = self._run_step(x_t, states, self.ctx_cells)

        predictions = []
        current_pred = self.decoder(states[-1][0])
        for t in range(F):
            era5_t = target_era5[:, t, :].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
            x_t = torch.cat([current_pred, era5_t, dem], dim=1)
            current_pred, states = self._run_step(x_t, states, self.fwd_cells)
            predictions.append(current_pred.squeeze(1))

        return torch.stack(predictions, dim=1)


class ConvGRUCell(nn.Module):
    def __init__(self, in_ch, hidden_ch, kernel_size=3):
        super().__init__()
        self.hidden_ch = hidden_ch
        pad = kernel_size // 2
        self.gates = nn.Conv2d(in_ch + hidden_ch, 2 * hidden_ch, kernel_size, padding=pad)
        self.candidate = nn.Conv2d(in_ch + hidden_ch, hidden_ch, kernel_size, padding=pad)

    def forward(self, x, h=None):
        if h is None:
            h = torch.zeros(x.size(0), self.hidden_ch, x.size(2), x.size(3),
                            device=x.device, dtype=x.dtype)
        gates = torch.sigmoid(self.gates(torch.cat([x, h], dim=1)))
        r, z = gates.chunk(2, dim=1)
        candidate = torch.tanh(self.candidate(torch.cat([x, r * h], dim=1)))
        return (1 - z) * h + z * candidate


class ContextUNet(nn.Module):
    """ConvGRU temporal encoder + U-Net decoder, non-autoregressive delta prediction."""

    delta_prediction = True

    def __init__(self, context_length=10, n_s2=4, n_era5=6,
                 forecast_horizon=20, base=48, vi_type="ndvi"):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.context_length = context_length
        self.vi_type = vi_type

        frame_in = n_s2 + n_era5 + 1
        self.frame_enc = nn.Sequential(
            nn.Conv2d(frame_in, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.GELU(),
        )
        self.conv_gru = ConvGRUCell(32, 64)

        self.era5_enc = nn.Sequential(
            nn.Linear(forecast_horizon * n_era5, 64), nn.GELU(),
            nn.Linear(64, 32), nn.GELU(),
        )

        # 64 (ConvGRU) + 1 (last_ndvi) + 32 (era5) + 1 (DEM)
        in_ch = 64 + 1 + 32 + 1

        self.enc1 = self._block(in_ch, base)
        self.enc2 = self._block(base, base * 2)
        self.enc3 = self._block(base * 2, base * 4)
        self.bottleneck = self._block(base * 4, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = self._block(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = self._block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = self._block(base * 2, base)
        self.final = nn.Conv2d(base, forecast_horizon, 1)
        self.pool = nn.MaxPool2d(2)

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.GELU(),
        )

    def forward(self, context_s2, context_era5, target_era5, dem, **kwargs):
        B, C, n_s2, H, W = context_s2.shape

        h = None
        for t in range(C):
            s2_t = context_s2[:, t]
            e5 = context_era5[:, t].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
            frame = torch.cat([s2_t, e5, dem], dim=1)
            h = self.conv_gru(self.frame_enc(frame), h)

        last_s2 = context_s2[:, -1]
        nir, red, blue = last_s2[:, 3:4], last_s2[:, 2:3], last_s2[:, 0:1]
        if self.vi_type == "evi":
            den = nir + 6.0 * red - 7.5 * blue + 1.0
            last_ndvi = (2.5 * (nir - red) / (den + 1e-8)).clamp(-1, 1)
        else:
            last_ndvi = ((nir - red) / (nir + red + 1e-8)).clamp(-1, 1)

        era5_feat = self.era5_enc(target_era5.reshape(B, -1))
        era5_spatial = era5_feat.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)

        x = torch.cat([h, last_ndvi, era5_spatial, dem], dim=1)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        delta = self.final(d1)
        return last_ndvi + delta


class TerraMindForecaster(nn.Module):
    """TerraMind embeddings + U-Net decoder, non-autoregressive delta prediction."""

    delta_prediction = True

    def __init__(self, emb_dim=768, n_s2=4, n_era5=6, forecast_horizon=20,
                 hidden=256, base=48):
        super().__init__()
        self.forecast_horizon = forecast_horizon

        self.emb_proj = nn.Sequential(
            nn.Conv2d(emb_dim, hidden, 1),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
        )
        self.conv_gru = ConvGRUCell(hidden, hidden)

        # Project GRU hidden (14x14) up to 64x64 feature map
        self.up_proj = nn.Sequential(
            nn.Conv2d(hidden, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
        )

        self.era5_enc = nn.Sequential(
            nn.Linear(forecast_horizon * n_era5, 64), nn.GELU(),
            nn.Linear(64, 32), nn.GELU(),
        )

        # U-Net at 64x64: 64 (upsampled GRU) + 4 (last S2) + 1 (last_ndvi) + 32 (era5) + 1 (DEM)
        in_ch = 64 + n_s2 + 1 + 32 + 1

        self.enc1 = self._block(in_ch, base)
        self.enc2 = self._block(base, base * 2)
        self.enc3 = self._block(base * 2, base * 4)
        self.bottleneck = self._block(base * 4, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = self._block(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = self._block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = self._block(base * 2, base)
        self.final = nn.Conv2d(base, forecast_horizon, 1)
        self.pool = nn.MaxPool2d(2)

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.GELU(),
        )

    def forward(self, context_emb, context_era5, target_era5, dem, last_ndvi,
                last_s2=None, **kwargs):
        B, C = context_emb.shape[:2]

        h = None
        for t in range(C):
            x_t = self.emb_proj(context_emb[:, t])
            h = self.conv_gru(x_t, h)

        # Upsample GRU features 14x14 -> 64x64
        h_up = nn.functional.interpolate(self.up_proj(h), size=(64, 64),
                                          mode='bilinear', align_corners=False)

        era5_feat = self.era5_enc(target_era5.reshape(B, -1))
        era5_spatial = era5_feat.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 64, 64)

        # Compute last NDVI from S2 if available
        nir, red = last_s2[:, 3:4], last_s2[:, 2:3]
        last_vi = ((nir - red) / (nir + red + 1e-8)).clamp(-1, 1)

        x = torch.cat([h_up, last_s2, last_vi, era5_spatial, dem], dim=1)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        delta = self.final(d1)
        return last_ndvi + delta


def build_model(name, context_length=10, forecast_horizon=20, n_s2=4, n_era5=6, vi_type="ndvi"):
    if name == "PixelMLP":
        return PixelMLP(context_length, n_s2, n_era5, forecast_horizon, hidden=256)
    elif name == "PixelLSTM":
        return PixelLSTM(n_s2, n_era5, hidden=64, num_layers=2, forecast_horizon=forecast_horizon)
    elif name == "MiniUNet":
        return MiniUNet(in_channels=1 + n_era5 + 1, out_channels=1, base=32)
    elif name == "ConvLSTM":
        return ConvLSTMForecaster(
            ctx_channels=n_s2 + n_era5 + 1,
            fwd_channels=1 + n_era5 + 1,
            hidden_channels=64, num_layers=2,
        )
    elif name == "ContextUNet":
        return ContextUNet(
            context_length=context_length, n_s2=n_s2, n_era5=n_era5,
            forecast_horizon=forecast_horizon, base=48, vi_type=vi_type,
        )
    elif name == "TerraMindForecaster":
        return TerraMindForecaster(
            emb_dim=768, n_s2=n_s2, n_era5=n_era5,
            forecast_horizon=forecast_horizon,
        )
    else:
        raise ValueError(f"Unknown model: {name}")
