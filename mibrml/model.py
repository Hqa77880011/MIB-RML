import torch
from torch import nn
from torch.nn import functional as F


VARIANTS = ("mibrml", "early_fusion", "moddrop", "uniform_ib", "no_rcra", "no_gate",
            "no_shared_private", "no_alignment", "no_consistency", "no_auxiliary")


class ModalityEncoder(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.projection = nn.Linear(features, 128)
        layer = nn.TransformerEncoderLayer(128, 4, 256, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, 2)
        self.mean = nn.Linear(128, 64)
        self.log_variance = nn.Linear(128, 64)

    def forward(self, x, lengths):
        time = torch.arange(x.shape[1], device=x.device)
        padding = time[None, :] >= lengths[:, None]
        projected = self.projection(x)
        positions = time.to(projected.dtype)[:, None]
        frequency = torch.exp(torch.arange(0, 128, 2, device=x.device) * (-torch.log(torch.tensor(10000.0, device=x.device)) / 128))
        positional = torch.zeros_like(projected[0])
        positional[:, 0::2] = torch.sin(positions * frequency)
        positional[:, 1::2] = torch.cos(positions * frequency)
        encoded = self.transformer(projected + positional, src_key_padding_mask=padding)
        pooled = encoded.masked_fill(padding[:, :, None], 0).sum(1) / lengths[:, None]
        return self.mean(pooled), self.log_variance(pooled).clamp(-8, 8)


class MIBRML(nn.Module):
    def __init__(self, dimensions, outputs, variant="mibrml", rho=0.5):
        super().__init__()
        self.variant = variant
        self.rho = rho
        self.modalities = len(dimensions)
        self.outputs = outputs
        self.encoders = nn.ModuleList(ModalityEncoder(dimension) for dimension in dimensions)
        self.shared = nn.ModuleList(nn.Linear(64, 32) for _ in dimensions)
        self.private = nn.ModuleList(nn.Linear(64, 32) for _ in dimensions)
        self.auxiliary = nn.ModuleList(nn.Linear(64, outputs) for _ in dimensions)
        self.gate = nn.Sequential(nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 1))
        self.gamma_unconstrained = nn.Parameter(torch.zeros(()))
        size = 64 * self.modalities if variant in ("early_fusion", "moddrop") else 64
        self.head = nn.Sequential(nn.Linear(size + self.modalities, 64), nn.ReLU(), nn.Linear(64, outputs))

    def forward(self, streams, lengths, availability):
        means, variances, latent = [], [], []
        for modality, values in enumerate(streams):
            observed = availability[:, modality].bool()
            mean = values.new_zeros((len(values), 64))
            log_variance = values.new_zeros((len(values), 64))
            if observed.any():
                encoded_mean, encoded_variance = self.encoders[modality](values[observed], lengths[modality][observed])
                mean = mean.index_copy(0, observed.nonzero().flatten(), encoded_mean)
                log_variance = log_variance.index_copy(0, observed.nonzero().flatten(), encoded_variance)
            code = mean
            if self.training and self.variant not in ("early_fusion", "moddrop"):
                code = mean + (0.5 * log_variance).exp() * torch.randn_like(mean)
            means.append(mean)
            variances.append(log_variance)
            latent.append(code * availability[:, modality:modality + 1])
        mean = torch.stack(means, 1)
        log_variance = torch.stack(variances, 1)
        code = torch.stack(latent, 1)
        uncertainty = log_variance.mean(-1)
        logits = self.gate(mean).squeeze(-1) - F.softplus(self.gamma_unconstrained) * uncertainty
        if self.variant in ("no_gate", "uniform_ib"):
            logits = torch.zeros_like(logits)
        weights = logits.masked_fill(availability == 0, -torch.inf).softmax(1)
        rate_logits = self.rho * uncertainty
        if self.variant in ("no_rcra", "uniform_ib"):
            rate_logits = torch.zeros_like(rate_logits)
        rate = rate_logits.masked_fill(availability == 0, -torch.inf).softmax(1) * availability.sum(1, keepdim=True)
        shared = torch.stack([self.shared[m](code[:, m]) for m in range(self.modalities)], 1)
        private = torch.stack([self.private[m](code[:, m]) for m in range(self.modalities)], 1)
        if self.variant in ("early_fusion", "moddrop"):
            fused = code.flatten(1)
        elif self.variant == "no_shared_private":
            fused = (weights[:, :, None] * code).sum(1)
        else:
            fused = torch.cat(((weights[:, :, None] * shared).sum(1),
                               (weights[:, :, None] * private).sum(1)), 1)
        prediction = self.head(torch.cat((fused, availability), 1))
        auxiliary = torch.stack([self.auxiliary[m](code[:, m]) for m in range(self.modalities)], 1)
        return dict(prediction=prediction, auxiliary=auxiliary, mean=mean, log_variance=log_variance,
                    weights=weights, rate=rate, shared=shared, private=private)


def task_loss(prediction, target, task):
    if task == "sentiment":
        return (prediction[:, 0] - target).abs() + F.cross_entropy(
            prediction[:, 1:3], (target >= 0).long(), reduction="none")
    if task == "multilabel":
        return F.binary_cross_entropy_with_logits(prediction, target, reduction="none").mean(1)
    return F.cross_entropy(prediction, target.long(), reduction="none")


def probabilities(prediction, task):
    if task == "sentiment":
        return prediction[:, 1:3].softmax(1)
    if task == "multilabel":
        return prediction.sigmoid()
    return prediction.softmax(1)


def symmetric_kl(first, second, task):
    first = first.clamp(1e-6, 1 - 1e-6)
    second = second.clamp(1e-6, 1 - 1e-6)
    if task == "multilabel":
        first = torch.stack((first, 1 - first), -1)
        second = torch.stack((second, 1 - second), -1)
    value = 0.5 * (first - second) * (first.log() - second.log())
    return value.sum(-1).mean()


def representation_loss(output, target, availability, task, epoch, args):
    loss = task_loss(output["prediction"], target, task).mean()
    if args.variant in ("early_fusion", "moddrop"):
        return loss
    if args.variant != "no_auxiliary":
        aux = sum((task_loss(output["auxiliary"][:, m], target, task) * availability[:, m]).sum()
                  for m in range(availability.shape[1])) / availability.sum()
        loss = loss + args.auxiliary_weight * aux
    kl = -0.5 * (1 + output["log_variance"] - output["mean"].square() - output["log_variance"].exp()).sum(-1)
    rate = (output["rate"] * kl).sum(1) / availability.sum(1)
    loss = loss + args.beta * min(1, epoch / 10) * rate.mean()
    if args.variant == "no_shared_private":
        return loss
    covariance = []
    alignment = []
    modalities = availability.shape[1]
    for first in range(modalities):
        observed = availability[:, first].bool()
        if observed.sum() >= 2:
            shared = output["shared"][observed, first]
            private = output["private"][observed, first]
            centered_shared = shared - shared.mean(0)
            centered_private = private - private.mean(0)
            covariance.append((centered_shared.T @ centered_private / (len(shared) - 1)).square().sum())
        for second in range(modalities):
            if first == second:
                continue
            paired = (availability[:, first] * availability[:, second]).bool()
            if not paired.any():
                continue
            a = F.normalize(output["shared"][paired, first], dim=1)
            b = F.normalize(output["shared"][paired, second], dim=1)
            similarity = a @ b.T / args.temperature
            labels = target[paired]
            if task == "sentiment":
                distance = (labels[:, None] - labels[None, :]).abs()
            elif task == "multilabel":
                distance = (labels[:, None, :] - labels[None, :, :]).abs().mean(-1)
            else:
                distance = (labels[:, None] != labels[None, :]).to(a)
            omega = 1 - torch.exp(-distance / args.label_temperature)
            omega.fill_diagonal_(1)
            denominator = torch.logsumexp(similarity + omega.log(), 1)
            alignment.append((denominator - similarity.diagonal()).mean())
    if covariance:
        loss = loss + args.covariance_weight * torch.stack(covariance).mean()
    if alignment and args.variant != "no_alignment":
        loss = loss + args.alignment_weight * torch.stack(alignment).mean()
    return loss
