"""
Diffusion Beta Schedule 실험: Linear vs Cosine
==============================================
- MNIST 데이터셋으로 두 가지 noise schedule을 비교
- 학습 + 샘플링 전체 파이프라인 포함

실행:
    python diffusion_schedule_experiment.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import os

# =========================================================
# 1. Beta Schedule 정의 (핵심!)
# =========================================================

def linear_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    """DDPM 원본 논문 방식. β_1=1e-4 ~ β_T=0.02 까지 선형 증가."""
    return torch.linspace(beta_start, beta_end, T)


def cosine_beta_schedule(T, s=0.008):
    """
    Nichol & Dhariwal 2021 (Improved DDPM) 방식.
    누적 신호 보존율 ᾱ_t가 cosine 곡선을 따르도록 설계.

    핵심 아이디어:
    - Linear은 후반부에 너무 빨리 망가짐 → 뒷부분 timestep 낭비
    - Cosine은 천천히 망가뜨려서 모든 timestep을 의미있게 사용
    """
    steps = T + 1
    t = torch.linspace(0, T, steps) / T
    alpha_bar = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


# =========================================================
# 2. Diffusion 유틸리티
# =========================================================

class DiffusionUtils:
    """β로부터 α, ᾱ, σ 등 필요한 값들을 미리 계산해둠"""

    def __init__(self, betas, device='cpu'):
        self.betas = betas.to(device)
        self.alphas = (1.0 - self.betas).to(device)
        self.alpha_bars = torch.cumprod(self.alphas, dim=0).to(device)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1 - self.alpha_bars)
        self.T = len(betas)
        self.device = device

    def q_sample(self, x0, t, noise=None):
        """
        Forward process: x_0에서 x_t로 한번에 점프
        x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self.sqrt_alpha_bars[t].view(-1, 1, 1, 1)
        sqrt_1mab = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1, 1)
        return sqrt_ab * x0 + sqrt_1mab * noise

    @torch.no_grad()
    def p_sample_loop(self, model, shape):
        """
        Reverse process (DDPM sampling): 순수 노이즈에서 시작해 이미지 생성
        """
        x = torch.randn(shape, device=self.device)
        for t in reversed(range(self.T)):
            t_batch = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
            pred_noise = model(x, t_batch)

            alpha_t = self.alphas[t]
            alpha_bar_t = self.alpha_bars[t]
            beta_t = self.betas[t]

            # x_{t-1} 계산
            mean = (1 / torch.sqrt(alpha_t)) * (
                x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * pred_noise
            )
            if t > 0:
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(beta_t) * noise
            else:
                x = mean
        return x


# =========================================================
# 3. 간단한 U-Net (시간 임베딩 포함)
# =========================================================

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class SimpleUNet(nn.Module):
    """MNIST(28x28)용 작은 U-Net. 노이즈 ε를 예측."""

    def __init__(self, time_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            TimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        # Encoder
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1, stride=2)
        # Time projection
        self.t_proj = nn.Linear(time_dim, 128)
        # Decoder
        self.up1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(128, 32, 4, stride=2, padding=1)
        self.out = nn.Conv2d(64, 1, 3, padding=1)
        self.act = nn.SiLU()

    def forward(self, x, t):
        temb = self.time_mlp(t)
        h1 = self.act(self.conv1(x))      # 28
        h2 = self.act(self.conv2(h1))     # 14
        h3 = self.act(self.conv3(h2))     # 7
        # add time
        h3 = h3 + self.t_proj(temb)[:, :, None, None]
        u1 = self.act(self.up1(h3))                       # 14
        u2 = self.act(self.up2(torch.cat([u1, h2], 1)))   # 28
        return self.out(torch.cat([u2, h1], 1))


# =========================================================
# 4. 학습 + 샘플링 파이프라인
# =========================================================

def train(model, diff, dataloader, epochs, device, label):
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=2e-4)
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for x, _ in dataloader:
            x = x.to(device)
            B = x.size(0)
            t = torch.randint(0, diff.T, (B,), device=device)
            noise = torch.randn_like(x)
            x_t = diff.q_sample(x, t, noise)
            pred = model(x_t, t)
            loss = F.mse_loss(pred, noise)  # ε-prediction
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss += loss.item()
        avg = epoch_loss / len(dataloader)
        losses.append(avg)
        print(f"[{label}] Epoch {epoch+1}/{epochs} - loss: {avg:.4f}")
    return losses


def run_experiment(T=200, epochs=3, batch_size=128, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}, T: {T}, epochs: {epochs}")

    # 데이터
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # [-1, 1]
    ])
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # 두 스케줄로 따로 학습
    results = {}
    for name, schedule_fn in [('linear', linear_beta_schedule),
                              ('cosine', cosine_beta_schedule)]:
        print(f"\n=== {name.upper()} schedule 학습 시작 ===")
        betas = schedule_fn(T)
        diff = DiffusionUtils(betas, device=device)
        model = SimpleUNet().to(device)
        losses = train(model, diff, loader, epochs, device, name)

        # 샘플링
        print(f"[{name}] 샘플 생성 중...")
        model.eval()
        samples = diff.p_sample_loop(model, (16, 1, 28, 28))
        samples = (samples.clamp(-1, 1) + 1) / 2  # [0,1]로 변환

        results[name] = {
            'betas': betas.cpu(),
            'alpha_bars': diff.alpha_bars.cpu(),
            'losses': losses,
            'samples': samples.cpu(),
        }

    # 결과 저장
    os.makedirs('outputs', exist_ok=True)
    plot_schedules(results, 'outputs/schedules.png')
    save_image(results['linear']['samples'], 'outputs/samples_linear.png', nrow=4)
    save_image(results['cosine']['samples'], 'outputs/samples_cosine.png', nrow=4)
    print("\n결과가 outputs/ 폴더에 저장됐어요.")
    return results


def plot_schedules(results, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # β_t
    for name in ['linear', 'cosine']:
        axes[0].plot(results[name]['betas'], label=name)
    axes[0].set_title('β_t (단계별 노이즈 양)')
    axes[0].set_xlabel('timestep t')
    axes[0].legend()

    # ᾱ_t (가장 중요!)
    for name in ['linear', 'cosine']:
        axes[1].plot(results[name]['alpha_bars'], label=name)
    axes[1].set_title('ᾱ_t (누적 신호 보존율)')
    axes[1].set_xlabel('timestep t')
    axes[1].legend()

    # loss
    for name in ['linear', 'cosine']:
        axes[2].plot(results[name]['losses'], label=name, marker='o')
    axes[2].set_title('학습 loss')
    axes[2].set_xlabel('epoch')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    run_experiment(T=200, epochs=3)