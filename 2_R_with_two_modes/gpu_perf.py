"""Phase 2·3 공용 GPU 가속 유틸 (CUDA 서버, 현재 = RTX 4090 / Ada Lovelace).

여기 모인 설정들은 **모든 schedule·seed에 동일하게** 적용되는 '처리량(throughput)'
옵션이라, schedule 간 비교의 공정성을 해치지 않는다(통제 요소로 고정). 학습 결과를
좌우하는 hyperparameter(batch/lr/steps/seed/schedule)는 여기서 건드리지 않는다.

GPU별 주의: bf16/TF32는 Ampere+ 전용이다. RTX 4090(Ada Lovelace, compute 8.9)은 둘 다
지원하므로 resolve_amp("auto")가 bf16(GradScaler 불필요)을 고른다. 반대로 bf16 미지원
GPU(예: Turing/2080 Ti)에서는 자동으로 fp16(+GradScaler)로 폴백하고 TF32 플래그는
무시된다(켜 둬도 안전). 따라서 같은 코드가 Turing~Ada(및 이후) 모두에서 동작한다.

제공 기능:
  - configure_backends() : cudnn autotuner + TF32 활성화 (Ampere+에서만 효과, Turing은 무시).
  - resolve_amp()        : 혼합정밀(AMP) 설정 결정. bf16 미지원 GPU면 fp16으로 폴백.
  - autocast_ctx()       : AMP autocast 컨텍스트(또는 no-op).
  - make_grad_scaler()   : fp16일 때만 필요한 GradScaler(버전 호환 래퍼).
  - maybe_compile()      : torch.compile 시도(실패하면 eager로 안전 폴백).
"""
from __future__ import annotations

import contextlib

import torch


def configure_backends() -> None:
    """행렬/conv 연산 TF32 + cudnn autotuner를 켠다.

    CIFAR(32×32)·MNIST(28→32)처럼 입력 크기가 고정이면 cudnn.benchmark가 가장 빠른
    커널을 한 번 찾아 캐싱하므로 반복 학습이 빨라진다. TF32는 Ada/Ampere Tensor Core를
    써서 matmul/conv를 가속하며 학습 정확도 손실은 사실상 없다.
    """
    if not torch.cuda.is_available():
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def resolve_amp(device: str, precision: str = "auto") -> tuple[bool, torch.dtype, bool]:
    """(autocast 사용여부, dtype, GradScaler 사용여부)를 반환한다.

    precision:
      - "auto" : CUDA + bf16 지원이면 bf16(Ampere+), bf16 미지원이면 fp16(Turing/2080 Ti),
                 비CUDA면 fp32. bf16은 GradScaler 불필요, fp16은 필요.
      - "bf16" / "fp16" / "fp32" : 명시 지정.
    """
    if device != "cuda" or precision == "fp32":
        return False, torch.float32, False
    if precision == "fp16":
        return True, torch.float16, True            # fp16은 underflow 방지용 scaler 필요
    # auto or bf16
    if precision in ("auto", "bf16") and torch.cuda.is_bf16_supported():
        return True, torch.bfloat16, False          # bf16은 scaler 불필요
    return True, torch.float16, True


def autocast_ctx(device: str, enabled: bool, dtype: torch.dtype):
    """AMP autocast 컨텍스트. 비활성/비CUDA면 아무 것도 하지 않는 컨텍스트."""
    if device == "cuda" and enabled:
        return torch.autocast("cuda", dtype=dtype)
    return contextlib.nullcontext()


def make_grad_scaler(use_scaler: bool):
    """torch 버전 차이를 흡수하는 GradScaler 생성기 (fp16일 때만 실제 동작)."""
    try:                                              # torch>=2.4 권장 API
        from torch.amp import GradScaler
        return GradScaler("cuda", enabled=use_scaler)
    except (ImportError, TypeError):                  # 구버전 폴백
        return torch.cuda.amp.GradScaler(enabled=use_scaler)


def maybe_compile(model, enabled: bool):
    """torch.compile로 모델을 JIT 컴파일한다(실패 시 원본 모델 그대로 반환).

    중요: 반환된 컴파일 모델은 원본과 **같은 파라미터 텐서**를 공유한다. 따라서
    optimizer/EMA/state_dict 은 계속 '원본 model'을 대상으로 쓰고, forward 호출만
    컴파일 객체로 하면 된다(파라미터는 자동으로 함께 갱신됨).
    """
    if not enabled or not torch.cuda.is_available():
        return model
    try:
        return torch.compile(model)
    except Exception as e:  # noqa: BLE001 — 컴파일 실패는 치명적이지 않으므로 폴백
        print(f"  [compile] torch.compile 실패 — eager 모드로 진행합니다: {e}")
        return model
