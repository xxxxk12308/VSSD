"""
Sanity-check for MC-NCD: at initialization (W_A == 0), the modified Mamba2
must produce **exactly the same** output as VSSD baseline (use_mc_ncd=False).
"""
import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.mamba2 import Mamba2


def main():
    torch.manual_seed(0)
    B, L, D = 2, 56 * 56, 96       # mimic VSSD-T stage 1
    H_, W_ = 56, 56
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32

    # Common kwargs for both baseline and MC-NCD
    common = dict(
        d_model=D, expand=2, headdim=64, ngroups=1, d_state=64,
        chunk_size=256, linear_attn_duality=True,
        device=device, dtype=dtype,
    )

    # Two instances sharing the same random seed for fair comparison
    torch.manual_seed(42)
    m_base = Mamba2(use_mc_ncd=False, **common).to(device).eval()
    torch.manual_seed(42)
    m_mc   = Mamba2(use_mc_ncd=True, mc_rank=8, **common).to(device).eval()

    # Manually copy all shared weights so the comparison is meaningful.
    # (W_A is zero-init in m_mc, a_bias is derived from A_log, so MC-NCD branch
    # contributes zero on top of VSSD's effective A.)
    state_base = m_base.state_dict()
    state_mc   = m_mc.state_dict()
    for k in state_base.keys():
        if k in state_mc:
            state_mc[k] = state_base[k].clone()
    m_mc.load_state_dict(state_mc, strict=False)

    # Verify W_A is near-zero (we use tiny std=1e-4 init instead of strict 0
    # so that W_g receives gradient from step 1 under DDP).
    assert m_mc.W_A.weight.abs().max().item() < 1e-2, "W_A should be near-zero at init"
    print(f"[OK] W_A.weight is near-zero, max|w|={m_mc.W_A.weight.abs().max().item():.2e}, shape={tuple(m_mc.W_A.weight.shape)}")

    # Random input
    u = torch.randn(B, L, D, device=device, dtype=dtype)
    relpos = None

    with torch.no_grad():
        y_base = m_base(u, H_, W_, relpos)
        y_mc   = m_mc(u, H_, W_, relpos)

    diff = (y_base - y_mc).abs().max().item()
    rel  = diff / (y_base.abs().max().item() + 1e-9)
    print(f"[Sanity] max |y_base - y_mc| = {diff:.3e}  (rel={rel:.3e})")
    if diff < 1e-2:
        print("[PASS] MC-NCD at init is numerically close to VSSD baseline.")
    else:
        print("[FAIL] MC-NCD at init deviates too much from baseline. Check init scale of W_A.")

    # Test gradient flow: after a step, W_A should get non-zero grad.
    m_mc.train()
    u2 = torch.randn(B, L, D, device=device, dtype=dtype, requires_grad=False)
    out = m_mc(u2, H_, W_, relpos)
    loss = out.mean()
    loss.backward()
    print(f"[Grad] W_A.grad.abs().mean() = {m_mc.W_A.weight.grad.abs().mean().item():.3e}")
    print(f"[Grad] W_g.grad.abs().mean() = {m_mc.W_g.weight.grad.abs().mean().item():.3e}")
    print(f"[Grad] a_bias.grad.abs().mean() = {m_mc.a_bias.grad.abs().mean().item():.3e}")

    # Param count
    n_extra = sum(p.numel() for n, p in m_mc.named_parameters()
                  if n in ('W_g.weight', 'W_A.weight', 'a_bias'))
    n_total = sum(p.numel() for p in m_mc.parameters())
    print(f"[Params] MC-NCD extra params = {n_extra} / {n_total} ({100*n_extra/n_total:.2f}%)")


if __name__ == '__main__':
    main()