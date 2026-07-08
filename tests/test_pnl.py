import pytest
import torch
from python.training.pnl import compute_pnl


def test_no_hedge_pnl():
    # 2 paths, 3 steps: S = [100, 105, 110], S = [100, 95, 90]
    # K = 100, call option.
    S = torch.tensor([
        [100.0, 105.0, 110.0],
        [100.0, 95.0, 90.0]
    ], dtype=torch.float32)
    deltas = torch.zeros((2, 3), dtype=torch.float32)
    
    pnl = compute_pnl(S, deltas, 100.0, "call", cost_rate=0.0)
    
    # Path 1: payoff = max(110-100, 0) = 10. P&L = -10
    # Path 2: payoff = max(90-100, 0) = 0. P&L = 0
    assert torch.allclose(pnl, torch.tensor([-10.0, 0.0]))


def test_perfect_hedge_itm():
    # If S0 = K and option is ITM call throughout, delta = 1.0 should give P&L = 0
    S = torch.tensor([[80.0, 100.0, 110.0]], dtype=torch.float32)
    deltas = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32)
    
    pnl = compute_pnl(S, deltas, 80.0, "call", cost_rate=0.0)
    
    # Payoff = 110 - 80 = 30
    # Trading P&L = 1.0*(100-80) + 1.0*(110-100) = 30
    # Total P&L = -30 + 30 = 0
    assert torch.allclose(pnl, torch.tensor([0.0]))


def test_pnl_transaction_costs():
    S = torch.tensor([[100.0, 102.0, 101.0]], dtype=torch.float32)
    # Delta changes from 0.5 to 0.7 to 0.6
    deltas = torch.tensor([[0.5, 0.7, 0.6]], dtype=torch.float32)
    
    pnl_no_costs = compute_pnl(S, deltas, 100.0, "call", cost_rate=0.0)
    pnl_with_costs = compute_pnl(S, deltas, 100.0, "call", cost_rate=0.01)
    
    # Cost at t=0: 0.01 * 100.0 * |0.7 - 0.5| = 0.01 * 100 * 0.2 = 0.2
    # Cost at t=1: 0.01 * 102.0 * |0.6 - 0.7| = 0.01 * 102 * 0.1 = 0.102
    # Total costs = 0.302
    expected_diff = -0.302
    
    assert torch.allclose(pnl_with_costs - pnl_no_costs, torch.tensor([expected_diff]))


def test_autograd_flow():
    S = torch.tensor([[100.0, 105.0, 110.0]], dtype=torch.float32)
    deltas = torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float32, requires_grad=True)
    
    pnl = compute_pnl(S, deltas, 100.0, "call", cost_rate=0.0)
    loss = -pnl.mean()
    loss.backward()
    
    assert deltas.grad is not None
    # dLoss/dDelta_0 = -(105 - 100) = -5. (mean of 1 path)
    # dLoss/dDelta_1 = -(110 - 105) = -5.
    # dLoss/dDelta_2 = 0 (since delta_2 does not trade)
    assert torch.allclose(deltas.grad, torch.tensor([[-5.0, -5.0, 0.0]]))
