import pytest
import torch
import io
from python.models.dense_hedger import DenseHedger
from python.models.lstm_hedger import LSTMHedger


def test_dense_hedger_forward():
    model = DenseHedger(input_dim=3, hidden_dim=16)
    
    # 5 samples of 3 features: [S_norm, BS_delta, delta_prev]
    x = torch.randn(5, 3)
    out = model(x)
    
    assert out.shape == (5, 1)
    # Output delta should be bounded in [-1, 1] due to output activation (e.g. Tanh or custom mapping)
    assert torch.all(out >= -1.0)
    assert torch.all(out <= 1.0)


def test_lstm_hedger_forward():
    # input_dim=8, hidden_dim=16, num_layers=2, use_gru=False
    model = LSTMHedger(input_dim=8, hidden_dim=16, num_layers=2, dropout=0.0, use_gru=False)
    
    # (batch_size=4, seq_len=10, input_dim=8)
    x = torch.randn(4, 10, 8)
    out = model(x)
    
    assert out.shape == (4, 10, 1)
    assert torch.all(out >= -1.0)
    assert torch.all(out <= 1.0)


def test_lstm_hedger_gru():
    # Same with GRU
    model = LSTMHedger(input_dim=8, hidden_dim=16, num_layers=2, dropout=0.0, use_gru=True)
    x = torch.randn(4, 10, 8)
    out = model(x)
    assert out.shape == (4, 10, 1)


def test_gradient_flow():
    model = LSTMHedger(input_dim=8, hidden_dim=16)
    x = torch.randn(2, 5, 8)
    
    out = model(x)
    loss = out.sum()
    loss.backward()
    
    for name, param in model.named_parameters():
        assert param.grad is not None
        assert not torch.isnan(param.grad).any()


def test_model_save_load():
    model = DenseHedger(input_dim=3, hidden_dim=16)
    x = torch.randn(3, 3)
    out1 = model(x)
    
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    
    model2 = DenseHedger(input_dim=3, hidden_dim=16)
    model2.load_state_dict(torch.load(buffer, weights_only=True))
    out2 = model2(x)
    
    assert torch.allclose(out1, out2)
