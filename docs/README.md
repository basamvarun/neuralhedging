# QuantEdge — Documentation

## Contents

- [Architecture Overview](architecture.md)
- [Research Design](research_design.md)
- [API Reference](api_reference.md)

## Build Instructions

### C++ Engine
```bash
cd cpp
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Python
```bash
pip install -r requirements.txt
```

### Run Experiments
```bash
python -m python.training.train --config configs/default.yaml
```

### Dashboard
```bash
streamlit run python/dashboard/app.py
```
