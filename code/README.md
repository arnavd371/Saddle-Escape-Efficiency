# code/

`algorithm.py` is the SEE evaluation algorithm.

1. `algorithm.py`: oracles, shared-trajectory loop, SEE scalar, rank agreement
2. `core.py`: losses, saddles, Hessians, optimizer constructors
3. `run_exp1.py`: 2D clients
4. `run_exp2.py`: 10D / 50D
5. `nn.py` then `run_exp3.py`: XOR MLP
6. `run_analysis.py`: printed summary
7. `smoke.py`: short check

```bash
python algorithm.py
```

Output from the experiment scripts goes to `../results/`.
