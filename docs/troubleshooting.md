# Troubleshooting

## FAISS GPU is not visible

If PyTorch sees CUDA but FAISS reports zero GPUs:

```text
torch cuda: True
faiss gpus: 0
```

the environment likely installed a CPU-only or CUDA-mismatched FAISS package.
For the cluster CUDA 12 setup, use the project setup script:

```bash
bash setup_env.sh
conda activate vlm4cluster
```

For an existing environment:

```bash
bash setup_env.sh --update
conda activate vlm4cluster
```

The script creates the conda environment from `environment.yaml`, then installs
`faiss-gpu-cu12` with `--no-deps`. The `--no-deps` flag is required because the
wheel dependency metadata can otherwise let pip upgrade `numpy` to 2.x, which is
not compatible with the current PyTorch/scipy/sklearn stack used here.

If a broken or CPU-only FAISS package was installed earlier, remove it first and
rebuild the environment:

```bash
pip uninstall -y faiss faiss-cpu faiss-gpu faiss-gpu-cu12
bash setup_env.sh --update
```

Verify the result with:

```bash
python -c "import faiss; print(faiss.__version__, faiss.get_num_gpus())"
```

On CPU-only machines, skip the CUDA wheel and install `faiss-cpu` with conda
instead:

```bash
VLM4CLUSTER_FAISS_PACKAGE=none bash setup_env.sh
conda install -n vlm4cluster -c conda-forge faiss-cpu
```

The benchmark methods that use FAISS also keep scikit-learn fallbacks for
environments where FAISS is unavailable or cannot access GPU resources.
