# For PyTorch (GPU 0) and JAX (GPU 1)
# CUDA_VISIBLE_DEVICES=0,1 XLA_PYTHON_CLIENT_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=25695 train_esm.py
CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node 1 --master_port=25695 train_esm.py
