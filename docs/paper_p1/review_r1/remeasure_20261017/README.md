# Re-measurement on the new AutoDL machine (2026-10-17)

Two accounting items found on 2026-10-16:

1. Table 5 / ST12d report the GPU memory of route (c) as `torch.cuda.max_memory_allocated()`, which does not include the
   cuDSS factor of K_PP, the CUDA context or the allocator cache. `remeasure.sh` re-runs the Table 5 learned route with
   the command and environment of `chain_t5_retime.sh` (FINAL3) and samples the whole-device memory with nvidia-smi every
   100 ms (`gpu_<lattice>.csv`).
2. ST13 measured the NICE state with the earlier CSR network layers and before the correction caches existed, and its K
   column is the assembled CSR, which the deployment code never forms. `st13_deploy_mem.py` measures the deployed state of
   G1–G4 on the deployment code path, grouped into stiffness representation, correction, K_PP triplets, geometry and
   network.

`remeasure.sh` waits for the data transfer to finish, then runs both; outputs go to `/root/remeasure_20261017` on the
server (status in `status.txt`, last line `ALL_DONE`). `jrun.py` runs shell commands on the server through the Jupyter
kernel API (host and token from the environment variables `JH` and `JT`; neither is stored in the repository).
