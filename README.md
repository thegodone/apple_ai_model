# AI Apple Model AttentiveFP based on original(*) (Dense version)

You need a M1/M2/M3/... mac!

You need to install MLX, pytorch and rdkit, just create a conda env and install MLX, pytorch and rdkit using via pip or conda.

please create a saved_models folder to store torch model backup.

This is an example of usage of MLX implementation of AttentiveFP original code (code is not exactly similar to the paper "equations"!).

The goal is also to have a benchmark of the MLX, Pytorch Jax and Tensorflow implementation on GPU (Apple Metal) using exactly the same model.

I use the ESOL dataset for the experiment.

To do:
- [x] Add Pytorch version
- [ ] Add Jax version
- [ ] Add Tensorflow version

(*) original version https://github.com/OpenDrugAI/AttentiveFP
