# AI Apple Model AttentiveFP based on original(*) (this is a "Dense" graph version)

You need a M1/M2/M3/... mac!

You need to install MLX, tensorflow, jupyperlab, pytorch and rdkit, just create a conda env and install MLX, tensorflow, jupyperlab, pytorch and rdkit using via pip or conda.

Clone the repository and access the cloned repository (run in terminal the following command) 
````
git clone https://github.com/thegodone/apple_ai_model && cd apple_ai_model
````

please create a saved_models folder to store torch model backup (run in terminal the following command)
````
mkdir saved_models
````

This is an example of usage of MLX implementation of AttentiveFP original code (code is not exactly similar to the paper "equations"!).

The goal is also to have a benchmark of the MLX, Pytorch Jax and Tensorflow implementation on GPU (Apple Metal) using exactly the same model.

I use the ESOL dataset for the experiment.

Run jupyter notebooks to reproduce the results. Tensorflow and MLX are not using CPU, while pytorch is using both intensively.


# Results shown a big difference in convergence between MLX and the two other packages:
# after around 20 runs with MLX only few run have a performance closer to pytorch and tensorflow that are much more stable over the runs!


To do:
- [x] Add MLX version (0.19.1) ->  0.28 sec / epochs using "fast" version : improving Dropout/Bernoulli in MLX and add compile, take_along_axis (aka gather), eval/train switch.
- [x] Add Pytorch version (2.5) -> 0.46 sec / epochs
- [x] Add Tensorflow version (2.15.0) -> 0.48 sec / epochs
- [ ] compare losses & performance (there is an important difference between the MLX, Pytorch and Tensorflow performances so far!)
- [ ] Add Jax version -> too slow compare to the other implementations!
 
(*) original version https://github.com/OpenDrugAI/AttentiveFP

License (do what you want ;-))
