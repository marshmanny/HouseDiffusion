conda install -c conda-forge mamba

mamba shell init --shell zsh --root-prefix=root/to/where/envs/will/be/stored
# reload terminal
mamba env create -f house_diffusion_msd.yaml # python=3.10
