cd /workspace-SR006.nfs2/mkuznetsov/mkuznetsov/GenPlans/house_diffusion/house_diffusion_trainer

export CUDA_VISIBLE_DEVICES=1

# python training/train.py root=./ exp.config_dir=./ exp.config=configs/train/base.yaml exp.name='base' \
    # 'training.debug=True'

# python training/train.py root=./ exp.config_dir=./ exp.config=configs/train/base.yaml exp.name='base|noise_sch=cosine' \
#     'diffusion.params.noise_schedule=cosine' \
    # 'training.debug=True'

# python training/train.py root=./ exp.config_dir=./ exp.config=configs/train/base.yaml exp.name='base|noise_sch=cosine|clip_grad_norm=false' \
#     'diffusion.params.noise_schedule=cosine' \
#     'model.clip_grad_norm=99' \
#     # 'training.debug=True'

# python training/train.py root=./ exp.config_dir=./ exp.config=configs/train/space_diffusion.yaml exp.name='base|noise_sch=cosine|space_diffusion=true' \
#     'diffusion.params.noise_schedule=cosine' \
#     # 'training.debug=True'

# python training/train.py root=./ exp.config_dir=./ exp.config=configs/train/base.yaml exp.name='model=msd|data=msd|noise_sch=cosine|bs=512|lr=1e-3' \
#     'diffusion.params.noise_schedule=cosine' \
#     'training.batch_size=512' \
#     'model.opt_params.lr=0.001' \
#     # 'training.debug=True'

# python training/train.py root=./ exp.config_dir=./ exp.config=configs/train/original.yaml exp.name='model=hd|data=msd' \
#     'training.debug=True'

# python training/train.py root=./ exp.config_dir=./ exp.config=configs/train/original_data_original.yaml exp.name='model=hd|data=hd' \
    # 'training.debug=True'

# python training/train.py root=./ exp.config_dir=./ exp.config=configs/train/original_square.yaml exp.name='model=hd|data=msd|mod=square' \
    # 'training.debug=True'

# python training/train.py root=./ exp.config_dir=./ exp.config=configs/train/base_sq.yaml exp.name='model=msd|data=msd|mod=square' \
    # 'training.debug=True'

# git config user.name "MikhailKuz"
# git config user.email "lkmikhailkl@gmail.com"
