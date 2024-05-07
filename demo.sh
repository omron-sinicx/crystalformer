#!/usr/bin/bash

save_path="result/latticeformer/jarvis"

# | target_set                      | targets |
# jarvis__megnet                    | e_form | bandgap | 
# jarvis__megnet-bulk               | bulk_modulus
# jarvis__megnet-shear              | shear_modulus
# jarvis__dft_3d_2021               | formation_energy | total_energy | opt_bandgap |
# jarvis__dft_3d_2021-ehull         | ehull |
# jarvis__dft_3d_2021-mbj_bandgap   | mbj_bandgap |

target_set=jarvis__megnet
targets=e_form
gpu=0
exp_name=demo
reproduciblity_state=3
layer=4 # {4, 7}

CUDA_VISIBLE_DEVICES=${gpu} python demo.py -p latticeformer/default.json \
    --seed 123 \
    --save_path ${save_path} \
    --domain real \
    --num_layers ${layer} \
    --experiment_name ${exp_name}/${targets} \
    --target_set ${target_set} \
    --targets ${targets} \
    --batch_size 256 \
    --reproduciblity_state ${reproduciblity_state} \
    --pretrained_model weights/megnet-${targets}-layer${layer}.ckpt
