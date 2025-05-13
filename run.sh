
echo "Running NeRV"

#--not_resume_epoch
# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --not_resume_epoch --prune_ratio 0.4  --eval_only --quant_bit 16 --quant_axis 0 --num_frames 20

# No pruning
# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --eval_only --quant_bit 8 --quant_axis 0 --num_frames 20

# limiting number of frames to be used
# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_bit 8 --quant_axis 0 --num_frames 20


# INT

###############
# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only \
#     --num_frames 132 --num_prec_layers 2  --quant_bit 6 --quant_bit_enh 5 5 1 --qmode integer \
#     --dump_images --run_id _1


python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
    --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
    --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
    -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
    --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only \
    --num_frames 2 --num_prec_layers 3  --quant_bit 6 --quant_bit_enh 5 5 1 --qmode integer
    #--dump_images --run_id _2
#     ###############




# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only \
#     --num_frames 132 --num_prec_layers 1  --quant_bit 1 --quant_bit_enh 1 1 1 --qmode integer

# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only \
#     --num_frames 132 --num_prec_layers 1  --quant_bit 2 --quant_bit_enh 1 1 1 --qmode integer

# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only \
#     --num_frames 132 --num_prec_layers 1  --quant_bit 3 --quant_bit_enh 1 1 1 --qmode integer

# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only \
#     --num_frames 132 --num_prec_layers 1  --quant_bit 4 --quant_bit_enh 1 1 1 --qmode integer

# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only \
#     --num_frames 132 --num_prec_layers 1  --quant_bit 5 --quant_bit_enh 1 1 1 --qmode integer

# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only \
#     --num_frames 132 --num_prec_layers 1  --quant_bit 6 --quant_bit_enh 1 1 1 --qmode integer \
#     --dump_images


# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only \
#     --num_frames 132 --num_prec_layers 1  --quant_bit 7 --quant_bit_enh 1 1 1 --qmode integer

# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only \
#     --num_frames 132 --num_prec_layers 1  --quant_bit 8 --quant_bit_enh 1 1 1 --qmode integer


# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only \
#     --num_frames 132 --num_prec_layers 1  --quant_bit 9 --quant_bit_enh 1 1 1 --qmode integer


# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only \
#     --num_frames 132 --num_prec_layers 1  --quant_bit 10 --quant_bit_enh 1 1 1 --qmode integer

# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only \
#     --num_frames 132 --num_prec_layers 1  --quant_bit 11 --quant_bit_enh 1 1 1 --qmode integer


# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only \
#     --num_frames 132 --num_prec_layers 1  --quant_bit 12 --quant_bit_enh 1 1 1 --qmode integer


# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only \
#     --num_frames 132 --num_prec_layers 1  --quant_bit 13 --quant_bit_enh 1 1 1 --qmode integer

# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only \
#     --num_frames 132 --num_prec_layers 1  --quant_bit 14 --quant_bit_enh 1 1 1 --qmode integer

# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only \
#     --num_frames 132 --num_prec_layers 1  --quant_bit 15 --quant_bit_enh 1 1 1 --qmode integer


# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only \
#     --num_frames 132 --num_prec_layers 1  --quant_bit 16 --quant_bit_enh 1 1 1 --qmode integer



# You need more dynamic range as you head towards higher precision layers
#Evaluate mdlns scalable quantization

# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_axis 0 --num_frames 132 --num_prec_layers 1 --quant_bit_enh 4 4 4 \
#     --qmode mdlns \
#     --mdlns_sweep_start 0.1 --mdlns_sweep_end 10.0 --mdlns_sweep_step 0.1 --mdlns_auto_scale 0 \
#     --mdlns_first_base 2 --mdlns_second_base 3 --quant_bit 3  --mdlns_second_base_exp_num_bits 1 2 2 2


# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_axis 0 --num_frames 132 --num_prec_layers 1 --quant_bit_enh 4 4 4 \
#     --qmode mdlns \
#     --mdlns_sweep_start 0.1 --mdlns_sweep_end 10.0 --mdlns_sweep_step 0.1 --mdlns_auto_scale 0 \
#     --mdlns_first_base 2 --mdlns_second_base 3 --quant_bit 4  --mdlns_second_base_exp_num_bits 1 2 2 2


# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_axis 0 --num_frames 132 --num_prec_layers 1 --quant_bit_enh 4 4 4 \
#     --qmode mdlns \
#     --mdlns_sweep_start 0.1 --mdlns_sweep_end 10.0 --mdlns_sweep_step 0.1 --mdlns_auto_scale 0 \
#     --mdlns_first_base 2 --mdlns_second_base 3 --quant_bit 4  --mdlns_second_base_exp_num_bits 2 2 2 2


# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_axis 0 --num_frames 132 --num_prec_layers 1 --quant_bit_enh 4 4 4 \
#     --qmode mdlns \
#     --mdlns_sweep_start 0.1 --mdlns_sweep_end 10.0 --mdlns_sweep_step 0.1 --mdlns_auto_scale 0 \
#     --mdlns_first_base 2 --mdlns_second_base 3 --quant_bit 5  --mdlns_second_base_exp_num_bits 2 2 2 2


# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_axis 0 --num_frames 132 --num_prec_layers 1 --quant_bit_enh 4 4 4 \
#     --qmode mdlns \
#     --mdlns_sweep_start 0.1 --mdlns_sweep_end 10.0 --mdlns_sweep_step 0.1 --mdlns_auto_scale 0 \
#     --mdlns_first_base 2 --mdlns_second_base 3 --quant_bit 6  --mdlns_second_base_exp_num_bits 2 2 2 2


# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_axis 0 --num_frames 132 --num_prec_layers 1 --quant_bit_enh 4 4 4 \
#     --qmode mdlns \
#     --mdlns_sweep_start 0.1 --mdlns_sweep_end 10.0 --mdlns_sweep_step 0.1 --mdlns_auto_scale 0 \
#     --mdlns_first_base 2 --mdlns_second_base 3 --quant_bit 6  --mdlns_second_base_exp_num_bits 3 2 2 2


# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_axis 0 --num_frames 132 --num_prec_layers 1 --quant_bit_enh 4 4 4 \
#     --qmode mdlns \
#     --mdlns_sweep_start 0.1 --mdlns_sweep_end 10.0 --mdlns_sweep_step 0.1 --mdlns_auto_scale 0 \
#     --mdlns_first_base 2 --mdlns_second_base 3 --quant_bit 7  --mdlns_second_base_exp_num_bits 3 2 2 2



# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_axis 0 --num_frames 132 --num_prec_layers 1 --quant_bit_enh 4 4 4 \
#     --qmode mdlns \
#     --mdlns_sweep_start 0.1 --mdlns_sweep_end 10.0 --mdlns_sweep_step 0.1 --mdlns_auto_scale 0 \
#     --mdlns_first_base 2 --mdlns_second_base 3 --quant_bit 8  --mdlns_second_base_exp_num_bits 3 2 2 2


# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_axis 0 --num_frames 132 --num_prec_layers 1 --quant_bit_enh 4 4 4 \
#     --qmode mdlns \
#     --mdlns_sweep_start 0.1 --mdlns_sweep_end 10.0 --mdlns_sweep_step 0.1 --mdlns_auto_scale 0 \
#     --mdlns_first_base 2 --mdlns_second_base 3 --quant_bit 8  --mdlns_second_base_exp_num_bits 4 2 2 2



# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_axis 0 --num_frames 132 --num_prec_layers 1 --quant_bit_enh 4 4 4 \
#     --qmode mdlns \
#     --mdlns_sweep_start 0.1 --mdlns_sweep_end 10.0 --mdlns_sweep_step 0.1 --mdlns_auto_scale 0 \
#     --mdlns_first_base 2 --mdlns_second_base 3 --quant_bit 9  --mdlns_second_base_exp_num_bits 4 2 2 2


# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_axis 0 --num_frames 132 --num_prec_layers 1 --quant_bit_enh 4 4 4 \
#     --qmode mdlns \
#     --mdlns_sweep_start 0.1 --mdlns_sweep_end 10.0 --mdlns_sweep_step 0.1 --mdlns_auto_scale 0 \
#     --mdlns_first_base 2 --mdlns_second_base 3 --quant_bit 10  --mdlns_second_base_exp_num_bits 4 2 2 2


# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_axis 0 --num_frames 132 --num_prec_layers 1 --quant_bit_enh 4 4 4 \
#     --qmode mdlns \
#     --mdlns_sweep_start 0.3 --mdlns_sweep_end 10.0 --mdlns_sweep_step 0.3 --mdlns_auto_scale 0 \
#     --mdlns_first_base 2 --mdlns_second_base 3 --quant_bit 10  --mdlns_second_base_exp_num_bits 5 2 2 2



# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_axis 0 --num_frames 132 --num_prec_layers 1 --quant_bit_enh 4 4 4 \
#     --qmode mdlns \
#     --mdlns_sweep_start 0.3 --mdlns_sweep_end 10.0 --mdlns_sweep_step 0.3 --mdlns_auto_scale 0 \
#     --mdlns_first_base 2 --mdlns_second_base 3 --quant_bit 11  --mdlns_second_base_exp_num_bits 5 2 2 2



# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_axis 0 --num_frames 132 --num_prec_layers 1 --quant_bit_enh 4 4 4 \
#     --qmode mdlns \
#     --mdlns_sweep_start 0.3 --mdlns_sweep_end 10.0 --mdlns_sweep_step 0.3 --mdlns_auto_scale 0 \
#     --mdlns_first_base 2 --mdlns_second_base 3 --quant_bit 12  --mdlns_second_base_exp_num_bits 5 2 2 2


# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_axis 0 --num_frames 132 --num_prec_layers 1 --quant_bit_enh 4 4 4 \
#     --qmode mdlns \
#     --mdlns_sweep_start 0.3 --mdlns_sweep_end 10.0 --mdlns_sweep_step 0.3 --mdlns_auto_scale 0 \
#     --mdlns_first_base 2 --mdlns_second_base 3 --quant_bit 12  --mdlns_second_base_exp_num_bits 6 2 2 2



# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_axis 0 --num_frames 132 --num_prec_layers 1 --quant_bit_enh 4 4 4 \
#     --qmode mdlns \
#     --mdlns_sweep_start 0.3 --mdlns_sweep_end 10.0 --mdlns_sweep_step 0.3 --mdlns_auto_scale 0 \
#     --mdlns_first_base 2 --mdlns_second_base 3 --quant_bit 13  --mdlns_second_base_exp_num_bits 6 2 2 2



# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_axis 0 --num_frames 132 --num_prec_layers 1 --quant_bit_enh 4 4 4 \
#     --qmode mdlns \
#     --mdlns_sweep_start 0.3 --mdlns_sweep_end 10.0 --mdlns_sweep_step 0.3 --mdlns_auto_scale 0 \
#     --mdlns_first_base 2 --mdlns_second_base 3 --quant_bit 14  --mdlns_second_base_exp_num_bits 6 2 2 2


# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_axis 0 --num_frames 132 --num_prec_layers 1 --quant_bit_enh 4 4 4 \
#     --qmode mdlns \
#     --mdlns_sweep_start 0.3 --mdlns_sweep_end 10.0 --mdlns_sweep_step 0.3 --mdlns_auto_scale 0 \
#     --mdlns_first_base 2 --mdlns_second_base 3 --quant_bit 14  --mdlns_second_base_exp_num_bits 7 2 2 2


# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_axis 0 --num_frames 132 --num_prec_layers 1 --quant_bit_enh 4 4 4 \
#     --qmode mdlns \
#     --mdlns_sweep_start 0.3 --mdlns_sweep_end 10.0 --mdlns_sweep_step 0.3 --mdlns_auto_scale 0 \
#     --mdlns_first_base 2 --mdlns_second_base 3 --quant_bit 15  --mdlns_second_base_exp_num_bits 7 2 2 2



# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_axis 0 --num_frames 132 --num_prec_layers 1 --quant_bit_enh 4 4 4 \
#     --qmode mdlns \
#     --mdlns_sweep_start 0.3 --mdlns_sweep_end 10.0 --mdlns_sweep_step 0.3 --mdlns_auto_scale 0 \
#     --mdlns_first_base 2 --mdlns_second_base 3 --quant_bit 16  --mdlns_second_base_exp_num_bits 7 2 2 2


# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_axis 0 --num_frames 132 --num_prec_layers 1 --quant_bit_enh 4 4 4 \
#     --qmode mdlns \
#     --mdlns_sweep_start 0.3 --mdlns_sweep_end 10.0 --mdlns_sweep_step 0.3 --mdlns_auto_scale 0 \
#     --mdlns_first_base 2 --mdlns_second_base 3 --quant_bit 16  --mdlns_second_base_exp_num_bits 8 2 2 2


################################################################

    
# lns

# For 2 precision layers, you need at least 4 bits for exponent
                    # For 3 precision layers, you need at least 5 bits for exponent
                    # You need more dynamic range as you head towards higher precision layers


###################################################



# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --num_frames 132 --num_prec_layers 2 --quant_bit_enh 5 5 1 \
#     --qmode lns \
#     --lns_base 2 --lns_exp_num_bits 5 4 4 4 --quant_bit 6



# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --num_frames 132 --num_prec_layers 3 --quant_bit_enh 5 5 1 \
#     --qmode lns \
#     --lns_base 2 --lns_exp_num_bits 5 4 4 4 --quant_bit 6


##############

# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --num_frames 132 --num_prec_layers 2 --quant_bit_enh 5 5 1 \
#     --qmode lns --mdlns_second_base_exp_num_bits 2 2 2 2\
#     --mdlns_sweep_start 0.5 --mdlns_sweep_end 10.0 --mdlns_sweep_step 0.5 --mdlns_auto_scale 0\
#     --mdlns_first_base 2 --mdlns_second_base 1000\
#     --lns_base 2 --lns_exp_num_bits 4 4 4 4 --quant_bit 5
###################################################

# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --num_frames 132 --num_prec_layers 1 --quant_bit_enh 5 5 5 \
#     --qmode lns --mdlns_second_base_exp_num_bits 2 2 2 2\
#     --mdlns_sweep_start 0.5 --mdlns_sweep_end 10.0 --mdlns_sweep_step 0.5 --mdlns_auto_scale 0\
#     --mdlns_first_base 2 --mdlns_second_base 1000\
#     --lns_base 2 --lns_exp_num_bits 1 5 5 5 --quant_bit 2


# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --num_frames 132 --num_prec_layers 1 --quant_bit_enh 5 5 5 \
#     --qmode lns --mdlns_second_base_exp_num_bits 2 2 2 2\
#     --mdlns_sweep_start 0.5 --mdlns_sweep_end 10.0 --mdlns_sweep_step 0.5 --mdlns_auto_scale 0\
#     --mdlns_first_base 2 --mdlns_second_base 1000\
#     --lns_base 2 --lns_exp_num_bits 2 5 5 5 --quant_bit 3


# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --num_frames 132 --num_prec_layers 1 --quant_bit_enh 5 5 5 \
#     --qmode lns --mdlns_second_base_exp_num_bits 2 2 2 2\
#     --mdlns_sweep_start 0.5 --mdlns_sweep_end 10.0 --mdlns_sweep_step 0.5 --mdlns_auto_scale 0\
#     --mdlns_first_base 2 --mdlns_second_base 1000\
#     --lns_base 2 --lns_exp_num_bits 3 5 5 5 --quant_bit 4


# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --num_frames 132 --num_prec_layers 1 --quant_bit_enh 5 5 5 \
#     --qmode lns --mdlns_second_base_exp_num_bits 2 2 2 2\
#     --mdlns_sweep_start 0.5 --mdlns_sweep_end 10.0 --mdlns_sweep_step 0.5 --mdlns_auto_scale 0\
#     --mdlns_first_base 2 --mdlns_second_base 1000\
#     --lns_base 2 --lns_exp_num_bits 4 5 5 5 --quant_bit 5


# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --num_frames 132 --num_prec_layers 1 --quant_bit_enh 5 5 5 \
#     --qmode lns --mdlns_second_base_exp_num_bits 2 2 2 2\
#     --mdlns_sweep_start 0.5 --mdlns_sweep_end 10.0 --mdlns_sweep_step 0.5 --mdlns_auto_scale 0\
#     --mdlns_first_base 2 --mdlns_second_base 1000\
#     --lns_base 2 --lns_exp_num_bits 5 5 5 5 --quant_bit 6


# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --num_frames 132 --num_prec_layers 1 --quant_bit_enh 5 5 5 \
#     --qmode lns --mdlns_second_base_exp_num_bits 2 2 2 2\
#     --mdlns_sweep_start 0.5 --mdlns_sweep_end 10.0 --mdlns_sweep_step 0.5 --mdlns_auto_scale 0\
#     --mdlns_first_base 2 --mdlns_second_base 1000\
#     --lns_base 2 --lns_exp_num_bits 6 5 5 5 --quant_bit 7


# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --num_frames 132 --num_prec_layers 1 --quant_bit_enh 5 5 5 \
#     --qmode lns --mdlns_second_base_exp_num_bits 2 2 2 2\
#     --mdlns_sweep_start 0.5 --mdlns_sweep_end 10.0 --mdlns_sweep_step 0.5 --mdlns_auto_scale 0\
#     --mdlns_first_base 2 --mdlns_second_base 1000\
#     --lns_base 2 --lns_exp_num_bits 7 5 5 5 --quant_bit 8

#     python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --num_frames 132 --num_prec_layers 1 --quant_bit_enh 5 5 5 \
#     --qmode lns --mdlns_second_base_exp_num_bits 2 2 2 2\
#     --mdlns_sweep_start 0.5 --mdlns_sweep_end 10.0 --mdlns_sweep_step 0.5 --mdlns_auto_scale 0\
#     --mdlns_first_base 2 --mdlns_second_base 1000\
#     --lns_base 2 --lns_exp_num_bits 8 5 5 5 --quant_bit 9


# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --num_frames 132 --num_prec_layers 1 --quant_bit_enh 5 5 5 \
#     --qmode lns --mdlns_second_base_exp_num_bits 2 2 2 2\
#     --mdlns_sweep_start 0.5 --mdlns_sweep_end 10.0 --mdlns_sweep_step 0.5 --mdlns_auto_scale 0\
#     --mdlns_first_base 2 --mdlns_second_base 1000\
#     --lns_base 2 --lns_exp_num_bits 9 5 5 5 --quant_bit 10


# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --num_frames 132 --num_prec_layers 1 --quant_bit_enh 5 5 5 \
#     --qmode lns --mdlns_second_base_exp_num_bits 2 2 2 2\
#     --mdlns_sweep_start 0.5 --mdlns_sweep_end 10.0 --mdlns_sweep_step 0.5 --mdlns_auto_scale 0\
#     --mdlns_first_base 2 --mdlns_second_base 1000\
#     --lns_base 2 --lns_exp_num_bits 10 5 5 5 --quant_bit 11


# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --num_frames 132 --num_prec_layers 1 --quant_bit_enh 5 5 5 \
#     --qmode lns --mdlns_second_base_exp_num_bits 2 2 2 2\
#     --mdlns_sweep_start 0.5 --mdlns_sweep_end 10.0 --mdlns_sweep_step 0.5 --mdlns_auto_scale 0\
#     --mdlns_first_base 2 --mdlns_second_base 1000\
#     --lns_base 2 --lns_exp_num_bits 11 5 5 5 --quant_bit 12


# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --num_frames 132 --num_prec_layers 1 --quant_bit_enh 5 5 5 \
#     --qmode lns --mdlns_second_base_exp_num_bits 2 2 2 2\
#     --mdlns_sweep_start 0.5 --mdlns_sweep_end 10.0 --mdlns_sweep_step 0.5 --mdlns_auto_scale 0\
#     --mdlns_first_base 2 --mdlns_second_base 1000\
#     --lns_base 2 --lns_exp_num_bits 12 5 5 5 --quant_bit 13


# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --num_frames 132 --num_prec_layers 1 --quant_bit_enh 5 5 5 \
#     --qmode lns --mdlns_second_base_exp_num_bits 2 2 2 2\
#     --mdlns_sweep_start 0.5 --mdlns_sweep_end 10.0 --mdlns_sweep_step 0.5 --mdlns_auto_scale 0\
#     --mdlns_first_base 2 --mdlns_second_base 1000\
#     --lns_base 2 --lns_exp_num_bits 13 5 5 5 --quant_bit 14


# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --num_frames 132 --num_prec_layers 1 --quant_bit_enh 5 5 5 \
#     --qmode lns --mdlns_second_base_exp_num_bits 2 2 2 2\
#     --mdlns_sweep_start 0.5 --mdlns_sweep_end 10.0 --mdlns_sweep_step 0.5 --mdlns_auto_scale 0\
#     --mdlns_first_base 2 --mdlns_second_base 1000\
#     --lns_base 2 --lns_exp_num_bits 14 5 5 5 --quant_bit 15


# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --num_frames 132 --num_prec_layers 1 --quant_bit_enh 5 5 5 \
#     --qmode lns --mdlns_second_base_exp_num_bits 2 2 2 2\
#     --mdlns_sweep_start 0.5 --mdlns_sweep_end 10.0 --mdlns_sweep_step 0.5 --mdlns_auto_scale 0\
#     --mdlns_first_base 2 --mdlns_second_base 1000\
#     --lns_base 2 --lns_exp_num_bits 15 5 5 5 --quant_bit 16


###################################################


#minifloat
#################

##############
# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_bit 6 --quant_axis 0 --num_frames 132 --num_prec_layers 2 --quant_bit_enh 5 5 1 \
#     --qmode minifloat \
#     --minifloat_exp_num_bits 3 2 2 2


# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_bit 6 --quant_axis 0 --num_frames 132 --num_prec_layers 3 --quant_bit_enh 5 5 1 \
#     --qmode minifloat \
#     --minifloat_exp_num_bits 3 2 2 2
##############

# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_bit 3 --quant_axis 0 --num_frames 132 --num_prec_layers 1 --quant_bit_enh 6 6 6 \
#     --qmode minifloat \
#     --minifloat_exp_num_bits 1 4 4 4

# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_bit 4 --quant_axis 0 --num_frames 132 --num_prec_layers 1 --quant_bit_enh 6 6 6 \
#     --qmode minifloat \
#     --minifloat_exp_num_bits 1 4 4 4

# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_bit 4 --quant_axis 0 --num_frames 132 --num_prec_layers 1 --quant_bit_enh 6 6 6 \
#     --qmode minifloat \
#     --minifloat_exp_num_bits 2 4 4 4



# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_bit 5 --quant_axis 0 --num_frames 132 --num_prec_layers 1 --quant_bit_enh 6 6 6 \
#     --qmode minifloat \
#     --minifloat_exp_num_bits 2 4 4 4

# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_bit 6 --quant_axis 0 --num_frames 132 --num_prec_layers 1 --quant_bit_enh 6 6 6 \
#     --qmode minifloat \
#     --minifloat_exp_num_bits 2 4 4 4

# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_bit 6 --quant_axis 0 --num_frames 132 --num_prec_layers 1 --quant_bit_enh 6 6 6 \
#     --qmode minifloat \
#     --minifloat_exp_num_bits 3 4 4 4



# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_bit 7 --quant_axis 0 --num_frames 132 --num_prec_layers 1 --quant_bit_enh 6 6 6 \
#     --qmode minifloat \
#     --minifloat_exp_num_bits 3 4 4 4

# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_bit 8 --quant_axis 0 --num_frames 132 --num_prec_layers 1 --quant_bit_enh 6 6 6 \
#     --qmode minifloat \
#     --minifloat_exp_num_bits 3 4 4 4

# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_bit 8 --quant_axis 0 --num_frames 132 --num_prec_layers 1 --quant_bit_enh 6 6 6 \
#     --qmode minifloat \
#     --minifloat_exp_num_bits 4 4 4 4


# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_bit 9 --quant_axis 0 --num_frames 132 --num_prec_layers 1 --quant_bit_enh 6 6 6 \
#     --qmode minifloat \
#     --minifloat_exp_num_bits 4 4 4 4

# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_bit 10 --quant_axis 0 --num_frames 132 --num_prec_layers 1 --quant_bit_enh 6 6 6 \
#     --qmode minifloat \
#     --minifloat_exp_num_bits 4 4 4 4

# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_bit 10 --quant_axis 0 --num_frames 132 --num_prec_layers 1 --quant_bit_enh 6 6 6 \
#     --qmode minifloat \
#     --minifloat_exp_num_bits 5 4 4 4


# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_bit 11 --quant_axis 0 --num_frames 132 --num_prec_layers 1 --quant_bit_enh 6 6 6 \
#     --qmode minifloat \
#     --minifloat_exp_num_bits 5 4 4 4

# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_bit 12 --quant_axis 0 --num_frames 132 --num_prec_layers 1 --quant_bit_enh 6 6 6 \
#     --qmode minifloat \
#     --minifloat_exp_num_bits 5 4 4 4

# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_bit 12 --quant_axis 0 --num_frames 132 --num_prec_layers 1 --quant_bit_enh 6 6 6 \
#     --qmode minifloat \
#     --minifloat_exp_num_bits 6 4 4 4


# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_bit 13 --quant_axis 0 --num_frames 132 --num_prec_layers 1 --quant_bit_enh 6 6 6 \
#     --qmode minifloat \
#     --minifloat_exp_num_bits 6 4 4 4

# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_bit 14 --quant_axis 0 --num_frames 132 --num_prec_layers 1 --quant_bit_enh 6 6 6 \
#     --qmode minifloat \
#     --minifloat_exp_num_bits 6 4 4 4

# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_bit 14 --quant_axis 0 --num_frames 132 --num_prec_layers 1 --quant_bit_enh 6 6 6 \
#     --qmode minifloat \
#     --minifloat_exp_num_bits 7 4 4 4



# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_bit 15 --quant_axis 0 --num_frames 132 --num_prec_layers 1 --quant_bit_enh 6 6 6 \
#     --qmode minifloat \
#     --minifloat_exp_num_bits 7 4 4 4

# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_bit 16 --quant_axis 0 --num_frames 132 --num_prec_layers 1 --quant_bit_enh 6 6 6 \
#     --qmode minifloat \
#     --minifloat_exp_num_bits 7 4 4 4

# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_bit 16 --quant_axis 0 --num_frames 132 --num_prec_layers 1 --quant_bit_enh 6 6 6 \
#     --qmode minifloat \
#     --minifloat_exp_num_bits 8 4 4 4

#################

# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_bit 20 --quant_axis 0 --num_frames 2 --num_prec_layers 2 --quant_bit_enh 20 20 20 \
#     --qmode lns --mdlns_second_base_exp_num_bits 2 2 2 2\
#     --mdlns_sweep_start 0.5 --mdlns_sweep_end 10.0 --mdlns_sweep_step 0.5 --mdlns_auto_scale 0\
#     --mdlns_first_base 2 --mdlns_second_base 1000\
#     --lns_base 2 --lns_exp_num_bits 3 3 3 3
    

#Evaluate quantized model command line from Git
# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_bit 15 --quant_axis 0 --num_frames 2 --num_prec_layers 1 --quant_bit_enh 5 5 5

# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_bit 4 --num_frames 2 --num_prec_layers 2 --quant_bit_enh 4 4 4




# python train_nerv.py -e 300   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf bunny_ab --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0.2 --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none  --act swish \
#     --weight checkpoints/nerv_S.pth --eval_only 

# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf prune_ab --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S.pth --not_resume_epoch --prune_ratio 0.4 


# You need more dynamic range as you head towards higher precision layers
#Evaluate mdlns scalable quantization WITH the sweep feature


######################
# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_axis 0 --num_frames 132 --num_prec_layers 1 --quant_bit_enh 5 5 5 \
#     --qmode mdlns \
#     --mdlns_sweep_start 0.25 --mdlns_sweep_end 10.0 --mdlns_sweep_step 0.25 --mdlns_auto_scale 0 \
#     --mdlns_first_base 2 --mdlns_second_base 3 --quant_bit 3  --mdlns_second_base_exp_num_bits 1 2 2 2

######################
# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_axis 0 --num_frames 132 --num_prec_layers 2 --quant_bit_enh 5 5 5 \
#     --qmode mdlns \
#     --mdlns_sweep_start 0.25 --mdlns_sweep_end 10.0 --mdlns_sweep_step 0.25 --mdlns_auto_scale 0 \
#     --mdlns_first_base 2 --mdlns_second_base 3 --quant_bit 6  --mdlns_second_base_exp_num_bits 2 2 2 2



# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_axis 0 --num_frames 132 --num_prec_layers 3 --quant_bit_enh 5 5 5 \
#     --qmode mdlns \
#     --mdlns_sweep_start 0.25 --mdlns_sweep_end 10.0 --mdlns_sweep_step 0.25 --mdlns_auto_scale 0 \
#     --mdlns_first_base 2 --mdlns_second_base 3 --quant_bit 6  --mdlns_second_base_exp_num_bits 2 2 2 2


#****************************************************************
# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_axis 0 --num_frames 132 --num_prec_layers 2 --quant_bit_enh 5 5 5 \
#     --qmode mdlns \
#     --mdlns_sweep_start 0.25 --mdlns_sweep_end 10.0 --mdlns_sweep_step 0.25 --mdlns_auto_scale 0 \
#     --mdlns_first_base 2 --mdlns_second_base 1000 --quant_bit 6  --mdlns_second_base_exp_num_bits 2 2 2 2 \
#     --dump_images --run_id _1


# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_axis 0 --num_frames 132 --num_prec_layers 3 --quant_bit_enh 5 5 5 \
#     --qmode mdlns \
#     --mdlns_sweep_start 0.25 --mdlns_sweep_end 10.0 --mdlns_sweep_step 0.25 --mdlns_auto_scale 0 \
#     --mdlns_first_base 2 --mdlns_second_base 1000 --quant_bit 6  --mdlns_second_base_exp_num_bits 2 2 2 2
#     --dump_images --run_id _2


# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_axis 0 --num_frames 132 --num_prec_layers 2 --quant_bit_enh 5 5 5 \
#     --qmode mixed \
#     --mdlns_sweep_start 0.25 --mdlns_sweep_end 10.0 --mdlns_sweep_step 0.25 --mdlns_auto_scale 0 \
#     --mdlns_first_base 2 --mdlns_second_base 1000 --quant_bit 6  --mdlns_second_base_exp_num_bits 2 2 2 2 \
#     --dump_images --run_id _3


# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_axis 0 --num_frames 132 --num_prec_layers 3 --quant_bit_enh 5 5 5 \
#     --qmode mixed \
#     --mdlns_sweep_start 0.25 --mdlns_sweep_end 10.0 --mdlns_sweep_step 0.25 --mdlns_auto_scale 0 \
#     --mdlns_first_base 2 --mdlns_second_base 1000 --quant_bit 6  --mdlns_second_base_exp_num_bits 2 2 2 2 \
#     --dump_images --run_id _4
#****************************************************************
######################





# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_axis 0 --num_frames 132 --num_prec_layers 1 --quant_bit_enh 4 4 4 \
#     --qmode mdlns \
#     --mdlns_sweep_start 0.25 --mdlns_sweep_end 10.0 --mdlns_sweep_step 0.25 --mdlns_auto_scale 0 \
#     --mdlns_first_base 2 --mdlns_second_base 1000 --quant_bit 3  --mdlns_second_base_exp_num_bits 1 2 2 2


# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_axis 0 --num_frames 132 --num_prec_layers 1 --quant_bit_enh 4 4 4 \
#     --qmode mdlns \
#     --mdlns_sweep_start 0.25 --mdlns_sweep_end 10.0 --mdlns_sweep_step 0.25 --mdlns_auto_scale 0 \
#     --mdlns_first_base 2 --mdlns_second_base 1000 --quant_bit 4  --mdlns_second_base_exp_num_bits 1 2 2 2


# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_axis 0 --num_frames 132 --num_prec_layers 1 --quant_bit_enh 4 4 4 \
#     --qmode mdlns \
#     --mdlns_sweep_start 0.25 --mdlns_sweep_end 10.0 --mdlns_sweep_step 0.25 --mdlns_auto_scale 0 \
#     --mdlns_first_base 2 --mdlns_second_base 1000 --quant_bit 4  --mdlns_second_base_exp_num_bits 2 2 2 2


# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_axis 0 --num_frames 132 --num_prec_layers 1 --quant_bit_enh 4 4 4 \
#     --qmode mdlns \
#     --mdlns_sweep_start 0.25 --mdlns_sweep_end 10.0 --mdlns_sweep_step 0.25 --mdlns_auto_scale 0 \
#     --mdlns_first_base 2 --mdlns_second_base 1000 --quant_bit 5  --mdlns_second_base_exp_num_bits 2 2 2 2


# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_axis 0 --num_frames 132 --num_prec_layers 1 --quant_bit_enh 4 4 4 \
#     --qmode mdlns \
#     --mdlns_sweep_start 0.25 --mdlns_sweep_end 10.0 --mdlns_sweep_step 0.25 --mdlns_auto_scale 0 \
#     --mdlns_first_base 2 --mdlns_second_base 1000 --quant_bit 6  --mdlns_second_base_exp_num_bits 2 2 2 2 \
#     --dump_images


# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_axis 0 --num_frames 132 --num_prec_layers 1 --quant_bit_enh 4 4 4 \
#     --qmode mdlns \
#     --mdlns_sweep_start 0.25 --mdlns_sweep_end 10.0 --mdlns_sweep_step 0.25 --mdlns_auto_scale 0 \
#     --mdlns_first_base 2 --mdlns_second_base 1000 --quant_bit 6  --mdlns_second_base_exp_num_bits 3 2 2 2


# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_axis 0 --num_frames 132 --num_prec_layers 1 --quant_bit_enh 4 4 4 \
#     --qmode mdlns \
#     --mdlns_sweep_start 0.25 --mdlns_sweep_end 10.0 --mdlns_sweep_step 0.25 --mdlns_auto_scale 0 \
#     --mdlns_first_base 2 --mdlns_second_base 1000 --quant_bit 7  --mdlns_second_base_exp_num_bits 3 2 2 2



# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_axis 0 --num_frames 132 --num_prec_layers 1 --quant_bit_enh 4 4 4 \
#     --qmode mdlns \
#     --mdlns_sweep_start 0.25 --mdlns_sweep_end 10.0 --mdlns_sweep_step 0.25 --mdlns_auto_scale 0 \
#     --mdlns_first_base 2 --mdlns_second_base 1000 --quant_bit 8  --mdlns_second_base_exp_num_bits 3 2 2 2


# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_axis 0 --num_frames 132 --num_prec_layers 1 --quant_bit_enh 4 4 4 \
#     --qmode mdlns \
#     --mdlns_sweep_start 0.25 --mdlns_sweep_end 10.0 --mdlns_sweep_step 0.25 --mdlns_auto_scale 0 \
#     --mdlns_first_base 2 --mdlns_second_base 1000 --quant_bit 8  --mdlns_second_base_exp_num_bits 4 2 2 2



# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_axis 0 --num_frames 132 --num_prec_layers 1 --quant_bit_enh 4 4 4 \
#     --qmode mdlns \
#     --mdlns_sweep_start 0.25 --mdlns_sweep_end 10.0 --mdlns_sweep_step 0.25 --mdlns_auto_scale 0 \
#     --mdlns_first_base 2 --mdlns_second_base 1000 --quant_bit 9  --mdlns_second_base_exp_num_bits 4 2 2 2


# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_axis 0 --num_frames 132 --num_prec_layers 1 --quant_bit_enh 4 4 4 \
#     --qmode mdlns \
#     --mdlns_sweep_start 0.25 --mdlns_sweep_end 10.0 --mdlns_sweep_step 0.25 --mdlns_auto_scale 0 \
#     --mdlns_first_base 2 --mdlns_second_base 1000 --quant_bit 10  --mdlns_second_base_exp_num_bits 4 2 2 2


# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_axis 0 --num_frames 132 --num_prec_layers 1 --quant_bit_enh 4 4 4 \
#     --qmode mdlns \
#     --mdlns_sweep_start 0.25 --mdlns_sweep_end 10.0 --mdlns_sweep_step 0.25 --mdlns_auto_scale 0 \
#     --mdlns_first_base 2 --mdlns_second_base 1000 --quant_bit 10  --mdlns_second_base_exp_num_bits 5 2 2 2



# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_axis 0 --num_frames 132 --num_prec_layers 1 --quant_bit_enh 4 4 4 \
#     --qmode mdlns \
#     --mdlns_sweep_start 0.25 --mdlns_sweep_end 10.0 --mdlns_sweep_step 0.25 --mdlns_auto_scale 0 \
#     --mdlns_first_base 2 --mdlns_second_base 1000 --quant_bit 11  --mdlns_second_base_exp_num_bits 5 2 2 2 \
#     --dump_images --run_id _5



# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_axis 0 --num_frames 132 --num_prec_layers 1 --quant_bit_enh 4 4 4 \
#     --qmode mdlns \
#     --mdlns_sweep_start 0.25 --mdlns_sweep_end 10.0 --mdlns_sweep_step 0.25 --mdlns_auto_scale 0 \
#     --mdlns_first_base 2 --mdlns_second_base 1000 --quant_bit 12  --mdlns_second_base_exp_num_bits 5 2 2 2


# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_axis 0 --num_frames 132 --num_prec_layers 1 --quant_bit_enh 4 4 4 \
#     --qmode mdlns \
#     --mdlns_sweep_start 0.25 --mdlns_sweep_end 10.0 --mdlns_sweep_step 0.25 --mdlns_auto_scale 0 \
#     --mdlns_first_base 2 --mdlns_second_base 1000 --quant_bit 12  --mdlns_second_base_exp_num_bits 6 2 2 2



# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_axis 0 --num_frames 132 --num_prec_layers 1 --quant_bit_enh 4 4 4 \
#     --qmode mdlns \
#     --mdlns_sweep_start 0.25 --mdlns_sweep_end 10.0 --mdlns_sweep_step 0.25 --mdlns_auto_scale 0 \
#     --mdlns_first_base 2 --mdlns_second_base 1000 --quant_bit 13  --mdlns_second_base_exp_num_bits 6 2 2 2



# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_axis 0 --num_frames 132 --num_prec_layers 1 --quant_bit_enh 4 4 4 \
#     --qmode mdlns \
#     --mdlns_sweep_start 0.25 --mdlns_sweep_end 10.0 --mdlns_sweep_step 0.25 --mdlns_auto_scale 0 \
#     --mdlns_first_base 2 --mdlns_second_base 1000 --quant_bit 14  --mdlns_second_base_exp_num_bits 6 2 2 2


#####################################
# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_axis 0 --num_frames 132 --num_prec_layers 1 --quant_bit_enh 4 4 4 \
#     --qmode mdlns \
#     --mdlns_sweep_start 0.25 --mdlns_sweep_end 10.0 --mdlns_sweep_step 0.25 --mdlns_auto_scale 0 \
#     --mdlns_first_base 2 --mdlns_second_base 1000 --quant_bit 14  --mdlns_second_base_exp_num_bits 7 2 2 2


# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_axis 0 --num_frames 132 --num_prec_layers 1 --quant_bit_enh 4 4 4 \
#     --qmode mdlns \
#     --mdlns_sweep_start 0.25 --mdlns_sweep_end 10.0 --mdlns_sweep_step 0.25 --mdlns_auto_scale 0 \
#     --mdlns_first_base 2 --mdlns_second_base 1000 --quant_bit 15  --mdlns_second_base_exp_num_bits 7 2 2 2



# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_axis 0 --num_frames 132 --num_prec_layers 1 --quant_bit_enh 4 4 4 \
#     --qmode mdlns \
#     --mdlns_sweep_start 0.25 --mdlns_sweep_end 10.0 --mdlns_sweep_step 0.25 --mdlns_auto_scale 0 \
#     --mdlns_first_base 2 --mdlns_second_base 1000 --quant_bit 16  --mdlns_second_base_exp_num_bits 7 2 2 2 \
#     --dump_images --run_id _6


# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_axis 0 --num_frames 132 --num_prec_layers 1 --quant_bit_enh 4 4 4 \
#     --qmode mdlns \
#     --mdlns_sweep_start 0.25 --mdlns_sweep_end 10.0 --mdlns_sweep_step 0.25 --mdlns_auto_scale 0 \
#     --mdlns_first_base 2 --mdlns_second_base 1000 --quant_bit 16  --mdlns_second_base_exp_num_bits 8 2 2 2
    #####################################


################################################################


echo "Done"

