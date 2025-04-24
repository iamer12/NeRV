
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


# 4 bits
# python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
#     --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
#     --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
#     -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_bit 4 --quant_axis 0


#Evaluate mdlns scalable quantization
python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
    --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
    --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
    -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
    --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_bit 5 --quant_axis 0 --num_frames 2 --num_prec_layers 2 --quant_bit_enh 5 5 5 \
    --qmode mdlns --mdlns_second_base 1000 --mdlns_second_base_exp_num_bits 2 2 2 2\
    --mdlns_sweep_start 0.1 --mdlns_sweep_end 5.0 --mdlns_sweep_step 0.1

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
#     --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_bit 5 --num_frames 2 --num_prec_layers 1 --quant_bit_enh 5 5 5




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


echo "Done"

