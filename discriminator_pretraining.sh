id=baseline_refcoco
ckpt_path="log/log_"$id
if [ ! -d $ckpt_path ]; then
  mkdir $ckpt_path
fi

scst_path="log/log_scst_"$id
if [ ! -d $scst_path ]; then
  mkdir $scst_path
fi

if [ ! -f $ckpt_path"/infos_"$id".pkl" ]; then
start_from=""
else
start_from="--start_from "$ckpt_path
fi

export CUDA_VISIBLE_DEVICES=3,4,5

python2 train.py --id $id --caption_model SLD --train_mode discriminator \
    --label_smoothing 0.0 \
    --input_json data/refcoco.json --input_label_h5 data/refcoco_label.h5 \
    --input_fc_dir /usr0/home/yansenwa/mscoco_feature/boxes/bu_feature_gt/refcoco_unc/cocobu_ref_fc --input_att_dir /usr0/home/yansenwa/mscoco_feature/boxes/bu_feature_gt/refcoco_unc/cocobu_ref_att --start_from checkpoint/pretrain --discriminator_start_from checkpoint/discriminator\
    --seq_per_img 3 --batch_size 25 --beam_size 1 --num_layers 6 --input_encoding_size 512 --rnn_size 2048 \
    --reduce_on_plateau --learning_rate_decay_start 0 --scheduled_sampling_start 0 --checkpoint_path checkpoint/discriminator \
    --save_checkpoint_every 2000 --language_eval 0 --val_images_use 5000 --max_epochs 15 --learning_rate 5e-4 | tee $ckpt_path/train.log 

#python train.py --id $id --caption_model transformer --reduce_on_plateau \
#    --input_json data/refcoco.json --input_label_h5 data/refcoco_label.h5 --cached_tokens refcoco-all-idxs \
#    --input_fc_dir data/cocobu_ref_fc \
#    --input_att_dir data/cocobu_ref_att \
#    --seq_per_img 3 --batch_size 10 --beam_size 1 \
#    --learning_rate 1e-5 --num_layers 6 --input_encoding_size 512 --rnn_size 2048 \
#    --checkpoint_path $scst_path $start_from \
#    --save_checkpoint_every 3000 --language_eval 1 --val_images_use 5000 --self_critical_after 10 | tee $scst_path/train.log
