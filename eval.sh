export CUDA_VISIBLE_DEVICES=2

python2 eval.py --dump_images 0 --num_images 5000 --split val \
    --model log/log_refcocog/model-best.pth --infos_path log/log_refcocog/infos_refcocog-best.pkl --language_eval 1 \
    --input_box_dir data/cocotalk_box --input_fc_dir data/cocobu_ref_fc --input_att_dir data/cocobu_ref_att \
    --input_label_h5 data/refcocog_label.h5 --input_json data/refcocog.json --beam_size 2
