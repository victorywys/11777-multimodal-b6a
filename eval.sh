python eval.py --dump_images 0 --num_images 5000 --split testA \
    --model log/log_$1/model-best.pth --infos_path log/log_$1/infos_$1-best.pkl --language_eval 1 \
    --input_box_dir data/cocotalk_box --input_fc_dir data/cocobu_ref_fc --input_att_dir data/cocobu_ref_att \
    --input_label_h5 data/refcoco_label.h5 --input_json data/refcoco.json --beam_size 1
