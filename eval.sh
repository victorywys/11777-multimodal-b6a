python2 eval.py --dump_images 0 --num_images 5000 --split testA \
    --model checkpoint/pretrain/model-best.pth --infos_path checkpoint/pretrain/infos_baseline_refcoco-best.pkl --language_eval 1 \
    --input_box_dir data/cocotalk_box  --input_fc_dir /usr0/home/yansenwa/mscoco_feature/boxes/bu_feature_gt/refcoco_unc/cocobu_ref_fc  --input_att_dir /usr0/home/yansenwa/mscoco_feature/boxes/bu_feature_gt/refcoco_unc/cocobu_ref_att  \
    --input_label_h5 data/refcoco_label.h5 --input_json data/refcoco.json --beam_size 1
