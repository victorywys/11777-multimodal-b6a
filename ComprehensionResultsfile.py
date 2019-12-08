import numpy as np 
import ast
import pdb
def load_data(json_file):
    file1 = open(json_file, 'r')
    lines = file1.readlines()
    file1.close()
    print('Total Data: %s'%str(len(lines)))
    dict_lst = []
    for line in lines:
        dict_lst.append(ast.literal_eval(line))
    return dict_lst


def get_comprehension_result(dict_lst):
    cnt = 0
    acc = 0
    for i in range(len(dict_lst)):
        image_id = dict_lst[i]['image_id']
        image_ann_ids = dict_lst[i]['img_ann_ids']
        sent_ids = dict_lst[i]['sent_ids']
        gd_idx = dict_lst[i]['gd_ixs']
        
        for j, sent_ids in enumerate(sent_ids):
            cnt+=1
            gd_ix = gd_idx[j]
            #cossine_similarity should be elementwise multiplication of text and image features
            _, pos_sc, neg_sc = compute_margin_loss(cossine_similarity, gd_ix, 0)
           if(pos_sc>neg_sc):
               acc += 1
    return acc/cnt
    
def compute_margin_loss(scores, gd_ix, margin):
	scores = scores.copy()
	pos_sc = scores[gd_ix].copy()
	scores[gd_ix] = -1e5
	max_neg_sc = scores.max()
	loss = max([0, margin + max_neg_sc - pos_sc])
	return loss, pos_sc, max_neg_sc
            

if __name__=='__main__':
    f = load_data('./refcoco_testA_data.json')
    