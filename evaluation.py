import os
import os.path
import json
import numpy as np

rel2id = json.load(open('meta/rel2id.json', 'r'))
id2rel = {value: key for key, value in rel2id.items()}


def to_official(preds, features, title2mask=None, title2len=None, title2ent=None):
    h_idx, t_idx, title = [], [], []

    for f in features:
        hts = f["hts"]
        h_idx += [ht[0] for ht in hts]
        t_idx += [ht[1] for ht in hts]
        title += [f["title"] for ht in hts]

    res = []
    for i in range(preds.shape[0]):
        pred = preds[i]
        pred = np.nonzero(pred)[0].tolist()
        for p in pred:
            if p != 0:
                if title2mask is not None:
                    res.append(
                        {
                            'title': title[i],
                            'h_idx': h_idx[i],
                            't_idx': t_idx[i],
                            'r': id2rel[p],
                            'is_intra': title2mask[title[i]][h_idx[i]][t_idx[i]],
                            'len': title2len[title[i]],
                            'ent': title2ent[title[i]],
                        }
                    )
                else:
                    res.append(
                        {
                            'title': title[i],
                            'h_idx': h_idx[i],
                            't_idx': t_idx[i],
                            'r': id2rel[p],
                        }
                    )
    return res


def gen_train_facts(data_file_name, truth_dir):
    fact_file_name = data_file_name[data_file_name.find("train_"):]
    fact_file_name = os.path.join(truth_dir, fact_file_name.replace(".json", ".fact"))

    if os.path.exists(fact_file_name):
        fact_in_train = set([])
        triples = json.load(open(fact_file_name))
        for x in triples:
            fact_in_train.add(tuple(x))
        return fact_in_train

    fact_in_train = set([])
    ori_data = json.load(open(data_file_name))
    for data in ori_data:
        vertexSet = data['vertexSet']
        for label in data['labels']:
            rel = label['r']
            for n1 in vertexSet[label['h']]:
                for n2 in vertexSet[label['t']]:
                    fact_in_train.add((n1['name'], n2['name'], rel))

    json.dump(list(fact_in_train), open(fact_file_name, "w"))

    return fact_in_train


def official_evaluate(tmp, path, title2mask, title2len, title2ent, tag="dev"):
    '''
        Adapted from the official evaluation code
    '''
    truth_dir = os.path.join(path, 'ref')

    if not os.path.exists(truth_dir):
        os.makedirs(truth_dir)

    fact_in_train_annotated = gen_train_facts(os.path.join(path, "train_annotated.json"), truth_dir)
    fact_in_train_distant = gen_train_facts(os.path.join(path, "train_distant.json"), truth_dir)

    truth = json.load(open(os.path.join(path, tag + ".json")))

    std = {}
    tot_evidences = 0
    tot_intra_relations = 0
    tot_inter_relations = 0
    tot_in_len_bucket = np.zeros(5)
    tot_in_ent_bucket = np.zeros(7)
    titleset = set([])

    title2vectexSet = {}

    for x in truth:
        title = x['title']
        titleset.add(title)

        vertexSet = x['vertexSet']
        title2vectexSet[title] = vertexSet

        for label in x['labels']:
            r = label['r']
            h_idx = label['h']
            t_idx = label['t']
            std[(title, r, h_idx, t_idx)] = set(label['evidence'])
            tot_evidences += len(label['evidence'])
            if title2mask[title][h_idx][t_idx] == 1:
                tot_intra_relations += 1
            else:
                tot_inter_relations += 1
            tot_in_len_bucket[title2len[title]] += 1
            tot_in_ent_bucket[title2ent[title]] += 1

    tot_relations = len(std)
    tmp.sort(key=lambda x: (x['title'], x['h_idx'], x['t_idx'], x['r']))
    submission_answer = [tmp[0]]
    for i in range(1, len(tmp)):
        x = tmp[i]
        y = tmp[i - 1]
        if (x['title'], x['h_idx'], x['t_idx'], x['r']) != (y['title'], y['h_idx'], y['t_idx'], y['r']):
            submission_answer.append(tmp[i])

    correct_re = 0
    correct_evidence = 0
    pred_evi = 0

    correct_in_train_annotated = 0
    correct_in_train_distant = 0
    correct_in_intra = 0
    correct_in_inter = 0
    predicted_intra = 0
    predicted_inter = 0
    correct_in_len_bucket = np.zeros(5)
    correct_in_ent_bucket = np.zeros(7)
    predicted_in_len_bucket = np.zeros(5)
    predicted_in_ent_bucket = np.zeros(7)

    titleset2 = set([])
    for x in submission_answer:
        title = x['title']
        h_idx = x['h_idx']
        t_idx = x['t_idx']
        r = x['r']
        is_intra = x['is_intra']
        titleset2.add(title)
        if title not in title2vectexSet:
            continue
        vertexSet = title2vectexSet[title]

        if 'evidence' in x:
            evi = set(x['evidence'])
        else:
            evi = set([])
        pred_evi += len(evi)

        if is_intra == 1:
            predicted_intra += 1
        else:
            predicted_inter += 1
        predicted_in_len_bucket[x['len']] += 1
        predicted_in_ent_bucket[x['ent']] += 1

        if (title, r, h_idx, t_idx) in std:
            correct_re += 1
            stdevi = std[(title, r, h_idx, t_idx)]
            correct_evidence += len(stdevi & evi)
            in_train_annotated = in_train_distant = False
            for n1 in vertexSet[h_idx]:
                for n2 in vertexSet[t_idx]:
                    if (n1['name'], n2['name'], r) in fact_in_train_annotated:
                        in_train_annotated = True
                    if (n1['name'], n2['name'], r) in fact_in_train_distant:
                        in_train_distant = True

            if in_train_annotated:
                correct_in_train_annotated += 1
            if in_train_distant:
                correct_in_train_distant += 1
            if is_intra == 1:
                correct_in_intra += 1
            else:
                correct_in_inter += 1
            correct_in_len_bucket[x['len']] += 1
            correct_in_ent_bucket[x['ent']] += 1

    print(correct_re, len(submission_answer), tot_relations)
    re_p = 1.0 * correct_re / len(submission_answer)
    re_r = 1.0 * correct_re / tot_relations
    if re_p + re_r == 0:
        re_f1 = 0
    else:
        re_f1 = 2.0 * re_p * re_r / (re_p + re_r)
    

    evi_p = 1.0 * correct_evidence / pred_evi if pred_evi > 0 else 0
    evi_r = 1.0 * correct_evidence / tot_evidences
    if evi_p + evi_r == 0:
        evi_f1 = 0
    else:
        evi_f1 = 2.0 * evi_p * evi_r / (evi_p + evi_r)

    re_p_ignore_train_annotated = 1.0 * (correct_re - correct_in_train_annotated) / (len(submission_answer) - correct_in_train_annotated + 1e-5)
    re_p_ignore_train = 1.0 * (correct_re - correct_in_train_distant) / (len(submission_answer) - correct_in_train_distant + 1e-5)

    if re_p_ignore_train_annotated + re_r == 0:
        re_f1_ignore_train_annotated = 0
    else:
        re_f1_ignore_train_annotated = 2.0 * re_p_ignore_train_annotated * re_r / (re_p_ignore_train_annotated + re_r)

    if re_p_ignore_train + re_r == 0:
        re_f1_ignore_train = 0
    else:
        re_f1_ignore_train = 2.0 * re_p_ignore_train * re_r / (re_p_ignore_train + re_r)

    re_intra_p = correct_in_intra / predicted_intra if predicted_intra > 0 else 0
    re_intra_r = correct_in_intra / tot_intra_relations
    if re_intra_p + re_intra_r == 0:
        intra_f1 = 0
    else:
        intra_f1 = 2.0 * re_intra_p * re_intra_r / (re_intra_p + re_intra_r)

    re_inter_p = correct_in_inter / predicted_inter if predicted_inter > 0 else 0
    re_inter_r = correct_in_inter / tot_inter_relations
    if re_inter_p + re_inter_r == 0:
        inter_f1 = 0
    else:
        inter_f1 = 2.0 * re_inter_p * re_inter_r / (re_inter_p + re_inter_r)
    print(correct_in_intra, predicted_intra, tot_intra_relations)
    print(correct_in_inter, predicted_inter, tot_inter_relations)
    print("intra_f1: ", intra_f1 * 100, "inter_f1: ", inter_f1 * 100)

    print(re_p, re_r)
    p_len = correct_in_len_bucket / predicted_in_len_bucket
    r_len = correct_in_len_bucket / tot_in_len_bucket
    f1_len = 2.0 * p_len * r_len / (p_len + r_len + 1e-20)
    p_ent = correct_in_ent_bucket / predicted_in_ent_bucket
    r_ent = correct_in_ent_bucket / tot_in_ent_bucket
    f1_ent = 2.0 * p_ent * r_ent / (p_ent + r_ent + 1e-20)
    print(f1_len, f1_ent)
    return re_f1, evi_f1, re_f1_ignore_train_annotated, re_f1_ignore_train
