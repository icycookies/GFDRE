from tqdm import tqdm
import ujson as json
from graph import build_graph
import numpy as np

docred_rel2id = json.load(open('meta/rel2id.json', 'r'))
docred_rel2name = json.load(open('meta/rel_info.json', 'r'))
cdr_rel2id = {'1:NR:2': 0, '1:CID:2': 1}
gda_rel2id = {'1:NR:2': 0, '1:GDA:2': 1}

unused_tokens_bert = ['[unused41]', '[unused42]', '[unused43]', '[unused44]', '[unused45]', '[unused46]', '[unused47]', '[unused48]', '[unused49]', '[unused50]', '[unused51]', '[unused52]', '[unused53]', '[unused54]', '[unused55]', '[unused56]', '[unused57]', '[unused58]', '[unused59]', '[unused60]', '[unused61]', '[unused62]', '[unused63]', '[unused64]', '[unused65]', '[unused66]', '[unused67]', '[unused68]', '[unused69]', '[unused70]', '[unused71]', '[unused72]', '[unused73]', '[unused74]', '[unused75]', '[unused76]', '[unused77]', '[unused78]', '[unused79]', '[unused80]', '[unused81]', '[unused82]', '[unused83]', '[unused84]']
unused_tokens_roberta = ["Subscribe", "alsh", "repl", "Ġselector", "ĠLength", "Ġtemporal", "Tele", "ocalyptic", "ĠDeaths", "rl", "Target", "ĠOrn", "ongh", "Ġ1909", "Quest", "Place", "ĠDisabled", "Ġascending", "giene", "ĠMSI", "ivil", "Ġcaval", "Ġintermitt", "Ġsalts", "Apr", "059", "ĠKeeper", "emis", "ĠEternal", "SER", "estones", "Ġrudimentary", "Ġpooled", "ĠAlright", "Ġdiagrams", "ydia", "Jacob", "Ġarchitectures", "ĠUSPS", "Ġfootnote", "ĠBrav", "ĠLeopard", "Ġvirtuous", "ploma", "ĠHIP", "Ġhorizontally", "olith", "Prop", "ĠApocalypse", "Syria", "ĠShowdown", "constitutional", "Independent", "ĠMiliband", "ĠTracks", "adle", "ĠESL", "ĠFIGHT", "Ġjohn", "é", "benef", "eware", "ĠTABLE", "ĠVeg", "ainers", "Ġresolves", "Warren"]

def chunks(l, n):
    res = []
    for i in range(0, len(l), n):
        assert len(l[i:i + n]) == n
        res += [l[i:i + n]]
    return res

def compute_intra_mask(seq_length, max_ent_cnt, tok_to_sent, tok_to_ent):
    intra_mask = np.zeros((max_ent_cnt, max_ent_cnt))
    pos = []
    for i in range(seq_length):
        if tok_to_ent[i] != -1:
            pos.append(i)
    for i in pos:
        for j in pos:
            if tok_to_sent[i] == tok_to_sent[j]:
                intra_mask[tok_to_ent[i]][tok_to_ent[j]] = 1
    return intra_mask


def read_docred(file_in, tokenizer, max_seq_length=1024, bert_type="bert"):
    i_line = 0
    pos_samples = 0
    neg_samples = 0
    
    features = []
    if bert_type == "bert":
        unused_tokens = unused_tokens_bert
    elif bert_type == "roberta":
        unused_tokens = unused_tokens_roberta
    if file_in == "":
        return None
    with open(file_in, "r") as fh:
        data = json.load(fh)

    all_len = 0
    all_sents = 0
    all_men = 0
    num_edges = np.zeros(5)

    for sample in tqdm(data, desc="Example"):
        sents = []
        tok2sent = []
        sent_map = []

        entities = sample['vertexSet']
        entity_start, entity_end = {}, {}
        for i, entity in enumerate(entities):
            # sents.append(unused_tokens[i])
            # tok2sent.append(-1)
            for mention in entity:
                sent_id = mention["sent_id"]
                pos = mention["pos"]
                entity_start[(sent_id, pos[0],)] = i
                entity_end[(sent_id, pos[1] - 1,)] = i
        
        for i_s, sent in enumerate(sample['sents']):
            new_map = {}
            sents.append(unused_tokens[-2])
            tok2sent.append(i_s)
            all_sents += 1
            for i_t, token in enumerate(sent):
                tokens_wordpiece = tokenizer.tokenize(token)
                all_len += len(tokens_wordpiece)
                if (i_s, i_t) in entity_start:
                    tokens_wordpiece = [unused_tokens[entity_start[(i_s, i_t)]]] + tokens_wordpiece
                if (i_s, i_t) in entity_end:
                    tokens_wordpiece = tokens_wordpiece + [unused_tokens[entity_end[(i_s, i_t)]]]
                new_map[i_t] = len(sents)
                sents.extend(tokens_wordpiece)
                tok2sent.extend([i_s] * len(tokens_wordpiece))
            new_map[i_t + 1] = len(sents)
            sent_map.append(new_map)

        train_triple = {}
        if "labels" in sample:
            for label in sample['labels']:
                evidence = label['evidence']
                r = int(docred_rel2id[label['r']])
                if (label['h'], label['t']) not in train_triple:
                    train_triple[(label['h'], label['t'])] = [
                        {'relation': r, 'evidence': evidence}
                    ]
                else:
                    train_triple[(label['h'], label['t'])].append(
                        {'relation': r, 'evidence': evidence}
                    )

        entity_pos = []
        tok2ent = [-1] * len(sents)
        for i, e in enumerate(entities):
            # entity_pos.append([(i, i + 1)])
            entity_pos.append([])
            for m in e:
                start = sent_map[m["sent_id"]][m["pos"][0]]
                end = sent_map[m["sent_id"]][m["pos"][1]]
                entity_pos[-1].append((start, end,))
                for j in range(start, end):
                    tok2ent[j] = i
                all_men += 1

        relations, hts = [], []
        for h, t in train_triple.keys():
            relation = [0] * len(docred_rel2id)
            for mention in train_triple[h, t]:
                relation[mention["relation"]] = 1
                evidence = mention["evidence"]
            relations.append(relation)
            hts.append([h, t])
            pos_samples += 1

        for h in range(len(entities)):
            for t in range(len(entities)):
                if h != t and [h, t] not in hts:
                    relation = [1] + [0] * (len(docred_rel2id) - 1)
                    relations.append(relation)
                    hts.append([h, t])
                    neg_samples += 1

        assert len(relations) == len(entities) * (len(entities) - 1)

        sents = sents[:max_seq_length - 2]
        tok_type = [0] * len(sents)
        for i in range(len(sents)):
            if sents[i] in unused_tokens[:42]:
                tok_type[i] = 2
            elif sents[i] == unused_tokens[-2]:
                tok_type[i] = 1
        input_ids = tokenizer.convert_tokens_to_ids(sents)
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)
        tok2sent = [-1] + tok2sent[:max_seq_length - 2] + [-1]
        tok2ent = [-1] + tok2ent[:max_seq_length - 2] + [-1]
        tok_type = [-1] + tok_type + [-1]
        
        # print(len(input_ids), len(tok2sent), len(tok2ent), len(tok_type))
        graph = build_graph(len(entities), tok2sent, tok2ent, tok_type)
        try:
            num_edges += graph.num_edges_each_type()
        except Exception:
            num_edges += graph[0].num_edges_each_type() + graph[1].num_edges_each_type()
        intra_mask = compute_intra_mask(len(input_ids), len(entities), tok2sent, tok2ent)
        feature = {'input_ids': input_ids,
                   'entity_pos': entity_pos,
                   'labels': relations,
                   'hts': hts,
                   'graph': graph,
                   'title': sample['title'],
                   'intra_mask': intra_mask,
                   }
        features.append(feature)

        if i_line == 0 or sample['title'] == 'Harbour Esplanade, Docklands':
            print("num_ents: {}".format(len(entities)))
            print("sents: {}".format(" ".join(sents)))
            print("entity_pos for 1st entity: {}".format(str(entity_pos[0])))
            print("hts: {}".format(" ".join([str(x[0]) + '-' + str(x[1]) for x in hts])))
            print("labels for 1st hts: {}".format(str(relations[0])))
            print("num_edges: {}".format(graph.num_edges))
            for i in range(5):
                graph.print_example_edges(i)
        i_line += 1

    print("# of documents {}.".format(i_line))
    print("# of positive examples {}.".format(pos_samples))
    print("# of negative examples {}.".format(neg_samples))
    print("average document length", all_len / i_line)
    print("average sent", all_sents / i_line)
    print("average mention / sent", all_men / all_sents)
    print("average edge count", num_edges / i_line)
    return features

def gen_docred(file_in, preds):
    with open(file_in, "r") as fh:
        data = json.load(fh)

    results = {}
    for sample in data:
        result = {}
        doc = []
        tok2ent = []
        ents = []
        pred = []
        gt = []

        entities = sample['vertexSet']
        entity_start, entity_end = {}, {}
        for i, entity in enumerate(entities):
            ents.append(entity[0]["name"])
            for mention in entity:
                sent_id = mention["sent_id"]
                pos = mention["pos"]
                entity_start[(sent_id, pos[0],)] = i
                entity_end[(sent_id, pos[1] - 1,)] = i
        
        ent_id = -1
        for i_s, sent in enumerate(sample['sents']):
            for i_t, token in enumerate(sent):
                if (i_s, i_t) in entity_start:
                    ent_id = entity_start[(i_s, i_t)]
                doc.append(token)
                tok2ent.append(ent_id)
                if (i_s, i_t) in entity_end:
                    ent_id = -1
        for label in sample['labels']:
            gt.append({
                'h': label['h'],
                't': label['t'],
                'r': docred_rel2name[label['r']],
            })
        result["doc"] = " ".join(doc)
        result["tok2ent"] = tok2ent
        result["ents"] = ents
        result["gt"] = gt
        result["pred"] = []
        results[sample['title']] = result
    for pred in preds:
        if pred['title'] in results:
            results[pred["title"]]["pred"].append({
                'h': pred['h_idx'],
                't': pred['t_idx'],
                'r': docred_rel2name[pred['r']],
            })
    return results

def read_cdr(file_in, tokenizer, max_seq_length=1024):
    pmids = set()
    features = []
    maxlen = 0
    with open(file_in, 'r') as infile:
        lines = infile.readlines()
        for i_l, line in enumerate(tqdm(lines)):
            line = line.rstrip().split('\t')
            pmid = line[0]

            if pmid not in pmids:
                pmids.add(pmid)
                text = line[1]
                prs = chunks(line[2:], 17)

                ent2idx = {}
                train_triples = {}

                entity_pos = set()
                for p in prs:
                    es = list(map(int, p[8].split(':')))
                    ed = list(map(int, p[9].split(':')))
                    tpy = p[7]
                    for start, end in zip(es, ed):
                        entity_pos.add((start, end, tpy))

                    es = list(map(int, p[14].split(':')))
                    ed = list(map(int, p[15].split(':')))
                    tpy = p[13]
                    for start, end in zip(es, ed):
                        entity_pos.add((start, end, tpy))

                sents = [t.split(' ') for t in text.split('|')]
                new_sents = []
                sent_map = {}
                i_t = 0
                for sent in sents:
                    for token in sent:
                        tokens_wordpiece = tokenizer.tokenize(token)
                        for start, end, tpy in list(entity_pos):
                            if i_t == start:
                                tokens_wordpiece = ["*"] + tokens_wordpiece
                            if i_t + 1 == end:
                                tokens_wordpiece = tokens_wordpiece + ["*"]
                        sent_map[i_t] = len(new_sents)
                        new_sents.extend(tokens_wordpiece)
                        i_t += 1
                    sent_map[i_t] = len(new_sents)
                sents = new_sents

                entity_pos = []

                for p in prs:
                    if p[0] == "not_include":
                        continue
                    if p[1] == "L2R":
                        h_id, t_id = p[5], p[11]
                        h_start, t_start = p[8], p[14]
                        h_end, t_end = p[9], p[15]
                    else:
                        t_id, h_id = p[5], p[11]
                        t_start, h_start = p[8], p[14]
                        t_end, h_end = p[9], p[15]
                    h_start = map(int, h_start.split(':'))
                    h_end = map(int, h_end.split(':'))
                    t_start = map(int, t_start.split(':'))
                    t_end = map(int, t_end.split(':'))
                    h_start = [sent_map[idx] for idx in h_start]
                    h_end = [sent_map[idx] for idx in h_end]
                    t_start = [sent_map[idx] for idx in t_start]
                    t_end = [sent_map[idx] for idx in t_end]
                    if h_id not in ent2idx:
                        ent2idx[h_id] = len(ent2idx)
                        entity_pos.append(list(zip(h_start, h_end)))
                    if t_id not in ent2idx:
                        ent2idx[t_id] = len(ent2idx)
                        entity_pos.append(list(zip(t_start, t_end)))
                    h_id, t_id = ent2idx[h_id], ent2idx[t_id]

                    r = cdr_rel2id[p[0]]
                    if (h_id, t_id) not in train_triples:
                        train_triples[(h_id, t_id)] = [{'relation': r}]
                    else:
                        train_triples[(h_id, t_id)].append({'relation': r})

                relations, hts = [], []
                for h, t in train_triples.keys():
                    relation = [0] * len(cdr_rel2id)
                    for mention in train_triples[h, t]:
                        relation[mention["relation"]] = 1
                    relations.append(relation)
                    hts.append([h, t])

            maxlen = max(maxlen, len(sents))
            sents = sents[:max_seq_length - 2]
            input_ids = tokenizer.convert_tokens_to_ids(sents)
            input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

            if len(hts) > 0:
                feature = {'input_ids': input_ids,
                           'entity_pos': entity_pos,
                           'labels': relations,
                           'hts': hts,
                           'title': pmid,
                           }
                features.append(feature)
    print("Number of documents: {}.".format(len(features)))
    print("Max document length: {}.".format(maxlen))
    return features


def read_gda(file_in, tokenizer, max_seq_length=1024):
    pmids = set()
    features = []
    maxlen = 0
    with open(file_in, 'r') as infile:
        lines = infile.readlines()
        for i_l, line in enumerate(tqdm(lines)):
            line = line.rstrip().split('\t')
            pmid = line[0]

            if pmid not in pmids:
                pmids.add(pmid)
                text = line[1]
                prs = chunks(line[2:], 17)

                ent2idx = {}
                train_triples = {}

                entity_pos = set()
                for p in prs:
                    es = list(map(int, p[8].split(':')))
                    ed = list(map(int, p[9].split(':')))
                    tpy = p[7]
                    for start, end in zip(es, ed):
                        entity_pos.add((start, end, tpy))

                    es = list(map(int, p[14].split(':')))
                    ed = list(map(int, p[15].split(':')))
                    tpy = p[13]
                    for start, end in zip(es, ed):
                        entity_pos.add((start, end, tpy))

                sents = [t.split(' ') for t in text.split('|')]
                new_sents = []
                sent_map = {}
                i_t = 0
                for sent in sents:
                    for token in sent:
                        tokens_wordpiece = tokenizer.tokenize(token)
                        for start, end, tpy in list(entity_pos):
                            if i_t == start:
                                tokens_wordpiece = ["*"] + tokens_wordpiece
                            if i_t + 1 == end:
                                tokens_wordpiece = tokens_wordpiece + ["*"]
                        sent_map[i_t] = len(new_sents)
                        new_sents.extend(tokens_wordpiece)
                        i_t += 1
                    sent_map[i_t] = len(new_sents)
                sents = new_sents

                entity_pos = []

                for p in prs:
                    if p[0] == "not_include":
                        continue
                    if p[1] == "L2R":
                        h_id, t_id = p[5], p[11]
                        h_start, t_start = p[8], p[14]
                        h_end, t_end = p[9], p[15]
                    else:
                        t_id, h_id = p[5], p[11]
                        t_start, h_start = p[8], p[14]
                        t_end, h_end = p[9], p[15]
                    h_start = map(int, h_start.split(':'))
                    h_end = map(int, h_end.split(':'))
                    t_start = map(int, t_start.split(':'))
                    t_end = map(int, t_end.split(':'))
                    h_start = [sent_map[idx] for idx in h_start]
                    h_end = [sent_map[idx] for idx in h_end]
                    t_start = [sent_map[idx] for idx in t_start]
                    t_end = [sent_map[idx] for idx in t_end]
                    if h_id not in ent2idx:
                        ent2idx[h_id] = len(ent2idx)
                        entity_pos.append(list(zip(h_start, h_end)))
                    if t_id not in ent2idx:
                        ent2idx[t_id] = len(ent2idx)
                        entity_pos.append(list(zip(t_start, t_end)))
                    h_id, t_id = ent2idx[h_id], ent2idx[t_id]

                    r = gda_rel2id[p[0]]
                    if (h_id, t_id) not in train_triples:
                        train_triples[(h_id, t_id)] = [{'relation': r}]
                    else:
                        train_triples[(h_id, t_id)].append({'relation': r})

                relations, hts = [], []
                for h, t in train_triples.keys():
                    relation = [0] * len(gda_rel2id)
                    for mention in train_triples[h, t]:
                        relation[mention["relation"]] = 1
                    relations.append(relation)
                    hts.append([h, t])

            maxlen = max(maxlen, len(sents))
            sents = sents[:max_seq_length - 2]
            input_ids = tokenizer.convert_tokens_to_ids(sents)
            input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

            if len(hts) > 0:
                feature = {'input_ids': input_ids,
                           'entity_pos': entity_pos,
                           'labels': relations,
                           'hts': hts,
                           'title': pmid,
                           }
                features.append(feature)
    print("Number of documents: {}.".format(len(features)))
    print("Max document length: {}.".format(maxlen))
    return features
