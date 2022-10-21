from collections import Counter
import json
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment


INCLUDE_SINGLETONS = False


def muc(clusters, mention_to_gold):
    tp, p = 0, 0
    for c in clusters:
        p += len(c) - 1
        tp += len(c)
        linked = set()
        for m in c:
            m = tuple(m)
            if m in mention_to_gold:
                linked.add(mention_to_gold[m])
            else:
                tp -= 1
        tp -= len(linked)
    return tp, p


def b_cubed(clusters, mention_to_gold):
    num, dem = 0, 0

    for c in clusters:
        if len(c) == 1:
            continue
        gold_counts = Counter()
        correct = 0
        for m in c:
            m = tuple(m)
            if m in mention_to_gold:
                gold_counts[tuple(mention_to_gold[m])] += 1
        for c2, count in gold_counts.items():
            if len(c2) != 1:
                correct += count * count
        
        num += correct / float(len(c))
        dem += len(c)

    return num, dem

def lea(clusters, mention_to_gold):
    num, dem = 0, 0

    for c in clusters:
        if len(c) == 1:
            continue

        common_links = 0
        all_links = len(c) * (len(c) - 1) / 2.0
        for i, m in enumerate(c):
            m = tuple(m)
            if m in mention_to_gold:
                for m2 in c[i + 1:]:
                    m2 = tuple(m2)
                    if m2 in mention_to_gold and mention_to_gold[m] == mention_to_gold[m2]:
                        common_links += 1

        num += len(c) * common_links / float(all_links)
        dem += len(c)

    return num, dem







def phi4(c1, c2):
    return 2 * len([m for m in c1 if m in c2]) / float(len(c1) + len(c2))

def ceafe(clusters, gold_clusters):
    clusters = [c for c in clusters if len(c) != 1]
    scores = np.zeros((len(gold_clusters), len(clusters)))
    for i in range(len(gold_clusters)):
        for j in range(len(clusters)):
            scores[i, j] = phi4(gold_clusters[i], clusters[j])
    matching = linear_assignment(-scores)
    similarity = sum(scores[matching[:][0], matching[:][1]])
    return similarity, len(clusters), similarity, len(gold_clusters)



def calculate_recall_precision(predicted_file_path , actual_file_path , metric):
   
    predicted_file = open(predicted_file_path)
    predicted = [l for l in predicted_file.readlines()]
    predicted_file.close()
    annotated_file = open(actual_file_path)
    actual = [l for l in annotated_file.readlines()]
    annotated_file.close()
    list_recall , list_precision , list_f1_score = [] , [] , []
            
    
    for i in range(len(predicted)) :
        predicted_clusters , gold_clusters = json.loads(predicted[i])["clusters"] , json.loads(actual[i])["clusters"]
        clusters , gold = [tuple(tuple(m) for m in gc) for gc in predicted_clusters] , [tuple(tuple(m) for m in gc) for gc in gold_clusters]
        mention_to_predicted , mention_to_gold = {} , {}
        
        for gc in clusters:
            for mention in gc:
                mention_to_predicted[mention] = gc
            
        for gc in gold:
            for mention in gc:
                mention_to_gold[mention] = gc
        
        if metric == ceafe :
            
            precision_num , precision_den , recall_num , recall_den = metric(predicted_clusters , gold_clusters)
            
        else:
            
            recall_num , recall_den =  metric(gold_clusters, mention_to_predicted)
            precision_num , precision_den =  metric(predicted_clusters, mention_to_gold)
            
        recall = get_recall(recall_num , recall_den) * 100 
        precision = get_precision(precision_num , precision_den) * 100
        f1_score = f1(precision_num, precision_den, recall_num, recall_den, beta=1) * 100
        
        list_recall.append(recall)
        list_precision.append(precision)
        list_f1_score.append(f1_score)
        
       
    recall = sum(list_recall) / len(list_recall) 
    precision = sum(list_precision) / len(list_precision)
    f1_score = sum(list_f1_score) / len(list_f1_score)
    
    
    return recall , precision , f1_score , list_recall , list_precision , list_f1_score




def f1(p_num, p_den, r_num, r_den, beta=1):
    p = 0 if p_den == 0 else p_num / float(p_den)
    r = 0 if r_den == 0 else r_num / float(r_den)
    return 0 if p + r == 0 else (1 + beta * beta) * p * r / (beta * beta * p + r)

def get_recall(r_num , r_den):
    return 0 if r_num == 0 else r_num / float(r_den)

def get_precision(p_num , p_den):
    return 0 if p_num == 0 else p_num / float(p_den)