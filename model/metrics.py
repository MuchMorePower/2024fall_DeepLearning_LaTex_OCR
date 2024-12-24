import numpy as np
import distance
import json
import os
from nltk.translate.bleu_score import sentence_bleu

def evaluate(losses,  top1accs, top5accs, references, hypotheses, epoch):
    # 用于在验证集上计算各种评价指标指导模型早停
    # Calculate scores
    bleu4 = 0.0
    json_file='evaluation_results.json'
    for i, j in zip(references, hypotheses):
        bleu4 += max(sentence_bleu([i], j), 0.01)
    bleu4 = bleu4 / len(references)
    Exact_Match = exact_match_score(references, hypotheses)
    Edit_Distance = edit_distance(references, hypotheses)
    Score = bleu4 + Exact_Match + Edit_Distance / 10

    print(
        '\n * LOSS:{loss.avg:.3f},TOP-1 ACCRACY:{top1.avg:.3f} TOP-5 ACCURACY:{top5.avg:.3f},BLEU-4:{bleu:.3f},Exact Match:{Exact_Match:.1f},Edit Distance:{Edit_Distance:.3f},Score:{Score:.6f}'.format(
            loss=losses,
            top1=top1accs,
            top5=top5accs,
            bleu=bleu4,
            Exact_Match=Exact_Match,
            Edit_Distance=Edit_Distance,
            Score=Score))

    # 创建一个字典来存储当前 epoch 的数据
    epoch_data = {
        'epoch': epoch,
        'loss': losses.avg,
        'top1_accuracy': top1accs.avg,
        'top5_accuracy': top5accs.avg,
        'bleu4': bleu4,
        'exact_match': Exact_Match,
        'edit_distance': Edit_Distance,
        'score': Score
    }

    # 检查文件是否存在，如果存在则读取现有数据，添加新数据
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
    else:
        data = []

    # 将当前 epoch 数据添加到列表中
    data.append(epoch_data)

    # 将更新后的数据写回 JSON 文件
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4)

    return Score



def exact_match_score(references, hypotheses):
    """Computes exact match scores.

    Args:
        references: list of list of tokens (one ref)
        hypotheses: list of list of tokens (one hypothesis)

    Returns:
        exact_match: (float) 1 is perfect

    """
    exact_match = 0
    for ref, hypo in zip(references, hypotheses):
        if np.array_equal(ref, hypo):
            exact_match += 1

    return exact_match / float(max(len(hypotheses), 1))

def edit_distance(references, hypotheses):
    """Computes Levenshtein distance between two sequences.

    Args:
        references: list of list of token (one hypothesis)
        hypotheses: list of list of token (one hypothesis)

    Returns:
        1 - levenshtein distance: (higher is better, 1 is perfect)

    """
    d_leven, len_tot = 0, 0
    for ref, hypo in zip(references, hypotheses):
        d_leven += distance.levenshtein(ref, hypo)
        len_tot += float(max(len(ref), len(hypo)))

    return 1. - d_leven / len_tot