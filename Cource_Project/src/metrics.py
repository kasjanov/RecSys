import numpy as np


def recall(recommended_list, bought_list):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)

    flags = np.isin(bought_list, recommended_list)

    recall = flags.sum() / len(bought_list)

    return recall


def recall_at_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)

    if k < len(recommended_list):
        recommended_list = recommended_list[:k]

    flags = np.isin(bought_list, recommended_list)
    recall = flags.sum() / len(bought_list)

    return recall


# Money Recall@k = (revenue of recommended items @k that are relevant) / (revenue of relevant items)
def money_recall_at_k(recommended_list, bought_list, prices_recommended, prices_bought, k):
    bought_list = np.array(bought_list)
    prices_bought = np.array(prices_bought)
    recommended_list = np.array(recommended_list[:k])
    prices_recommended = np.array(prices_recommended[:k])

    flags = np.isin(recommended_list, bought_list)

    r1 = np.dot(flags, prices_recommended)
    r2 = prices_bought.sum()
    recall = r1 / r2

    return recall


def precision(recommended_list, bought_list):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)

    flags = np.isin(bought_list, recommended_list)

    precision = flags.sum() / len(recommended_list)

    return precision


def precision_at_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)

    bought_list = bought_list  # Тут нет [:k] !!

    if k < len(recommended_list):
        recommended_list = recommended_list[:k]

    flags = np.isin(bought_list, recommended_list)

    precision = flags.sum() / len(recommended_list)

    return precision


# (revenue of recommended items @k that are relevant) / (revenue of recommended items @k)
def money_precision_at_k(recommended_list, bought_list, prices_recommended, k):
    # your_code
    # Лучше считать через скалярное произведение, а не цикл

    bought_list = bought_list  # Тут нет [:k] !!
    recommended_list = np.array(recommended_list[:k])
    prices_recommended = np.array(prices_recommended[:k])

    flags = np.isin(recommended_list, bought_list)  # вернет размерность recommended_list, т.е. k

    precision = np.dot(flags, prices_recommended) / prices_recommended.sum()

    return precision


# был ли хотя бы 1 релевантный товар среди топ-k рекомендованных
def hit_rate_at_k(recommended_list, bought_list, k):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list[:k])

    flags = np.isin(bought_list, recommended_list)

    hit_rate = (flags.sum() > 0) * 1

    return hit_rate


def reciprocal_rank(recommended_list, bought_list):
    flags = np.isin(recommended_list, bought_list)

    if sum(flags) == 0:
        return 0

    sum_ = 0
    count = 0
    for i in range(1, len(flags) + 1):
        if flags[i - 1]:
            sum_ += 1 / i
            print(f'iteration:{i}', f'rank (1/i):{1 / i}')
            count += 1
    result = sum_ / count

    return result


def reciprocal_rank_at_k(recommended_list, bought_list, k):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)

    recommended_list = recommended_list[:k]
    rank = [0]
    flag = False
    for item_rec in recommended_list:
        for i, item_bought in enumerate(bought_list):
            if item_rec == item_bought:
                rank.append(1 / (i + 1))
    rank = max(rank)

    return rank

def ap_k(recommended_list, bought_list, k=5):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    flags = np.isin(recommended_list, bought_list)
    flag = []
    for i in range(k):
        if flags[i] == False:
            flag.append(0)
        else:
            flag.append(1)

    result = sum([precision_at_k(recommended_list, bought_list, k=i + 1) for i in flag])/ len(flag)
    return result

def mrr_at_k(data, k):
    return data.apply(lambda x: reciprocal_rank_at_k(x[1], x[2], k), 1).mean()


def ndcg_at_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)

    recommended_list = recommended_list[:k]
    
    flags = np.isin(recommended_list, bought_list)
    flag = []
    for i in range(k):
        if flags[i] == False:
            flag.append(0)
        else:
            flag.append(1)
                
    dcg = 0
    idcg = 0
   
    if len(flag) < k:
        k = len(flag)
 
    
    for i in range(k):
        dcg_i = (int(flag[i]) * 1) / np.log2(i + 2)
        idcg_i = 1 / np.log2(i + 2)
        dcg += dcg_i
        idcg += idcg_i

    ndcg_k = dcg / idcg

    return ndcg_k