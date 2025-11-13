from collections import defaultdict
import random

import numpy as np
from pandas.compat.pickle_compat import load_reduce
from pyJoules.device import DeviceFactory
from pyJoules.device.rapl_device import RaplPackageDomain, RaplDramDomain
from surprise import accuracy, Dataset, SVD, Reader, KNNWithMeans
from surprise.model_selection import train_test_split
import json
import pandas as pd
import pickle
import re
from pyJoules.energy_meter import measure_energy, EnergyMeter
import math
from datasketch import MinHash, MinHashLSH


k = 10
do_measure_energy = False
relevance_threshold = 3.5
popularity_threshold = 4

def load_object(filename):
    with open(filename, 'rb') as fp:
        ret = pickle.load(fp)
    return ret

def save_object(filename, object):
    with open(filename, 'wb') as fp:
        pickle.dump(object, fp, protocol=pickle.HIGHEST_PROTOCOL)

def get_minhash(tags):
    m = MinHash(num_perm=128)
    for tag in tags:
        m.update(tag.encode('utf8'))
    return m

def get_similar_items_lsh(item_id, minhashes, lsh, top_n=10):
    m = minhashes[item_id]
    candidates = lsh.query(m)
    candidates.remove(item_id)
    return candidates[:top_n]

def get_liked_items(user_raw_id, trainset, threshold=4.0):
    try:
        user_inner_id = trainset.to_inner_uid(user_raw_id)
    except ValueError:
        return []

    liked = []
    for item_inner_id, rating in trainset.ur[user_inner_id]:
        if rating >= threshold:
            item_raw_id = trainset.to_raw_iid(item_inner_id)
            liked.append((item_raw_id, rating))
    return liked

def content_based_predict(user_raw_id, trainset, minhashes, lsh, top_n=10, threshold=4.0):
    liked_items = get_liked_items(user_raw_id, trainset, threshold=threshold)
    if not liked_items:
        return []

    candidate_scores = defaultdict(float)
    similarity_sums = defaultdict(float)

    for item_id, rating in liked_items:
        try:
            similar_items = get_similar_items_lsh(item_id, minhashes, lsh)
        except KeyError:
            continue

        for sim_item in similar_items:
            if sim_item == item_id:
                continue
            # Use constant similarity score (approximate), or estimate similarity weight = 1
            candidate_scores[sim_item] += rating  # Can be weighted by similarity if you return scores
            similarity_sums[sim_item] += 1.0

    # Final predicted score = weighted average
    predictions = []
    already_rated = set(trainset.to_raw_iid(iid) for iid, _ in trainset.ur[trainset.to_inner_uid(user_raw_id)])

    for item_id in candidate_scores:
        if item_id in already_rated:
            continue
        if similarity_sums[item_id] == 0:
            continue
        score = candidate_scores[item_id] / similarity_sums[item_id]

        # Clip score to rating scale
        min_rating, max_rating = trainset.rating_scale
        score = max(min(score, max_rating), min_rating)

        predictions.append((item_id, score))

    # Sort by predicted score
    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
    return predictions[:top_n]

def content_based_predict_testset(testset, trainset, minhashes, lsh, products, threshold=4.0):
    # Get all liked items by the user
    user_est_true = defaultdict(list)
    for user_id, item_id, true_rating in testset:
        liked_items = get_liked_items(user_id, trainset, threshold=threshold)
        if not liked_items:
            if item_id in products:
                est_rating = products[item_id]["average_rating"]  # No profile for this user
            else:
                continue
        else:
            score_sum = 0.0
            weight_sum = 0.0

            for liked_item_id, liked_item_rating in liked_items:
                try:
                    similar_items = get_similar_items_lsh(liked_item_id, minhashes, lsh)
                except KeyError:
                    continue

                if item_id in similar_items:
                    # Assume similarity weight of 1.0 (since LSH doesn't return scores)
                    score_sum += liked_item_rating
                    weight_sum += 1.0

            if weight_sum == 0:
                if item_id in products:
                    est_rating = products[item_id]["average_rating"]  # No similar items → can't make a prediction
                else:
                    continue
            else:
                predicted_rating = score_sum / weight_sum
                min_rating, max_rating = trainset.rating_scale
                est_rating = max(min(predicted_rating, max_rating), min_rating)
        user_est_true[user_id].append((est_rating, true_rating))

    return user_est_true

def caluclate_rmse(user_est_true):
    sum = 0.0
    n = 0
    for u_id in user_est_true:
        for tuple in user_est_true[u_id]:
            n += 1
            sum += math.pow(tuple[1] - tuple[0], 2)
    if n == 0:
        return 999
    return math.sqrt(sum / n)

def recommend_popularity_based(recommendation_approach, products, trainset, testset):
    if do_measure_energy:
        domains = [RaplPackageDomain(0), RaplDramDomain(0)]
        devices = DeviceFactory.create_devices(domains)
        meter = EnergyMeter(devices)

    if do_measure_energy:
        ec_build = 0
        ec_build /= 100000  # energy is in uJ so converting to J
        save_object("temp_ec_build-" + recommendation_approach, ec_build)
    else:
        ec_build = load_object("temp_ec_build-" + recommendation_approach)


    top_n = get_top_n_recommendations_popularity_based(trainset, products, n=k)
    avg_car_f = calculate_avg_car_f(top_n, products)
    g_i_rec = calculate_g_i_rec(top_n, products)

    if do_measure_energy:
        meter.start()

    user_est_true = popularity_based_predict_testset(testset, products)

    if do_measure_energy:
        meter.stop()
        e_interface = meter.get_trace()[0].energy["package_0"] + meter.get_trace()[0].energy["dram_0"]
        e_interface /= 100000  # energy is in uJ so converting to J
        save_object("temp_e_interface-" + recommendation_approach, e_interface)
    else:
        e_interface = load_object("temp_e_interface-" + recommendation_approach)

    e_c_rec = calculate_e_c_rec(e_interface, len(testset))
    e_c_mod_b = calculate_e_c_mod_b(ec_build, 1)  # not sure if this should be 1
    e_c_mod_b_data = calculate_e_c_mod_b_data(e_c_mod_b, len(products.keys()))

    avg_list_du = calculate_avg_list_d_u(top_n, products)
    avg_ser_u = calculate_avg_ser_u(top_n, products)

    rmse = caluclate_rmse(user_est_true)
    precisions, recalls = precision_recall_at_k(None, user_est_true, k=k, threshold=relevance_threshold)
    precision = sum(prec for prec in precisions.values()) / len(precisions)
    recall = sum(rec for rec in recalls.values()) / len(recalls)

    res = Result(recommendation_approach, avg_car_f, g_i_rec, e_c_rec, e_c_mod_b, e_c_mod_b_data, avg_list_du, avg_ser_u, rmse, precision, recall)
    return res


def get_top_n_recommendations_popularity_based(trainset, products, n=10):
    top_n = defaultdict(list)

    items = [(k, v["bw_score"]) for k, v in products.items()]
    pop_n = sorted(items, key=lambda item: item[1], reverse=True)[:n]

    for user in [trainset.to_raw_uid(uid) for uid in trainset.all_users()]:
        top_n[user] = pop_n.copy()
    return top_n

def popularity_based_predict_testset(testset, products):
    # Get all liked items by the user
    user_est_true = defaultdict(list)
    for user_id, item_id, true_rating in testset:
        if item_id in products:
            user_est_true[user_id].append((products[item_id]["bw_score"], true_rating))

    return user_est_true

def recommend_content_based(recommendation_approach, products, trainset, testset):
    if do_measure_energy:
        domains = [RaplPackageDomain(0), RaplDramDomain(0)]
        devices = DeviceFactory.create_devices(domains)
        meter = EnergyMeter(devices)

    if do_measure_energy:
        meter.start()

    minhashes = {}
    for pid, product in products.items():
        minhashes[pid] = get_minhash(get_item_categories(product))

    lsh = MinHashLSH(threshold=0.7, num_perm=128)
    for item_id, m in minhashes.items():
        lsh.insert(item_id, m)


    if do_measure_energy:
        meter.stop()
        ec_build = meter.get_trace()[0].energy["package_0"] + meter.get_trace()[0].energy["dram_0"]
        ec_build /= 100000  # energy is in uJ so converting to J
        save_object("temp_ec_build-" + recommendation_approach, ec_build)
    else:
        ec_build = load_object("temp_ec_build-" + recommendation_approach)

    save_object("minhashes", minhashes)
    save_object("lsh", lsh)


    top_n = get_top_n_recommendations_content_based(trainset, minhashes, lsh, n=k)

    avg_car_f = calculate_avg_car_f(top_n, products)
    g_i_rec = calculate_g_i_rec(top_n, products)

    if do_measure_energy:
        meter.start()

    user_est_true = content_based_predict_testset(testset, trainset, minhashes, lsh, products)

    if do_measure_energy:
        meter.stop()
        e_interface = meter.get_trace()[0].energy["package_0"] + meter.get_trace()[0].energy["dram_0"]
        e_interface /= 100000  # energy is in uJ so converting to J
        save_object("temp_e_interface-" + recommendation_approach, e_interface)
    else:
        e_interface = load_object("temp_e_interface-" + recommendation_approach)

    e_c_rec = calculate_e_c_rec(e_interface, len(testset))
    e_c_mod_b = calculate_e_c_mod_b(ec_build, 1)  # not sure if this should be 1
    e_c_mod_b_data = calculate_e_c_mod_b_data(e_c_mod_b, len(products.keys()))

    avg_list_du = calculate_avg_list_d_u(top_n, products)
    avg_ser_u = calculate_avg_ser_u(top_n, products)

    rmse = caluclate_rmse(user_est_true)
    precisions, recalls = precision_recall_at_k(None, user_est_true, k=k, threshold=relevance_threshold)
    precision = sum(prec for prec in precisions.values()) / len(precisions)
    recall = sum(rec for rec in recalls.values()) / len(recalls)

    res = Result(recommendation_approach, avg_car_f, g_i_rec, e_c_rec, e_c_mod_b, e_c_mod_b_data, avg_list_du, avg_ser_u, rmse, precision, recall)
    return res

def get_top_n_recommendations_content_based(trainset, minhashes, lsh, n=10):
    top_n = defaultdict(list)
    for user in [trainset.to_raw_uid(uid) for uid in trainset.all_users()]:
        top_n[user] = content_based_predict(user, trainset, minhashes, lsh, top_n=n)
    return top_n

def main():
    data = load_object("ratings-data-reduced")
    products = load_object("extracted-products-reduced-supplemented")

    green_label_coverage = 0
    green_sum = 0
    car_f_label_coverage = 0
    car_f_sum = 0
    for p_id in products:
        item = products[p_id]
        if item["car_f"] is not None:
            car_f_label_coverage += 1
            car_f_sum += item["car_f"]
        if item["green"] is not None:
            green_label_coverage += 1
            if item["green"]:
                green_sum += 1
    print("AvgCarFAllProducts:", car_f_sum / car_f_label_coverage)
    print("AvgGreenAllProducts:", green_sum / green_label_coverage)
    green_label_coverage /= len(products)
    car_f_label_coverage /= len(products)
    print("LabelCoverage(green):", green_label_coverage)
    print("LabelCoverage(carF):", car_f_label_coverage)

    fullResults = FullResults()
    for iteration in range(1): # 0):
        # sample random trainset and testset
        # test set is made of 25% of the ratings.
        trainset, testset = train_test_split(data, test_size=0.25)

        for recommendation_approach in ["NeuMF"]: #["SVD", "KNNWithMeans", "ContentBasedFiltering", "PopularityBaseline", "NeuMF"]:
            if recommendation_approach == "ContentBasedFiltering":
                res = recommend_content_based(recommendation_approach, products, trainset, testset)
            elif recommendation_approach == "PopularityBaseline":
                res = recommend_popularity_based(recommendation_approach, products, trainset, testset)
            elif recommendation_approach == "NeuMF":
                res = recommend_popularity_based(recommendation_approach, products, trainset, testset)
            else:
                res = surprise_based_recommendations(recommendation_approach, products, trainset, testset)
            fullResults.add_result(recommendation_approach, res)
    save_object("fullResults", fullResults)
    fullResults.print_averages()

def surprise_based_recommendations(recommendation_approach, products, trainset, testset):
    if do_measure_energy:
        domains = [RaplPackageDomain(0), RaplDramDomain(0)]
        devices = DeviceFactory.create_devices(domains)
        meter = EnergyMeter(devices)
        meter.start()

    if recommendation_approach == "SVD":
        algo = SVD()
    elif recommendation_approach == "KNNWithMeans":
        algo = KNNWithMeans()

    algo.fit(trainset)

    if do_measure_energy:
        meter.stop()
        ec_build = meter.get_trace()[0].energy["package_0"] + meter.get_trace()[0].energy["dram_0"]
        ec_build /= 100000  # energy is in uJ so converting to J
        save_object("temp_ec_build-" + recommendation_approach, ec_build)
    else:
        ec_build = load_object("temp_ec_build-" + recommendation_approach)

    inverted_set = trainset.build_anti_testset()

    predictions = algo.test(inverted_set)
    top_n = get_top_n_recommendations(predictions, n=k)


    avg_car_f = calculate_avg_car_f(top_n, products)
    g_i_rec = calculate_g_i_rec(top_n, products)

    if do_measure_energy:
        meter.start()
    predictions = algo.test(testset)
    if do_measure_energy:
        meter.stop()
        e_interface = meter.get_trace()[0].energy["package_0"] + meter.get_trace()[0].energy["dram_0"]
        e_interface /= 100000  # energy is in uJ so converting to J
        save_object("temp_e_interface-" + recommendation_approach, e_interface)
    else:
        e_interface = load_object("temp_e_interface-" + recommendation_approach)

    e_c_rec = calculate_e_c_rec(e_interface, len(testset))
    e_c_mod_b = calculate_e_c_mod_b(ec_build, 1)  # not sure if this should be 1
    e_c_mod_b_data = calculate_e_c_mod_b_data(e_c_mod_b, len(trainset.ir))

    avg_list_du = calculate_avg_list_d_u(top_n, products)
    avg_ser_u = calculate_avg_ser_u(top_n, products)

    precisions, recalls = precision_recall_at_k(predictions, None, k=k, threshold=relevance_threshold)

    rmse = accuracy.rmse(predictions, verbose=False)
    precision = sum(prec for prec in precisions.values()) / len(precisions)
    recall = sum(rec for rec in recalls.values()) / len(recalls)
    res = Result(recommendation_approach, avg_car_f, g_i_rec, e_c_rec, e_c_mod_b, e_c_mod_b_data, avg_list_du,
                 avg_ser_u, rmse, precision, recall)
    return res

def calculate_avg_ser_u(top_n, products):
    sum = 0.0
    n = len(top_n)
    for uid, r_u in top_n.items():
        if len(r_u) > 0:
            t = calculate_ser_u(r_u, products)
            if t is not None:
                sum += t
            else:
                n -= 1
        else:
            n -= 1
    return sum / len(top_n)

def calculate_ser_u(r_u, products):
    top = 0
    number_rec = len(r_u)
    for i in range(len(r_u)):
        if r_u[i][0] not in products:
            number_rec -= 1
        else:
            item = products[r_u[i][0]]
            if r_u[i][1] > relevance_threshold and item["bw_score"] <= popularity_threshold:
                top += 1
    if number_rec == 0:
        return None
    return top / number_rec

def calculate_avg_list_d_u(top_n, products):
    sum = 0.0
    n = len(top_n)
    for uid, r_u in top_n.items():
        if len(r_u) > 0:
            t = calculate_list_d_u(r_u, products)
            if t is not None:
                sum += t
            else:
                n -= 1
        else:
            n -= 1
    return sum / len(top_n)

def calculate_list_d_u(r_u, products):
    top = 0
    number_rec = len(r_u)
    items = []
    for i in range(len(r_u)):
        if r_u[i][0] not in products:
            number_rec -= 1
            items.append(None)
        else:
            items.append(products[r_u[i][0]])
    for i in range(len(r_u)):
        if items[i] is None:
            continue
        for j in range(len(r_u)):
            if i != j:
                if items[j] is None:
                    continue
                top += calculate_sim(items[i], items[j])
    bottom = number_rec * (number_rec - 1)
    if bottom == 0:
        return None
    return 1 - (top / bottom)

def calculate_e_c_rec(e_interface, n_rec):
    return e_interface / n_rec

def calculate_e_c_mod_b(ec_build, n_epoch):
    return ec_build / n_epoch

def calculate_e_c_mod_b_data(ec_build, n_data_processed):
    return ec_build / n_data_processed

def calculate_avg_car_f(top_n, products):
    user_i = 0
    sum = 0
    alarm = 0
    for user in top_n:
        car_f_i = 0
        user_sum = 0
        for rec in top_n[user]:
            if rec[0] not in products:
                alarm += 1
                continue
            item = products[rec[0]]
            # car_f = estimate_carbon_footprint(item["weight"], item["volume"], item["electronics"])
            # if not math.isnan(car_f):
            if item["car_f"] is not None:
                user_sum += item["car_f"]
                car_f_i += 1
        if car_f_i > 0:
            sum += user_sum / car_f_i
            user_i += 1
    # print("ALARM:", alarm)
    if user_i == 0:
        return float("nan")
    else:
        return sum / user_i

def calculate_g_i_rec(top_n, products):
    sum_top = 0
    sum_bottom = 0
    alarm = 0
    for user in top_n:
        for rec in top_n[user]:
            if rec[0] not in products:
                alarm += 1
                continue
            item = products[rec[0]]
            if item["green"] is not None:
                sum_bottom += 1
                if item["green"]:
                    sum_top += 1
    # print("ALARM:", alarm)
    return sum_top / sum_bottom

def transforming_data(min = False):
    ratings_dict = {
        "itemID": [],
        "userID": [],
        "rating": [],
    }

    with open("Toys_and_Games.jsonl", 'r') as f:
        i = 0
        for line in f:
            json_data = json.loads(line)
            if json_data["verified_purchase"]:
                ratings_dict["itemID"].append(json_data["asin"])
                ratings_dict["userID"].append(json_data["user_id"])
                ratings_dict["rating"].append(json_data["rating"])
            if min and i >= 10000:
                break
            i += 1
    df = pd.DataFrame(ratings_dict)

    reader = Reader(rating_scale=(1, 5))

    data = Dataset.load_from_df(df[["userID", "itemID", "rating"]], reader)
    save_object("ratings-data" + "-min" if min else "", data)

def bayesian_weighted_score(R, v, C, m):
    return (v / (v + m)) * R + (m / (v + m)) * C

def transforming_meta_data():
    products = {}
    with open("meta_Toys_and_Games.jsonl", 'r') as f:
        ar = []
        nr = []
        for line in f:
            json_data = json.loads(line)
            id = json_data["parent_asin"]
            ar.append(json_data["average_rating"])
            nr.append(json_data["rating_number"])
            item = {
                "categories": get_categories(json_data),
                "average_rating": json_data["average_rating"],
                "number_ratings": json_data["rating_number"]
            }
            products.update({id: item})

        # Global mean and prior strength
        C = np.mean(np.array(ar))
        # Choose m as, e.g., the 80th percentile of vote counts (common heuristic)
        m = np.percentile(np.array(nr), 80)

        for k, p in products.items():
            p["bw_score"] = bayesian_weighted_score(p["average_rating"], p["number_ratings"], C, m)
        save_object("extracted-products", products)

def reduce_data():
    data = load_object("ratings-data")

    df = pd.DataFrame(data.raw_ratings, columns=["user", "item", "rating", "timestamp"])

    # Count number of ratings per user
    user_counts = df.groupby("user").size()

    # Filter users who have rated 16 or more items
    eligible_users = user_counts[user_counts >= 16].index

    # Randomly select 1,000 users (or fewer if not enough users meet the condition)
    selected_users = random.sample(list(eligible_users), min(1000, len(eligible_users)))

    # Filter the DataFrame to only include ratings from selected users
    filtered_df = df[df['user'].isin(selected_users)]

    # Drop timestamp column Surprise doesn't use it
    filtered_df = filtered_df.drop(columns=["timestamp"])

    # Convert back to Surprise Dataset
    reader = Reader(rating_scale=(df["rating"].min(), df["rating"].max()))
    filtered_data = Dataset.load_from_df(filtered_df, reader)

    save_object("ratings-data-reduced", filtered_data)
    products = load_object("extracted-products")
    valid_product_ids = set(filtered_data.df["item"].unique())

    # Filter the product_dict to include only valid items
    filtered_product_dict = {
        pid: pdata for pid, pdata in products.items()
        if pid in valid_product_ids
    }
    save_object("extracted-products-reduced", filtered_product_dict)


def count_dict_key(counting_dict, key):
    if key in counting_dict:
        counting_dict[key] += 1
    else:
        counting_dict.update({key: 1})

def get_item_categories(item):
    ret = list(item["categories"])
    return ret

# based on jaccard similarity
def calculate_sim(item1, item2):
    set1 = set(get_item_categories(item1))
    set2 = set(get_item_categories(item2))
    return len(set1 & set2) / len(set1 | set2)

def identify_user():
    users = set([])
    with open("Toys_and_Games.jsonl", 'r') as f:
        i = 0
        for line in f:
            json_data = json.loads(line)
            if json_data["verified_purchase"]:
                users.add(json_data["user_id"])
                i += 1
    save_object("users", list(users))

def get_categories(json_data):
    if json_data["main_category"] is None:
        ret = []
    else:
        ret = [json_data["main_category"]]
    for category in json_data["categories"]:
        if category not in ret:
            ret.append(category)
    return ret

# source: https://surprise.readthedocs.io/en/stable/FAQ.html
def precision_recall_at_k(predictions, user_est_true, k=10, threshold=3.5):
    """Return precision and recall at k metrics for each user"""

    if user_est_true is None:
        # First map the predictions to each user.
        user_est_true = defaultdict(list)
        for uid, _, true_r, est, _ in predictions:
            user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():
        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(
            ((true_r >= threshold) and (est >= threshold))
            for (est, true_r) in user_ratings[:k]
        )

        # Precision@K: Proportion of recommended items that are relevant
        # When n_rec_k is 0, Precision is undefined. We here set it to 0.

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

        # Recall@K: Proportion of relevant items that are recommended
        # When n_rel is 0, Recall is undefined. We here set it to 0.

        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    return precisions, recalls

def get_top_n_recommendations(predictions, n=10):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the n highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n

def testing_stuff():
    return


class Result:
    def __init__(self, recommendation_approach, AvgCarFI, GIRec, ECRec, ECTrain, ECPDat, AvgListDu, AvgSERu, RMSE, Precision, Recall):
        self.recommendation_approach = recommendation_approach
        self.AvgCarFI = AvgCarFI
        self.GIRec = GIRec
        self.ECRec = ECRec
        self.ECTrain = ECTrain
        self.ECPDat = ECPDat
        self.AvgListDu = AvgListDu
        self.AvgSERu = AvgSERu
        self.RMSE = RMSE
        self.Precision = Precision
        self.Recall = Recall
        self.F1 = 2 * (Precision * Recall) / (Precision + Recall)

    def print(self):
        print("=======================================================")
        print("Recommendation approach: ", self.recommendation_approach)
        print("=======================================================")
        print("Sustainability Metrics =======")
        print("AvgCarFI:", self.AvgCarFI, "kgCO2")
        print("GIRec:", self.GIRec)
        print("ECRec:", self.ECRec)
        print("ECTrain:", self.ECTrain)
        print("ECPDat:", self.ECPDat)
        print("AvgListDu:", self.AvgListDu)
        print("AvgSERu:", self.AvgSERu)
        print("Conventional Metrics =======")
        print("RMSE:", self.RMSE)
        print("Precision@K=10:", self.Precision)
        print("Recall@K=10:", self.Recall)
        print("F1@K=10:", self.F1)
        print("=======================================================")

class FullResults:
    def __init__(self):
        self.recommendation_approaches = []
        self.results = {}

    def add_result(self, recommendation_approach, result : Result):
        if recommendation_approach in self.recommendation_approaches:
            self.results[recommendation_approach].append(result)
        else:
            self.recommendation_approaches.append(recommendation_approach)
            self.results[recommendation_approach] = [result]

    def print_averages(self):
        for recommendation_approach in self.recommendation_approaches:
            print("=======================================================")
            print("Recommendation approach: ", recommendation_approach)
            print("==============================")
            print("Sustainability Metrics =======")
            print("AvgCarFI [kgCO2]: ============")
            a = np.array([i.AvgCarFI for i in self.results[recommendation_approach]])
            print("MEAN", np.mean(a))
            print("STDD", np.std(a))
            print("MEDI", np.median(a))
            print("GIRec: =======================")
            a = np.array([i.GIRec for i in self.results[recommendation_approach]])
            print("MEAN", np.mean(a))
            print("STDD", np.std(a))
            print("MEDI", np.median(a))
            print("ECRec: =======================")
            a = np.array([i.ECRec for i in self.results[recommendation_approach]])
            print("MEAN", np.mean(a))
            print("STDD", np.std(a))
            print("MEDI", np.median(a))
            print("ECTrain: =====================")
            a = np.array([i.ECTrain for i in self.results[recommendation_approach]])
            print("MEAN", np.mean(a))
            print("STDD", np.std(a))
            print("MEDI", np.median(a))
            print("ECPDat: ======================")
            a = np.array([i.ECPDat for i in self.results[recommendation_approach]])
            print("MEAN", np.mean(a))
            print("STDD", np.std(a))
            print("MEDI", np.median(a))
            print("AvgListDu: ===================")
            a = np.array([i.AvgListDu for i in self.results[recommendation_approach]])
            print("MEAN", np.mean(a))
            print("STDD", np.std(a))
            print("MEDI", np.median(a))
            print("AvgSERu: =====================")
            a = np.array([i.AvgSERu for i in self.results[recommendation_approach]])
            print("MEAN", np.mean(a))
            print("STDD", np.std(a))
            print("MEDI", np.median(a))
            print("Conventional Metrics =========")
            print("RMSE: ========================")
            a = np.array([i.RMSE for i in self.results[recommendation_approach]])
            print("MEAN", np.mean(a))
            print("STDD", np.std(a))
            print("MEDI", np.median(a))
            print("Precision@K=10: ==============")
            a = np.array([i.Precision for i in self.results[recommendation_approach]])
            print("MEAN", np.mean(a))
            print("STDD", np.std(a))
            print("MEDI", np.median(a))
            print("Recall@K=10: =================")
            a = np.array([i.Recall for i in self.results[recommendation_approach]])
            print("MEAN", np.mean(a))
            print("STDD", np.std(a))
            print("MEDI", np.median(a))
            print("F1@K=10: =====================")
            a = np.array([i.F1 for i in self.results[recommendation_approach]])
            print("MEAN", np.mean(a))
            print("STDD", np.std(a))
            print("MEDI", np.median(a))
            print("=======================================================")


if __name__ == "__main__":

    # transforming_data()
    # transforming_meta_data()
    # reduce_data()

    # Note that the supplemented product data was extended by the LLM sourced green flag and the Carbon footprint estimated by https://github.com/amazon-science/carbon-assessment-with-ml as discribed in the paper

    main()