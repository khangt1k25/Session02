import numpy as np
from sklearn.metrics import mean_squared_error
from collections import defaultdict

# Class Member has (vector features, label(), id )
class Member:
    def __init__(self, r_d, label, doc_id):
        self._r_d = r_d
        self._label = label
        self._doc_id = doc_id


# Class Cluster has centroid (vector) , list of Members
class Cluster:
    def __init__(self):
        self._centroid = None
        self._members = []

    def reset_Members(self):
        self._members = []

    def add_Members(self, new_member):
        self._members.append(new_member)


#  Kmeans by hand
class Kmeans:
    # Initializing
    def __init__(self, k_clusters):
        self._num_clusters = k_clusters
        self._clusters = [Cluster() for k in range(k_clusters)]
        self._E = []
        self._S = 0

    # Loading data (clustering in train dataset)
    def load_data(self, data_path):
        def sparse_to_dense(r_d, vocab_size):
            vector = [0.0 for i in range(vocab_size)]
            list_rd = r_d.split(" ")
            for one_rd in list_rd:
                one_rd_id = int(one_rd.split(":")[0])
                one_rd_tfidf = float(one_rd.split(":")[1])
                vector[one_rd_id] = one_rd_tfidf
            return np.array(vector)

        with open("../session01/20news-bydate/train_words_idfs.txt", "r") as f:
            vocab = f.read().splitlines()
            vocab_size = len(vocab)
        with open(data_path, "r") as f:
            lines = f.read().splitlines()
        self._data = []
        self._label_count = defaultdict(int)
        for data_id, data_d in enumerate(lines):
            (label, doc_id, r_d) = data_d.split("<fff>")
            label = int(label)
            doc_id = int(doc_id)
            self._label_count[label] += 1

            vector_rd = sparse_to_dense(r_d, vocab_size)

            self._data.append(Member(vector_rd, label, doc_id))

    # Initializing centroid for Kmeans
    def random_init(self, seed):
        np.random.seed(seed)
        length = len(self._data)
        rand_id = np.random.randint(0, length, size=self._num_clusters)
        self._clusters = []
        for k in range(0, self._num_clusters):
            cluster_k = Cluster()
            cluster_k._centroid = self._data[rand_id[k]]._r_d
            self._clusters.append(cluster_k)
        return

    # Computing Distance in L2 norm
    def compute_distance(self, member, centroid):
        member_rd = member._r_d
        return np.sum((member_rd - centroid) ** 2)

    # Selecting cluster for member
    def select_cluster_for(self, member):
        minDist = 10e9
        cluster_chosed = None
        for cluster in self._clusters:
            simi = self.compute_distance(member, cluster._centroid)
            if simi < minDist:
                minDist = simi
                cluster_chosed = cluster

        cluster_chosed.add_Members(member)
        return minDist

    # Updating centroid
    def update_centroid_of(self, cluster):
        member_rds = [member._r_d for member in cluster._members]
        ave_rd = np.mean(member_rds, axis=0)
        sqrt_sum = np.sqrt(np.sum(ave_rd ** 2))
        new_centroid = np.array([value / sqrt_sum for value in ave_rd])
        cluster.Centroid = new_centroid

    # Criteria
    def stopping_condition(self, criterion, threshold):
        criteria = ["centroid", "similarity", "max_iters"]
        assert criterion in criteria
        if criterion == "max_iters":
            if self._iteration >= threshold:
                return True
            else:
                return False
        elif criterion == "centroid":
            E_new = [list(cluster.Centroid) for cluster in self._clusters]
            E_new_minus_E = [centroid for centroid in E_new if centroid not in self._E]
            self._E = E_new
            if len(E_new_minus_E) <= threshold:
                return True
            else:
                return False
        else:
            new_S_minus_S = self._newS - self._S
            self._S = self._newS
            if new_S_minus_S <= threshold:
                return True
            else:
                return False

    # Computing purity
    def compute_purity(self):
        majority_sum = 0
        for cluster in self._clusters:
            member_labels = [member._label for member in cluster._members]
            max_count_labels = max([member_labels.count(label) for label in range(20)])
            majority_sum += max_count_labels
        print(majority_sum * 1.0 / len(self._data))
        return majority_sum * 1.0 / len(self._data)

    # Computing NMI
    def compute_NMI(self):
        I_value, H_omega, H_C, N = 0.0, 0.0, 0.0, len(self._data)
        for cluster in self._clusters:
            wk = 1.0 * len(cluster._members)
            H_omega += -wk / N * np.log10(wk / N)
            member_labels = [member._label for member in cluster._members]

            for label in range(20):
                wk_cj = member_labels.count(label) * 1.0
                cj = self._label_count[label]
                I_value += wk_cj / N * np.log10(N * wk_cj / (wk * cj) + 1e-12)

        for label in range(20):
            cj = self._label_count[label]
            H_C += cj / N * np.log10(cj / N)
        print(I_value * 2.0 / (H_omega + H_C))
        return I_value * 2.0 / (H_omega + H_C)

    # Run KMeans algorithm
    def run(self, seed, criterion, threshold):
        self.random_init(seed)
        self._iteration = 0
        while True:
            for cluster in self._clusters:
                cluster.reset_Members()
            self._newS = 0
            for member in self._data:
                max_s = self.select_cluster_for(member)
                self._newS += max_s

            for cluster in self._clusters:
                self.update_centroid_of(cluster)

            self._iteration += 1
            if self.stopping_condition(criterion, threshold):
                break
        a = self.compute_purity
        print(a)
        b = self.compute_NMI
        print(b)


# Main
if __name__ == "__main__":
    model = Kmeans(20)
    model.load_data("../session01/20news-bydate/train_words_tf_idf.txt")
    model.run(42, "max_iters", 2)
