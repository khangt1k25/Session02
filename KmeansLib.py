import numpy as np
from sklearn.metrics import mean_squared_error
from collections import defaultdict
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix


def load_data(data_path):
    def sparse_to_dense(r_d, vocab_size):
        vector = [0.0 for i in range(vocab_size)]
        list_rd = r_d.split(" ")
        for one_rd in list_rd:
            one_rd_id = int(one_rd.split(":")[0])
            one_rd_tfidf = float(one_rd.split(":")[1])
            vector[one_rd_id] = one_rd_tfidf
        return np.array(vector)

    with open("../session01/20news-bydate/words_idfs.txt", "r") as f:
        vocab = f.read().splitlines()
        vocab_size = len(vocab)
    with open(data_path, "r") as f:
        lines = f.read().splitlines()
    data = []
    labels = []
    label_count = defaultdict(int)
    for data_id, data_d in enumerate(lines):
        (label, doc_id, r_d) = data_d.split("<fff>")
        label = int(label)
        doc_id = int(doc_id)
        label_count[label] += 1

        vector_rd = sparse_to_dense(r_d, vocab_size)

        data.append(vector_rd)
        labels.append(label)
    return data, labels


def compute_accuracy(y_predicted, y_expected):
    matches = np.equal(y_expected, y_predicted)

    accuracy = np.sum(matches.astype(float)) / len(y_expected)

    return accuracy


def clustering_kmeans_sklearn(data, labels):
    X = csr_matrix(data)
    print("=====")
    kmeans = KMeans(
        n_clusters=20, init="random", n_init=5, tol=1e-3, random_state=42
    ).fit(X)
    print("inertia = " + str(kmeans.inertia_))


def clustering_svm_sklearn(data, labels):
    from sklearn.svm import LinearSVC

    classifier = LinearSVC(C=10.0, tol=0.001, verbose=True, random_state=42)

    classifier.fit(data, labels)

    y_predicted = classifier.predict(data)
    accuracy = compute_accuracy(y_predicted=y_predicted, y_expected=labels)

    print("Accuracy" + str(accuracy))


def clustering_svm_kernel_sklearn(data, labels):
    from sklearn.svm import SVC

    classifier = SVC(C=10.0, kernel="rbf", tol=1e-3, verbose=True, random_state=42)
    classifier.fit(data, labels)


if __name__ == "__main__":
    data, labels = load_data("../session01/20news-bydate/words_tf_idf.txt")

    print("Using sklearn kmeans")
    clustering_svm_sklearn(data, labels)
    print("=============")
    print("Using sklearn svm")
    clustering_svm_sklearn(data, labels)
