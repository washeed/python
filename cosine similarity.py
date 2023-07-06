import os
import shutil
import time

from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from multiprocessing import Pool
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from pdfminer.high_level import extract_text
import matplotlib.pyplot as plt


def hunter_spider(pdf_file):
    # The 'hunter spider' opens, reads a PDF file, and processes the text
    document = extract_text(pdf_file)

    # The 'hunter spider' now also does the lemmatization and stopwords removal
    lemmatizer = WordNetLemmatizer()
    document = " ".join(
        lemmatizer.lemmatize(word)
        for word in document.lower().split()
        if word not in stopwords.words("english")
    )
    return document


def worker_spider(documents):
    # The 'worker spider' now does the vectorization, clustering, and file moving
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(documents).toarray()

    # Use cosine similarity for creating a similarity matrix
    cosine_sim_matrix = cosine_similarity(X)

    # Use AgglomerativeClustering with cosine similarity
    clustering_model = AgglomerativeClustering(n_clusters=3, affinity="precomputed", linkage='complete')
    clusters = clustering_model.fit_predict(1 - cosine_sim_matrix)

    pdf_directory = "pdf"
    pdf_files = [os.path.join(pdf_directory, f) for f in os.listdir(pdf_directory) if f.endswith(".pdf")]

    for i, cluster in enumerate(clusters):
        cluster_folder = os.path.join(pdf_directory, f"cluster_{cluster}")
        os.makedirs(cluster_folder, exist_ok=True)
        shutil.move(pdf_files[i], os.path.join(cluster_folder, os.path.basename(pdf_files[i])))

    return X, clusters


if __name__ == "__main__":
    start = time.time()

    pdf_directory = "pdf"
    pdf_files = [os.path.join(pdf_directory, f) for f in os.listdir(pdf_directory) if f.endswith(".pdf")]

    num_hunter_spiders = 4

    # Create hunter spiders pool
    hunter_pool = Pool(num_hunter_spiders)
    documents = hunter_pool.map(hunter_spider, pdf_files)

    # Now 'worker spider' is responsible for the clustering
    X, clusters = worker_spider(documents)

    end = time.time()
    print(end - start)

    # Continue with PCA and plotting as before
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(X)

    plt.figure(figsize=(10, 5))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=clusters)
    plt.title("Cluster Visualization")
    plt.show()
