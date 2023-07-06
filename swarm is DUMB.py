import os
import shutil
import numpy as np
from io import StringIO
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Spider:
    def __init__(self, position):
        self.position = position
        self.fitness = 0

class SocialSpiderOptimization:
    def __init__(self, docs_tfidf, num_clusters=2, Nf=20, Nm=20, iterations=100, alpha=0.5, beta=0.5):
        self.Nf = Nf
        self.Nm = Nm
        self.iterations = iterations
        self.num_clusters = num_clusters
        self.alpha = alpha
        self.beta = beta
        self.population = [Spider(np.random.uniform(0, 1, (num_clusters, docs_tfidf.shape[1]))) for _ in range(Nf+Nm)]
        self.docs_tfidf = docs_tfidf

    def calculate_fitness(self, spider):
        total_similarity = 0
        for i in range(self.num_clusters):
            centroid = spider.position[i]
            similarities = cosine_similarity(centroid.reshape(1, -1), self.docs_tfidf)
            total_similarity += np.sum(similarities)
        return total_similarity

    def update_spider(self, spider):
        best_position = self.population[0].position
        for i in range(self.num_clusters):
            if np.random.random() < 0.5:
                spider.position[i] += self.alpha * best_position[i] + self.beta * np.random.uniform(-0.1, 0.1, spider.position[i].shape)
            else:
                spider.position[i] -= self.alpha * best_position[i] + self.beta * np.random.uniform(-0.1, 0.1, spider.position[i].shape)
        spider.fitness = self.calculate_fitness(spider)

    def run_optimization(self):
        for iteration in range(self.iterations):
            for spider in self.population:
                self.update_spider(spider)
            self.population.sort(key=lambda x: x.fitness, reverse=True)
        return self.population[0].position  # Return best solution

def read_pdf_file(path):
    output_string = StringIO()
    with open(path, 'rb') as in_file:
        parser = PDFParser(in_file)
        doc = PDFDocument(parser)
        rsrcmgr = PDFResourceManager()
        device = TextConverter(rsrcmgr, output_string, laparams=LAParams())
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        for page in PDFPage.create_pages(doc):
            interpreter.process_page(page)
    return output_string.getvalue()

# Read documents from PDF files
input_dir = 'pdf'
documents = []
file_names = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith('.pdf')]
doc_to_file = {}

for file_name in file_names:
    doc_text = read_pdf_file(os.path.join(input_dir, file_name))
    documents.append(doc_text)

    doc_to_file[doc_text] = file_name

# Vectorize documents
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

# Run Social Spider Optimization algorithm
ssa = SocialSpiderOptimization(X.toarray(), num_clusters=3, Nf=70, Nm=70, iterations=700, alpha=0.5, beta=0.5)
best_solution = ssa.run_optimization()

# Cluster documents
clusters = {}
for doc_text in documents:
    doc_tfidf = vectorizer.transform([doc_text]).toarray()[0]
    similarities = cosine_similarity(doc_tfidf.reshape(1, -1), best_solution)
    cluster_index = np.argmax(similarities)
    if cluster_index in clusters:
        clusters[cluster_index].append(doc_text)
    else:
        clusters[cluster_index] = [doc_text]

# Move documents to corresponding folders
for cluster_index, cluster in clusters.items():
    # We create the new directories inside the input directory
    cluster_dir = os.path.join(input_dir, f'cluster_{cluster_index}')
    if not os.path.exists(cluster_dir):
        os.makedirs(cluster_dir)
    for doc_text in cluster:
        original_file_name = doc_to_file[doc_text]
        # We move the files to the new directories inside the input directory
        shutil.move(os.path.join(input_dir, original_file_name), os.path.join(cluster_dir, original_file_name))
