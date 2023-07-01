import os
import multiprocessing as mp
import time
import concurrent.futures
import re
from pdfminer.high_level import extract_text
import numpy as np


def scoutBee(file_path):
    text = extract_text(file_path)
    return text


def Onlookerbee1(text, keyword_list):
    return all(keyword in text for keyword in keyword_list)


def Onlookerbee(text, keyword_list):
    for keyword in keyword_list:
        if re.search(r'\b' + keyword + r'\b', text) is None:
            return False
    return True


def categorize_pdf(pdf_path, categories_keywords_dict):
    text = scoutBee(pdf_path)
    for category, keywords in categories_keywords_dict.items():
        if Onlookerbee(text, keywords):
            return category
    return 'Uncategorized'


def employedBee(file_paths, categories_keywords_dict, output_dir):
    categorized_files = {}  # Stores file path and its respective category
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_pdf = {executor.submit(categorize_pdf, pdf_path, categories_keywords_dict): pdf_path for pdf_path in file_paths}
        for future in concurrent.futures.as_completed(future_to_pdf):
            pdf_path = future_to_pdf[future]
            try:
                category = future.result()
            except Exception as exc:
                print('%r generated an exception: %s' % (pdf_path, exc))
            else:
                categorized_files[pdf_path] = category

    # Move files in single threaded environment
    for pdf_path, category in categorized_files.items():
        category_dir = os.path.join(output_dir, category)
        os.makedirs(category_dir, exist_ok=True)
        os.rename(pdf_path, os.path.join(category_dir, os.path.basename(pdf_path)))

def WorkPool(input_dir, output_dir, categories_keywords_dict, num_workers):
    pdf_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.pdf')]
    chunks = np.array_split(pdf_files, num_workers)
    processes = []
    for i in range(num_workers):
        p = mp.Process(target=employedBee, args=(chunks[i], categories_keywords_dict, output_dir))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()


# usage
if __name__ == '__main__':
    start = time.time()
    categories_keywords_dict = {'Automata': ['finite', 'state', 'machines'], 'AI': ['Artificial', 'Intelligence'],
                                'DT': ['game', 'theory']}
    WorkPool('pdf', 'pdf', categories_keywords_dict, 1)

    end = time.time()
    print(end - start)

# 87.37399411201477 manual search
# 89.74445843696594 regex
# 110.88900136947632 in 10 workers
# 86.02990007400513 in 20 workers

#108.83784294128418 30w
#106.13278388977051 20w re
#113.14934515953064 30w re
#88.63368105888367 20w
#87.64759230613708 16w
