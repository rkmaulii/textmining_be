import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

def load_data(file_path):
    return pd.read_csv(file_path)

def clean(data):
    data['clean_content'] = data['clean_content'].fillna('')
    return data

def apply_tfidf(data, output_csv_path):
    tfidf_vectorizer = TfidfVectorizer()
    matrix = tfidf_vectorizer.fit_transform(data['clean_content'])
    
    df_tfidf = pd.DataFrame(matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
    df_tfidf.to_csv(output_csv_path, index=False)
    
    return tfidf_vectorizer

if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    input_file_path = os.path.join(current_dir, '..', 'data', 'processed', 'nc_preprocessed_dataset.csv')
    tfidf_model_path = os.path.join(current_dir, '..', 'models', 'nc_tfidf_vectorizer.pkl')
    tfidf_csv_path = os.path.join(current_dir, '..', 'data', 'processed', 'nc_processed_tfidf.csv')

    data = load_data(input_file_path)
    data = clean(data)
    tfidf_vectorizer = apply_tfidf(data, tfidf_csv_path)
    joblib.dump(tfidf_vectorizer, tfidf_model_path)

    print(f"✅ Model TF-IDF disimpan di {tfidf_model_path}")
    print(f"✅ Hasil TF-IDF disimpan di {tfidf_csv_path}")