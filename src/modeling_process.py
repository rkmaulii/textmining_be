import os
import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import json
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix)
import seaborn as sns
from sklearn.svm import LinearSVC


def train_svm(X_train, y_train):
    print("Melatih model SVM...")
    model = LinearSVC()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, confmet_path):
    print("\nMengevaluasi model...")
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    report_str = classification_report(y_test, y_pred)

    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(confmet_path)
    plt.close()

    return model, report, report_str

def show_top_features(model, feature_names,  top_features_path, top_n=50):
    print("\nMenampilkan 50 fitur dengan bobot tertinggi per kelas:")
    if not hasattr(model, "coef_"):
        print("Model ini tidak memiliki koefisien bobot (coef_). Pastikan menggunakan kernel='linear'.")
        return
    
    coef = model.coef_.flatten()
    top_indikasi_idx = coef.argsort()[-top_n:][::-1]
    top_non_idx = coef.argsort()[:top_n]

    lines = []
    lines.append("Kelas indikasi (fitur yang mendorong prediksi 'indikasi'):\n")
    for i in top_indikasi_idx:
        line = f"{feature_names[i]:<50} {coef[i]:.4f}"
        print(line)
        lines.append(line + "\n")

    print("\n Kelas non (fitur yang mendorong prediksi 'non'):")
    for i in top_non_idx:
        line = f"{feature_names[i]:<50} {coef[i]:.4f}"
        print(line)
        lines.append(line + "\n")
    
    with open(top_features_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"✅ Top features disimpan di: {top_features_path}")

def save_model(model, report, report_str, model_path, txt_report_path, json_report_path):
    joblib.dump(model, model_path)
    with open(txt_report_path, "w", encoding="utf-8") as f:
        f.write(report_str)
    with open(json_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
    print("✅ Semua file berhasil disimpan.")

if __name__ == "__main__":
    print("Memulai proses pemodelan...\n")
    current_dir = os.path.dirname(__file__)
    tfidf_csv_path = os.path.join(current_dir, '..', 'data', 'processed', 'nc_processed_tfidf.csv')
    original_data_path = os.path.join(current_dir, '..', 'data', 'raw', 'final_dataset.csv')
    model_path = os.path.join(current_dir, '..', 'models', 'svm_model9010.pkl')
    report_dir = os.path.join(current_dir, '..', 'data', 'report')
    txt_report_path = os.path.join(report_dir, "classification_report_9010.txt")
    json_report_path = os.path.join(report_dir, "classification_report_9010.json")
    top_features_path = os.path.join(report_dir, "top_features_9010.txt")
    confmet_path = os.path.join(report_dir, "confusion_matrix_9010.png")

    # Load data
    X = pd.read_csv(tfidf_csv_path)
    original_df = pd.read_csv(original_data_path)

    if 'label' not in original_df.columns:
        raise ValueError("Kolom 'label' tidak ditemukan dalam data asli!")

    y = original_df['label'].fillna('').reset_index(drop=True)
    original_df = original_df.reset_index(drop=True)
    X = X.reset_index(drop=True)

    # Split data
    X_train, X_test, y_train, y_test, original_df_train, original_df_test = train_test_split(
        X, y, original_df, test_size=0.1, stratify=y, random_state=42
    )
    model = train_svm(X_train, y_train)
    model, report, report_str = evaluate_model(model, X_test, y_test, confmet_path)
    save_model(model, report, report_str, model_path, txt_report_path, json_report_path)

    feature_names = X.columns.tolist()
    show_top_features(model, feature_names, top_features_path, top_n=50)

    print("Proses selesai!")