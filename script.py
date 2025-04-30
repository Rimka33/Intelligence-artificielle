import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
import optuna
import os

st.title("Détection de Malware")

st.sidebar.header("Configuration")

# Section pour le dataset d'entraînement
st.sidebar.subheader("Dataset d'entraînement")
default_file_path = "C:/Users/DELL/PycharmProjects/malware/DatasetmalwareExtrait.csv" 
use_default = st.sidebar.checkbox("Utiliser le dataset par défaut", value=True)

if use_default:
    try:
        dataset = pd.read_csv(default_file_path)
        st.success(f"Dataset par défaut chargé depuis : {default_file_path}")
    except FileNotFoundError:
        st.error(f"Fichier par défaut non trouvé au chemin : {default_file_path}")
        dataset = None
else:
    uploaded_file = st.sidebar.file_uploader("Upload Dataset CSV", type=["csv"])
    if uploaded_file is not None:
        dataset = pd.read_csv(uploaded_file)
        st.success("Dataset chargé avec succès depuis le téléversement.")
    else:
        st.warning("Veuillez téléverser un fichier CSV ou utiliser le fichier par défaut.")
        dataset = None

# Section pour la prédiction
st.sidebar.subheader("Prédiction de nouveaux fichiers")
prediction_file = st.sidebar.file_uploader("Upload fichier à analyser", type=["csv"])

if dataset is not None:
    st.write("Aperçu des données d'entraînement :", dataset.head())

    # Séparation des features et des labels
    x = dataset.drop(columns=["legitimate"])
    y = dataset["legitimate"]

    # Division des données
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=42)

    # Optimisation avec Optuna
    st.sidebar.header("Optimisation des paramètres")
    n_trials = st.sidebar.slider("Nombre d'essais pour Optuna", 10, 100, 20)

    def objective(trial):
        max_depth = trial.suggest_int("max_depth", 5, 20)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])

        model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            criterion=criterion,
            random_state=42
        )
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        return f1_score(y_test, y_pred)

    with st.spinner("Optimisation en cours..."):
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

    st.success("Optimisation terminée !")
    st.write("Meilleurs paramètres :", study.best_params)

    # Modèle avec paramètres optimisés
    optimized_model = DecisionTreeClassifier(**study.best_params, random_state=42)
    optimized_model.fit(x_train, y_train)
    y_pred_optimized = optimized_model.predict(x_test)

    # Calcul des métriques pour le modèle optimisé
    optimized_precision = precision_score(y_test, y_pred_optimized)
    optimized_recall = recall_score(y_test, y_pred_optimized)
    optimized_f1 = f1_score(y_test, y_pred_optimized)
    optimized_conf_matrix = confusion_matrix(y_test, y_pred_optimized)

    # Affichage des résultats
    st.subheader("Performance du modèle")
    results = pd.DataFrame({
        "Métrique": ["Precision", "Recall", "F1-Score"],
        "Valeur": [optimized_precision, optimized_recall, optimized_f1]
    })
    st.write(results)

    # Section de prédiction
    if prediction_file is not None:
        st.subheader("Analyse du nouveau fichier")
        try:
            new_data = pd.read_csv(prediction_file)
            prediction = optimized_model.predict(new_data)
            
            # Affichage des résultats de prédiction
            for i, pred in enumerate(prediction):
                if pred == 1:
                    st.error(f"⚠️ Fichier {i+1} : Malware détecté !")
                else:
                    st.success(f"✅ Fichier {i+1} : Fichier légitime")
            
            # Statistiques globales
            total_files = len(prediction)
            malware_count = sum(prediction)
            legitimate_count = total_files - malware_count
            
            st.write(f"Résumé de l'analyse :")
            st.write(f"- Nombre total de fichiers analysés : {total_files}")
            st.write(f"- Nombre de malwares détectés : {malware_count}")
            st.write(f"- Nombre de fichiers légitimes : {legitimate_count}")
            
        except Exception as e:
            st.error(f"Erreur lors de l'analyse du fichier : {str(e)}")

else:
    st.warning("Veuillez d'abord charger un dataset d'entraînement.")
