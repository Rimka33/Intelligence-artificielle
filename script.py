import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
import optuna
import os

st.title("Détection de Malware")

st.sidebar.header("Configuration")

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

if dataset is not None:
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

    with st.spinner("Entraînement du modèle en cours..."):
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

    # Modèle avec paramètres optimisés
    optimized_model = DecisionTreeClassifier(**study.best_params, random_state=42)
    optimized_model.fit(x_train, y_train)

    # Section pour la prédiction
    st.header("Analyse de fichier")
    new_file = st.file_uploader("Upload fichier à analyser", type=["csv"])
    
    if new_file is not None:
        try:
            new_data = pd.read_csv(new_file)
            prediction = optimized_model.predict(new_data)
            
            st.write("---")
            if prediction[0] == 1:
                st.error("⚠️ MALWARE DÉTECTÉ !")
                st.write("Ce fichier présente des caractéristiques malveillantes.")
            else:
                st.success("✅ FICHIER LÉGITIME")
                st.write("Ce fichier ne présente pas de caractéristiques malveillantes.")
            
        except Exception as e:
            st.error(f"Erreur lors de l'analyse : {str(e)}")

else:
    st.warning("Veuillez d'abord charger un dataset d'entraînement.")
