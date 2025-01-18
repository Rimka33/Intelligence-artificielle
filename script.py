import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
import optuna

st.title("Malware Detection Optimization")

# Chargement des données
st.sidebar.header("Dataset Upload")

# Option de chargement d'un fichier par défaut
default_file_path = "C:/Users/DELL/PycharmProjects/malware/DatasetmalwareExtrait.csv"  # Chemin par défaut du fichier
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
    st.write("Aperçu des données :", dataset.head())

    # Séparation des features et des labels
    x = dataset.drop(columns=["legitimate"])
    y = dataset["legitimate"]

    # Division des données
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=42)

    # Modèle sans optimisation
    baseline_model = DecisionTreeClassifier(random_state=42)
    baseline_model.fit(x_train, y_train)
    baseline_y_pred = baseline_model.predict(x_test)

    # Calcul des métriques pour le modèle sans optimisation
    baseline_precision = precision_score(y_test, baseline_y_pred)
    baseline_recall = recall_score(y_test, baseline_y_pred)
    baseline_f1 = f1_score(y_test, baseline_y_pred)
    baseline_conf_matrix = confusion_matrix(y_test, baseline_y_pred)

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

    st.write("Running Optuna optimization...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

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

    # Tableau de comparaison des résultats
    results = pd.DataFrame({
        "Modèle": ["Sans optimisation", "Avec optimisation"],
        "Precision": [baseline_precision, optimized_precision],
        "Recall": [baseline_recall, optimized_recall],
        "F1-Score": [baseline_f1, optimized_f1]
    })

    st.write("Comparaison des modèles :")
    st.write(results)

    # Affichage des matrices de confusion
    st.write("Matrice de confusion - Modèle sans optimisation :")
    st.text(baseline_conf_matrix)

    st.write("Matrice de confusion - Modèle avec optimisation :")
    st.text(optimized_conf_matrix)

else:
    st.warning("Aucun dataset disponible pour le traitement.")
