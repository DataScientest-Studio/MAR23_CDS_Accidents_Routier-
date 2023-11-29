import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, plot_precision_recall_curve, plot_confusion_matrix

# Importation du dataset
@st.cache_data
def load_data():
    data = pd.read_csv('C:/Users/LENOVO/Documents/GitHub/thales/fusion3.csv', sep=';', low_memory=False)
    #data = 'C:/Users/LENOVO/Documents/GitHub/thales/X'
    return data

def main():
    st.title("App. ML pour la Prediction de la Gravité des Accidents Routiers en France de 2005 - 2020")
    st.subheader("Auteurs: Pilon C., Fadimatou A., Maillard S., Levra C.,Tall A.")
    st.write("------------------------------")
    # Load data
    data = load_data()
    data_head = data.head(10)

    # Display the DataFrame
    if st.sidebar.checkbox("Affichage du dataset...", False):
        st.write(data_head)
        st.subheader("Jeu de données nettoyés et encodées des accidents routiers de 2005 -2020")

    seed = 123
    # Séparation des données en train et test
    def split(dataframe):
        y = dataframe['grav']
        X = dataframe.drop('grav', axis=1)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            stratify=y,
            random_state=seed
        )
        return X_train, X_test, y_train, y_test

    X_train, X_test, y_train, y_test = split(dataframe=data) 

    # Analyse Performance des modèles
    def plot_perf(graphes):
        if 'Confusion Matrix' in graphes:
            st.subheader("Matrice de Confusion")
            plot_confusion_matrix(model_rf, X_test, y_test)
            st.pyplot(clear_figure=True)
        if 'ROC Curve' in graphes:
            st.subheader("Courbe ROC")
            plot_roc_curve(model_rf, X_test, y_test)
            st.pyplot(clear_figure=True)
        if 'Précision-Recall Curve' in graphes:
            st.subheader("Courbe Précision-Recall")
            plot_precision_recall_curve(model_rf, X_test, y_test)
            st.pyplot(clear_figure=True)

    classifier = st.sidebar.selectbox("Classificateur", ("Modèles", "Logistic Regression", "XGBoost", "Random Forest"))

    if classifier == "Logistic Regression":
        # Modèle Logistique regresseur
        st.sidebar.subheader("Hyperparamètres du modèle LR")
        penalty = st.sidebar.number_input("La norme de pénalité à applique", 8, 20, step=1)
        C = st.sidebar.number_input("Inverse de la force de régularisation", 0, 5, step=1)
        max_iter = st.sidebar.number_input("Nombre maximal d'itérations pour les solveurs à converger)", 100, 200, step=10)
        
                
        # Choix du graphe de performance du modèle ML
        graphes_perf = st.sidebar.multiselect("Choisir le graphe de performance du modèle ML", 
                                              ("Confusion Matrix", "ROC Curve", "Précision-Recall Curve"))
        
        if st.sidebar.button("Exécution", key=classifier):
            st.subheader("Résultats du modèle LogicRegression")
            
            # Initialiser un object LogicRegression
            model_lr = LogisticRegression(
                penalty=penalty,
                C=C,
                max_iter=max_iter,
                fit_intercept=fit_intercept
            )
            # Entrainement du modèle LogicRegression
            model_lr.fit(X_train, y_train) 

            # Prédiction du modèle LogicRegression
            y_pred = model_lr.predict(X_test) 

            # Métriques d'évaluation de la performnce du modèle LogicRegression
            accuracy = model_lr.score(X_test, y_test) 
            precision = precision_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)

            # Affichage des métriques d'évaluation de la performnce du modèle LogicRegression
            st.write("Accuracy LR", accuracy, round(3)) 
            st.write("Précision LR", precision, round(3))
            st.write("F1 Score LR", f1, round(3))
            st.write("Recall LR", recall, round(3))

    elif classifier == "XGBoost":
        # Modèle XGBoost
        st.sidebar.subheader("Hyperparamètres du modèle XGB")
        n_estimators = st.sidebar.number_input("Choisir le Nombre d'arbres à créer dans le processus d'ensemble", 20, 200, step=2)
        learning_rate = st.sidebar.number_input("Taux d'apprentissage utilisé pour réduire la contribution de chaque arbre", 0.1, 0.9, step=0.1)
        max_depth = st.sidebar.number_input("La profondeur maximale de chaque arbre", 1, 20, step=1)

        # Choix du graphe de performance du modèle ML
        graphes_perf = st.sidebar.multiselect("Choisir le graphe de performance du modèle ML", 
                                              ("Confusion Matrix", "ROC Curve", "Précision-Recall Curve"))
                                         
        if st.sidebar.button("Exécution", key=classifier):
            st.subheader("Résultats du modèle XGBClassifier")
        
            # Initialiser algo XGBOOST Classifier
            model_xgb = GradientBoostingClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth, 
                learning_rate=learning_rate
            )
            # Entrainement du modèle XGB
            model_xgb.fit(X_train, y_train) 

            # Prédiction du modèle XGB
            y_pred = model_xgb.predict(X_test) 

            # Métriques d'évaluation de la performnce du modèle XGB
            accuracy = model_xgb.score(X_test, y_test) 
            precision = precision_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)

            # Affichage des métriques d'évaluation de la performnce du modèle XGB
            st.write("Accuracy XGB", accuracy, round(3)) 
            st.write("Précision XGB ", precision, round(3))
            st.write("F1 Score XGB", f1, round(3))
            st.write("Recall XGB", recall, round(3))

    elif classifier == "Random Forest":
        # Modèle Random Forest 
        st.sidebar.subheader("Hyperparamètres du modèle RF")
        n_estimators = st.sidebar.number_input("Choisir le Nombre d'arbres dans la forêt", 20, 200, step=2)
        max_depth = st.sidebar.number_input("Choisir la Profondeur maximale d'un arbre (max_depth)", 2, 100, step=2)
        min_samples_split = st.sidebar.number_input("Nombre minimum d'échantillons valable (min_sample_split)", 1, 20, step=1)
        min_samples_leaf = st.sidebar.number_input("Nombre minimum d'échantillons acceptable (min_samples_leaf)", 1, 20, step=1)
                
        # Choix du graphe de performance du modèle ML
        graphes_perf = st.sidebar.multiselect("Choisir le graphe de performance du modèle ML", 
                                              ("Confusion Matrix", "ROC Curve", "Précision-Recall Curve"))
        
        if st.sidebar.button("Exécution", key=classifier):
            st.subheader("Résultats du modèle Random Forest")
            
            # Initialiser un object RandomForestClassifier
            model_rf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth, 
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf
            )
            # Entrainement du modèle RandomForest
            model_rf.fit(X_train, y_train) 

            # Prédiction du modèle RandomForest
            y_pred = model_rf.predict(X_test) 

            # Métriques d'évaluation de la performnce du modèle RandomForest
            accuracy = model_rf.score(X_test, y_test) 
            precision = precision_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)

            # Affichage des métriques d'évaluation de la performnce du modèle RandomForest
            st.write("Accuracy RF", accuracy, round(3)) 
            st.write("Précision RF", precision, round(3))
            st.write("F1 Score RF", f1, round(3))
            st.write("Recall RF", recall, round(3))
          
            # Affichage des graphes des performances
            plot_perf(graphes_perf)

if __name__ == '__main__':
    main()
