import streamlit as st
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
import google.generativeai as genai
from io import StringIO
import pypdf
from sqlalchemy import create_engine, text
import pymongo
from neo4j import GraphDatabase
import chromadb
import duckdb
from sklearn.preprocessing import LabelEncoder
import plotly.express as px

# --- Configuration de la Page et de l'API ---
st.set_page_config(layout="wide", page_title="Chatbot Analytique Ultime")

# Charger la clé API Google Gemini
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configurer l'API
try:
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
    else:
        st.error("Clé API Google non trouvée. Veuillez la définir dans votre fichier .env.")
        st.stop()
except Exception as e:
    st.error(f"Erreur de configuration Gemini : {e}")
    st.stop()

# --- Initialisation de l'état de la session ---
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.df_original = None
    st.session_state.df_prepared = None
    st.session_state.chat_history = []
    st.session_state.show_chat_modal = False
    st.session_state.show_reset_modal = False
    st.session_state.selected_model = 'gemini-1.5-flash' # Modèle par défaut

# --- Fonctions Utilitaires ---
def notify(message, type='success'):
    """Affiche une notification toast."""
    icons = {'success': '✅', 'error': '❌', 'warning': '⚠️', 'info': 'ℹ️'}
    st.toast(f'{icons.get(type, "✅")} {message}')

# --- Logique du Chatbot (Helper Functions) ---
def get_df_schema(df):
    """Génère une description textuelle du schéma du DataFrame pour le prompt de l'IA."""
    schema_buffer = StringIO()
    df.info(buf=schema_buffer)
    schema_info = schema_buffer.getvalue()
    head_str = df.head().to_string()
    
    return f"""
    Informations sur le DataFrame (df.info()):
    {schema_info}
    
    Début du DataFrame (df.head()):
    {head_str}
    """

def handle_chat_query(user_query, df, model_name):
    """Génère et exécute du code Python pour répondre à une question utilisateur en utilisant le modèle choisi."""
    if not GOOGLE_API_KEY:
        return "Erreur : La clé API Google Gemini n'est pas configurée."
    
    try:
        model = genai.GenerativeModel(model_name)
    except Exception as e:
        return f"Erreur lors de la sélection du modèle '{model_name}': {e}"
        
    df_schema = get_df_schema(df)
    
    prompt = f"""
    Rôle: Tu es un data scientist expert en Python, spécialisé dans les bibliothèques Pandas et Plotly.
    Tâche: Ta tâche est de répondre à la question de l'utilisateur en générant une unique ligne de code Python exécutable qui opère sur un DataFrame pandas nommé `df`.
    
    Contexte:
    1. Le DataFrame est déjà chargé et disponible dans une variable nommée `df`.
    2. Le code que tu génères sera exécuté directement avec `eval()`. Il doit donc retourner un résultat affichable (valeur, DataFrame, figure Plotly).
    3. Les bibliothèques `pandas as pd`, `plotly.express as px` et `numpy as np` sont déjà importées et disponibles.
    
    Règles de génération de code:
    - **NE PAS** inclure `df = ...` ou `print(...)`. Le code doit juste être l'expression à évaluer.
    - Pour une visualisation, génère du code utilisant `plotly.express as px` et retourne l'objet figure. Exemple: `px.scatter(df, x='col1', y='col2')`.
    - Pour une réponse textuelle ou un calcul (moyenne, somme, etc.), retourne directement la valeur. Exemple: `df['col1'].mean()`.
    - Pour une réponse sous forme de tableau, retourne le DataFrame ou la Série filtrée/groupée. Exemple: `df[df['col1'] > 10]`.
    - Si la question est une salutation, une question générale non liée aux données ou ne peut être répondue par du code, réponds directement en texte simple sans générer de code. Commence ta réponse par "TEXT:". Exemple: "TEXT: Bonjour! Comment puis-je vous aider avec vos données?"
    - Si tu ne peux absolument pas répondre à la question avec le contexte fourni, retourne le texte "I CAN'T ANSWER".

    Informations sur le DataFrame disponible:
    {df_schema}

    ---
    Question de l'utilisateur:
    "{user_query}"
    ---

    Code Python à générer (une seule ligne) :
    """
    
    generated_code = ""
    try:
        response = model.generate_content(prompt)
        generated_code = response.text.strip()
        
        if generated_code.startswith("```python"): generated_code = generated_code[9:]
        if generated_code.startswith("```"): generated_code = generated_code[3:]
        if generated_code.endswith("```"): generated_code = generated_code[:-3]
        generated_code = generated_code.strip()

        if generated_code == "I CAN'T ANSWER":
             return "Désolé, je ne peux pas répondre à cette question avec les données fournies."
        
        if generated_code.startswith("TEXT:"):
            return generated_code[5:].strip()

        result = eval(generated_code, {"df": df, "pd": pd, "px": px, "np": np})
        return result

    except Exception as e:
        error_message = f"""**Erreur lors de l'exécution du code.**
        **Modèle utilisé :** `{model_name}`
        **Code généré :**
        ```python
        {generated_code}
        ```
        **Erreur Python :**
        `{e}`
        """
        return error_message

# --- Fonctions de Chargement (inchangées) ---
def load_from_file(uploaded_file):
    try:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        if file_extension == '.csv': return pd.read_csv(uploaded_file)
        if file_extension in ['.xlsx', '.xls']: return pd.read_excel(uploaded_file)
        if file_extension == '.json': return pd.read_json(uploaded_file)
        if file_extension == '.parquet': return pd.read_parquet(uploaded_file)
        if file_extension == '.html': return pd.read_html(uploaded_file)[0]
        if file_extension == '.xml': return pd.read_xml(uploaded_file)
        if file_extension == '.txt': return pd.read_csv(uploaded_file, delimiter='\t')
        if file_extension == '.pdf':
            reader = pypdf.PdfReader(uploaded_file)
            text_content = "".join(page.extract_text() for page in reader.pages)
            return pd.read_csv(StringIO(text_content))
        if file_extension == '.sql':
            sql_script = uploaded_file.read().decode('utf-8')
            with duckdb.connect(database=':memory:') as conn:
                conn.execute(sql_script)
                tables = conn.execute("SHOW TABLES;").fetchdf()
                if not tables.empty:
                    return conn.execute(f"SELECT * FROM {tables['name'][0]};").fetchdf()
                else:
                    st.error("Aucune table créée par le script SQL.")
                    return None
        notify(f"Type de fichier non supporté : {file_extension}", "error")
        return None
    except Exception as e:
        st.error(f"Erreur de lecture du fichier : {e}")
        return None

def load_from_db(db_type, conn_params, query):
    try:
        if db_type in ["PostgreSQL", "MySQL", "SQL Server", "Oracle"]:
            dialect_map = {"PostgreSQL": "postgresql+psycopg2", "MySQL": "mysql+mysqlconnector", "SQL Server": "mssql+pyodbc", "Oracle": "oracle+cx_oracle"}
            conn_str = f"{dialect_map[db_type]}://{conn_params['user']}:{conn_params['password']}@{conn_params['host']}:{conn_params['port']}/{conn_params['dbname']}"
            engine = create_engine(conn_str)
            with engine.connect() as connection:
                return pd.read_sql(text(query), connection)
        elif db_type == "SQLite":
            engine = create_engine(f"sqlite:///{conn_params['path']}")
            with engine.connect() as connection:
                return pd.read_sql(text(query), connection)
        elif db_type == "MongoDB":
            conn_str = f"mongodb://{conn_params.get('user', '')}:{conn_params.get('password', '')}@{conn_params['host']}:{conn_params['port']}/".replace(":@", "")
            client = pymongo.MongoClient(conn_str)
            db_name, collection_name = query.split('.')
            return pd.DataFrame(list(client[db_name][collection_name].find()))
        elif db_type == "Neo4j":
            driver = GraphDatabase.driver(conn_params['uri'], auth=(conn_params['user'], conn_params['password']))
            with driver.session() as session:
                return pd.DataFrame([r.data() for r in session.run(query)])
        elif db_type == "ChromaDB":
            client = chromadb.PersistentClient(path=conn_params['path'])
            collection = client.get_collection(name=query)
            data = collection.get()
            df = pd.DataFrame(data['metadatas'])
            df['document'] = data['documents']
            df['id'] = data['ids']
            return df
    except Exception as e:
        st.error(f"Erreur de connexion/requête : {e}")
        return None

# --- Fonctions de Nettoyage (inchangées) ---
def remove_outliers_iqr(df, column):
    if pd.api.types.is_numeric_dtype(df[column]):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

# --- Interface Utilisateur (UI) ---
st.title("🤖 Chatbot Analytique Ultime")
st.markdown("Chargez vos données, préparez-les, puis discutez avec un analyste IA pour en extraire des insights.")

# --- BARRE LATERALE ---
with st.sidebar:
    st.header("1. Source des Données")
    source_type = st.radio("Type de source", ["Fichier Local", "Base de Données"], label_visibility="collapsed")

    if source_type == "Fichier Local":
        uploaded_file = st.file_uploader("Choisissez un fichier", type=["csv", "xlsx", "xls", "json", "parquet", "html", "xml", "pdf", "txt", "sql"])
        if uploaded_file and st.button("Charger les données", use_container_width=True, type="primary"):
            with st.spinner("Chargement..."):
                df = load_from_file(uploaded_file)
                if df is not None:
                    st.session_state.df_original = df.copy()
                    st.session_state.df_prepared = df.copy()
                    st.session_state.data_loaded = True
                    st.session_state.chat_history = []
                    notify("Données chargées avec succès !")
    else: # Base de Données
        db_type = st.selectbox("Type de BD", ["PostgreSQL", "MySQL", "SQLite", "MongoDB", "Neo4j", "ChromaDB", "SQL Server", "Oracle"])
        conn_params = {}
        query = ""

        if db_type in ["PostgreSQL", "MySQL", "SQL Server", "Oracle"]:
            col1, col2 = st.columns(2); conn_params['host'] = col1.text_input("Hôte", "localhost"); conn_params['port'] = col2.text_input("Port", {"PostgreSQL": "5432", "MySQL": "3306"}.get(db_type, "1521")); conn_params['dbname'] = st.text_input("Nom de la base"); col3, col4 = st.columns(2); conn_params['user'] = col3.text_input("Utilisateur"); conn_params['password'] = col4.text_input("Mot de passe", type="password"); query = st.text_area("Requête SQL", "SELECT * FROM ...")
        elif db_type == "SQLite":
            conn_params['path'] = st.text_input("Chemin du fichier .db", "path/to/db.sqlite"); query = st.text_area("Requête SQL", "SELECT * FROM ...")
        elif db_type == "MongoDB":
            col1, col2 = st.columns(2); conn_params['host'] = col1.text_input("Hôte", "localhost"); conn_params['port'] = col2.text_input("Port", "27017"); col3, col4 = st.columns(2); conn_params['user'] = col3.text_input("Utilisateur (opt.)"); conn_params['password'] = col4.text_input("Mot de passe (opt.)", type="password"); query = st.text_input("Base.Collection", "ma_base.ma_collection")
        elif db_type == "Neo4j":
            conn_params['uri'] = st.text_input("URI (bolt://...)", "bolt://localhost:7687"); col1, col2 = st.columns(2); conn_params['user'] = col1.text_input("Utilisateur", "neo4j"); conn_params['password'] = col2.text_input("Mot de passe", type="password"); query = st.text_area("Requête Cypher", "MATCH (n) RETURN n LIMIT 25")
        elif db_type == "ChromaDB":
            conn_params['path'] = st.text_input("Chemin du dossier de la BD", "/path/to/chroma"); query = st.text_input("Nom de la collection")

        if st.button("Charger depuis la BD", use_container_width=True, type="primary"):
            if all(v for k, v in conn_params.items() if k not in ['user', 'password']) and query:
                with st.spinner("Connexion..."):
                    df = load_from_db(db_type, conn_params, query)
                    if df is not None:
                        st.session_state.df_original, st.session_state.df_prepared = df.copy(), df.copy(); st.session_state.data_loaded = True; st.session_state.chat_history = []; notify("Données chargées depuis la BD !")
            else:
                notify("Veuillez remplir tous les champs requis.", "warning")
    
    st.divider()
    if st.button("💥 Réinitialiser l'Application", use_container_width=True):
        st.session_state.show_reset_modal = True

# --- MODALE DE CONFIRMATION DE RÉINITIALISATION ---
@st.dialog("Confirmation de Réinitialisation")
def run_reset_confirmation_modal():
    st.warning("Êtes-vous sûr de vouloir réinitialiser l'application ?\nToutes les données chargées, les préparations et l'historique du chat seront perdus.")
    c1, c2 = st.columns(2)
    if c1.button("Oui, Réinitialiser", use_container_width=True, type="primary"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.session_state.show_reset_modal = False
        notify("Application réinitialisée.")
        st.rerun()
    if c2.button("Annuler", use_container_width=True):
        st.session_state.show_reset_modal = False
        st.rerun()

if st.session_state.get('show_reset_modal', False):
    run_reset_confirmation_modal()

# --- ZONE PRINCIPALE ---
if st.session_state.data_loaded:
    st.header("2. Exploration et Préparation")
    data_to_show = st.radio("Afficher :", ["Données Originales", "Données Préparées"], horizontal=True, key="data_view_select")
    df_display = st.session_state.df_original if data_to_show == "Données Originales" else st.session_state.df_prepared
    st.markdown(f"**Dimensions :** `{df_display.shape[0]}` lignes, `{df_display.shape[1]}` colonnes.")
    st.dataframe(df_display, use_container_width=True)

    # --- SECTION CORRIGÉE ---
    with st.expander("🛠️ Outils de Préparation et Nettoyage (modifie les 'Données Préparées')"):
        tabs = st.tabs(["Manquants", "Types", "Colonnes", "Valeurs", "Encodage", "Outliers"])
        df_prep = st.session_state.df_prepared

        with tabs[0]: # Gérer les valeurs manquantes
            col, strat = st.columns(2)
            col_missing = col.selectbox("Colonne", df_prep.columns, key="missing_col")
            strategy = strat.selectbox("Stratégie", ["Supprimer les lignes", "Remplir avec une valeur"], key="missing_strat")
            fill_val = st.text_input("Valeur de remplacement", "") if strategy == "Remplir avec une valeur" else None
            if st.button("Appliquer Manquants"):
                if strategy == 'Supprimer les lignes':
                    df_prep.dropna(subset=[col_missing], inplace=True)
                else:
                    df_prep[col_missing].fillna(fill_val, inplace=True)
                notify("Valeurs manquantes traitées.")
                st.rerun()

        with tabs[1]: # Changer le type
            col, typ = st.columns(2)
            col_type = col.selectbox("Colonne", df_prep.columns, key="type_col")
            new_type = typ.selectbox("Nouveau type", ["object", "int64", "float64", "datetime64[ns]"], key="type_new")
            if st.button("Appliquer Type"):
                try:
                    if 'int' in new_type or 'float' in new_type:
                        df_prep[col_type] = pd.to_numeric(df_prep[col_type], errors='coerce')
                    else:
                        df_prep[col_type] = df_prep[col_type].astype(new_type)
                    notify(f"Type de '{col_type}' changé.")
                    st.rerun()
                except Exception as e:
                    notify(f"Erreur de conversion : {e}", "error")

        with tabs[2]: # Gérer les colonnes
            st.subheader("Supprimer des colonnes")
            cols_to_drop = st.multiselect("Colonnes à supprimer", options=df_prep.columns, key="drop_cols")
            if st.button("Supprimer Colonnes"):
                df_prep.drop(columns=cols_to_drop, inplace=True)
                notify("Colonnes supprimées.")
                st.rerun()
            st.subheader("Renommer une colonne")
            col1, col2 = st.columns(2)
            col_to_rename = col1.selectbox("Colonne à renommer", options=df_prep.columns, key="rename_select")
            new_name = col2.text_input("Nouveau nom", value=col_to_rename)
            if st.button("Renommer Colonne"):
                df_prep.rename(columns={col_to_rename: new_name}, inplace=True)
                notify("Colonne renommée.")
                st.rerun()
        
        with tabs[3]: # Remplacer des valeurs
            col, old, new = st.columns(3)
            col_replace = col.selectbox("Colonne", df_prep.columns, key="replace_col")
            old_val = old.text_input("Valeur à remplacer")
            new_val = new.text_input("Nouvelle valeur")
            if st.button("Remplacer Valeur"):
                df_prep[col_replace].replace(old_val, new_val, inplace=True)
                notify("Valeur remplacée.")
                st.rerun()

        with tabs[4]: # Encodage
            cat_cols = df_prep.select_dtypes(include=['object', 'category']).columns
            if len(cat_cols) > 0:
                col_to_encode = st.selectbox("Colonne catégorielle à encoder", options=cat_cols, key="encode_select")
                method = st.radio("Méthode", ["Label Encoding", "One-Hot Encoding"], horizontal=True)
                if st.button("Encoder Colonne"):
                    if method == "Label Encoding":
                        le = LabelEncoder()
                        df_prep[col_to_encode] = le.fit_transform(df_prep[col_to_encode])
                    else: # One-Hot
                        dummies = pd.get_dummies(df_prep[col_to_encode], prefix=col_to_encode, drop_first=True)
                        st.session_state.df_prepared = pd.concat([df_prep.drop(col_to_encode, axis=1), dummies], axis=1)
                    notify("Encodage effectué.")
                    st.rerun()
            else: st.info("Aucune colonne catégorielle détectée.")

        with tabs[5]: # Outliers
            num_cols = df_prep.select_dtypes(include=np.number).columns
            if len(num_cols) > 0:
                col_outliers = st.selectbox("Colonne numérique", options=num_cols, key="outlier_select")
                if st.button("Supprimer Outliers (IQR)"):
                    rows_before = len(df_prep)
                    st.session_state.df_prepared = remove_outliers_iqr(df_prep, col_outliers)
                    rows_after = len(st.session_state.df_prepared)
                    notify(f"{rows_before - rows_after} outliers supprimés.")
                    st.rerun()
            else: st.info("Aucune colonne numérique détectée.")

    st.header("3. Chatbot Analytique")
    if st.button("💬 Ouvrir le Chatbot", use_container_width=True):
        st.session_state.show_chat_modal = True

    @st.dialog("Chatbot Analytique Expert", width="large")
    def run_chatbot_modal():
        st.info("Vous analysez les **Données Préparées**. L'IA génère du code Python pour répondre.")
        
        model_options = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro']
        selected_model = st.selectbox(
            "Choisissez le modèle Gemini :",
            options=model_options,
            index=model_options.index(st.session_state.get('selected_model', 'gemini-1.5-flash')),
            key='selected_model'
        )
        st.divider()

        chat_container = st.container(height=400)
        with chat_container:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    content = message["content"]
                    if isinstance(content, pd.DataFrame): st.dataframe(content)
                    elif isinstance(content, pd.Series): st.dataframe(content)
                    elif 'plotly.graph_objs._figure.Figure' in str(type(content)): st.plotly_chart(content, use_container_width=True)
                    else: st.markdown(content)

        if prompt := st.chat_input("Ex: 'Montre-moi un histogramme de la colonne age'"):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            st.rerun()

        if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
            last_prompt = st.session_state.chat_history[-1]["content"]
            with st.spinner(f"Le modèle '{st.session_state.selected_model}' génère et exécute le code..."):
                response_content = handle_chat_query(last_prompt, st.session_state.df_prepared, st.session_state.selected_model)
                st.session_state.chat_history.append({"role": "assistant", "content": response_content})
                st.rerun()
        
        c1, c2 = st.columns([1, 1])
        if c1.button("🗑️ Nettoyer l'historique", use_container_width=True):
            st.session_state.chat_history = []; notify("Historique du chat effacé."); st.rerun()
        if c2.button("Fermer le Chat", use_container_width=True, type="primary"):
            st.session_state.show_chat_modal = False; st.rerun()

    if st.session_state.get('show_chat_modal', False):
        run_chatbot_modal()
else:
    st.info("👈 Commencez par charger un jeu de données via la barre latérale.")