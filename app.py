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

# --- Dictionnaire de Traductions (i18n) ---
translations = {
    "fr": {
        "page_title": "Chatbot Analytique Ultime",
        "lang_selector": "Langue",
        "sidebar": {
            "data_source_header": "1. Source des Données",
            "source_type": "Type de source",
            "local_file": "Fichier Local",
            "database": "Base de Données",
            "file_uploader_label": "Choisissez un fichier",
            "load_button": "Charger les données",
            "loading_spinner": "Chargement...",
            "db_type_label": "Type de BD",
            "db_host": "Hôte", "db_port": "Port", "db_name": "Nom de la base", "db_user": "Utilisateur", "db_password": "Mot de passe",
            "db_path": "Chemin du fichier .db", "db_sql_query": "Requête SQL", "db_mongo_conn": "Base.Collection",
            "db_neo4j_uri": "URI (bolt://...)", "db_cypher_query": "Requête Cypher",
            "db_chroma_path": "Chemin du dossier de la BD", "db_chroma_collection": "Nom de la collection",
            "load_from_db_button": "Charger depuis la BD",
            "db_connecting_spinner": "Connexion...",
            "db_fields_required_warning": "Veuillez remplir tous les champs requis.",
            "reset_app_button": "💥 Réinitialiser l'Application",
        },
        "reset_modal": {
            "title": "Confirmation de Réinitialisation",
            "warning": "Êtes-vous sûr de vouloir réinitialiser l'application ?\nToutes les données chargées, les préparations et l'historique du chat seront perdus.",
            "confirm_button": "Oui, Réinitialiser",
            "cancel_button": "Annuler"
        },
        "main": {
            "title": "🤖 Chatbot Analytique Ultime",
            "subtitle": "Chargez vos données, préparez-les, puis discutez avec un analyste IA pour en extraire des insights.",
            "info_start": "👈 Commencez par charger un jeu de données via la barre latérale.",
            "header_preparation": "2. Exploration et Préparation",
            "radio_show_data": "Afficher :",
            "original_data": "Données Originales",
            "prepared_data": "Données Préparées",
            "dimensions": "Dimensions", "rows": "lignes", "cols": "colonnes",
            "prep_tools_expander": "🛠️ Outils de Préparation et Nettoyage (modifie les 'Données Préparées')",
            "header_chatbot": "3. Chatbot Analytique",
            "open_chatbot_button": "💬 Ouvrir le Chatbot",
        },
        "prep_tabs": {
            "missing": "Manquants", "types": "Types", "columns": "Colonnes", "values": "Valeurs", "encoding": "Encodage", "outliers": "Outliers",
            "column": "Colonne", "strategy": "Stratégie", "del_rows": "Supprimer les lignes", "fill_value": "Remplir avec une valeur",
            "fill_input_label": "Valeur de remplacement", "apply_button": "Appliquer",
            "new_type": "Nouveau type",
            "drop_cols_header": "Supprimer des colonnes", "cols_to_drop": "Colonnes à supprimer", "drop_cols_button": "Supprimer Colonnes",
            "rename_col_header": "Renommer une colonne", "col_to_rename": "Colonne à renommer", "new_name": "Nouveau nom", "rename_col_button": "Renommer Colonne",
            "old_value": "Valeur à remplacer", "new_value": "Nouvelle valeur", "replace_val_button": "Remplacer Valeur",
            "categorical_col_label": "Colonne catégorielle à encoder", "encoding_method": "Méthode", "encode_button": "Encoder Colonne",
            "no_categorical_cols": "Aucune colonne catégorielle détectée.",
            "numeric_col_label": "Colonne numérique", "remove_outliers_button": "Supprimer Outliers (IQR)",
            "no_numeric_cols": "Aucune colonne numérique détectée."
        },
        "chatbot_modal": {
            "title": "Chatbot Analytique Expert",
            "info": "Vous analysez les **Données Préparées**. L'IA génère du code Python pour répondre.",
            "model_select": "Choisissez le modèle Gemini :",
            "chat_input_placeholder": "Ex: 'Montre-moi un histogramme de la colonne age'",
            "spinner_text": "Le modèle '{model}' génère et exécute le code...",
            "clear_history_button": "🗑️ Nettoyer l'historique",
            "close_button": "Fermer le Chat",
        },
        "notifications": {
            "success": "✅ {message}", "error": "❌ {message}", "warning": "⚠️ {message}", "info": "ℹ️ {message}",
            "gemini_config_error": "Erreur de configuration Gemini : {e}",
            "api_key_not_found": "Clé API Google non trouvée. Veuillez la définir dans votre fichier .env.",
            "unsupported_file": "Type de fichier non supporté : {ext}",
            "file_read_error": "Erreur de lecture du fichier : {e}",
            "sql_no_table": "Aucune table créée par le script SQL.",
            "db_conn_error": "Erreur de connexion/requête : {e}",
            "data_loaded_success": "Données chargées avec succès !",
            "data_loaded_from_db": "Données chargées depuis la BD !",
            "app_reset": "Application réinitialisée.",
            "missing_applied": "Valeurs manquantes traitées.",
            "type_changed": "Type de '{col}' changé.",
            "type_conversion_error": "Erreur de conversion : {e}",
            "cols_dropped": "Colonnes supprimées.",
            "col_renamed": "Colonne renommée.",
            "value_replaced": "Valeur remplacée.",
            "encoding_done": "Encodage effectué.",
            "outliers_removed": "{count} outliers supprimés.",
            "history_cleared": "Historique du chat effacé."
        },
        "ai": {
            "api_key_error": "Erreur : La clé API Google Gemini n'est pas configurée.",
            "model_select_error": "Erreur lors de la sélection du modèle '{model}': {e}",
            "cant_answer": "Désolé, je ne peux pas répondre à cette question avec les données fournies.",
            "exec_error": "**Erreur lors de l'exécution du code.**\n**Modèle utilisé :** `{model}`\n**Code généré :**\n```python\n{code}\n```\n**Erreur Python :**\n`{error}`",
            "prompt": """Rôle: Tu es un data scientist expert en Python, spécialisé dans les bibliothèques Pandas et Plotly.
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
        }
    },
    "en": {
        "page_title": "Ultimate Analytical Chatbot",
        "lang_selector": "Language",
        "sidebar": {
            "data_source_header": "1. Data Source",
            "source_type": "Source type",
            "local_file": "Local File",
            "database": "Database",
            "file_uploader_label": "Choose a file",
            "load_button": "Load Data",
            "loading_spinner": "Loading...",
            "db_type_label": "DB Type",
            "db_host": "Host", "db_port": "Port", "db_name": "Database Name", "db_user": "User", "db_password": "Password",
            "db_path": ".db file path", "db_sql_query": "SQL Query", "db_mongo_conn": "Database.Collection",
            "db_neo4j_uri": "URI (bolt://...)", "db_cypher_query": "Cypher Query",
            "db_chroma_path": "DB folder path", "db_chroma_collection": "Collection Name",
            "load_from_db_button": "Load from DB",
            "db_connecting_spinner": "Connecting...",
            "db_fields_required_warning": "Please fill in all required fields.",
            "reset_app_button": "💥 Reset Application",
        },
        "reset_modal": {
            "title": "Reset Confirmation",
            "warning": "Are you sure you want to reset the application?\nAll loaded data, preparations, and chat history will be lost.",
            "confirm_button": "Yes, Reset",
            "cancel_button": "Cancel"
        },
        "main": {
            "title": "🤖 Ultimate Analytical Chatbot",
            "subtitle": "Load your data, prepare it, then chat with an AI analyst to extract insights.",
            "info_start": "👈 Get started by loading a dataset from the sidebar.",
            "header_preparation": "2. Exploration and Preparation",
            "radio_show_data": "Display:",
            "original_data": "Original Data",
            "prepared_data": "Prepared Data",
            "dimensions": "Dimensions", "rows": "rows", "cols": "columns",
            "prep_tools_expander": "🛠️ Preparation and Cleaning Tools (modifies 'Prepared Data')",
            "header_chatbot": "3. Analytical Chatbot",
            "open_chatbot_button": "💬 Open Chatbot",
        },
        "prep_tabs": {
            "missing": "Missing", "types": "Types", "columns": "Columns", "values": "Values", "encoding": "Encoding", "outliers": "Outliers",
            "column": "Column", "strategy": "Strategy", "del_rows": "Delete rows", "fill_value": "Fill with a value",
            "fill_input_label": "Replacement value", "apply_button": "Apply",
            "new_type": "New type",
            "drop_cols_header": "Drop columns", "cols_to_drop": "Columns to drop", "drop_cols_button": "Drop Columns",
            "rename_col_header": "Rename a column", "col_to_rename": "Column to rename", "new_name": "New name", "rename_col_button": "Rename Column",
            "old_value": "Value to replace", "new_value": "New value", "replace_val_button": "Replace Value",
            "categorical_col_label": "Categorical column to encode", "encoding_method": "Method", "encode_button": "Encode Column",
            "no_categorical_cols": "No categorical columns detected.",
            "numeric_col_label": "Numerical column", "remove_outliers_button": "Remove Outliers (IQR)",
            "no_numeric_cols": "No numerical columns detected."
        },
        "chatbot_modal": {
            "title": "Expert Analytical Chatbot",
            "info": "You are analyzing the **Prepared Data**. The AI generates Python code to answer.",
            "model_select": "Choose the Gemini model:",
            "chat_input_placeholder": "E.g., 'Show me a histogram of the age column'",
            "spinner_text": "The '{model}' model is generating and executing the code...",
            "clear_history_button": "🗑️ Clear History",
            "close_button": "Close Chat",
        },
        "notifications": {
            "success": "✅ {message}", "error": "❌ {message}", "warning": "⚠️ {message}", "info": "ℹ️ {message}",
            "gemini_config_error": "Gemini configuration error: {e}",
            "api_key_not_found": "Google API key not found. Please set it in your .env file.",
            "unsupported_file": "Unsupported file type: {ext}",
            "file_read_error": "Error reading file: {e}",
            "sql_no_table": "No table created by the SQL script.",
            "db_conn_error": "Connection/query error: {e}",
            "data_loaded_success": "Data loaded successfully!",
            "data_loaded_from_db": "Data loaded from DB!",
            "app_reset": "Application reset.",
            "missing_applied": "Missing values handled.",
            "type_changed": "Type of '{col}' changed.",
            "type_conversion_error": "Conversion error: {e}",
            "cols_dropped": "Columns dropped.",
            "col_renamed": "Column renamed.",
            "value_replaced": "Value replaced.",
            "encoding_done": "Encoding complete.",
            "outliers_removed": "{count} outliers removed.",
            "history_cleared": "Chat history cleared."
        },
        "ai": {
            "api_key_error": "Error: Google Gemini API key is not configured.",
            "model_select_error": "Error selecting model '{model}': {e}",
            "cant_answer": "Sorry, I cannot answer this question with the provided data.",
            "exec_error": "**Error during code execution.**\n**Model used:** `{model}`\n**Generated code:**\n```python\n{code}\n```\n**Python Error:**\n`{error}`",
            "prompt": """Role: You are an expert Python data scientist, specializing in the Pandas and Plotly libraries.
Task: Your task is to answer the user's question by generating a single, executable line of Python code that operates on a pandas DataFrame named `df`.
Context:
1. The DataFrame is already loaded and available in a variable named `df`.
2. The code you generate will be executed directly with `eval()`. It must therefore return a displayable result (a value, a DataFrame, a Plotly figure).
3. The libraries `pandas as pd`, `plotly.express as px`, and `numpy as np` are already imported and available.
Code Generation Rules:
- **DO NOT** include `df = ...` or `print(...)`. The code must just be the expression to be evaluated.
- For a visualization, generate code using `plotly.express as px` and return the figure object. Example: `px.scatter(df, x='col1', y='col2')`.
- For a textual answer or a calculation (mean, sum, etc.), return the value directly. Example: `df['col1'].mean()`.
- For an answer in table form, return the filtered/grouped DataFrame or Series. Example: `df[df['col1'] > 10]`.
- If the question is a greeting, a general question not related to the data, or cannot be answered by code, respond directly in plain text without generating code. Start your response with "TEXT:". Example: "TEXT: Hello! How can I help you with your data?"
- If you absolutely cannot answer the question with the provided context, return the text "I CAN'T ANSWER".
Information about the available DataFrame:
{df_schema}
---
User's Question:
"{user_query}"
---
Python code to generate (one single line):
"""
        }
    }
}

# --- Initialisation de l'état de la session ---
if 'lang' not in st.session_state:
    st.session_state.lang = "Français"
    st.session_state.lang_code = "fr"
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.df_original = None
    st.session_state.df_prepared = None
    st.session_state.chat_history = []
    st.session_state.show_chat_modal = False
    st.session_state.show_reset_modal = False
    st.session_state.selected_model = 'gemini-1.5-flash'

# --- Fonctions Utilitaires et i18n ---
def get_translation(key):
    """Helper to get nested dictionary values."""
    try:
        keys = key.split('.')
        val = translations[st.session_state.lang_code]
        for k in keys:
            val = val[k]
        return val
    except (KeyError, AttributeError):
        # Fallback to the last part of the key if not found
        return key.split('.')[-1]

t = get_translation

def notify(message_key, type='success', **kwargs):
    """Affiche une notification toast en utilisant une clé de traduction."""
    message_template = t(f"notifications.{type}")
    message = t(f"notifications.{message_key}").format(**kwargs)
    st.toast(message_template.format(message=message))

# --- Configuration de la Page et de l'API ---
st.set_page_config(layout="wide", page_title=t("page_title"))

# Charger la clé API Google Gemini
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

try:
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
    else:
        st.error(t("notifications.api_key_not_found"))
        st.stop()
except Exception as e:
    st.error(t("notifications.gemini_config_error").format(e=e))
    st.stop()

# --- Logique du Chatbot (Helper Functions) ---
def get_df_schema(df):
    """Génère une description textuelle du schéma du DataFrame pour le prompt de l'IA."""
    schema_buffer = StringIO()
    df.info(buf=schema_buffer)
    schema_info = schema_buffer.getvalue()
    head_str = df.head().to_string()
    
    return f"DataFrame Info (df.info()):\n{schema_info}\n\nDataFrame Head (df.head()):\n{head_str}"

def handle_chat_query(user_query, df, model_name):
    if not GOOGLE_API_KEY: return t("ai.api_key_error")
    try:
        model = genai.GenerativeModel(model_name)
    except Exception as e:
        return t("ai.model_select_error").format(model=model_name, e=e)
        
    df_schema = get_df_schema(df)
    prompt = t("ai.prompt").format(df_schema=df_schema, user_query=user_query)
    
    generated_code = ""
    try:
        response = model.generate_content(prompt)
        generated_code = response.text.strip().replace("```python", "").replace("```", "").strip()

        if generated_code == "I CAN'T ANSWER":
             return t("ai.cant_answer")
        if generated_code.startswith("TEXT:"):
            return generated_code[5:].strip()

        return eval(generated_code, {"df": df, "pd": pd, "px": px, "np": np})

    except Exception as e:
        return t("ai.exec_error").format(model=model_name, code=generated_code, error=e)


# --- Fonctions de Chargement ---
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
            return pd.read_csv(StringIO("".join(page.extract_text() for page in reader.pages)))
        if file_extension == '.sql':
            sql_script = uploaded_file.read().decode('utf-8')
            with duckdb.connect(database=':memory:') as conn:
                conn.execute(sql_script)
                tables = conn.execute("SHOW TABLES;").fetchdf()
                if not tables.empty:
                    return conn.execute(f"SELECT * FROM {tables['name'][0]};").fetchdf()
                else:
                    st.error(t("notifications.sql_no_table")); return None
        notify("unsupported_file", "error", ext=file_extension)
        return None
    except Exception as e:
        st.error(t("notifications.file_read_error").format(e=e)); return None

def load_from_db(db_type, conn_params, query):
    try:
        if db_type in ["PostgreSQL", "MySQL", "SQL Server", "Oracle"]:
            dialect_map = {"PostgreSQL": "postgresql+psycopg2", "MySQL": "mysql+mysqlconnector", "SQL Server": "mssql+pyodbc", "Oracle": "oracle+cx_oracle"}
            conn_str = f"{dialect_map[db_type]}://{conn_params['user']}:{conn_params['password']}@{conn_params['host']}:{conn_params['port']}/{conn_params['dbname']}"
            engine = create_engine(conn_str)
            with engine.connect() as connection: return pd.read_sql(text(query), connection)
        elif db_type == "SQLite":
            engine = create_engine(f"sqlite:///{conn_params['path']}")
            with engine.connect() as connection: return pd.read_sql(text(query), connection)
        elif db_type == "MongoDB":
            conn_str = f"mongodb://{conn_params.get('user', '')}:{conn_params.get('password', '')}@{conn_params['host']}:{conn_params['port']}/".replace(":@", "")
            client = pymongo.MongoClient(conn_str)
            db_name, collection_name = query.split('.')
            return pd.DataFrame(list(client[db_name][collection_name].find()))
        elif db_type == "Neo4j":
            driver = GraphDatabase.driver(conn_params['uri'], auth=(conn_params['user'], conn_params['password']))
            with driver.session() as session: return pd.DataFrame([r.data() for r in session.run(query)])
        elif db_type == "ChromaDB":
            client = chromadb.PersistentClient(path=conn_params['path'])
            collection = client.get_collection(name=query)
            data = collection.get()
            df = pd.DataFrame(data['metadatas']); df['document'] = data['documents']; df['id'] = data['ids']
            return df
    except Exception as e:
        st.error(t("notifications.db_conn_error").format(e=e)); return None

# --- Fonctions de Nettoyage (inchangées) ---
def remove_outliers_iqr(df, column):
    if pd.api.types.is_numeric_dtype(df[column]):
        Q1, Q3 = df[column].quantile(0.25), df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

# --- Interface Utilisateur (UI) ---
st.title(t("main.title"))
st.markdown(t("main.subtitle"))

# --- BARRE LATERALE ---
with st.sidebar:
    lang_map = {"Français": "fr", "English": "en"}
    selected_lang_name = st.selectbox(
        t("lang_selector"), options=lang_map.keys(),
        index=list(lang_map.keys()).index(st.session_state.lang)
    )
    if st.session_state.lang != selected_lang_name:
        st.session_state.lang = selected_lang_name
        st.session_state.lang_code = lang_map[selected_lang_name]
        st.rerun()

    st.header(t("sidebar.data_source_header"))
    source_type = st.radio(t("sidebar.source_type"), [t("sidebar.local_file"), t("sidebar.database")], label_visibility="collapsed")

    if source_type == t("sidebar.local_file"):
        uploaded_file = st.file_uploader(t("sidebar.file_uploader_label"), type=["csv", "xlsx", "xls", "json", "parquet", "html", "xml", "pdf", "txt", "sql"])
        if uploaded_file and st.button(t("sidebar.load_button"), use_container_width=True, type="primary"):
            with st.spinner(t("sidebar.loading_spinner")):
                df = load_from_file(uploaded_file)
                if df is not None:
                    st.session_state.update(df_original=df.copy(), df_prepared=df.copy(), data_loaded=True, chat_history=[])
                    notify("data_loaded_success")
    else: # Database
        db_type = st.selectbox(t("sidebar.db_type_label"), ["PostgreSQL", "MySQL", "SQLite", "MongoDB", "Neo4j", "ChromaDB", "SQL Server", "Oracle"])
        conn_params, query = {}, ""
        if db_type in ["PostgreSQL", "MySQL", "SQL Server", "Oracle"]:
            col1, col2 = st.columns(2); conn_params['host'] = col1.text_input(t("sidebar.db_host"), "localhost"); conn_params['port'] = col2.text_input(t("sidebar.db_port"), {"PostgreSQL": "5432", "MySQL": "3306"}.get(db_type, "1521")); conn_params['dbname'] = st.text_input(t("sidebar.db_name")); col3, col4 = st.columns(2); conn_params['user'] = col3.text_input(t("sidebar.db_user")); conn_params['password'] = col4.text_input(t("sidebar.db_password"), type="password"); query = st.text_area(t("sidebar.db_sql_query"), "SELECT * FROM ...")
        elif db_type == "SQLite":
            conn_params['path'] = st.text_input(t("sidebar.db_path"), "path/to/db.sqlite"); query = st.text_area(t("sidebar.db_sql_query"), "SELECT * FROM ...")
        elif db_type == "MongoDB":
            col1, col2 = st.columns(2); conn_params['host'] = col1.text_input(t("sidebar.db_host"), "localhost"); conn_params['port'] = col2.text_input(t("sidebar.db_port"), "27017"); col3, col4 = st.columns(2); conn_params['user'] = col3.text_input(f"{t('sidebar.db_user')} (opt.)"); conn_params['password'] = col4.text_input(f"{t('sidebar.db_password')} (opt.)", type="password"); query = st.text_input(t("sidebar.db_mongo_conn"), "my_db.my_collection")
        elif db_type == "Neo4j":
            conn_params['uri'] = st.text_input(t("sidebar.db_neo4j_uri"), "bolt://localhost:7687"); col1, col2 = st.columns(2); conn_params['user'] = col1.text_input(t("sidebar.db_user"), "neo4j"); conn_params['password'] = col2.text_input(t("sidebar.db_password"), type="password"); query = st.text_area(t("sidebar.db_cypher_query"), "MATCH (n) RETURN n LIMIT 25")
        elif db_type == "ChromaDB":
            conn_params['path'] = st.text_input(t("sidebar.db_chroma_path"), "/path/to/chroma"); query = st.text_input(t("sidebar.db_chroma_collection"))

        if st.button(t("sidebar.load_from_db_button"), use_container_width=True, type="primary"):
            if all(v for k, v in conn_params.items() if k not in ['user', 'password']) and query:
                with st.spinner(t("sidebar.db_connecting_spinner")):
                    df = load_from_db(db_type, conn_params, query)
                    if df is not None:
                        st.session_state.update(df_original=df.copy(), df_prepared=df.copy(), data_loaded=True, chat_history=[])
                        notify("data_loaded_from_db")
            else:
                notify("db_fields_required_warning", "warning")
    
    st.divider()
    if st.button(t("sidebar.reset_app_button"), use_container_width=True):
        st.session_state.show_reset_modal = True

# --- MODALES ---

# MODALE DE CONFIRMATION DE RÉINITIALISATION
@st.dialog(t("reset_modal.title"))
def run_reset_confirmation_modal():
    st.warning(t("reset_modal.warning"))
    c1, c2 = st.columns(2)
    if c1.button(t("reset_modal.confirm_button"), use_container_width=True, type="primary"):
        keys_to_keep = ['lang', 'lang_code']
        for key in list(st.session_state.keys()):
            if key not in keys_to_keep:
                del st.session_state[key]
        st.session_state.show_reset_modal = False
        notify("app_reset")
        st.rerun()
    if c2.button(t("reset_modal.cancel_button"), use_container_width=True):
        st.session_state.show_reset_modal = False; st.rerun()

# MODALE DU CHATBOT
@st.dialog(t("chatbot_modal.title"), width="large")
def run_chatbot_modal():
    st.info(t("chatbot_modal.info"))
    model_options = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro']
    st.selectbox(t("chatbot_modal.model_select"), options=model_options, index=model_options.index(st.session_state.selected_model), key='selected_model')
    st.divider()

    with st.container(height=400):
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                content = msg["content"]
                if isinstance(content, (pd.DataFrame, pd.Series)): st.dataframe(content)
                elif 'plotly.graph_objs._figure.Figure' in str(type(content)): st.plotly_chart(content, use_container_width=True)
                else: st.markdown(content)

    if prompt := st.chat_input(t("chatbot_modal.chat_input_placeholder")):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.rerun()

    if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
        last_prompt = st.session_state.chat_history[-1]["content"]
        spinner_text = t("chatbot_modal.spinner_text").format(model=st.session_state.selected_model)
        with st.spinner(spinner_text):
            response_content = handle_chat_query(last_prompt, st.session_state.df_prepared, st.session_state.selected_model)
            st.session_state.chat_history.append({"role": "assistant", "content": response_content})
            st.rerun()
    
    c1, c2 = st.columns([1, 1])
    if c1.button(t("chatbot_modal.clear_history_button"), use_container_width=True):
        st.session_state.chat_history = []; notify("history_cleared"); st.rerun()
    if c2.button(t("chatbot_modal.close_button"), use_container_width=True, type="primary"):
        st.session_state.show_chat_modal = False; st.rerun()

# --- LOGIQUE D'AFFICHAGE DES MODALES (CORRIGÉE) ---
if st.session_state.get('show_reset_modal', False):
    run_reset_confirmation_modal()
elif st.session_state.get('show_chat_modal', False):
    run_chatbot_modal()


# --- ZONE PRINCIPALE ---
if st.session_state.data_loaded:
    st.header(t("main.header_preparation"))
    data_to_show = st.radio(t("main.radio_show_data"), [t("main.original_data"), t("main.prepared_data")], horizontal=True)
    df_display = st.session_state.df_original if data_to_show == t("main.original_data") else st.session_state.df_prepared
    st.markdown(f"**{t('main.dimensions')}:** `{df_display.shape[0]}` {t('main.rows')}, `{df_display.shape[1]}` {t('main.cols')}.")
    st.dataframe(df_display, use_container_width=True)

    with st.expander(t("main.prep_tools_expander")):
        tab_keys = ["missing", "types", "columns", "values", "encoding", "outliers"]
        tab_names = [t(f"prep_tabs.{k}") for k in tab_keys]
        tabs = st.tabs(tab_names)
        df_prep = st.session_state.df_prepared

        with tabs[0]: # Manquants
            col, strat = st.columns(2)
            col_missing = col.selectbox(t("prep_tabs.column"), df_prep.columns, key="missing_col")
            strat_opts = {t("prep_tabs.del_rows"): "del", t("prep_tabs.fill_value"): "fill"}
            strategy = strat.selectbox(t("prep_tabs.strategy"), strat_opts.keys(), key="missing_strat")
            fill_val = st.text_input(t("prep_tabs.fill_input_label"), "") if strat_opts[strategy] == "fill" else None
            if st.button(t("prep_tabs.apply_button"), key="apply_missing"):
                if strat_opts[strategy] == 'del': df_prep.dropna(subset=[col_missing], inplace=True)
                else: df_prep[col_missing].fillna(fill_val, inplace=True)
                notify("missing_applied"); st.rerun()

        with tabs[1]: # Types
            col, typ = st.columns(2)
            col_type = col.selectbox(t("prep_tabs.column"), df_prep.columns, key="type_col")
            new_type = typ.selectbox(t("prep_tabs.new_type"), ["object", "int64", "float64", "datetime64[ns]"], key="type_new")
            if st.button(t("prep_tabs.apply_button"), key="apply_type"):
                try:
                    df_prep[col_type] = pd.to_numeric(df_prep[col_type], errors='coerce') if 'int' in new_type or 'float' in new_type else df_prep[col_type].astype(new_type)
                    notify("type_changed", col=col_type); st.rerun()
                except Exception as e: notify("type_conversion_error", "error", e=e)

        with tabs[2]: # Colonnes
            st.subheader(t("prep_tabs.drop_cols_header"))
            cols_to_drop = st.multiselect(t("prep_tabs.cols_to_drop"), options=df_prep.columns, key="drop_cols")
            if st.button(t("prep_tabs.drop_cols_button"), key="drop_cols_btn"):
                df_prep.drop(columns=cols_to_drop, inplace=True); notify("cols_dropped"); st.rerun()
            st.subheader(t("prep_tabs.rename_col_header"))
            col1, col2 = st.columns(2)
            col_to_rename = col1.selectbox(t("prep_tabs.col_to_rename"), options=df_prep.columns, key="rename_select")
            new_name = col2.text_input(t("prep_tabs.new_name"), value=col_to_rename)
            if st.button(t("prep_tabs.rename_col_button"), key="rename_col_btn"):
                df_prep.rename(columns={col_to_rename: new_name}, inplace=True); notify("col_renamed"); st.rerun()
        
        with tabs[3]: # Valeurs
            col, old, new = st.columns(3)
            col_replace = col.selectbox(t("prep_tabs.column"), df_prep.columns, key="replace_col")
            old_val, new_val = old.text_input(t("prep_tabs.old_value")), new.text_input(t("prep_tabs.new_value"))
            if st.button(t("prep_tabs.replace_val_button"), key="replace_val_btn"):
                df_prep[col_replace].replace(old_val, new_val, inplace=True); notify("value_replaced"); st.rerun()

        with tabs[4]: # Encodage
            cat_cols = df_prep.select_dtypes(include=['object', 'category']).columns
            if len(cat_cols) > 0:
                col_to_encode = st.selectbox(t("prep_tabs.categorical_col_label"), options=cat_cols, key="encode_select")
                method = st.radio(t("prep_tabs.encoding_method"), ["Label Encoding", "One-Hot Encoding"], horizontal=True)
                if st.button(t("prep_tabs.encode_button"), key="encode_btn"):
                    if method == "Label Encoding":
                        df_prep[col_to_encode] = LabelEncoder().fit_transform(df_prep[col_to_encode])
                    else: # One-Hot
                        dummies = pd.get_dummies(df_prep[col_to_encode], prefix=col_to_encode, drop_first=True)
                        st.session_state.df_prepared = pd.concat([df_prep.drop(col_to_encode, axis=1), dummies], axis=1)
                    notify("encoding_done"); st.rerun()
            else: st.info(t("prep_tabs.no_categorical_cols"))

        with tabs[5]: # Outliers
            num_cols = df_prep.select_dtypes(include=np.number).columns
            if len(num_cols) > 0:
                col_outliers = st.selectbox(t("prep_tabs.numeric_col_label"), options=num_cols, key="outlier_select")
                if st.button(t("prep_tabs.remove_outliers_button"), key="outlier_btn"):
                    rows_before = len(df_prep)
                    st.session_state.df_prepared = remove_outliers_iqr(df_prep, col_outliers)
                    notify("outliers_removed", count=(rows_before - len(st.session_state.df_prepared))); st.rerun()
            else: st.info(t("prep_tabs.no_numeric_cols"))

    st.header(t("main.header_chatbot"))
    if st.button(t("main.open_chatbot_button"), use_container_width=True):
        st.session_state.show_chat_modal = True

else:
    st.info(t("main.info_start"))