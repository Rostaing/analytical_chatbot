# 🤖 Ultimate Analytical Chatbot

A powerful, web-based data analysis tool built with Streamlit. This application allows you to load data from multiple sources, clean it through an intuitive UI, and analyze it by conversing with an advanced AI analyst powered by Google's Gemini models.

![image](https://user-images.githubusercontent.com/12832533/226190325-19e64a13-3398-4665-8b2c-62383827d0c3.png)
*(Tip: Replace this image with a screenshot or GIF of your application in action!)*

---

## ✨ Key Features

### 📂 1. Versatile Data Loading
Load your data from almost anywhere:
- **Local Files**:
  - `CSV`, `Excel (.xlsx, .xls)`
  - `JSON`, `Parquet`, `XML`, `HTML`
  - Text files (`.txt`), `PDF` (text extraction)
  - `SQL` Scripts (executed in-memory via DuckDB)
- **Databases**:
  - **SQL**: PostgreSQL, MySQL, SQL Server, Oracle, SQLite
  - **NoSQL**: MongoDB
  - **Graph**: Neo4j
  - **Vector**: ChromaDB

### 🛠️ 2. Intuitive Data Preparation Tools
Prepare your data without writing a single line of code:
- **Missing Value Handling**: Drop rows or fill missing values with a specific input.
- **Type Conversion**: Change the data type of columns (numeric, text, datetime).
- **Column Management**: Easily drop or rename columns.
- **Value Replacement**: Find and replace specific values within a column.
- **Categorical Encoding**: Transform text columns into numerical values using `Label Encoding` or `One-Hot Encoding`.
- **Outlier Detection**: Identify and remove outliers using the Interquartile Range (IQR) method.

### 💬 3. Intelligent Analytical Chatbot
The core of the application. Chat with your data!
- **On-the-Fly Code Generation**: Ask questions in natural language (e.g., "Show me the distribution of ages"). The AI generates and executes the necessary Python code (`pandas`, `plotly`) to answer.
- **Multi-Format Responses**: The chatbot can respond with:
  - Text and numbers (e.g., "The average sales amount is $1245").
  - Data tables (`DataFrame`).
  - Interactive charts and graphs (`Plotly`).
- **AI Model Selection**: Choose the best Gemini model for your needs (`gemini-1.5-pro`, `gemini-1.5-flash`, `gemini-pro`) to balance performance and speed.
- **Application Management**: A safe reset button with a confirmation dialog to start over.

---

## 🚀 Tech Stack

- **Web Framework**: Streamlit
- **Data Analysis**: Pandas, NumPy, Scikit-learn
- **Artificial Intelligence**: Google Generative AI (Gemini)
- **Visualization**: Plotly Express
- **Data Connectors**: SQLAlchemy, PyMongo, Neo4j, ChromaDB, PyPDF, DuckDB

---

## ⚙️ Installation and Setup

Follow these steps to run the application on your local machine.

### 1. Prerequisites
- Python 3.8 or higher
- Git

### 2. Clone the Repository
Open your terminal and clone this repository:
```bash
git clone <YOUR_REPOSITORY_URL>
cd <REPOSITORY_FOLDER_NAME>
```

### 3. Create a Virtual Environment (Recommended)
This isolates the project's dependencies.
```bash
# Create the environment
python -m venv venv

# Activate the environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 4. Install Dependencies
Create a `requirements.txt` file with the content below, then install it.

**`requirements.txt`**:
```txt
streamlit
pandas
numpy
scikit-learn
google-generativeai
python-dotenv
pypdf
openpyxl
plotly
# --- Database Connectors ---
sqlalchemy
# For PostgreSQL
psycopg2-binary
# For MySQL
mysql-connector-python
# For SQL Server (requires ODBC drivers)
pyodbc
# For Oracle (requires Oracle Instant Client)
oracledb
# For MongoDB
pymongo
# For Neo4j
neo4j
# For ChromaDB
chromadb-client
# For .sql files
duckdb
```

Execute the following command in your terminal:
```bash
pip install -r requirements.txt
```
> **Note**: If you encounter a `numpy.dtype size changed` error, it indicates a binary incompatibility. Run the following command to fix it:
> `pip install --upgrade --force-reinstall --no-cache-dir numpy pandas scikit-learn`

### 5. Configure the Gemini API Key
The application needs an API key to communicate with Google's Gemini models.

1.  Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
2.  Create a file named `.env` in the root of your project.
3.  Add your key to this file as follows:

    **`.env`**:
    ```
    GOOGLE_API_KEY="YOUR_API_KEY_HERE"
    ```

### 6. Launch the Application
Once the installation is complete, run the app with the command:
```bash
streamlit run app.py
```
Open your browser to the URL displayed (usually `http://localhost:8501`).

---

## 📖 How to Use

1.  **Load Data**: Use the sidebar to select your data source (local file or database) and fill in the required information. Click the "Load" button.

2.  **Explore and Prepare**: Once loaded, your data is displayed. You can switch between the "Original" and "Prepared" data views. Use the "Preparation and Cleaning Tools" expander to clean your data. Each action modifies the "Prepared" DataFrame.

3.  **Chat with the AI**: Click the "💬 Open Chatbot" button.
    - Select the AI model you wish to use.
    - Ask your questions in the input box at the bottom.
    - Example questions:
      - "What is the total number of rows?"
      - "Show me the top 5 best-selling products."
      - "Create a histogram for the 'age' column."
      - "Is there a correlation between 'price' and 'customer_rating'?"
    - The AI will respond with text, a table, or an interactive chart.

4.  **Reset**: If you want to start over with a new dataset, use the "💥 Reset Application" button in the sidebar.

---

### ⚠️ Security Warning

This project uses the `eval()` function to execute Python code generated by the AI. While this provides incredible flexibility, `eval()` can pose a security risk if exposed to malicious input.

-   **Personal Use**: The application is safe for personal and local use.
-   **Public Deployment**: If you plan to deploy this application in a multi-user or public environment, it is **imperative** to implement additional security measures (such as sandboxing or analyzing the generated code) to prevent arbitrary code execution.