# assets.py
import os
import json
import pandas as pd
from datetime import datetime

from dagster import asset

from scrapy.crawler import CrawlerProcess
from crawler_noticia.economia.economia.spiders.noticia import NoticiasSpider
from crawler_noticia.governo.governo.spiders.noticia import G1Spider

import duckdb
from filelock import FileLock
import subprocess
import hashlib

# Bibliotecas para IA e análise de sentimento
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)

# Conexão com MongoDB
from db_mongo.conexao_mongo import salvar_no_mongo

# -----------------------------------------------------------------------------
# CONFIGURAÇÕES DE DIRETÓRIO E ARQUIVOS
# -----------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTADOS_DIR = os.path.join(BASE_DIR, "projeto_dados", "resultados")
os.makedirs(RESULTADOS_DIR, exist_ok=True)

DUCKDB_FILE = os.path.join(BASE_DIR, "noticias.duckdb")
DBT_PROJECT_PATH = os.path.join(BASE_DIR, "dbt_project")

# Lock para evitar concorrência no acesso ao arquivo DuckDB
duckdb_lock = FileLock(os.path.join(BASE_DIR, "duckdb.lock"))

# -----------------------------------------------------------------------------
# FUNÇÕES AUXILIARES
# -----------------------------------------------------------------------------

def run_spider(spider, raw_table_name, collection_name):
    """
    Executa um spider do Scrapy, salva o resultado em JSON, depois no DuckDB e no MongoDB.
    - Principio SOLID (S) e Clean Code: faz apenas a parte de coleta e salvamento inicial.
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_file = f"projeto_dados/data/{raw_table_name}_{timestamp}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Configura e roda o CrawlerProcess do Scrapy
    process = CrawlerProcess(settings={
        "FEEDS": {output_file: {"format": "json"}},
        "FEED_EXPORT_ENCODING": "utf-8",
    })
    process.crawl(spider)
    process.start()

    # Carrega JSON e processa dados
    with open(output_file, "r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)

    processed_data = []
    if raw_table_name == "economia_raw":
        # Estrutura para economia
        for item in data:
            for time_key, titles in item.items():
                for title in titles:
                    processed_data.append({
                        "id": hashlib.md5(f"{title}{time_key}".encode()).hexdigest(),
                        "titulo_noticia": title,
                        "texto": "",
                        "data_publicacao": None,
                        "time_ago": time_key,
                    })
    elif raw_table_name == "governo_raw":
        # Estrutura para governo
        for item in data:
            processed_data.append({
                "id": hashlib.md5(str(item).encode()).hexdigest(),
                "titulo_noticia": item.get("title", ""),
                "texto": item.get("body", ""),
                "data_publicacao": item.get("data_publicacao", ""),
            })

    df = pd.DataFrame(processed_data)

    if 'id' not in df.columns:
        df['id'] = df.apply(lambda row: hashlib.md5(str(row).encode()).hexdigest(), axis=1)

    if 'data_publicacao' not in df.columns:
        df['data_publicacao'] = pd.NaT

    # Salvar no DuckDB (com lock para evitar conflitos)
    with duckdb_lock:
        with duckdb.connect(DUCKDB_FILE) as conn:
            conn.register("df_view", df)
            try:
                count = conn.execute(f"SELECT COUNT(*) FROM {raw_table_name}").fetchone()[0]
            except duckdb.BinderException:
                count = 0

            if count > 0:
                # Insere somente os novos (evita duplicatas)
                conn.execute(f"""
                    INSERT INTO {raw_table_name}
                    SELECT * FROM df_view
                    WHERE id NOT IN (SELECT id FROM {raw_table_name})
                """)
                print(f"Novos dados apendados na tabela '{raw_table_name}' no DuckDB.")
            else:
                # Cria tabela se não existir
                conn.execute(f"CREATE TABLE {raw_table_name} AS SELECT * FROM df_view")
                print(f"Tabela '{raw_table_name}' criada e dados inseridos no DuckDB.")

    print(f"Dados salvos na tabela '{raw_table_name}' no DuckDB.")

    # Salvar no MongoDB
    salvar_no_mongo(processed_data, collection_name)
    print(f"Novos dados adicionados à coleção '{collection_name}' no MongoDB.")


def carregar_dados_duckdb(table_name):
    """
    Carrega dados de uma tabela do DuckDB em um DataFrame do pandas.
    """
    with duckdb_lock:
        with duckdb.connect(DUCKDB_FILE) as conn:
            try:
                return conn.execute(f"SELECT * FROM {table_name}").fetchdf()
            except duckdb.BinderException:
                print(f"Tabela {table_name} não encontrada.")
                return pd.DataFrame()


def tratar_dados_func(transformed_table, processed_table, resultados_dir, collection_name):
    """
    Toma a tabela transformada (via DBT) e realiza análise de sentimento, salvando resultados.
    """
    df = carregar_dados_duckdb(transformed_table)
    if df.empty:
        print(f"Nenhum dado encontrado na tabela {transformed_table}.")
        return pd.DataFrame()

    analyzer = SentimentIntensityAnalyzer()

    # Usa 'texto' se existir, caso contrário usa 'titulo_noticia'
    df['texto_para_sentimento'] = df.apply(
        lambda row: row['texto'] if isinstance(row['texto'], str) and row['texto'].strip()
        else row['titulo_noticia'],
        axis=1
    )

    # Aplica VADER
    df['sentimento'] = df['texto_para_sentimento'].apply(
        lambda x: analyzer.polarity_scores(x)['compound'] if isinstance(x, str) else 0
    )

    # Classificação em positivo, negativo, neutro (limiares do VADER)
    df['sentimento_classificacao'] = df['sentimento'].apply(
        lambda x: 'positivo' if x >= 0.05 else ('negativo' if x <= -0.05 else 'neutro')
    )
    # Cria coluna binária (1 = positivo, 0 = não positivo)
    df['target'] = df['sentimento_classificacao'].apply(
        lambda x: 1 if x == 'positivo' else 0
    )

    # Gera CSV de resumo
    sentimento_summary = df['sentimento_classificacao'].value_counts().to_frame().reset_index()
    sentimento_summary.columns = ['Sentimento', 'Contagem']
    sentimento_file = os.path.join(resultados_dir, f"{transformed_table}_sentimento_summary.csv")
    sentimento_summary.to_csv(sentimento_file, index=False)
    print(f"Resumo de sentimentos salvo em '{sentimento_file}'.")

    # Salva no Mongo (dados processados)
    salvar_no_mongo(df.to_dict(orient='records'), f"{collection_name}_processed")
    print(f"Dados processados salvos na coleção '{collection_name}_processed' no MongoDB.")

    # Escreve (ou sobrescreve) tabela de dados processados
    with duckdb_lock:
        with duckdb.connect(DUCKDB_FILE) as conn:
            conn.register("df_view", df)
            conn.execute(f"CREATE OR REPLACE TABLE {processed_table} AS SELECT * FROM df_view")

    print(f"Dados tratados salvos na tabela '{processed_table}' no DuckDB.")
    return df


def treinar_ia(df: pd.DataFrame, nome_modelo: str, resultados_dir: str):
    """
    Recebe DataFrame processado, treina 3 modelos (RF, LR, SVM).
    Gera métricas e relatório mesmo que dados sejam insuficientes.
    """
    if df.empty:
        print(f"Nenhum dado disponível para treinamento em '{nome_modelo}'.")
        return

    X = df[['sentimento']].fillna(0)
    y = df['target']

    # Se só há uma classe, gera relatório mínimo
    if y.nunique() <= 1:
        print(f"Dados insuficientes para treinar IA em '{nome_modelo}'.")
        # Ao menos gerar um CSV de sentimento
        sentimento_summary = df['sentimento_classificacao'].value_counts().to_frame().reset_index()
        sentimento_summary.columns = ['Sentimento', 'Contagem']
        sentimento_file = os.path.join(resultados_dir, f"{nome_modelo}_sentimento_summary.csv")
        sentimento_summary.to_csv(sentimento_file, index=False)

        # Gera um relatório simples
        relatorio_path = os.path.join(resultados_dir, f"{nome_modelo}_relatorio.txt")
        with open(relatorio_path, 'w') as relatorio:
            relatorio.write(f"Relatório de Treinamento de IA para {nome_modelo}\n")
            relatorio.write("="*50 + "\n\n")
            relatorio.write("Dados insuficientes para treinamento.\n\n")
            relatorio.write("Distribuição de Sentimentos:\n")
            relatorio.write(str(sentimento_summary) + "\n\n")
            relatorio.write("Nenhuma métrica foi gerada.\n")
        return

    # Split com estratificação
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError as e:
        print(f"Erro ao fazer split: {e}")
        return

    # Se após split ainda só tem uma classe em y_train/y_test
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        print(f"Classes insuficientes após split em '{nome_modelo}'.")
        return

    # Algoritmos
    algoritmos = {
        "RandomForest": RandomForestClassifier(random_state=42),
        "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000),
        "SVM": SVC(random_state=42)
    }

    metrics_list = []

    for nome_algoritmo, modelo in algoritmos.items():
        try:
            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)

            acuracia = accuracy_score(y_test, y_pred)
            precisao = precision_score(y_test, y_pred, zero_division=0)
            recall_ = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            metrics_list.append({
                "Algoritmo": nome_algoritmo,
                "Acurácia": acuracia,
                "Precisão": precisao,
                "Recall": recall_,
                "F1-Score": f1
            })

            # Matriz de confusão
            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
            cm_df = pd.DataFrame(cm, index=["Negativo", "Positivo"], columns=["Negativo", "Positivo"])
            cm_file = os.path.join(resultados_dir, f"{nome_modelo}_{nome_algoritmo}_confusion_matrix.csv")
            cm_df.to_csv(cm_file)
            print(f"Matriz de Confusão para {nome_algoritmo} salva em '{cm_file}'.")
        except ValueError as e:
            print(f"Erro ao treinar {nome_algoritmo} em {nome_modelo}: {e}")

    sentimento_summary = df['sentimento_classificacao'].value_counts().to_frame().reset_index()
    sentimento_summary.columns = ['Sentimento', 'Contagem']

    sentimento_file = os.path.join(resultados_dir, f"{nome_modelo}_sentimento_summary.csv")
    sentimento_summary.to_csv(sentimento_file, index=False)
    print(f"Resumo de sentimentos salvo em '{sentimento_file}'.")

    # Gera CSV de métricas e relatório
    if metrics_list:
        metrics_df = pd.DataFrame(metrics_list)
        metrics_file = os.path.join(resultados_dir, f"{nome_modelo}_metrics.csv")
        metrics_df.to_csv(metrics_file, index=False)
        print(f"Métricas salvas em '{metrics_file}'.")

        relatorio_path = os.path.join(resultados_dir, f"{nome_modelo}_relatorio.txt")
        with open(relatorio_path, 'w') as relatorio:
            relatorio.write(f"Relatório de Treinamento de IA para {nome_modelo}\n")
            relatorio.write("="*50 + "\n\n")
            relatorio.write("Distribuição de Sentimentos:\n")
            relatorio.write(str(sentimento_summary) + "\n\n")
            relatorio.write("Métricas dos Modelos:\n")
            relatorio.write(str(metrics_df) + "\n\n")
            relatorio.write("Observações:\n")
            relatorio.write("- IA analisando sentimento com pontuação do VADER.\n")
            relatorio.write("- Classes: Positivo (1) vs. Não-positivo (0).\n")
    else:
        print("Nenhuma métrica foi coletada (erro no treinamento ou dados insuficientes).")


# -----------------------------------------------------------------------------
# ASSETS (Dagster)
# -----------------------------------------------------------------------------

@asset(
    description="Executa o crawler para coletar notícias de economia e salva no DuckDB/MongoDB.",
    kinds={"python"}
)
def crawler_economia():
    run_spider(NoticiasSpider, "economia_raw", "economia_raw")

@asset(
    description="Executa o crawler para coletar notícias de governo e salva no DuckDB/MongoDB.",
    kinds={"python"}
)
def crawler_governo():
    run_spider(G1Spider, "governo_raw", "governo_raw")

@asset(
    description="Executa os modelos DBT para transformar dados brutos em tabelas transformadas.",
    non_argument_deps={"crawler_economia", "crawler_governo"},
    kinds={"dbt"}
)
def executar_dbt():
    if not os.path.exists(DBT_PROJECT_PATH):
        raise NotADirectoryError(f"O diretório DBT não existe: {DBT_PROJECT_PATH}")
    # Roda: dbt run
    subprocess.run(["dbt", "run"], check=True, cwd=DBT_PROJECT_PATH)
    print("Modelos DBT executados com sucesso.")

@asset(
    description="Trata os dados transformados de economia (análise de sentimento, etc.).",
    non_argument_deps={"executar_dbt"},
    kinds={"duckdb", "pandas", "mongodb"}
)
def tratar_dados_economia():
    return tratar_dados_func("economia_transformed", "economia_processed", RESULTADOS_DIR, "economia_processed")

@asset(
    description="Trata os dados transformados de governo (análise de sentimento, etc.).",
    non_argument_deps={"executar_dbt"},
    kinds={"duckdb", "pandas", "mongodb"}
)
def tratar_dados_governo():
    return tratar_dados_func("governo_transformed", "governo_processed", RESULTADOS_DIR, "governo_processed")

@asset(
    description="Treina IA usando dados processados de economia.",
    non_argument_deps={"tratar_dados_economia"},
    kinds={"python", "scikitlearn"}
)
def treinar_ia_economia():
    df = carregar_dados_duckdb("economia_processed")
    treinar_ia(df, "economia", RESULTADOS_DIR)

@asset(
    description="Treina IA usando dados processados de governo.",
    non_argument_deps={"tratar_dados_governo"},
    kinds={"python", "scikitlearn"}
)
def treinar_ia_governo():
    df = carregar_dados_duckdb("governo_processed")
    treinar_ia(df, "governo", RESULTADOS_DIR)

@asset(
    description="Verifica amostras dos dados transformados (economia/governo).",
    non_argument_deps={"tratar_dados_economia", "tratar_dados_governo"},
    kinds={"python"}
)
def verificar_dados_transformados():
    with duckdb_lock:
        with duckdb.connect(DUCKDB_FILE) as conn:
            gov_df = conn.execute("SELECT * FROM governo_transformed LIMIT 5").fetchdf()
            print("Amostra de governo_transformed:")
            print(gov_df)

            eco_df = conn.execute("SELECT * FROM economia_transformed LIMIT 5").fetchdf()
            print("Amostra de economia_transformed:")
            print(eco_df)

@asset(
    description="Verifica quantos registros existem para IA.",
    non_argument_deps={"tratar_dados_economia", "tratar_dados_governo"},
    kinds={"python"}
)
def verificar_dados_disponiveis():
    eco_df = carregar_dados_duckdb("economia_processed")
    gov_df = carregar_dados_duckdb("governo_processed")
    print(f"Economia: {eco_df.shape[0]} registros.")
    print(f"Governo: {gov_df.shape[0]} registros.")
