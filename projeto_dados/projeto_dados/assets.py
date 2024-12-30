'''# assets.py
import os
import json
import pandas as pd
from datetime import datetime
from dagster import asset
from scrapy.crawler import CrawlerProcess
from crawler_noticia.economia.economia.spiders.noticia import NoticiasSpider
from crawler_noticia.governo.governo.spiders.noticia import G1Spider
import duckdb
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from filelock import FileLock
import hashlib
import subprocess

# Importar funções de MongoDB
from db_mongo.conexao_mongo import salvar_no_mongo

# Diretório base do projeto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Pasta para armazenar resultados
RESULTADOS_DIR = os.path.join(BASE_DIR, "projeto_dados", "resultados")
os.makedirs(RESULTADOS_DIR, exist_ok=True)

# Arquivo DuckDB
DUCKDB_FILE = os.path.join(BASE_DIR, "noticias.duckdb")
DBT_PROJECT_PATH = os.path.join(BASE_DIR, "dbt_project")

# Lock para controle de acesso ao DuckDB
duckdb_lock = FileLock(os.path.join(BASE_DIR, "duckdb.lock"))

def run_spider(spider, raw_table_name, collection_name):
    """
    Executa o spider do Scrapy para coletar dados e salva no DuckDB e MongoDB.

    Args:
        spider: Classe do spider do Scrapy.
        raw_table_name (str): Nome da tabela bruta no DuckDB.
        collection_name (str): Nome da coleção no MongoDB.
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_file = f"projeto_dados/data/{raw_table_name}_{timestamp}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    process = CrawlerProcess(settings={
        "FEEDS": {output_file: {"format": "json"}},
        "FEED_EXPORT_ENCODING": "utf-8",
    })
    process.crawl(spider)
    process.start()

    with open(output_file, "r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)
    
    # Processar dados para estrutura consistente
    processed_data = []
    if raw_table_name == "economia_raw":
        for item in data:
            for time_key, titles in item.items():
                for title in titles:
                    processed_data.append({
                        "id": hashlib.md5(f"{title}{time_key}".encode()).hexdigest(),
                        "titulo_noticia": title,
                        "texto": "",  # Preencher se disponível
                        "data_publicacao": None,  # Preencher se disponível
                        "time_ago": time_key,
                    })
    elif raw_table_name == "governo_raw":
        for item in data:
            processed_data.append({
                "id": hashlib.md5(str(item).encode()).hexdigest(),
                "titulo_noticia": item.get("title", ""),
                "texto": item.get("body", ""),
                "data_publicacao": item.get("data_publicacao", ""),
            })

    df = pd.DataFrame(processed_data)
    
    # Garantir coluna 'id' presente
    if 'id' not in df.columns:
        df['id'] = df.apply(lambda row: hashlib.md5(str(row).encode()).hexdigest(), axis=1)

    # Preencher 'data_publicacao' se ausente
    if 'data_publicacao' not in df.columns:
        df['data_publicacao'] = pd.NaT

    with duckdb_lock:
        conn = duckdb.connect(DUCKDB_FILE)
        conn.register("df_view", df)
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {raw_table_name}").fetchone()[0]  # type: ignore
        except duckdb.BinderException:
            count = 0

        if count > 0:
            # Se a tabela já existe e possui dados, apende os novos dados sem duplicatas
            conn.execute(f"""
                INSERT INTO {raw_table_name} 
                SELECT * FROM df_view 
                WHERE id NOT IN (SELECT id FROM {raw_table_name})
            """)
            print(f"Novos dados apendados na tabela '{raw_table_name}' no DuckDB.")
        else:
            # Se a tabela não existe ou está vazia, cria a tabela
            conn.execute(f"CREATE TABLE {raw_table_name} AS SELECT * FROM df_view")
            print(f"Tabela '{raw_table_name}' criada e dados inseridos no DuckDB.")
        conn.close()
    print(f"Dados salvos na tabela '{raw_table_name}' no DuckDB.")

    # Salvar dados brutos no MongoDB
    salvar_no_mongo(processed_data, collection_name)  # type: ignore
    print(f"Novos dados adicionados à coleção '{collection_name}' no MongoDB.")

# Asset: Executar o crawler de economia
@asset(
    description="Executa o crawler para coletar notícias de economia e salva os dados brutos no DuckDB e MongoDB.",
    kinds={"python"},
)
def crawler_economia() -> None:
    run_spider(NoticiasSpider, "economia_raw", "economia_raw")

# Asset: Executar o crawler de governo
@asset(
    description="Executa o crawler para coletar notícias de governo e salva os dados brutos no DuckDB e MongoDB.",
    kinds={"python"},
)
def crawler_governo() -> None:
    run_spider(G1Spider, "governo_raw", "governo_raw")

# Asset: Executar DBT para transformação de dados
@asset(
    description="Executa os modelos DBT para transformar os dados brutos.",
    non_argument_deps={"crawler_economia", "crawler_governo"},
    kinds={"dbt"},
)
def executar_dbt() -> None:
    if not os.path.exists(DBT_PROJECT_PATH):
        raise NotADirectoryError(f"O diretório especificado para o DBT não existe: {DBT_PROJECT_PATH}")
    subprocess.run(["dbt", "run"], check=True, cwd=DBT_PROJECT_PATH)
    print("Modelos DBT executados com sucesso.")

# Função para carregar dados do DuckDB
def carregar_dados_duckdb(table_name):
    """
    Carrega dados de uma tabela DuckDB.

    Args:
        table_name (str): Nome da tabela no DuckDB.

    Returns:
        pd.DataFrame: DataFrame contendo os dados da tabela.
    """
    with duckdb_lock:
        conn = duckdb.connect(DUCKDB_FILE)
        try:
            df = conn.execute(f"SELECT * FROM {table_name}").fetchdf()
        except duckdb.BinderException:
            print(f"Tabela {table_name} não encontrada no DuckDB.")
            df = pd.DataFrame()
        conn.close()
    return df

# Função para tratar os dados processados
def tratar_dados_func(transformed_table, processed_table, resultados_dir, collection_name):
    """
    Trata os dados transformados, realiza análise de sentimento e prepara para treinamento de IA.

    Args:
        transformed_table (str): Nome da tabela transformada no DuckDB.
        processed_table (str): Nome da tabela processada no DuckDB.
        resultados_dir (str): Diretório para salvar os resultados.
        collection_name (str): Nome da coleção no MongoDB para salvar os dados processados.

    Returns:
        pd.DataFrame: DataFrame processado.
    """
    df = carregar_dados_duckdb(transformed_table)
    if df.empty:
        print(f"Nenhum dado encontrado na tabela {transformed_table}.")
        return pd.DataFrame()

    analyzer = SentimentIntensityAnalyzer()
    # Se 'texto' estiver vazio, usar 'titulo_noticia' para análise de sentimento
    df['texto_para_sentimento'] = df.apply(
        lambda row: row['texto'] if isinstance(row['texto'], str) and row['texto'].strip() else row['titulo_noticia'],
        axis=1
    )
    df['sentimento'] = df['texto_para_sentimento'].apply(lambda x: analyzer.polarity_scores(x)['compound'] if isinstance(x, str) else 0)
    
    # Preencher valores None com 0
    df['sentimento'] = df['sentimento'].fillna(0)
    
    df['sentimento_classificacao'] = df['sentimento'].apply(
        lambda x: 'positivo' if x > 0 else ('negativo' if x < 0 else 'neutro')
    )
    df['target'] = df['sentimento_classificacao'].apply(lambda x: 1 if x == 'positivo' else 0)  # Mantendo binário

    # Salvar um resumo da análise de sentimento
    sentimento_summary = df['sentimento_classificacao'].value_counts().to_frame().reset_index()
    sentimento_summary.columns = ['Sentimento', 'Contagem']
    sentimento_file = os.path.join(resultados_dir, f"{transformed_table}_sentimento_summary.csv")
    sentimento_summary.to_csv(sentimento_file, index=False)
    print(f"Resumo de sentimentos salvo em '{sentimento_file}'.")

    # Salvar dados processados no MongoDB
    salvar_no_mongo(df.to_dict(orient='records'), f"{collection_name}_processed")  # type: ignore
    print(f"Dados processados salvos na coleção '{collection_name}_processed' no MongoDB.")

    # Verificar se há dados suficientes para treinamento
    contagem_targets = df['target'].value_counts()
    if contagem_targets.shape[0] < 2:
        print(f"Não há classes suficientes para treinar IA em {transformed_table}.")
        return df

    with duckdb_lock:
        conn = duckdb.connect(DUCKDB_FILE)
        conn.register("df_view", df)
        conn.execute(f"CREATE OR REPLACE TABLE {processed_table} AS SELECT * FROM df_view")
        conn.close()
    print(f"Dados tratados salvos na tabela '{processed_table}' no DuckDB.")
    return df

# Asset: Tratar dados de economia
@asset(
    description="Trata os dados transformados de economia e salva no DuckDB e MongoDB.",
    non_argument_deps={"executar_dbt"},
    kinds={"duckdb", "pandas" , "mongodb"},
)
def tratar_dados_economia() -> pd.DataFrame:
    return tratar_dados_func("economia_transformed", "economia_processed", RESULTADOS_DIR, "economia_processed")

# Asset: Tratar dados de governo
@asset(
    description="Trata os dados transformados de governo e salva no DuckDB e MongoDB.",
    non_argument_deps={"executar_dbt"},
    kinds={"duckdb", "pandas" , "mongodb"},
)
def tratar_dados_governo() -> pd.DataFrame:
    return tratar_dados_func("governo_transformed", "governo_processed", RESULTADOS_DIR, "governo_processed")

# Função para treinar IA
def treinar_ia(df: pd.DataFrame, nome_modelo: str, resultados_dir: str):
    """
    Treina modelos de IA utilizando os dados fornecidos e salva as métricas e relatórios.

    Args:
        df (pd.DataFrame): DataFrame com os dados para treinamento.
        nome_modelo (str): Nome do modelo (economia/governo).
        resultados_dir (str): Diretório para salvar os resultados.
    """
    if df.empty:
        print(f"Nenhum dado disponível para treinamento em {nome_modelo}.")
        return
    
    X = df[['sentimento']].fillna(0)
    y = df['target']
    if y.nunique() <= 1:
        print(f"Dados insuficientes para treinar IA em {nome_modelo}.")
        return

    # Dividir os dados
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Verificar se ambas as classes estão presentes em y_train e y_test
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        print(f"Dados insuficientes para treinar IA em {nome_modelo}.")
        return

    # Definir os algoritmos a serem utilizados
    algoritmos = {
        "RandomForest": RandomForestClassifier(random_state=42),
        "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000),
        "SVM": SVC(random_state=42)
    }
    
    # DataFrame para armazenar métricas
    metrics_df = pd.DataFrame(columns=["Algoritmo", "Acurácia", "Precisão", "Recall", "F1-Score"])
    metrics_list = []  # Coletar métricas em uma lista

    # Matriz de Confusão para cada algoritmo
    for nome_algoritmo, modelo in algoritmos.items():
        try:
            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)
            
            acuracia = accuracy_score(y_test, y_pred)
            precisao = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            # Adicionar métricas à lista
            metrics_list.append({
                "Algoritmo": nome_algoritmo,
                "Acurácia": acuracia,
                "Precisão": precisao,
                "Recall": recall,
                "F1-Score": f1
            })
            
            # Salvar a Matriz de Confusão com labels especificados para evitar ValueError
            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
            cm_df = pd.DataFrame(cm, index=["Negativo", "Positivo"], columns=["Negativo", "Positivo"])
            cm_file = os.path.join(resultados_dir, f"{nome_modelo}_{nome_algoritmo}_confusion_matrix.csv")
            cm_df.to_csv(cm_file)
            print(f"Matriz de Confusão para {nome_algoritmo} salva em '{cm_file}'.")
        except ValueError as e:
            print(f"Erro ao treinar {nome_algoritmo} em {nome_modelo}: {e}")
            continue

    # Criar DataFrame de métricas a partir da lista
    if metrics_list:
        new_metrics = pd.DataFrame(metrics_list)
        metrics_df = pd.concat([metrics_df, new_metrics], ignore_index=True)
    else:
        print(f"Nenhuma métrica foi coletada para {nome_modelo}.")

    # Salvar as métricas
    if not metrics_df.empty:
        metrics_file = os.path.join(resultados_dir, f"{nome_modelo}_metrics.csv")
        metrics_df.to_csv(metrics_file, index=False)
        print(f"Métricas dos modelos salvas em '{metrics_file}'.")
    else:
        print(f"Nenhuma métrica foi salva para {nome_modelo}.")

    # Gerar relatório de análise de sentimento
    sentimento_summary = df['sentimento_classificacao'].value_counts().to_frame().reset_index()
    sentimento_summary.columns = ['Sentimento', 'Contagem']
    sentimento_file = os.path.join(resultados_dir, f"{nome_modelo}_sentimento_summary.csv")
    sentimento_summary.to_csv(sentimento_file, index=False)
    print(f"Resumo de sentimentos salvo em '{sentimento_file}'.")

    # Gerar Relatório Detalhado das Métricas
    if not metrics_df.empty:
        relatorio_path = os.path.join(resultados_dir, f"{nome_modelo}_relatorio.txt")
        with open(relatorio_path, 'w') as relatorio:
            relatorio.write(f"Relatório de Treinamento de IA para {nome_modelo}\n")
            relatorio.write("="*50 + "\n\n")
            
            relatorio.write("Distribuição de Sentimentos:\n")
            relatorio.write(sentimento_summary.to_string(index=False))
            relatorio.write("\n\nMétricas dos Modelos:\n")
            relatorio.write(metrics_df.to_string(index=False))
            relatorio.write("\n\nObservações:\n")
            relatorio.write("A IA está analisando o sentimento das notícias com base na pontuação de sentimento fornecida pelo VADER.\n")
            relatorio.write("Classes utilizadas: Positivo (1), Negativo (0)\n")
        
        print(f"Relatório detalhado salvo em '{relatorio_path}'.")
    else:
        print(f"Nenhum relatório foi gerado para {nome_modelo}.")

# Asset: Treinar IA com dados de economia
@asset(
    description="Treina um modelo de IA com dados de economia processados e salva métricas no DuckDB e MongoDB.",
    non_argument_deps={"tratar_dados_economia"},
    kinds={"python", "scikitlearn"},
)
def treinar_ia_economia() -> None:
    df = carregar_dados_duckdb("economia_processed")
    treinar_ia(df, "economia", RESULTADOS_DIR)

# Asset: Treinar IA com dados de governo
@asset(
    description="Treina um modelo de IA com dados de governo processados e salva métricas no DuckDB e MongoDB.",
    non_argument_deps={"tratar_dados_governo"},
    kinds={"python", "scikitlearn"},
)
def treinar_ia_governo() -> None:
    df = carregar_dados_duckdb("governo_processed")
    treinar_ia(df, "governo", RESULTADOS_DIR)

# Asset: Verificar dados transformados
@asset(
    description="Verifica e imprime amostras dos dados transformados no DuckDB.",
    non_argument_deps={"tratar_dados_economia", "tratar_dados_governo"},
    kinds={"python"},
)
def verificar_dados_transformados() -> None:
    import duckdb

    conn = duckdb.connect(DUCKDB_FILE)

    # Verificar dados transformados de governo
    governo_transformed_df = conn.execute("SELECT * FROM governo_transformed LIMIT 5").fetchdf()
    print("Dados transformados de governo:")
    print(governo_transformed_df)

    # Verificar dados transformados de economia
    economia_transformed_df = conn.execute("SELECT * FROM economia_transformed LIMIT 5").fetchdf()
    print("Dados transformados de economia:")
    print(economia_transformed_df)

    conn.close()

# Asset: Verificar dados disponíveis para IA
@asset(
    description="Verifica a quantidade de dados disponíveis para treinamento de IA.",
    non_argument_deps={"tratar_dados_economia", "tratar_dados_governo"},
    kinds={"python"},
)
def verificar_dados_disponiveis() -> None:
    economia_df = carregar_dados_duckdb("economia_processed")
    governo_df = carregar_dados_duckdb("governo_processed")
    
    print(f"Dados disponíveis para Economia: {economia_df.shape[0]} registros.")
    print(f"Dados disponíveis para Governo: {governo_df.shape[0]} registros.")'''

''''
import os
import json
import pandas as pd
from datetime import datetime
from dagster import asset
from scrapy.crawler import CrawlerProcess
from crawler_noticia.economia.economia.spiders.noticia import NoticiasSpider
from crawler_noticia.governo.governo.spiders.noticia import G1Spider
import duckdb
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from filelock import FileLock
import hashlib
import subprocess

# Diretório base do projeto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Arquivo DuckDB
DUCKDB_FILE = os.path.join(BASE_DIR, "noticias.duckdb")
DBT_PROJECT_PATH = os.path.join(BASE_DIR, "dbt_project")

# Lock para controle de acesso ao DuckDB
duckdb_lock = FileLock(os.path.join(BASE_DIR, "duckdb.lock"))

def run_spider(spider, raw_table_name):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_file = f"projeto_dados/data/{raw_table_name}_{timestamp}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    process = CrawlerProcess(settings={
        "FEEDS": {output_file: {"format": "json"}},
        "FEED_EXPORT_ENCODING": "utf-8",
    })
    process.crawl(spider)
    process.start()

    with open(output_file, "r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)
    
    # Processar dados para estrutura consistente
    processed_data = []
    if raw_table_name == "economia_raw":
        for item in data:
            for time_key, titles in item.items():
                for title in titles:
                    processed_data.append({
                        "id": hashlib.md5(f"{title}{time_key}".encode()).hexdigest(),
                        "titulo_noticia": title,
                        "texto": "",  # Preencher se disponível
                        "data_publicacao": None,  # Preencher se disponível
                        "time_ago": time_key,
                    })
    elif raw_table_name == "governo_raw":
        for item in data:
            processed_data.append({
                "id": hashlib.md5(str(item).encode()).hexdigest(),
                "titulo_noticia": item.get("title", ""),
                "texto": item.get("body", ""),
                "data_publicacao": item.get("data_publicacao", ""),
            })

    df = pd.DataFrame(processed_data)
    
    # Garantir coluna 'id' presente
    if 'id' not in df.columns:
        df['id'] = df.apply(lambda row: hashlib.md5(str(row).encode()).hexdigest(), axis=1)

    # Preencher 'data_publicacao' se ausente
    if 'data_publicacao' not in df.columns:
        df['data_publicacao'] = pd.NaT

    with duckdb_lock:
        conn = duckdb.connect(DUCKDB_FILE)
        conn.register("df_view", df)
        # Substituir 'CREATE TABLE IF NOT EXISTS' por 'CREATE OR REPLACE TABLE'
        conn.execute(f"CREATE OR REPLACE TABLE {raw_table_name} AS SELECT * FROM df_view")
        conn.close()
    print(f"Dados salvos na tabela '{raw_table_name}' no DuckDB.")

# Asset: Executar o crawler de economia
@asset(
    description="Executa o crawler para coletar notícias de economia e salva os dados brutos no DuckDB.",
    kinds={"python"},
)
def crawler_economia() -> None:
    run_spider(NoticiasSpider, "economia_raw")

# Asset: Executar o crawler de governo
@asset(
    description="Executa o crawler para coletar notícias de governo e salva os dados brutos no DuckDB.",
    kinds={"python"},
)
def crawler_governo() -> None:
    run_spider(G1Spider, "governo_raw")

# Asset: Executar DBT para transformação de dados
@asset(
    description="Executa os modelos DBT para transformar os dados brutos.",
    non_argument_deps={"crawler_economia", "crawler_governo"},
    kinds={"dbt"},
)
def executar_dbt() -> None:
    if not os.path.exists(DBT_PROJECT_PATH):
        raise NotADirectoryError(f"O diretório especificado para o DBT não existe: {DBT_PROJECT_PATH}")
    subprocess.run(["dbt", "run"], check=True, cwd=DBT_PROJECT_PATH)
    print("Modelos DBT executados com sucesso.")

# Função para carregar dados do DuckDB
def carregar_dados_duckdb(table_name):
    with duckdb_lock:
        conn = duckdb.connect(DUCKDB_FILE)
        try:
            df = conn.execute(f"SELECT * FROM {table_name}").fetchdf()
        except duckdb.BinderException:
            print(f"Tabela {table_name} não encontrada no DuckDB.")
            df = pd.DataFrame()
        conn.close()
    return df

# Função para tratar os dados processados
def tratar_dados_func(transformed_table, processed_table):
    df = carregar_dados_duckdb(transformed_table)
    if df.empty:
        print(f"Nenhum dado encontrado na tabela {transformed_table}.")
        return pd.DataFrame()

    analyzer = SentimentIntensityAnalyzer()
    df['sentimento'] = df['texto'].apply(lambda x: analyzer.polarity_scores(x)['compound'] if isinstance(x, str) else None)
    
    # Preencher valores None com 0
    df['sentimento'] = df['sentimento'].fillna(0)
    
    df['sentimento_classificacao'] = df['sentimento'].apply(
        lambda x: 'positivo' if x > 0 else ('negativo' if x < 0 else 'neutro')
    )
    df['target'] = df['sentimento_classificacao'].apply(lambda x: 1 if x == 'positivo' else 0)

    with duckdb_lock:
        conn = duckdb.connect(DUCKDB_FILE)
        conn.register("df_view", df)
        conn.execute(f"CREATE OR REPLACE TABLE {processed_table} AS SELECT * FROM df_view")
        conn.close()
    print(f"Dados tratados salvos na tabela '{processed_table}' no DuckDB.")
    return df

# Asset: Tratar dados de economia
@asset(
    description="Trata os dados transformados de economia e salva no DuckDB.",
    non_argument_deps={"executar_dbt"},
    kinds={"duckdb", "pandas"},
)
def tratar_dados_economia() -> pd.DataFrame:
    return tratar_dados_func("economia_transformed", "economia_processed")

# Asset: Tratar dados de governo
@asset(
    description="Trata os dados transformados de governo e salva no DuckDB.",
    non_argument_deps={"executar_dbt"},
    kinds={"duckdb", "pandas"},
)
def tratar_dados_governo() -> pd.DataFrame:
    return tratar_dados_func("governo_transformed", "governo_processed")

# Função para treinar IA
def treinar_ia(df: pd.DataFrame, nome_modelo: str):
    if df.empty:
        print(f"Nenhum dado disponível para treinamento em {nome_modelo}.")
        return

    X = df[['sentimento']].fillna(0)
    y = df['target']
    if y.nunique() <= 1:
        print(f"Dados insuficientes para treinar IA em {nome_modelo}.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modelo = RandomForestClassifier()
    modelo.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, modelo.predict(X_test))
    print(f"Acurácia do modelo {nome_modelo}: {accuracy}")

# Asset: Treinar IA com dados de economia
@asset(
    description="Treina um modelo de IA com dados de economia processados.",
    non_argument_deps={"tratar_dados_economia"},
    kinds={"python", "scikitlearn"},
)
def treinar_ia_economia() -> None:
    df = carregar_dados_duckdb("economia_processed")
    treinar_ia(df, "economia")

# Asset: Treinar IA com dados de governo
@asset(
    description="Treina um modelo de IA com dados de governo processados.",
    non_argument_deps={"tratar_dados_governo"},
    kinds={"python", "scikitlearn"},
)
def treinar_ia_governo() -> None:
    df = carregar_dados_duckdb("governo_processed")
    treinar_ia(df, "governo")

# Asset: Verificar dados transformados
@asset(
    description="Verifica e imprime amostras dos dados transformados no DuckDB.",
    non_argument_deps={"tratar_dados_economia", "tratar_dados_governo"},
    kinds={"python"},
)
def verificar_dados_transformados() -> None:
    import duckdb

    conn = duckdb.connect(DUCKDB_FILE)

    # Verificar dados transformados de governo
    governo_transformed_df = conn.execute("SELECT * FROM governo_transformed LIMIT 5").fetchdf()
    print("Dados transformados de governo:")
    print(governo_transformed_df)

    # Verificar dados transformados de economia
    economia_transformed_df = conn.execute("SELECT * FROM economia_transformed LIMIT 5").fetchdf()
    print("Dados transformados de economia:")
    print(economia_transformed_df)

    conn.close()'''
    
'''
import os
import json
import pandas as pd
from datetime import datetime
from dagster import asset
from scrapy.crawler import CrawlerProcess
from crawler_noticia.economia.economia.spiders.noticia import NoticiasSpider
from crawler_noticia.governo.governo.spiders.noticia import G1Spider
import duckdb
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from filelock import FileLock
import hashlib
import subprocess

# Diretório base do projeto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Pasta para armazenar resultados
RESULTADOS_DIR = os.path.join(BASE_DIR, "projeto_dados", "resultados")
os.makedirs(RESULTADOS_DIR, exist_ok=True)

# Arquivo DuckDB
DUCKDB_FILE = os.path.join(BASE_DIR, "noticias.duckdb")
DBT_PROJECT_PATH = os.path.join(BASE_DIR, "dbt_project")

# Lock para controle de acesso ao DuckDB
duckdb_lock = FileLock(os.path.join(BASE_DIR, "duckdb.lock"))

def run_spider(spider, raw_table_name):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_file = f"projeto_dados/data/{raw_table_name}_{timestamp}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    process = CrawlerProcess(settings={
        "FEEDS": {output_file: {"format": "json"}},
        "FEED_EXPORT_ENCODING": "utf-8",
    })
    process.crawl(spider)
    process.start()

    with open(output_file, "r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)
    
    # Processar dados para estrutura consistente
    processed_data = []
    if raw_table_name == "economia_raw":
        for item in data:
            for time_key, titles in item.items():
                for title in titles:
                    processed_data.append({
                        "id": hashlib.md5(f"{title}{time_key}".encode()).hexdigest(),
                        "titulo_noticia": title,
                        "texto": "",  # Preencher se disponível
                        "data_publicacao": None,  # Preencher se disponível
                        "time_ago": time_key,
                    })
    elif raw_table_name == "governo_raw":
        for item in data:
            processed_data.append({
                "id": hashlib.md5(str(item).encode()).hexdigest(),
                "titulo_noticia": item.get("title", ""),
                "texto": item.get("body", ""),
                "data_publicacao": item.get("data_publicacao", ""),
            })

    df = pd.DataFrame(processed_data)
    
    # Garantir coluna 'id' presente
    if 'id' not in df.columns:
        df['id'] = df.apply(lambda row: hashlib.md5(str(row).encode()).hexdigest(), axis=1)

    # Preencher 'data_publicacao' se ausente
    if 'data_publicacao' not in df.columns:
        df['data_publicacao'] = pd.NaT

    with duckdb_lock:
        conn = duckdb.connect(DUCKDB_FILE)
        conn.register("df_view", df)
        # Substituir 'CREATE TABLE IF NOT EXISTS' por 'CREATE OR REPLACE TABLE'
        conn.execute(f"CREATE OR REPLACE TABLE {raw_table_name} AS SELECT * FROM df_view")
        conn.close()
    print(f"Dados salvos na tabela '{raw_table_name}' no DuckDB.")

# Asset: Executar o crawler de economia
@asset(
    description="Executa o crawler para coletar notícias de economia e salva os dados brutos no DuckDB.",
    kinds={"python"},
)
def crawler_economia() -> None:
    run_spider(NoticiasSpider, "economia_raw")

# Asset: Executar o crawler de governo
@asset(
    description="Executa o crawler para coletar notícias de governo e salva os dados brutos no DuckDB.",
    kinds={"python"},
)
def crawler_governo() -> None:
    run_spider(G1Spider, "governo_raw")

# Asset: Executar DBT para transformação de dados
@asset(
    description="Executa os modelos DBT para transformar os dados brutos.",
    non_argument_deps={"crawler_economia", "crawler_governo"},
    kinds={"dbt"},
)
def executar_dbt() -> None:
    if not os.path.exists(DBT_PROJECT_PATH):
        raise NotADirectoryError(f"O diretório especificado para o DBT não existe: {DBT_PROJECT_PATH}")
    subprocess.run(["dbt", "run"], check=True, cwd=DBT_PROJECT_PATH)
    print("Modelos DBT executados com sucesso.")

# Função para carregar dados do DuckDB
def carregar_dados_duckdb(table_name):
    with duckdb_lock:
        conn = duckdb.connect(DUCKDB_FILE)
        try:
            df = conn.execute(f"SELECT * FROM {table_name}").fetchdf()
        except duckdb.BinderException:
            print(f"Tabela {table_name} não encontrada no DuckDB.")
            df = pd.DataFrame()
        conn.close()
    return df

# Função para tratar os dados processados
def tratar_dados_func(transformed_table, processed_table, resultados_dir):
    df = carregar_dados_duckdb(transformed_table)
    if df.empty:
        print(f"Nenhum dado encontrado na tabela {transformed_table}.")
        return pd.DataFrame()

    analyzer = SentimentIntensityAnalyzer()
    df['sentimento'] = df['texto'].apply(lambda x: analyzer.polarity_scores(x)['compound'] if isinstance(x, str) else None)
    
    # Preencher valores None com 0
    df['sentimento'] = df['sentimento'].fillna(0)
    
    df['sentimento_classificacao'] = df['sentimento'].apply(
        lambda x: 'positivo' if x > 0 else ('negativo' if x < 0 else 'neutro')
    )
    df['target'] = df['sentimento_classificacao'].apply(lambda x: 1 if x == 'positivo' else 0)

    # Salvar um resumo da análise de sentimento
    sentimento_summary = df['sentimento_classificacao'].value_counts().to_frame().reset_index()
    sentimento_summary.columns = ['Sentimento', 'Contagem']
    sentimento_file = os.path.join(resultados_dir, f"{transformed_table}_sentimento_summary.csv")
    sentimento_summary.to_csv(sentimento_file, index=False)
    print(f"Resumo de sentimentos salvo em '{sentimento_file}'.")

    with duckdb_lock:
        conn = duckdb.connect(DUCKDB_FILE)
        conn.register("df_view", df)
        conn.execute(f"CREATE OR REPLACE TABLE {processed_table} AS SELECT * FROM df_view")
        conn.close()
    print(f"Dados tratados salvos na tabela '{processed_table}' no DuckDB.")
    return df

# Asset: Tratar dados de economia
@asset(
    description="Trata os dados transformados de economia e salva no DuckDB.",
    non_argument_deps={"executar_dbt"},
    kinds={"duckdb", "pandas"},
)
def tratar_dados_economia() -> pd.DataFrame:
    return tratar_dados_func("economia_transformed", "economia_processed", RESULTADOS_DIR)

# Asset: Tratar dados de governo
@asset(
    description="Trata os dados transformados de governo e salva no DuckDB.",
    non_argument_deps={"executar_dbt"},
    kinds={"duckdb", "pandas"},
)
def tratar_dados_governo() -> pd.DataFrame:
    return tratar_dados_func("governo_transformed", "governo_processed", RESULTADOS_DIR)

# Função para treinar IA
def treinar_ia(df: pd.DataFrame, nome_modelo: str, resultados_dir: str):
    if df.empty:
        print(f"Nenhum dado disponível para treinamento em {nome_modelo}.")
        return
    
    X = df[['sentimento']].fillna(0)
    y = df['target']
    if y.nunique() <= 1:
        print(f"Dados insuficientes para treinar IA em {nome_modelo}.")
        return

    # Dividir os dados
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Definir os algoritmos a serem utilizados
    algoritmos = {
        "RandomForest": RandomForestClassifier(random_state=42),
        "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000),
        "SVM": SVC(random_state=42)
    }
    
    # DataFrame para armazenar métricas
    metrics_df = pd.DataFrame(columns=["Algoritmo", "Acurácia", "Precisão", "Recall", "F1-Score"])
    
    # Matriz de Confusão para cada algoritmo
    for nome_algoritmo, modelo in algoritmos.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        
        acuracia = accuracy_score(y_test, y_pred)
        precisao = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        metrics_df = metrics_df.append({
            "Algoritmo": nome_algoritmo,
            "Acurácia": acuracia,
            "Precisão": precisao,
            "Recall": recall,
            "F1-Score": f1
        }, ignore_index=True) # type: ignore
        
        # Salvar a Matriz de Confusão
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, index=["Negativo", "Positivo"], columns=["Negativo", "Positivo"])
        cm_file = os.path.join(resultados_dir, f"{nome_modelo}_{nome_algoritmo}_confusion_matrix.csv")
        cm_df.to_csv(cm_file)
    
    # Salvar as métricas
    metrics_file = os.path.join(resultados_dir, f"{nome_modelo}_metrics.csv")
    metrics_df.to_csv(metrics_file, index=False)
    
    print(f"Métricas e matrizes de confusão salvas em '{resultados_dir}'.")
    
    # Gerar relatório de análise de sentimento
    sentimento_summary = df['sentimento_classificacao'].value_counts().to_frame().reset_index()
    sentimento_summary.columns = ['Sentimento', 'Contagem']
    sentimento_file = os.path.join(resultados_dir, f"{nome_modelo}_sentimento_summary.csv")
    sentimento_summary.to_csv(sentimento_file, index=False)
    
    print(f"Resumo de sentimentos salvo em '{sentimento_file}'.")

# Asset: Treinar IA com dados de economia
@asset(
    description="Treina um modelo de IA com dados de economia processados.",
    non_argument_deps={"tratar_dados_economia"},
    kinds={"python", "scikitlearn"},
)
def treinar_ia_economia() -> None:
    df = carregar_dados_duckdb("economia_processed")
    treinar_ia(df, "economia", RESULTADOS_DIR)

# Asset: Treinar IA com dados de governo
@asset(
    description="Treina um modelo de IA com dados de governo processados.",
    non_argument_deps={"tratar_dados_governo"},
    kinds={"python", "scikitlearn"},
)
def treinar_ia_governo() -> None:
    df = carregar_dados_duckdb("governo_processed")
    treinar_ia(df, "governo", RESULTADOS_DIR)

# Asset: Verificar dados transformados
@asset(
    description="Verifica e imprime amostras dos dados transformados no DuckDB.",
    non_argument_deps={"tratar_dados_economia", "tratar_dados_governo"},
    kinds={"python"},
)
def verificar_dados_transformados() -> None:
    import duckdb

    conn = duckdb.connect(DUCKDB_FILE)

    # Verificar dados transformados de governo
    governo_transformed_df = conn.execute("SELECT * FROM governo_transformed LIMIT 5").fetchdf()
    print("Dados transformados de governo:")
    print(governo_transformed_df)

    # Verificar dados transformados de economia
    economia_transformed_df = conn.execute("SELECT * FROM economia_transformed LIMIT 5").fetchdf()
    print("Dados transformados de economia:")
    print(economia_transformed_df)

    conn.close()'''
    
    
# assets.py

# assets.py
'''
import os
import json
import pandas as pd
from datetime import datetime
from dagster import asset
from scrapy.crawler import CrawlerProcess
from crawler_noticia.economia.economia.spiders.noticia import NoticiasSpider
from crawler_noticia.governo.governo.spiders.noticia import G1Spider
import duckdb
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from filelock import FileLock
import hashlib
import subprocess

# Importar funções de MongoDB
from db_mongo.conexao_mongo import salvar_no_mongo

# Diretório base do projeto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Pasta para armazenar resultados
RESULTADOS_DIR = os.path.join(BASE_DIR, "projeto_dados", "resultados")
os.makedirs(RESULTADOS_DIR, exist_ok=True)

# Arquivo DuckDB
DUCKDB_FILE = os.path.join(BASE_DIR, "noticias.duckdb")
DBT_PROJECT_PATH = os.path.join(BASE_DIR, "dbt_project")

# Lock para controle de acesso ao DuckDB
duckdb_lock = FileLock(os.path.join(BASE_DIR, "duckdb.lock"))

def run_spider(spider, raw_table_name, collection_name):
    """
    Executa o spider do Scrapy para coletar dados e salva no DuckDB e MongoDB.

    Args:
        spider: Classe do spider do Scrapy.
        raw_table_name (str): Nome da tabela bruta no DuckDB.
        collection_name (str): Nome da coleção no MongoDB.
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_file = f"projeto_dados/data/{raw_table_name}_{timestamp}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    process = CrawlerProcess(settings={
        "FEEDS": {output_file: {"format": "json"}},
        "FEED_EXPORT_ENCODING": "utf-8",
    })
    process.crawl(spider)
    process.start()

    with open(output_file, "r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)
    
    # Processar dados para estrutura consistente
    processed_data = []
    if raw_table_name == "economia_raw":
        for item in data:
            for time_key, titles in item.items():
                for title in titles:
                    processed_data.append({
                        "id": hashlib.md5(f"{title}{time_key}".encode()).hexdigest(),
                        "titulo_noticia": title,
                        "texto": "",  # Preencher se disponível
                        "data_publicacao": None,  # Preencher se disponível
                        "time_ago": time_key,
                    })
    elif raw_table_name == "governo_raw":
        for item in data:
            processed_data.append({
                "id": hashlib.md5(str(item).encode()).hexdigest(),
                "titulo_noticia": item.get("title", ""),
                "texto": item.get("body", ""),
                "data_publicacao": item.get("data_publicacao", ""),
            })

    df = pd.DataFrame(processed_data)
    
    # Garantir coluna 'id' presente
    if 'id' not in df.columns:
        df['id'] = df.apply(lambda row: hashlib.md5(str(row).encode()).hexdigest(), axis=1)

    # Preencher 'data_publicacao' se ausente
    if 'data_publicacao' not in df.columns:
        df['data_publicacao'] = pd.NaT

    with duckdb_lock:
        conn = duckdb.connect(DUCKDB_FILE)
        conn.register("df_view", df)
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {raw_table_name}").fetchone()[0]  # type: ignore
        except duckdb.BinderException:
            count = 0

        if count > 0:
            # Se a tabela já existe e possui dados, apende os novos dados sem duplicatas
            conn.execute(f"""
                INSERT INTO {raw_table_name} 
                SELECT * FROM df_view 
                WHERE id NOT IN (SELECT id FROM {raw_table_name})
            """)
            print(f"Novos dados apendados na tabela '{raw_table_name}' no DuckDB.")
        else:
            # Se a tabela não existe ou está vazia, cria a tabela
            conn.execute(f"CREATE TABLE {raw_table_name} AS SELECT * FROM df_view")
            print(f"Tabela '{raw_table_name}' criada e dados inseridos no DuckDB.")
        conn.close()
    print(f"Dados salvos na tabela '{raw_table_name}' no DuckDB.")

    # Salvar dados brutos no MongoDB
    salvar_no_mongo(processed_data, collection_name)  # type: ignore
    print(f"Novos dados adicionados à coleção '{collection_name}' no MongoDB.")

# Asset: Executar o crawler de economia
@asset(
    description="Executa o crawler para coletar notícias de economia e salva os dados brutos no DuckDB e MongoDB.",
    kinds={"python"},
)
def crawler_economia() -> None:
    run_spider(NoticiasSpider, "economia_raw", "economia_raw")

# Asset: Executar o crawler de governo
@asset(
    description="Executa o crawler para coletar notícias de governo e salva os dados brutos no DuckDB e MongoDB.",
    kinds={"python"},
)
def crawler_governo() -> None:
    run_spider(G1Spider, "governo_raw", "governo_raw")

# Asset: Executar DBT para transformação de dados
@asset(
    description="Executa os modelos DBT para transformar os dados brutos.",
    non_argument_deps={"crawler_economia", "crawler_governo"},
    kinds={"dbt"},
)
def executar_dbt() -> None:
    if not os.path.exists(DBT_PROJECT_PATH):
        raise NotADirectoryError(f"O diretório especificado para o DBT não existe: {DBT_PROJECT_PATH}")
    subprocess.run(["dbt", "run"], check=True, cwd=DBT_PROJECT_PATH)
    print("Modelos DBT executados com sucesso.")

# Função para carregar dados do DuckDB
def carregar_dados_duckdb(table_name):
    """
    Carrega dados de uma tabela DuckDB.

    Args:
        table_name (str): Nome da tabela no DuckDB.

    Returns:
        pd.DataFrame: DataFrame contendo os dados da tabela.
    """
    with duckdb_lock:
        conn = duckdb.connect(DUCKDB_FILE)
        try:
            df = conn.execute(f"SELECT * FROM {table_name}").fetchdf()
        except duckdb.BinderException:
            print(f"Tabela {table_name} não encontrada no DuckDB.")
            df = pd.DataFrame()
        conn.close()
    return df

# Função para tratar os dados processados
def tratar_dados_func(transformed_table, processed_table, resultados_dir, collection_name):
    """
    Trata os dados transformados, realiza análise de sentimento e prepara para treinamento de IA.

    Args:
        transformed_table (str): Nome da tabela transformada no DuckDB.
        processed_table (str): Nome da tabela processada no DuckDB.
        resultados_dir (str): Diretório para salvar os resultados.
        collection_name (str): Nome da coleção no MongoDB para salvar os dados processados.

    Returns:
        pd.DataFrame: DataFrame processado.
    """
    df = carregar_dados_duckdb(transformed_table)
    if df.empty:
        print(f"Nenhum dado encontrado na tabela {transformed_table}.")
        return pd.DataFrame()

    analyzer = SentimentIntensityAnalyzer()
    # Se 'texto' estiver vazio, usar 'titulo_noticia' para análise de sentimento
    df['texto_para_sentimento'] = df.apply(
        lambda row: row['texto'] if isinstance(row['texto'], str) and row['texto'].strip() else row['titulo_noticia'],
        axis=1
    )
    df['sentimento'] = df['texto_para_sentimento'].apply(lambda x: analyzer.polarity_scores(x)['compound'] if isinstance(x, str) else 0)
    
    # Preencher valores None com 0
    df['sentimento'] = df['sentimento'].fillna(0)
    
    df['sentimento_classificacao'] = df['sentimento'].apply(
        lambda x: 'positivo' if x > 0 else ('negativo' if x < 0 else 'neutro')
    )
    df['target'] = df['sentimento_classificacao'].apply(lambda x: 1 if x == 'positivo' else 0)  # Mantendo binário

    # Salvar um resumo da análise de sentimento
    sentimento_summary = df['sentimento_classificacao'].value_counts().to_frame().reset_index()
    sentimento_summary.columns = ['Sentimento', 'Contagem']
    sentimento_file = os.path.join(resultados_dir, f"{transformed_table}_sentimento_summary.csv")
    sentimento_summary.to_csv(sentimento_file, index=False)
    print(f"Resumo de sentimentos salvo em '{sentimento_file}'.")

    # Salvar dados processados no MongoDB
    salvar_no_mongo(df.to_dict(orient='records'), f"{collection_name}_processed")  # type: ignore
    print(f"Dados processados salvos na coleção '{collection_name}_processed' no MongoDB.")

    # Verificar se há dados suficientes para treinamento
    contagem_targets = df['target'].value_counts()
    if contagem_targets.shape[0] < 2:
        print(f"Não há classes suficientes para treinar IA em {transformed_table}.")
        return df

    with duckdb_lock:
        conn = duckdb.connect(DUCKDB_FILE)
        conn.register("df_view", df)
        conn.execute(f"CREATE OR REPLACE TABLE {processed_table} AS SELECT * FROM df_view")
        conn.close()
    print(f"Dados tratados salvos na tabela '{processed_table}' no DuckDB.")
    return df

# Asset: Tratar dados de economia
@asset(
    description="Trata os dados transformados de economia e salva no DuckDB e MongoDB.",
    non_argument_deps={"executar_dbt"},
    kinds={"duckdb", "pandas" , "mongodb"},
)
def tratar_dados_economia() -> pd.DataFrame:
    return tratar_dados_func("economia_transformed", "economia_processed", RESULTADOS_DIR, "economia_processed")

# Asset: Tratar dados de governo
@asset(
    description="Trata os dados transformados de governo e salva no DuckDB e MongoDB.",
    non_argument_deps={"executar_dbt"},
    kinds={"duckdb", "pandas" , "mongodb"},
)
def tratar_dados_governo() -> pd.DataFrame:
    return tratar_dados_func("governo_transformed", "governo_processed", RESULTADOS_DIR, "governo_processed")

# Função para treinar IA
def treinar_ia(df: pd.DataFrame, nome_modelo: str, resultados_dir: str):
    """
    Treina modelos de IA utilizando os dados fornecidos e salva as métricas e relatórios.

    Args:
        df (pd.DataFrame): DataFrame com os dados para treinamento.
        nome_modelo (str): Nome do modelo (economia/governo).
        resultados_dir (str): Diretório para salvar os resultados.
    """
    if df.empty:
        print(f"Nenhum dado disponível para treinamento em {nome_modelo}.")
        return
    
    X = df[['sentimento']].fillna(0)
    y = df['target']
    if y.nunique() <= 1:
        print(f"Dados insuficientes para treinar IA em {nome_modelo}.")
        return

    # Dividir os dados
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Verificar se ambas as classes estão presentes em y_train e y_test
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        print(f"Dados insuficientes para treinar IA em {nome_modelo}.")
        return

    # Definir os algoritmos a serem utilizados
    algoritmos = {
        "RandomForest": RandomForestClassifier(random_state=42),
        "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000),
        "SVM": SVC(random_state=42)
    }
    
    # DataFrame para armazenar métricas
    metrics_df = pd.DataFrame(columns=["Algoritmo", "Acurácia", "Precisão", "Recall", "F1-Score"])
    metrics_list = []  # Coletar métricas em uma lista

    # Matriz de Confusão para cada algoritmo
    for nome_algoritmo, modelo in algoritmos.items():
        try:
            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)
            
            acuracia = accuracy_score(y_test, y_pred)
            precisao = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            # Adicionar métricas à lista
            metrics_list.append({
                "Algoritmo": nome_algoritmo,
                "Acurácia": acuracia,
                "Precisão": precisao,
                "Recall": recall,
                "F1-Score": f1
            })
            
            # Salvar a Matriz de Confusão com labels especificados para evitar ValueError
            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
            cm_df = pd.DataFrame(cm, index=["Negativo", "Positivo"], columns=["Negativo", "Positivo"])
            cm_file = os.path.join(resultados_dir, f"{nome_modelo}_{nome_algoritmo}_confusion_matrix.csv")
            cm_df.to_csv(cm_file)
            print(f"Matriz de Confusão para {nome_algoritmo} salva em '{cm_file}'.")
        except ValueError as e:
            print(f"Erro ao treinar {nome_algoritmo} em {nome_modelo}: {e}")
            continue

    # Criar DataFrame de métricas a partir da lista
    if metrics_list:
        new_metrics = pd.DataFrame(metrics_list)
        metrics_df = pd.concat([metrics_df, new_metrics], ignore_index=True)
    else:
        print(f"Nenhuma métrica foi coletada para {nome_modelo}.")

    # Salvar as métricas
    if not metrics_df.empty:
        metrics_file = os.path.join(resultados_dir, f"{nome_modelo}_metrics.csv")
        metrics_df.to_csv(metrics_file, index=False)
        print(f"Métricas dos modelos salvas em '{metrics_file}'.")
    else:
        print(f"Nenhuma métrica foi salva para {nome_modelo}.")

    # Gerar relatório de análise de sentimento
    sentimento_summary = df['sentimento_classificacao'].value_counts().to_frame().reset_index()
    sentimento_summary.columns = ['Sentimento', 'Contagem']
    sentimento_file = os.path.join(resultados_dir, f"{nome_modelo}_sentimento_summary.csv")
    sentimento_summary.to_csv(sentimento_file, index=False)
    print(f"Resumo de sentimentos salvo em '{sentimento_file}'.")

    # Gerar Relatório Detalhado das Métricas
    if not metrics_df.empty:
        relatorio_path = os.path.join(resultados_dir, f"{nome_modelo}_relatorio.txt")
        with open(relatorio_path, 'w') as relatorio:
            relatorio.write(f"Relatório de Treinamento de IA para {nome_modelo}\n")
            relatorio.write("="*50 + "\n\n")
            
            relatorio.write("Distribuição de Sentimentos:\n")
            relatorio.write(sentimento_summary.to_string(index=False))
            relatorio.write("\n\nMétricas dos Modelos:\n")
            relatorio.write(metrics_df.to_string(index=False))
            relatorio.write("\n\nObservações:\n")
            relatorio.write("A IA está analisando o sentimento das notícias com base na pontuação de sentimento fornecida pelo VADER.\n")
            relatorio.write("Classes utilizadas: Positivo (1), Negativo (0)\n")
        
        print(f"Relatório detalhado salvo em '{relatorio_path}'.")
    else:
        print(f"Nenhum relatório foi gerado para {nome_modelo}.")

# Asset: Treinar IA com dados de economia
@asset(
    description="Treina um modelo de IA com dados de economia processados e salva métricas no DuckDB e MongoDB.",
    non_argument_deps={"tratar_dados_economia"},
    kinds={"python", "scikitlearn"},
)
def treinar_ia_economia() -> None:
    df = carregar_dados_duckdb("economia_processed")
    treinar_ia(df, "economia", RESULTADOS_DIR)

# Asset: Treinar IA com dados de governo
@asset(
    description="Treina um modelo de IA com dados de governo processados e salva métricas no DuckDB e MongoDB.",
    non_argument_deps={"tratar_dados_governo"},
    kinds={"python", "scikitlearn"},
)
def treinar_ia_governo() -> None:
    df = carregar_dados_duckdb("governo_processed")
    treinar_ia(df, "governo", RESULTADOS_DIR)

# Asset: Verificar dados transformados
@asset(
    description="Verifica e imprime amostras dos dados transformados no DuckDB.",
    non_argument_deps={"tratar_dados_economia", "tratar_dados_governo"},
    kinds={"python"},
)
def verificar_dados_transformados() -> None:
    import duckdb

    conn = duckdb.connect(DUCKDB_FILE)

    # Verificar dados transformados de governo
    governo_transformed_df = conn.execute("SELECT * FROM governo_transformed LIMIT 5").fetchdf()
    print("Dados transformados de governo:")
    print(governo_transformed_df)

    # Verificar dados transformados de economia
    economia_transformed_df = conn.execute("SELECT * FROM economia_transformed LIMIT 5").fetchdf()
    print("Dados transformados de economia:")
    print(economia_transformed_df)

    conn.close()

# Asset: Verificar dados disponíveis para IA
@asset(
    description="Verifica a quantidade de dados disponíveis para treinamento de IA.",
    non_argument_deps={"tratar_dados_economia", "tratar_dados_governo"},
    kinds={"python"},
)
def verificar_dados_disponiveis() -> None:
    economia_df = carregar_dados_duckdb("economia_processed")
    governo_df = carregar_dados_duckdb("governo_processed")
    
    print(f"Dados disponíveis para Economia: {economia_df.shape[0]} registros.")
    print(f"Dados disponíveis para Governo: {governo_df.shape[0]} registros.")'''
    
    
# assets.py
'''
import os
import json
import pandas as pd
from datetime import datetime
from dagster import asset
from scrapy.crawler import CrawlerProcess
from crawler_noticia.economia.economia.spiders.noticia import NoticiasSpider
from crawler_noticia.governo.governo.spiders.noticia import G1Spider
import duckdb
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from filelock import FileLock
import hashlib
import subprocess

# Importar funções de MongoDB
from db_mongo.conexao_mongo import salvar_no_mongo

# Diretório base do projeto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Pasta para armazenar resultados
RESULTADOS_DIR = os.path.join(BASE_DIR, "projeto_dados", "resultados")
os.makedirs(RESULTADOS_DIR, exist_ok=True)

# Arquivo DuckDB
DUCKDB_FILE = os.path.join(BASE_DIR, "noticias.duckdb")
DBT_PROJECT_PATH = os.path.join(BASE_DIR, "dbt_project")

# Lock para controle de acesso ao DuckDB
duckdb_lock = FileLock(os.path.join(BASE_DIR, "duckdb.lock"))

def run_spider(spider, raw_table_name, collection_name):
    """
    Executa o spider do Scrapy para coletar dados e salva no DuckDB e MongoDB.

    Args:
        spider: Classe do spider do Scrapy.
        raw_table_name (str): Nome da tabela bruta no DuckDB.
        collection_name (str): Nome da coleção no MongoDB.
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_file = f"projeto_dados/data/{raw_table_name}_{timestamp}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    process = CrawlerProcess(settings={
        "FEEDS": {output_file: {"format": "json"}},
        "FEED_EXPORT_ENCODING": "utf-8",
    })
    process.crawl(spider)
    process.start()

    with open(output_file, "r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)
    
    # Processar dados para estrutura consistente
    processed_data = []
    if raw_table_name == "economia_raw":
        for item in data:
            for time_key, titles in item.items():
                for title in titles:
                    processed_data.append({
                        "id": hashlib.md5(f"{title}{time_key}".encode()).hexdigest(),
                        "titulo_noticia": title,
                        "texto": "",  # Preencher se disponível
                        "data_publicacao": None,  # Preencher se disponível
                        "time_ago": time_key,
                    })
    elif raw_table_name == "governo_raw":
        for item in data:
            processed_data.append({
                "id": hashlib.md5(str(item).encode()).hexdigest(),
                "titulo_noticia": item.get("title", ""),
                "texto": item.get("body", ""),
                "data_publicacao": item.get("data_publicacao", ""),
            })

    df = pd.DataFrame(processed_data)
    
    # Garantir coluna 'id' presente
    if 'id' not in df.columns:
        df['id'] = df.apply(lambda row: hashlib.md5(str(row).encode()).hexdigest(), axis=1)

    # Preencher 'data_publicacao' se ausente
    if 'data_publicacao' not in df.columns:
        df['data_publicacao'] = pd.NaT

    with duckdb_lock:
        conn = duckdb.connect(DUCKDB_FILE)
        conn.register("df_view", df)
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {raw_table_name}").fetchone()[0]  # type: ignore
        except duckdb.BinderException:
            count = 0

        if count > 0:
            # Se a tabela já existe e possui dados, apende os novos dados sem duplicatas
            conn.execute(f"""
                INSERT INTO {raw_table_name} 
                SELECT * FROM df_view 
                WHERE id NOT IN (SELECT id FROM {raw_table_name})
            """)
            print(f"Novos dados apendados na tabela '{raw_table_name}' no DuckDB.")
        else:
            # Se a tabela não existe ou está vazia, cria a tabela
            conn.execute(f"CREATE TABLE {raw_table_name} AS SELECT * FROM df_view")
            print(f"Tabela '{raw_table_name}' criada e dados inseridos no DuckDB.")
        conn.close()
    print(f"Dados salvos na tabela '{raw_table_name}' no DuckDB.")

    # Salvar dados brutos no MongoDB
    salvar_no_mongo(processed_data, collection_name)  # type: ignore
    print(f"Novos dados adicionados à coleção '{collection_name}' no MongoDB.")

# Asset: Executar o crawler de economia
@asset(
    description="Executa o crawler para coletar notícias de economia e salva os dados brutos no DuckDB e MongoDB.",
    kinds={"python"},
)
def crawler_economia() -> None:
    run_spider(NoticiasSpider, "economia_raw", "economia_raw")

# Asset: Executar o crawler de governo
@asset(
    description="Executa o crawler para coletar notícias de governo e salva os dados brutos no DuckDB e MongoDB.",
    kinds={"python"},
)
def crawler_governo() -> None:
    run_spider(G1Spider, "governo_raw", "governo_raw")

# Asset: Executar DBT para transformação de dados
@asset(
    description="Executa os modelos DBT para transformar os dados brutos.",
    non_argument_deps={"crawler_economia", "crawler_governo"},
    kinds={"dbt"},
)
def executar_dbt() -> None:
    if not os.path.exists(DBT_PROJECT_PATH):
        raise NotADirectoryError(f"O diretório especificado para o DBT não existe: {DBT_PROJECT_PATH}")
    subprocess.run(["dbt", "run"], check=True, cwd=DBT_PROJECT_PATH)
    print("Modelos DBT executados com sucesso.")

# Função para carregar dados do DuckDB
def carregar_dados_duckdb(table_name):
    """
    Carrega dados de uma tabela DuckDB.

    Args:
        table_name (str): Nome da tabela no DuckDB.

    Returns:
        pd.DataFrame: DataFrame contendo os dados da tabela.
    """
    with duckdb_lock:
        conn = duckdb.connect(DUCKDB_FILE)
        try:
            df = conn.execute(f"SELECT * FROM {table_name}").fetchdf()
        except duckdb.BinderException:
            print(f"Tabela {table_name} não encontrada no DuckDB.")
            df = pd.DataFrame()
        conn.close()
    return df

# Função para tratar os dados processados
def tratar_dados_func(transformed_table, processed_table, resultados_dir, collection_name):
    """
    Trata os dados transformados, realiza análise de sentimento e prepara para treinamento de IA.

    Args:
        transformed_table (str): Nome da tabela transformada no DuckDB.
        processed_table (str): Nome da tabela processada no DuckDB.
        resultados_dir (str): Diretório para salvar os resultados.
        collection_name (str): Nome da coleção no MongoDB para salvar os dados processados.

    Returns:
        pd.DataFrame: DataFrame processado.
    """
    df = carregar_dados_duckdb(transformed_table)
    if df.empty:
        print(f"Nenhum dado encontrado na tabela {transformed_table}.")
        return pd.DataFrame()

    analyzer = SentimentIntensityAnalyzer()
    # Se 'texto' estiver vazio, usar 'titulo_noticia' para análise de sentimento
    df['texto_para_sentimento'] = df.apply(
        lambda row: row['texto'] if isinstance(row['texto'], str) and row['texto'].strip() else row['titulo_noticia'],
        axis=1
    )
    df['sentimento'] = df['texto_para_sentimento'].apply(lambda x: analyzer.polarity_scores(x)['compound'] if isinstance(x, str) else 0)
    
    # Preencher valores None com 0
    df['sentimento'] = df['sentimento'].fillna(0)
    
    # Ajustar a classificação para seguir os limiares padrão do VADER
    df['sentimento_classificacao'] = df['sentimento'].apply(
        lambda x: 'positivo' if x >= 0.05 else ('negativo' if x <= -0.05 else 'neutro')
    )
    df['target'] = df['sentimento_classificacao'].apply(lambda x: 1 if x == 'positivo' else 0)  # Mantendo binário

    # Salvar um resumo da análise de sentimento
    sentimento_summary = df['sentimento_classificacao'].value_counts().to_frame().reset_index()
    sentimento_summary.columns = ['Sentimento', 'Contagem']
    sentimento_file = os.path.join(resultados_dir, f"{transformed_table}_sentimento_summary.csv")
    sentimento_summary.to_csv(sentimento_file, index=False)
    print(f"Resumo de sentimentos salvo em '{sentimento_file}'.")

    # Salvar dados processados no MongoDB
    salvar_no_mongo(df.to_dict(orient='records'), f"{collection_name}_processed")  # type: ignore
    print(f"Dados processados salvos na coleção '{collection_name}_processed' no MongoDB.")

    # Verificar se há dados suficientes para treinamento
    contagem_targets = df['target'].value_counts()
    if contagem_targets.shape[0] < 2:
        print(f"Não há classes suficientes para treinar IA em {transformed_table}.")
        return df

    with duckdb_lock:
        conn = duckdb.connect(DUCKDB_FILE)
        conn.register("df_view", df)
        conn.execute(f"CREATE OR REPLACE TABLE {processed_table} AS SELECT * FROM df_view")
        conn.close()
    print(f"Dados tratados salvos na tabela '{processed_table}' no DuckDB.")
    return df

# Asset: Tratar dados de economia
@asset(
    description="Trata os dados transformados de economia e salva no DuckDB e MongoDB.",
    non_argument_deps={"executar_dbt"},
    kinds={"duckdb", "pandas" , "mongodb"},
)
def tratar_dados_economia() -> pd.DataFrame:
    return tratar_dados_func("economia_transformed", "economia_processed", RESULTADOS_DIR, "economia_processed")

# Asset: Tratar dados de governo
@asset(
    description="Trata os dados transformados de governo e salva no DuckDB e MongoDB.",
    non_argument_deps={"executar_dbt"},
    kinds={"duckdb", "pandas" , "mongodb"},
)
def tratar_dados_governo() -> pd.DataFrame:
    return tratar_dados_func("governo_transformed", "governo_processed", RESULTADOS_DIR, "governo_processed")

# Função para treinar IA
def treinar_ia(df: pd.DataFrame, nome_modelo: str, resultados_dir: str):
    """
    Treina modelos de IA utilizando os dados fornecidos e salva as métricas e relatórios.

    Args:
        df (pd.DataFrame): DataFrame com os dados para treinamento.
        nome_modelo (str): Nome do modelo (economia/governo).
        resultados_dir (str): Diretório para salvar os resultados.
    """
    if df.empty:
        print(f"Nenhum dado disponível para treinamento em {nome_modelo}.")
        return
    
    X = df[['sentimento']].fillna(0)
    y = df['target']
    if y.nunique() <= 1:
        print(f"Dados insuficientes para treinar IA em {nome_modelo}.")
        return

    # Dividir os dados com estratificação para manter a distribuição das classes
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError as e:
        print(f"Erro ao dividir os dados para {nome_modelo}: {e}")
        return
    
    # Verificar se ambas as classes estão presentes em y_train e y_test
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        print(f"Dados insuficientes para treinar IA em {nome_modelo} após divisão.")
        return

    # Definir os algoritmos a serem utilizados
    algoritmos = {
        "RandomForest": RandomForestClassifier(random_state=42),
        "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000),
        "SVM": SVC(random_state=42)
    }
    
    # DataFrame para armazenar métricas
    metrics_list = []  # Coletar métricas em uma lista

    # Matriz de Confusão para cada algoritmo
    for nome_algoritmo, modelo in algoritmos.items():
        try:
            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)
            
            acuracia = accuracy_score(y_test, y_pred)
            precisao = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            # Adicionar métricas à lista
            metrics_list.append({
                "Algoritmo": nome_algoritmo,
                "Acurácia": acuracia,
                "Precisão": precisao,
                "Recall": recall,
                "F1-Score": f1
            })
            
            # Salvar a Matriz de Confusão com labels especificados para evitar ValueError
            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
            cm_df = pd.DataFrame(cm, index=["Negativo", "Positivo"], columns=["Negativo", "Positivo"])
            cm_file = os.path.join(resultados_dir, f"{nome_modelo}_{nome_algoritmo}_confusion_matrix.csv")
            cm_df.to_csv(cm_file)
            print(f"Matriz de Confusão para {nome_algoritmo} salva em '{cm_file}'.")
        except ValueError as e:
            print(f"Erro ao treinar {nome_algoritmo} em {nome_modelo}: {e}")
            continue

    # Criar DataFrame de métricas a partir da lista
    if metrics_list:
        metrics_df = pd.DataFrame(metrics_list)
    else:
        metrics_df = pd.DataFrame(columns=["Algoritmo", "Acurácia", "Precisão", "Recall", "F1-Score"])
        print(f"Nenhuma métrica foi coletada para {nome_modelo}.")

    # Salvar as métricas
    if not metrics_df.empty:
        metrics_file = os.path.join(resultados_dir, f"{nome_modelo}_metrics.csv")
        metrics_df.to_csv(metrics_file, index=False)
        print(f"Métricas dos modelos salvas em '{metrics_file}'.")
    else:
        print(f"Nenhuma métrica foi salva para {nome_modelo}.")

    # Gerar relatório de análise de sentimento
    sentimento_summary = df['sentimento_classificacao'].value_counts().to_frame().reset_index()
    sentimento_summary.columns = ['Sentimento', 'Contagem']
    sentimento_file = os.path.join(resultados_dir, f"{nome_modelo}_sentimento_summary.csv")
    sentimento_summary.to_csv(sentimento_file, index=False)
    print(f"Resumo de sentimentos salvo em '{sentimento_file}'.")

    # Gerar Relatório Detalhado das Métricas
    if not metrics_df.empty:
        relatorio_path = os.path.join(resultados_dir, f"{nome_modelo}_relatorio.txt")
        with open(relatorio_path, 'w') as relatorio:
            relatorio.write(f"Relatório de Treinamento de IA para {nome_modelo}\n")
            relatorio.write("="*50 + "\n\n")
            
            relatorio.write("Distribuição de Sentimentos:\n")
            relatorio.write(sentimento_summary.to_string(index=False))
            relatorio.write("\n\nMétricas dos Modelos:\n")
            relatorio.write(metrics_df.to_string(index=False))
            relatorio.write("\n\nObservações:\n")
            relatorio.write("A IA está analisando o sentimento das notícias com base na pontuação de sentimento fornecida pelo VADER.\n")
            relatorio.write("Classes utilizadas: Positivo (1), Negativo (0)\n")
        
        print(f"Relatório detalhado salvo em '{relatorio_path}'.")
    else:
        print(f"Nenhum relatório foi gerado para {nome_modelo}.")

# Asset: Treinar IA com dados de economia
@asset(
    description="Treina um modelo de IA com dados de economia processados e salva métricas no DuckDB e MongoDB.",
    non_argument_deps={"tratar_dados_economia"},
    kinds={"python", "scikitlearn"},
)
def treinar_ia_economia() -> None:
    df = carregar_dados_duckdb("economia_processed")
    treinar_ia(df, "economia", RESULTADOS_DIR)

# Asset: Treinar IA com dados de governo
@asset(
    description="Treina um modelo de IA com dados de governo processados e salva métricas no DuckDB e MongoDB.",
    non_argument_deps={"tratar_dados_governo"},
    kinds={"python", "scikitlearn"},
)
def treinar_ia_governo() -> None:
    df = carregar_dados_duckdb("governo_processed")
    treinar_ia(df, "governo", RESULTADOS_DIR)

# Asset: Verificar dados transformados
@asset(
    description="Verifica e imprime amostras dos dados transformados no DuckDB.",
    non_argument_deps={"tratar_dados_economia", "tratar_dados_governo"},
    kinds={"python"},
)
def verificar_dados_transformados() -> None:
    import duckdb

    conn = duckdb.connect(DUCKDB_FILE)

    # Verificar dados transformados de governo
    governo_transformed_df = conn.execute("SELECT * FROM governo_transformed LIMIT 5").fetchdf()
    print("Dados transformados de governo:")
    print(governo_transformed_df)

    # Verificar dados transformados de economia
    economia_transformed_df = conn.execute("SELECT * FROM economia_transformed LIMIT 5").fetchdf()
    print("Dados transformados de economia:")
    print(economia_transformed_df)

    conn.close()

# Asset: Verificar dados disponíveis para IA
@asset(
    description="Verifica a quantidade de dados disponíveis para treinamento de IA.",
    non_argument_deps={"tratar_dados_economia", "tratar_dados_governo"},
    kinds={"python"},
)
def verificar_dados_disponiveis() -> None:
    economia_df = carregar_dados_duckdb("economia_processed")
    governo_df = carregar_dados_duckdb("governo_processed")
    
    print(f"Dados disponíveis para Economia: {economia_df.shape[0]} registros.")
    print(f"Dados disponíveis para Governo: {governo_df.shape[0]} registros.")

'''
'''
import os
import json
import pandas as pd
from datetime import datetime
from dagster import asset, AssetIn
from scrapy.crawler import CrawlerProcess
from crawler_noticia.economia.economia.spiders.noticia import NoticiasSpider
from crawler_noticia.governo.governo.spiders.noticia import G1Spider
import duckdb
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from filelock import FileLock
import hashlib
import subprocess

# Importar funções de MongoDB
from db_mongo.conexao_mongo import salvar_no_mongo

# Diretório base do projeto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Pasta para armazenar resultados
RESULTADOS_DIR = os.path.join(BASE_DIR, "projeto_dados", "resultados")
os.makedirs(RESULTADOS_DIR, exist_ok=True)

# Arquivo DuckDB
DUCKDB_FILE = os.path.join(BASE_DIR, "noticias.duckdb")
DBT_PROJECT_PATH = os.path.join(BASE_DIR, "dbt_project")

# Lock para controle de acesso ao DuckDB
duckdb_lock = FileLock(os.path.join(BASE_DIR, "duckdb.lock"))

def run_spider(spider, raw_table_name, collection_name):
    """
    Executa o spider do Scrapy para coletar dados e salva no DuckDB e MongoDB.

    Args:
        spider: Classe do spider do Scrapy.
        raw_table_name (str): Nome da tabela bruta no DuckDB.
        collection_name (str): Nome da coleção no MongoDB.
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_file = f"projeto_dados/data/{raw_table_name}_{timestamp}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    process = CrawlerProcess(settings={
        "FEEDS": {output_file: {"format": "json"}},
        "FEED_EXPORT_ENCODING": "utf-8",
    })
    process.crawl(spider)
    process.start()

    with open(output_file, "r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)
    
    # Processar dados para estrutura consistente
    processed_data = []
    if raw_table_name == "economia_raw":
        for item in data:
            for time_key, titles in item.items():
                for title in titles:
                    processed_data.append({
                        "id": hashlib.md5(f"{title}{time_key}".encode()).hexdigest(),
                        "titulo_noticia": title,
                        "texto": "",  # Preencher se disponível
                        "data_publicacao": None,  # Preencher se disponível
                        "time_ago": time_key,
                    })
    elif raw_table_name == "governo_raw":
        for item in data:
            processed_data.append({
                "id": hashlib.md5(str(item).encode()).hexdigest(),
                "titulo_noticia": item.get("title", ""),
                "texto": item.get("body", ""),
                "data_publicacao": item.get("data_publicacao", ""),
            })

    df = pd.DataFrame(processed_data)
    
    # Garantir coluna 'id' presente
    if 'id' not in df.columns:
        df['id'] = df.apply(lambda row: hashlib.md5(str(row).encode()).hexdigest(), axis=1)

    # Preencher 'data_publicacao' se ausente
    if 'data_publicacao' not in df.columns:
        df['data_publicacao'] = pd.NaT

    with duckdb_lock:
        conn = duckdb.connect(DUCKDB_FILE)
        conn.register("df_view", df)
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {raw_table_name}").fetchone()[0]  # type: ignore
        except duckdb.BinderException:
            count = 0

        if count > 0:
            # Se a tabela já existe e possui dados, apende os novos dados sem duplicatas
            conn.execute(f"""
                INSERT INTO {raw_table_name} 
                SELECT * FROM df_view 
                WHERE id NOT IN (SELECT id FROM {raw_table_name})
            """)
            print(f"Novos dados apendados na tabela '{raw_table_name}' no DuckDB.")
        else:
            # Se a tabela não existe ou está vazia, cria a tabela
            conn.execute(f"CREATE TABLE {raw_table_name} AS SELECT * FROM df_view")
            print(f"Tabela '{raw_table_name}' criada e dados inseridos no DuckDB.")
        conn.close()
    print(f"Dados salvos na tabela '{raw_table_name}' no DuckDB.")

    # Salvar dados brutos no MongoDB
    salvar_no_mongo(processed_data, collection_name)  # type: ignore
    print(f"Novos dados adicionados à coleção '{collection_name}' no MongoDB.")

# Asset: Executar o crawler de economia
@asset(
    description="Executa o crawler para coletar notícias de economia e salva os dados brutos no DuckDB e MongoDB.",
    kinds={"python"},
)
def crawler_economia() -> None:
    run_spider(NoticiasSpider, "economia_raw", "economia_raw")

# Asset: Executar o crawler de governo
@asset(
    description="Executa o crawler para coletar notícias de governo e salva os dados brutos no DuckDB e MongoDB.",
    kinds={"python"},
)
def crawler_governo() -> None:
    run_spider(G1Spider, "governo_raw", "governo_raw")

# Asset: Executar DBT para transformação de dados
@asset(
    description="Executa os modelos DBT para transformar os dados brutos.",
    non_argument_deps={"crawler_economia", "crawler_governo"},
    kinds={"dbt"},
)
def executar_dbt() -> None:
    if not os.path.exists(DBT_PROJECT_PATH):
        raise NotADirectoryError(f"O diretório especificado para o DBT não existe: {DBT_PROJECT_PATH}")
    subprocess.run(["dbt", "run"], check=True, cwd=DBT_PROJECT_PATH)
    print("Modelos DBT executados com sucesso.")

# Função para carregar dados do DuckDB
def carregar_dados_duckdb(table_name):
    """
    Carrega dados de uma tabela DuckDB.

    Args:
        table_name (str): Nome da tabela no DuckDB.

    Returns:
        pd.DataFrame: DataFrame contendo os dados da tabela.
    """
    with duckdb_lock:
        conn = duckdb.connect(DUCKDB_FILE)
        try:
            df = conn.execute(f"SELECT * FROM {table_name}").fetchdf()
        except duckdb.BinderException:
            print(f"Tabela {table_name} não encontrada no DuckDB.")
            df = pd.DataFrame()
        conn.close()
    return df

# Função para tratar os dados processados
def tratar_dados_func(transformed_table, processed_table, resultados_dir, collection_name):
    """
    Trata os dados transformados, realiza análise de sentimento e prepara para treinamento de IA.

    Args:
        transformed_table (str): Nome da tabela transformada no DuckDB.
        processed_table (str): Nome da tabela processada no DuckDB.
        resultados_dir (str): Diretório para salvar os resultados.
        collection_name (str): Nome da coleção no MongoDB para salvar os dados processados.

    Returns:
        pd.DataFrame: DataFrame processado.
    """
    df = carregar_dados_duckdb(transformed_table)
    if df.empty:
        print(f"Nenhum dado encontrado na tabela {transformed_table}.")
        return pd.DataFrame()

    analyzer = SentimentIntensityAnalyzer()
    # Se 'texto' estiver vazio, usar 'titulo_noticia' para análise de sentimento
    df['texto_para_sentimento'] = df.apply(
        lambda row: row['texto'] if isinstance(row['texto'], str) and row['texto'].strip() else row['titulo_noticia'],
        axis=1
    )
    df['sentimento'] = df['texto_para_sentimento'].apply(lambda x: analyzer.polarity_scores(x)['compound'] if isinstance(x, str) else 0)
    
    # Preencher valores None com 0
    df['sentimento'] = df['sentimento'].fillna(0)
    
    # Ajustar a classificação para seguir os limiares padrão do VADER
    df['sentimento_classificacao'] = df['sentimento'].apply(
        lambda x: 'positivo' if x >= 0.05 else ('negativo' if x <= -0.05 else 'neutro')
    )
    df['target'] = df['sentimento_classificacao'].apply(lambda x: 1 if x == 'positivo' else 0)  # Mantendo binário

    # Salvar um resumo da análise de sentimento
    sentimento_summary = df['sentimento_classificacao'].value_counts().to_frame().reset_index()
    sentimento_summary.columns = ['Sentimento', 'Contagem']
    sentimento_file = os.path.join(resultados_dir, f"{transformed_table}_sentimento_summary.csv")
    sentimento_summary.to_csv(sentimento_file, index=False)
    print(f"Resumo de sentimentos salvo em '{sentimento_file}'.")

    # Salvar dados processados no MongoDB
    salvar_no_mongo(df.to_dict(orient='records'), f"{collection_name}_processed")  # type: ignore
    print(f"Dados processados salvos na coleção '{collection_name}_processed' no MongoDB.")

    # Verificar se há dados suficientes para treinamento
    contagem_targets = df['target'].value_counts()
    if contagem_targets.shape[0] < 2:
        print(f"Não há classes suficientes para treinar IA em {transformed_table}.")
        return df

    with duckdb_lock:
        conn = duckdb.connect(DUCKDB_FILE)
        conn.register("df_view", df)
        conn.execute(f"CREATE OR REPLACE TABLE {processed_table} AS SELECT * FROM df_view")
        conn.close()
    print(f"Dados tratados salvos na tabela '{processed_table}' no DuckDB.")
    return df

# Função para treinar IA
def treinar_ia(df: pd.DataFrame, nome_modelo: str, resultados_dir: str):
    """
    Treina modelos de IA utilizando os dados fornecidos e salva as métricas e relatórios.

    Args:
        df (pd.DataFrame): DataFrame com os dados para treinamento.
        nome_modelo (str): Nome do modelo (economia/governo).
        resultados_dir (str): Diretório para salvar os resultados.
    """
    if df.empty:
        print(f"Nenhum dado disponível para treinamento em {nome_modelo}.")
        return
    
    X = df[['sentimento']].fillna(0)
    y = df['target']
    if y.nunique() <= 1:
        print(f"Dados insuficientes para treinar IA em {nome_modelo}.")
        return

    # Dividir os dados com estratificação para manter a distribuição das classes
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError as e:
        print(f"Erro ao dividir os dados para {nome_modelo}: {e}")
        return

    # Verificar se ambas as classes estão presentes em y_train e y_test
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        print(f"Dados insuficientes para treinar IA em {nome_modelo} após divisão.")
        return

    # Definir os algoritmos a serem utilizados
    algoritmos = {
        "RandomForest": RandomForestClassifier(random_state=42),
        "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000),
        "SVM": SVC(random_state=42)
    }
    
    # DataFrame para armazenar métricas
    metrics_list = []  # Coletar métricas em uma lista

    # Matriz de Confusão para cada algoritmo
    for nome_algoritmo, modelo in algoritmos.items():
        try:
            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)
            
            acuracia = accuracy_score(y_test, y_pred)
            precisao = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            # Adicionar métricas à lista
            metrics_list.append({
                "Algoritmo": nome_algoritmo,
                "Acurácia": acuracia,
                "Precisão": precisao,
                "Recall": recall,
                "F1-Score": f1
            })
            
            # Salvar a Matriz de Confusão com labels especificados para evitar ValueError
            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
            cm_df = pd.DataFrame(cm, index=["Negativo", "Positivo"], columns=["Negativo", "Positivo"])
            cm_file = os.path.join(resultados_dir, f"{nome_modelo}_{nome_algoritmo}_confusion_matrix.csv")
            cm_df.to_csv(cm_file)
            print(f"Matriz de Confusão para {nome_algoritmo} salva em '{cm_file}'.")
        except ValueError as e:
            print(f"Erro ao treinar {nome_algoritmo} em {nome_modelo}: {e}")
            continue

    # Criar DataFrame de métricas a partir da lista
    if metrics_list:
        metrics_df = pd.DataFrame(metrics_list)
    else:
        metrics_df = pd.DataFrame(columns=["Algoritmo", "Acurácia", "Precisão", "Recall", "F1-Score"])
        print(f"Nenhuma métrica foi coletada para {nome_modelo}.")

    # Salvar as métricas
    if not metrics_df.empty:
        metrics_file = os.path.join(resultados_dir, f"{nome_modelo}_metrics.csv")
        metrics_df.to_csv(metrics_file, index=False)
        print(f"Métricas dos modelos salvas em '{metrics_file}'.")
    else:
        print(f"Nenhuma métrica foi salva para {nome_modelo}.")

    # Gerar relatório de análise de sentimento
    sentimento_summary = df['sentimento_classificacao'].value_counts().to_frame().reset_index()
    sentimento_summary.columns = ['Sentimento', 'Contagem']
    sentimento_file = os.path.join(resultados_dir, f"{nome_modelo}_sentimento_summary.csv")
    sentimento_summary.to_csv(sentimento_file, index=False)
    print(f"Resumo de sentimentos salvo em '{sentimento_file}'.")

    # Gerar Relatório Detalhado das Métricas
    if not metrics_df.empty:
        relatorio_path = os.path.join(resultados_dir, f"{nome_modelo}_relatorio.txt")
        with open(relatorio_path, 'w') as relatorio:
            relatorio.write(f"Relatório de Treinamento de IA para {nome_modelo}\n")
            relatorio.write("="*50 + "\n\n")
            
            relatorio.write("Distribuição de Sentimentos:\n")
            relatorio.write(sentimento_summary.to_string(index=False))
            relatorio.write("\n\nMétricas dos Modelos:\n")
            relatorio.write(metrics_df.to_string(index=False))
            relatorio.write("\n\nObservações:\n")
            relatorio.write("A IA está analisando o sentimento das notícias com base na pontuação de sentimento fornecida pelo VADER.\n")
            relatorio.write("Classes utilizadas: Positivo (1), Negativo (0)\n")
        
        print(f"Relatório detalhado salvo em '{relatorio_path}'.")
    else:
        print(f"Nenhum relatório foi gerado para {nome_modelo}.")

# Asset: Tratar dados de economia
@asset(
    description="Trata os dados transformados de economia e salva no DuckDB e MongoDB.",
    non_argument_deps={"executar_dbt"},
    kinds={"duckdb", "pandas", "mongodb"},
)
def tratar_dados_economia() -> pd.DataFrame:
    return tratar_dados_func("economia_transformed", "economia_processed", RESULTADOS_DIR, "economia_processed")

# Asset: Tratar dados de governo
@asset(
    description="Trata os dados transformados de governo e salva no DuckDB e MongoDB.",
    non_argument_deps={"executar_dbt"},
    kinds={"duckdb", "pandas", "mongodb"},
)
def tratar_dados_governo() -> pd.DataFrame:
    return tratar_dados_func("governo_transformed", "governo_processed", RESULTADOS_DIR, "governo_processed")

# Asset: Treinar IA com dados de economia
@asset(
    description="Treina um modelo de IA com dados de economia processados e salva métricas no DuckDB e MongoDB.",
    non_argument_deps={"tratar_dados_economia"},
    kinds={"python", "scikitlearn"},
)
def treinar_ia_economia() -> None:
    df = carregar_dados_duckdb("economia_processed")
    treinar_ia(df, "economia", RESULTADOS_DIR)

# Asset: Treinar IA com dados de governo
@asset(
    description="Treina um modelo de IA com dados de governo processados e salva métricas no DuckDB e MongoDB.",
    non_argument_deps={"tratar_dados_governo"},
    kinds={"python", "scikitlearn"},
)
def treinar_ia_governo() -> None:
    df = carregar_dados_duckdb("governo_processed")
    treinar_ia(df, "governo", RESULTADOS_DIR)

# Asset: Verificar dados transformados
@asset(
    description="Verifica e imprime amostras dos dados transformados no DuckDB.",
    non_argument_deps={"tratar_dados_economia", "tratar_dados_governo"},
    kinds={"python"},
)
def verificar_dados_transformados() -> None:
    import duckdb

    conn = duckdb.connect(DUCKDB_FILE)

    # Verificar dados transformados de governo
    governo_transformed_df = conn.execute("SELECT * FROM governo_transformed LIMIT 5").fetchdf()
    print("Dados transformados de governo:")
    print(governo_transformed_df)

    # Verificar dados transformados de economia
    economia_transformed_df = conn.execute("SELECT * FROM economia_transformed LIMIT 5").fetchdf()
    print("Dados transformados de economia:")
    print(economia_transformed_df)

    conn.close()

# Asset: Verificar dados disponíveis para IA
@asset(
    description="Verifica a quantidade de dados disponíveis para treinamento de IA.",
    non_argument_deps={"tratar_dados_economia", "tratar_dados_governo"},
    kinds={"python"},
)
def verificar_dados_disponiveis() -> None:
    economia_df = carregar_dados_duckdb("economia_processed")
    governo_df = carregar_dados_duckdb("governo_processed")
    
    print(f"Dados disponíveis para Economia: {economia_df.shape[0]} registros.")
    print(f"Dados disponíveis para Governo: {governo_df.shape[0]} registros.")'''

'''
import os
import json
import pandas as pd
from datetime import datetime

# dagster: Framework de orquestração de dados
from dagster import asset, AssetIn

# scrapy: Para coleta de dados da web (crawlers)
from scrapy.crawler import CrawlerProcess

# Import dos spiders customizados para economia e governo
from crawler_noticia.economia.economia.spiders.noticia import NoticiasSpider
from crawler_noticia.governo.governo.spiders.noticia import G1Spider

# duckdb: Banco de dados colunar em um único arquivo, rápido e eficiente
import duckdb

# vaderSentiment: Análise de sentimento pré-treinada baseada em léxicos
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# sklearn: Biblioteca de Machine Learning para treinamento de modelos simples
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# filelock: Evita acesso concorrente problemático ao arquivo DuckDB
from filelock import FileLock

import hashlib
import subprocess

# Importar função do MongoDB
from db_mongo.conexao_mongo import salvar_no_mongo

# ---------------------------------------------
# Configurações de diretórios e arquivos
# ---------------------------------------------

# BASE_DIR: Diretório base do projeto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# RESULTADOS_DIR: Onde resultados (métricas, relatórios) serão armazenados
RESULTADOS_DIR = os.path.join(BASE_DIR, "projeto_dados", "resultados")
os.makedirs(RESULTADOS_DIR, exist_ok=True)

# DUCKDB_FILE: Arquivo único do DuckDB para persistência local de dados
DUCKDB_FILE = os.path.join(BASE_DIR, "noticias.duckdb")

# DBT_PROJECT_PATH: Diretório do projeto DBT
DBT_PROJECT_PATH = os.path.join(BASE_DIR, "dbt_project")

# Lock para evitar concorrência ao acessar DuckDB
duckdb_lock = FileLock(os.path.join(BASE_DIR, "duckdb.lock"))

# ---------------------------------------------
# Funções auxiliares
# ---------------------------------------------

def run_spider(spider, raw_table_name, collection_name, logger):
    """
    Executa um spider do Scrapy, salva os dados brutos em JSON, carrega em DuckDB e MongoDB.
    
    - SRP: Esta função tem a única responsabilidade de rodar o spider, extrair dados e persistir.
    - OCP: Podemos adicionar novos spiders sem alterar esta função, apenas passando novos parâmetros.
    - Clean Code: Nome claro (run_spider), parâmetros descritivos.
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_file = f"projeto_dados/data/{raw_table_name}_{timestamp}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # CrawlerProcess: executa o spider sem precisar de comando no terminal
    process = CrawlerProcess(settings={
        "FEEDS": {output_file: {"format": "json"}},
        "FEED_EXPORT_ENCODING": "utf-8",
    })
    process.crawl(spider)
    process.start()

    # Carrega dados do JSON gerado pelo spider
    with open(output_file, "r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)
    
    # Processamento dos dados brutos para uma estrutura tabular padronizada
    processed_data = []
    if raw_table_name == "economia_raw":
        # Para economia, o spider retorna um dict com time_key e lista de títulos
        for item in data:
            for time_key, titles in item.items():
                for title in titles:
                    processed_data.append({
                        "id": hashlib.md5(f"{title}{time_key}".encode()).hexdigest(),
                        "titulo_noticia": title,
                        "texto": "",  # Aqui poderia estar o texto completo se disponível
                        "data_publicacao": None,
                        "time_ago": time_key,
                    })
    elif raw_table_name == "governo_raw":
        # Para governo, o spider retorna dicionários com campos específicos
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

    # Acesso ao DuckDB protegido por lock (Thread-Safety)
    with duckdb_lock:
        conn = duckdb.connect(DUCKDB_FILE)
        conn.register("df_view", df)
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {raw_table_name}").fetchone()[0] # type: ignore
        except duckdb.BinderException:
            count = 0

        if count > 0:
            # Inserção apenas de novos registros não duplicados
            conn.execute(f"""
                INSERT INTO {raw_table_name} 
                SELECT * FROM df_view 
                WHERE id NOT IN (SELECT id FROM {raw_table_name})
            """)
            logger.info(f"Novos dados apendados na tabela '{raw_table_name}' no DuckDB.")
        else:
            # Cria tabela caso ainda não exista
            conn.execute(f"CREATE TABLE {raw_table_name} AS SELECT * FROM df_view")
            logger.info(f"Tabela '{raw_table_name}' criada e dados inseridos no DuckDB.")
        conn.close()

    logger.info(f"Dados salvos na tabela '{raw_table_name}' no DuckDB.")

    # Salvar dados no MongoDB
    salvar_no_mongo(processed_data, collection_name)
    logger.info(f"Novos dados adicionados à coleção '{collection_name}' no MongoDB.")

def carregar_dados_duckdb(table_name, logger):
    """
    Carrega dados de uma tabela DuckDB em um DataFrame.
    - SRP: Apenas carregar dados.
    - Clean Code: Nome simples e descritivo.
    """
    with duckdb_lock:
        conn = duckdb.connect(DUCKDB_FILE)
        try:
            df = conn.execute(f"SELECT * FROM {table_name}").fetchdf()
            logger.info(f"Dados carregados da tabela '{table_name}' no DuckDB.")
        except duckdb.BinderException:
            logger.error(f"Tabela {table_name} não encontrada no DuckDB.")
            df = pd.DataFrame()
        conn.close()
    return df

def tratar_dados_func(transformed_table, processed_table, resultados_dir, collection_name, logger):
    """
    Trata dados transformados (após DBT), realiza análise de sentimento e prepara dados para IA.
    
    - SOLID (SRP): Uma única função para tratar dados (responsabilidade clara).
    - Caso precisemos mudar a forma de análise de sentimento, podemos injetar outra estratégia (Strategy Pattern).
    """
    df = carregar_dados_duckdb(transformed_table, logger)
    if df.empty:
        logger.warning(f"Nenhum dado encontrado na tabela {transformed_table}.")
        return pd.DataFrame()

    analyzer = SentimentIntensityAnalyzer()
    df['texto_para_sentimento'] = df.apply(
        lambda row: row['texto'] if isinstance(row['texto'], str) and row['texto'].strip() else row['titulo_noticia'],
        axis=1
    )
    df['sentimento'] = df['texto_para_sentimento'].apply(lambda x: analyzer.polarity_scores(x)['compound'] if isinstance(x, str) else 0)
    df['sentimento'] = df['sentimento'].fillna(0)
    
    # Classificação de acordo com VADER
    df['sentimento_classificacao'] = df['sentimento'].apply(
        lambda x: 'positivo' if x >= 0.05 else ('negativo' if x <= -0.05 else 'neutro')
    )
    df['target'] = df['sentimento_classificacao'].apply(lambda x: 1 if x == 'positivo' else 0)

    # Salvar resumo de sentimento
    sentimento_summary = df['sentimento_classificacao'].value_counts().to_frame().reset_index()
    sentimento_summary.columns = ['Sentimento', 'Contagem']
    sentimento_file = os.path.join(resultados_dir, f"{transformed_table}_sentimento_summary.csv")
    sentimento_summary.to_csv(sentimento_file, index=False)
    logger.info(f"Resumo de sentimentos salvo em '{sentimento_file}'.")

    # Salvar dados processados no MongoDB
    salvar_no_mongo(df.to_dict(orient='records'), f"{collection_name}_processed")
    logger.info(f"Dados processados salvos na coleção '{collection_name}_processed' no MongoDB.")

    # Verificar se temos classes suficientes
    contagem_targets = df['target'].value_counts()
    if contagem_targets.shape[0] < 2:
        logger.warning(f"Não há classes suficientes para treinar IA em {transformed_table}.")
        return df

    with duckdb_lock:
        conn = duckdb.connect(DUCKDB_FILE)
        conn.register("df_view", df)
        conn.execute(f"CREATE OR REPLACE TABLE {processed_table} AS SELECT * FROM df_view")
        conn.close()

    logger.info(f"Dados tratados salvos na tabela '{processed_table}' no DuckDB.")
    return df

def treinar_ia(df: pd.DataFrame, nome_modelo: str, resultados_dir: str, logger):
    """
    Treina modelos de IA (RandomForest, LogisticRegression, SVM) e salva métricas e relatórios.
    
    - SOLID (SRP): Função dedicada ao treinamento e avaliação de modelos.
    - Podemos usar o Strategy Pattern caso queiramos trocar o algoritmo de IA facilmente.
    """
    if df.empty:
        logger.warning(f"Nenhum dado disponível para treinamento em {nome_modelo}.")
        return
    
    X = df[['sentimento']].fillna(0)
    y = df['target']
    if y.nunique() <= 1:
        logger.warning(f"Dados insuficientes para treinar IA em {nome_modelo}.")
        return

    try:
        # Estratificação garante distribuição semelhante de classes no train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError as e:
        logger.error(f"Erro ao dividir os dados para {nome_modelo}: {e}")
        return

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        logger.warning(f"Dados insuficientes para treinar IA em {nome_modelo} após divisão.")
        return

    # Três modelos simples para demonstrar múltiplas abordagens
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
            recall_val = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            metrics_list.append({
                "Algoritmo": nome_algoritmo,
                "Acurácia": acuracia,
                "Precisão": precisao,
                "Recall": recall_val,
                "F1-Score": f1
            })
            
            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
            cm_df = pd.DataFrame(cm, index=["Negativo", "Positivo"], columns=["Negativo", "Positivo"])
            cm_file = os.path.join(resultados_dir, f"{nome_modelo}_{nome_algoritmo}_confusion_matrix.csv")
            cm_df.to_csv(cm_file)
            logger.info(f"Matriz de Confusão para {nome_algoritmo} salva em '{cm_file}'.")
        except ValueError as e:
            logger.error(f"Erro ao treinar {nome_algoritmo} em {nome_modelo}: {e}")
            continue

    if metrics_list:
        metrics_df = pd.DataFrame(metrics_list)
    else:
        metrics_df = pd.DataFrame(columns=["Algoritmo", "Acurácia", "Precisão", "Recall", "F1-Score"])
        logger.warning(f"Nenhuma métrica foi coletada para {nome_modelo}.")

    if not metrics_df.empty:
        metrics_file = os.path.join(resultados_dir, f"{nome_modelo}_metrics.csv")
        metrics_df.to_csv(metrics_file, index=False)
        logger.info(f"Métricas dos modelos salvas em '{metrics_file}'.")
    else:
        logger.warning(f"Nenhuma métrica foi salva para {nome_modelo}.")

    sentimento_summary = df['sentimento_classificacao'].value_counts().to_frame().reset_index()
    sentimento_summary.columns = ['Sentimento', 'Contagem']
    sentimento_file = os.path.join(resultados_dir, f"{nome_modelo}_sentimento_summary.csv")
    sentimento_summary.to_csv(sentimento_file, index=False)
    logger.info(f"Resumo de sentimentos salvo em '{sentimento_file}'.")

    if not metrics_df.empty:
        relatorio_path = os.path.join(resultados_dir, f"{nome_modelo}_relatorio.txt")
        with open(relatorio_path, 'w') as relatorio:
            relatorio.write(f"Relatório de Treinamento de IA para {nome_modelo}\n")
            relatorio.write("="*50 + "\n\n")
            relatorio.write("Distribuição de Sentimentos:\n")
            relatorio.write(sentimento_summary.to_string(index=False))
            relatorio.write("\n\nMétricas dos Modelos:\n")
            relatorio.write(metrics_df.to_string(index=False))
            relatorio.write("\n\nObservações:\n")
            relatorio.write("A IA está analisando o sentimento das notícias com base na pontuação de sentimento fornecida pelo VADER.\n")
            relatorio.write("Classes utilizadas: Positivo (1), Negativo (0)\n")
        
        logger.info(f"Relatório detalhado salvo em '{relatorio_path}'.")
    else:
        logger.warning(f"Nenhum relatório foi gerado para {nome_modelo}.")

# ---------------------------------------------
# Definição de Assets do Dagster
# ---------------------------------------------

@asset(
    description="Executa o crawler para coletar notícias de economia e salva os dados brutos no DuckDB e MongoDB.",
    kinds={"python"},
)
def crawler_economia(context) -> None:
    run_spider(NoticiasSpider, "economia_raw", "economia_raw", context.log)

@asset(
    description="Executa o crawler para coletar notícias de governo e salva os dados brutos no DuckDB e MongoDB.",
    kinds={"python"},
)
def crawler_governo(context) -> None:
    run_spider(G1Spider, "governo_raw", "governo_raw", context.log)

@asset(
    description="Executa os modelos DBT para transformar os dados brutos.",
    kinds={"dbt"},
)
def executar_dbt(context) -> None:
    if not os.path.exists(DBT_PROJECT_PATH):
        raise NotADirectoryError(f"O diretório especificado para o DBT não existe: {DBT_PROJECT_PATH}")
    subprocess.run(["dbt", "run"], check=True, cwd=DBT_PROJECT_PATH)
    context.log.info("Modelos DBT executados com sucesso.")

@asset(
    description="Trata os dados transformados de economia e salva no DuckDB e MongoDB.",
    kinds={"duckdb", "pandas" , "mongodb"},
    # Sem non_argument_deps, assumimos que o DBT rodou antes por depender da execução
    # Estes dados devem ser consumidos após executar_dbt (utilizar planejamento do pipeline)
)
def tratar_dados_economia(context) -> pd.DataFrame:
    return tratar_dados_func(
        transformed_table="economia_transformed",
        processed_table="economia_processed",
        resultados_dir=RESULTADOS_DIR,
        collection_name="economia_processed",
        logger=context.log
    )

@asset(
    description="Trata os dados transformados de governo e salva no DuckDB e MongoDB.",
    kinds={"duckdb", "pandas" , "mongodb"},
)
def tratar_dados_governo(context) -> pd.DataFrame:
    return tratar_dados_func(
        transformed_table="governo_transformed",
        processed_table="governo_processed",
        resultados_dir=RESULTADOS_DIR,
        collection_name="governo_processed",
        logger=context.log
    )

@asset(
    description="Treina um modelo de IA com dados de economia processados e salva métricas no DuckDB e MongoDB.",
    kinds={"python", "scikitlearn"},
    # Agora especificamos que ele depende da saída do asset tratar_dados_economia via ins
    ins={"tratar_dados_economia": AssetIn()},
)
def treinar_ia_economia(context, tratar_dados_economia: pd.DataFrame) -> None:
    # Aqui usamos o parâmetro tratar_dados_economia, correspondendo à chave ins acima
    treinar_ia(
        df=tratar_dados_economia,
        nome_modelo="economia",
        resultados_dir=RESULTADOS_DIR,
        logger=context.log
    )

@asset(
    description="Treina um modelo de IA com dados de governo processados e salva métricas no DuckDB e MongoDB.",
    kinds={"python", "scikitlearn"},
    ins={"tratar_dados_governo": AssetIn()},
)
def treinar_ia_governo(context, tratar_dados_governo: pd.DataFrame) -> None:
    # Aqui usamos o parâmetro tratar_dados_governo, correspondendo à chave ins acima
    treinar_ia(
        df=tratar_dados_governo,
        nome_modelo="governo",
        resultados_dir=RESULTADOS_DIR,
        logger=context.log
    )

@asset(
    description="Verifica e imprime amostras dos dados transformados no DuckDB.",
    kinds={"python"},
    ins={"tratar_dados_economia": AssetIn(), "tratar_dados_governo": AssetIn()},
)
def verificar_dados_transformados(context, tratar_dados_economia: pd.DataFrame, tratar_dados_governo: pd.DataFrame) -> None:
    # Recebemos como input os DataFrames já tratados, garantindo que rodamos após eles.
    conn = duckdb.connect(DUCKDB_FILE)

    try:
        governo_transformed_df = conn.execute("SELECT * FROM governo_transformed LIMIT 5").fetchdf()
        context.log.info("Dados transformados de governo:")
        context.log.info(governo_transformed_df)
    except Exception as e:
        context.log.error(f"Erro ao verificar dados transformados de governo: {e}")

    try:
        economia_transformed_df = conn.execute("SELECT * FROM economia_transformed LIMIT 5").fetchdf()
        context.log.info("Dados transformados de economia:")
        context.log.info(economia_transformed_df)
    except Exception as e:
        context.log.error(f"Erro ao verificar dados transformados de economia: {e}")

    conn.close()

@asset(
    description="Verifica a quantidade de dados disponíveis para treinamento de IA.",
    kinds={"python"},
    ins={"tratar_dados_economia": AssetIn(), "tratar_dados_governo": AssetIn()},
)
def verificar_dados_disponiveis(context, tratar_dados_economia: pd.DataFrame, tratar_dados_governo: pd.DataFrame) -> None:
    # Aqui já temos os dados tratados, podemos contar quantos registros temos
    economia_df = tratar_dados_economia
    governo_df = tratar_dados_governo
    
    context.log.info(f"Dados disponíveis para Economia: {economia_df.shape[0]} registros.")
    context.log.info(f"Dados disponíveis para Governo: {governo_df.shape[0]} registros.")'''
'''
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
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from filelock import FileLock
import hashlib
import subprocess

# Importar funções de MongoDB
from db_mongo.conexao_mongo import salvar_no_mongo

# ---------------------------- CONFIGURAÇÃO ----------------------------
# Diretório base do projeto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Pasta para armazenar resultados
RESULTADOS_DIR = os.path.join(BASE_DIR, "projeto_dados", "resultados")
os.makedirs(RESULTADOS_DIR, exist_ok=True)

# Arquivo DuckDB
DUCKDB_FILE = os.path.join(BASE_DIR, "noticias.duckdb")
DBT_PROJECT_PATH = os.path.join(BASE_DIR, "dbt_project")

# Lock para controle de acesso ao DuckDB (evita acessos concorrentes)
duckdb_lock = FileLock(os.path.join(BASE_DIR, "duckdb.lock"))

# ---------------------------- FUNÇÕES AUXILIARES ----------------------------
def run_spider(spider, raw_table_name, collection_name):
    """
    Executa o spider do Scrapy para coletar dados e salva no DuckDB e MongoDB.
    - Responsabilidade Única: Esta função apenas cuida da coleta.
    - Clean Code: Nomes descritivos, lógica clara.
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_file = f"projeto_dados/data/{raw_table_name}_{timestamp}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Executa o spider do Scrapy
    process = CrawlerProcess(settings={
        "FEEDS": {output_file: {"format": "json"}},
        "FEED_EXPORT_ENCODING": "utf-8",
    })
    process.crawl(spider)
    process.start()  # Bloqueante até terminar a coleta

    # Carrega os dados coletados
    with open(output_file, "r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)

    # Processa os dados para um formato consistente antes de inserir no DuckDB
    processed_data = []
    if raw_table_name == "economia_raw":
        for item in data:
            for time_key, titles in item.items():
                for title in titles:
                    processed_data.append({
                        "id": hashlib.md5(f"{title}{time_key}".encode()).hexdigest(),
                        "titulo_noticia": title,
                        "texto": "",  # Placeholder se não houver texto completo
                        "data_publicacao": None,
                        "time_ago": time_key,
                    })
    elif raw_table_name == "governo_raw":
        for item in data:
            processed_data.append({
                "id": hashlib.md5(str(item).encode()).hexdigest(),
                "titulo_noticia": item.get("title", ""),
                "texto": item.get("body", ""),
                "data_publicacao": item.get("data_publicacao", ""),
            })

    df = pd.DataFrame(processed_data)

    # Garantir colunas essenciais
    if 'id' not in df.columns:
        df['id'] = df.apply(lambda row: hashlib.md5(str(row).encode()).hexdigest(), axis=1)
    if 'data_publicacao' not in df.columns:
        df['data_publicacao'] = pd.NaT

    # Insere no DuckDB
    with duckdb_lock:
        conn = duckdb.connect(DUCKDB_FILE)
        conn.register("df_view", df)
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {raw_table_name}").fetchone()[0]  # type: ignore
        except duckdb.BinderException:
            count = 0

        if count > 0:
            # Apende novos dados sem duplicatas
            conn.execute(f"""
                INSERT INTO {raw_table_name} 
                SELECT * FROM df_view 
                WHERE id NOT IN (SELECT id FROM {raw_table_name})
            """)
            print(f"Novos dados apendados na tabela '{raw_table_name}' no DuckDB.")
        else:
            # Cria a tabela se não existir
            conn.execute(f"CREATE TABLE {raw_table_name} AS SELECT * FROM df_view")
            print(f"Tabela '{raw_table_name}' criada e dados inseridos no DuckDB.")
        conn.close()

    print(f"Dados salvos na tabela '{raw_table_name}' no DuckDB.")

    # Salvar dados no MongoDB
    salvar_no_mongo(processed_data, collection_name)
    print(f"Novos dados adicionados à coleção '{collection_name}' no MongoDB.")

def carregar_dados_duckdb(table_name):
    """
    Carrega dados de uma tabela DuckDB.
    - Responsabilidade Única: apenas carrega dados da tabela.
    - Caso a tabela não exista, retorna DataFrame vazio.
    """
    with duckdb_lock:
        conn = duckdb.connect(DUCKDB_FILE)
        try:
            df = conn.execute(f"SELECT * FROM {table_name}").fetchdf()
        except duckdb.BinderException:
            print(f"Tabela {table_name} não encontrada no DuckDB.")
            df = pd.DataFrame()
        conn.close()
    return df

def tratar_dados_func(transformed_table, processed_table, resultados_dir, collection_name):
    """
    Trata os dados transformados, realiza análise de sentimento e prepara para treinamento de IA.
    - Aplicação de Clean Code:
      * Função coesa: trata dados, extrai sentimentos, salva resumo.
      * Nomes descritivos.
    - SOLID: O princípio da responsabilidade única é aplicado; esta função não faz coleta nem treinamento, só trata dados.
    """
    df = carregar_dados_duckdb(transformed_table)
    if df.empty:
        print(f"Nenhum dado encontrado na tabela {transformed_table}.")
        return pd.DataFrame()

    # Análise de sentimento
    analyzer = SentimentIntensityAnalyzer()
    df['texto_para_sentimento'] = df.apply(
        lambda row: row['texto'] if isinstance(row['texto'], str) and row['texto'].strip() else row['titulo_noticia'],
        axis=1
    )
    df['sentimento'] = df['texto_para_sentimento'].apply(
        lambda x: analyzer.polarity_scores(x)['compound'] if isinstance(x, str) else 0
    )
    df['sentimento'] = df['sentimento'].fillna(0)

    # Classificação com base no VADER:
    # compound >=0.05 => positivo; <= -0.05 => negativo; caso contrário neutro
    df['sentimento_classificacao'] = df['sentimento'].apply(
        lambda x: 'positivo' if x >= 0.05 else ('negativo' if x <= -0.05 else 'neutro')
    )
    # Target binário para simplificar o treinamento
    df['target'] = df['sentimento_classificacao'].apply(lambda x: 1 if x == 'positivo' else 0)

    # Salva resumo dos sentimentos
    sentimento_summary = df['sentimento_classificacao'].value_counts().to_frame().reset_index()
    sentimento_summary.columns = ['Sentimento', 'Contagem']
    sentimento_file = os.path.join(resultados_dir, f"{transformed_table}_sentimento_summary.csv")
    sentimento_summary.to_csv(sentimento_file, index=False)
    print(f"Resumo de sentimentos salvo em '{sentimento_file}'.")

    # Salvar dados processados no MongoDB
    salvar_no_mongo(df.to_dict(orient='records'), f"{collection_name}_processed")
    print(f"Dados processados salvos na coleção '{collection_name}_processed' no MongoDB.")

    # Verificar se há classes suficientes para treinamento
    contagem_targets = df['target'].value_counts()
    if contagem_targets.shape[0] < 2:
        print(f"Não há classes suficientes para treinar IA em {transformed_table}.")
        # Mesmo assim salvamos a tabela processed (pode ser útil)
        with duckdb_lock:
            conn = duckdb.connect(DUCKDB_FILE)
            conn.register("df_view", df)
            conn.execute(f"CREATE OR REPLACE TABLE {processed_table} AS SELECT * FROM df_view")
            conn.close()
        return df

    # Salva tabela processed no DuckDB
    with duckdb_lock:
        conn = duckdb.connect(DUCKDB_FILE)
        conn.register("df_view", df)
        conn.execute(f"CREATE OR REPLACE TABLE {processed_table} AS SELECT * FROM df_view")
        conn.close()
    print(f"Dados tratados salvos na tabela '{processed_table}' no DuckDB.")

    return df

def treinar_ia(df: pd.DataFrame, nome_modelo: str, resultados_dir: str):
    """
    Treina modelos de IA e gera métricas, matrizes de confusão e relatórios.
    - Clean Code: função clara, comentários explicando, nomes descritivos.
    - SOLID: Esta função não faz coleta nem transformação, apenas treina e gera relatórios.
    - Design Patterns: Poderíamos aplicar Strategy se quisermos trocar os algoritmos facilmente.
    """
    if df.empty:
        print(f"Nenhum dado disponível para treinamento em {nome_modelo}.")
        return

    X = df[['sentimento']].fillna(0)
    y = df['target']

    if y.nunique() <= 1:
        print(f"Dados insuficientes para treinar IA em {nome_modelo}.")
        return

    # Divisão estratificada
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError as e:
        print(f"Erro ao dividir os dados para {nome_modelo}: {e}")
        return

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        print(f"Dados insuficientes após divisão para treinar IA em {nome_modelo}.")
        return

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
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            metrics_list.append({
                "Algoritmo": nome_algoritmo,
                "Acurácia": acuracia,
                "Precisão": precisao,
                "Recall": recall,
                "F1-Score": f1
            })

            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
            cm_df = pd.DataFrame(cm, index=["Negativo", "Positivo"], columns=["Negativo", "Positivo"])
            cm_file = os.path.join(resultados_dir, f"{nome_modelo}_{nome_algoritmo}_confusion_matrix.csv")
            cm_df.to_csv(cm_file)
            print(f"Matriz de Confusão para {nome_algoritmo} salva em '{cm_file}'.")

        except ValueError as e:
            print(f"Erro ao treinar {nome_algoritmo} em {nome_modelo}: {e}")
            continue

    if metrics_list:
        metrics_df = pd.DataFrame(metrics_list)
        metrics_file = os.path.join(resultados_dir, f"{nome_modelo}_metrics.csv")
        metrics_df.to_csv(metrics_file, index=False)
        print(f"Métricas dos modelos salvas em '{metrics_file}'.")
    else:
        print(f"Nenhuma métrica foi coletada para {nome_modelo}.")
        return

    # Resumo de sentimentos
    sentimento_summary = df['sentimento_classificacao'].value_counts().to_frame().reset_index()
    sentimento_summary.columns = ['Sentimento', 'Contagem']
    sentimento_file = os.path.join(resultados_dir, f"{nome_modelo}_sentimento_summary.csv")
    sentimento_summary.to_csv(sentimento_file, index=False)
    print(f"Resumo de sentimentos salvo em '{sentimento_file}'.")

    # Relatório detalhado
    if not metrics_df.empty:
        relatorio_path = os.path.join(resultados_dir, f"{nome_modelo}_relatorio.txt")
        with open(relatorio_path, 'w') as relatorio:
            relatorio.write(f"Relatório de Treinamento de IA para {nome_modelo}\n")
            relatorio.write("="*50 + "\n\n")
            relatorio.write("Distribuição de Sentimentos:\n")
            relatorio.write(sentimento_summary.to_string(index=False))
            relatorio.write("\n\nMétricas dos Modelos:\n")
            relatorio.write(metrics_df.to_string(index=False))
            relatorio.write("\n\nObservações:\n")
            relatorio.write("A IA está analisando o sentimento das notícias com base na pontuação do VADER.\n")
            relatorio.write("Classes: Positivo (1), Negativo (0), Neutro não usado no target (0 ou 1).\n")
        print(f"Relatório detalhado salvo em '{relatorio_path}'.")
    else:
        print(f"Nenhum relatório gerado para {nome_modelo}.")

# ---------------------------- ASSETS ----------------------------

@asset(
    description="Executa o crawler para coletar notícias de economia e salva no DuckDB e MongoDB.",
    kinds={"python"},  # Mantém ícone python
)
def crawler_economia() -> None:
    run_spider(NoticiasSpider, "economia_raw", "economia_raw")

@asset(
    description="Executa o crawler para coletar notícias de governo e salva no DuckDB e MongoDB.",
    kinds={"python"},  # Mantém ícone python
)
def crawler_governo() -> None:
    run_spider(G1Spider, "governo_raw", "governo_raw")

@asset(
    description="Executa os modelos DBT para transformar os dados brutos.",
    non_argument_deps={"crawler_economia", "crawler_governo"},
    kinds={"dbt"}  # Mantém ícone dbt
)
def executar_dbt() -> None:
    if not os.path.exists(DBT_PROJECT_PATH):
        raise NotADirectoryError(f"O diretório DBT não existe: {DBT_PROJECT_PATH}")
    subprocess.run(["dbt", "run"], check=True, cwd=DBT_PROJECT_PATH)
    print("Modelos DBT executados com sucesso.")

@asset(
    description="Trata os dados transformados de economia e salva no DuckDB e MongoDB.",
    non_argument_deps={"executar_dbt"},
    kinds={"duckdb", "pandas", "mongodb"}  # Ícones
)
def tratar_dados_economia() -> pd.DataFrame:
    return tratar_dados_func("economia_transformed", "economia_processed", RESULTADOS_DIR, "economia_processed")

@asset(
    description="Trata os dados transformados de governo e salva no DuckDB e MongoDB.",
    non_argument_deps={"executar_dbt"},
    kinds={"duckdb", "pandas", "mongodb"}
)
def tratar_dados_governo() -> pd.DataFrame:
    return tratar_dados_func("governo_transformed", "governo_processed", RESULTADOS_DIR, "governo_processed")

@asset(
    description="Treina IA com dados de economia e gera métricas e relatórios.",
    non_argument_deps={"tratar_dados_economia"},
    kinds={"python", "scikitlearn"}  # Ícones python, scikitlearn
)
def treinar_ia_economia() -> None:
    df = carregar_dados_duckdb("economia_processed")
    treinar_ia(df, "economia", RESULTADOS_DIR)

@asset(
    description="Treina IA com dados de governo e gera métricas e relatórios.",
    non_argument_deps={"tratar_dados_governo"},
    kinds={"python", "scikitlearn"}
)
def treinar_ia_governo() -> None:
    df = carregar_dados_duckdb("governo_processed")
    treinar_ia(df, "governo", RESULTADOS_DIR)

@asset(
    description="Verifica amostras dos dados transformados no DuckDB.",
    non_argument_deps={"tratar_dados_economia", "tratar_dados_governo"},
    kinds={"python"}
)
def verificar_dados_transformados() -> None:
    conn = duckdb.connect(DUCKDB_FILE)

    # Verificando governo
    governo_transformed_df = conn.execute("SELECT * FROM governo_transformed LIMIT 5").fetchdf()
    print("Amostra dados transformados de governo:")
    print(governo_transformed_df)

    # Verificando economia
    economia_transformed_df = conn.execute("SELECT * FROM economia_transformed LIMIT 5").fetchdf()
    print("Amostra dados transformados de economia:")
    print(economia_transformed_df)

    conn.close()

@asset(
    description="Verifica quantidade de dados disponíveis para IA.",
    non_argument_deps={"tratar_dados_economia", "tratar_dados_governo"},
    kinds={"python"}
)
def verificar_dados_disponiveis() -> None:
    economia_df = carregar_dados_duckdb("economia_processed")
    governo_df = carregar_dados_duckdb("governo_processed")

    print(f"Dados disponíveis para Economia: {economia_df.shape[0]} registros.")
    print(f"Dados disponíveis para Governo: {governo_df.shape[0]} registros.")'''

'''

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
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from filelock import FileLock
import hashlib
import subprocess

# Importar funções de MongoDB
from db_mongo.conexao_mongo import salvar_no_mongo

# ---------------------------- CONFIG ----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTADOS_DIR = os.path.join(BASE_DIR, "projeto_dados", "resultados")
os.makedirs(RESULTADOS_DIR, exist_ok=True)
DUCKDB_FILE = os.path.join(BASE_DIR, "noticias.duckdb")
DBT_PROJECT_PATH = os.path.join(BASE_DIR, "dbt_project")
duckdb_lock = FileLock(os.path.join(BASE_DIR, "duckdb.lock"))

# ---------------------------- FUNÇÕES AUXILIARES ----------------------------
def run_spider(spider, raw_table_name, collection_name):
    """
    Executa spider do Scrapy, salva dados no DuckDB e MongoDB.
    - Clean Code: nomes claros, funções coesas.
    - SOLID (S): Função faz apenas coleta e inserção inicial.
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_file = f"projeto_dados/data/{raw_table_name}_{timestamp}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    process = CrawlerProcess(settings={
        "FEEDS": {output_file: {"format": "json"}},
        "FEED_EXPORT_ENCODING": "utf-8",
    })
    process.crawl(spider)
    process.start()

    with open(output_file, "r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)

    processed_data = []
    if raw_table_name == "economia_raw":
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

    with duckdb_lock:
        conn = duckdb.connect(DUCKDB_FILE)
        conn.register("df_view", df)
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {raw_table_name}").fetchone()[0] # type: ignore
        except duckdb.BinderException:
            count = 0

        if count > 0:
            conn.execute(f"""
                INSERT INTO {raw_table_name} 
                SELECT * FROM df_view 
                WHERE id NOT IN (SELECT id FROM {raw_table_name})
            """)
            print(f"Novos dados apendados na tabela '{raw_table_name}' no DuckDB.")
        else:
            conn.execute(f"CREATE TABLE {raw_table_name} AS SELECT * FROM df_view")
            print(f"Tabela '{raw_table_name}' criada e dados inseridos no DuckDB.")
        conn.close()

    print(f"Dados salvos na tabela '{raw_table_name}' no DuckDB.")
    salvar_no_mongo(processed_data, collection_name)
    print(f"Novos dados adicionados à coleção '{collection_name}' no MongoDB.")

def carregar_dados_duckdb(table_name):
    """
    Carrega dados de uma tabela no DuckDB.
    - Clean Code: função simples e clara.
    """
    with duckdb_lock:
        conn = duckdb.connect(DUCKDB_FILE)
        try:
            df = conn.execute(f"SELECT * FROM {table_name}").fetchdf()
        except duckdb.BinderException:
            print(f"Tabela {table_name} não encontrada no DuckDB.")
            df = pd.DataFrame()
        conn.close()
    return df

def tratar_dados_func(transformed_table, processed_table, resultados_dir, collection_name):
    """
    Trata dados transformados: análise de sentimento, salva resumo e dados processados.
    - Sempre salva a tabela processed, mesmo sem classes suficientes.
    """
    df = carregar_dados_duckdb(transformed_table)
    if df.empty:
        print(f"Nenhum dado encontrado na tabela {transformed_table}.")
        return pd.DataFrame()

    analyzer = SentimentIntensityAnalyzer()
    df['texto_para_sentimento'] = df.apply(
        lambda row: row['texto'] if isinstance(row['texto'], str) and row['texto'].strip() else row['titulo_noticia'],
        axis=1
    )
    df['sentimento'] = df['texto_para_sentimento'].apply(
        lambda x: analyzer.polarity_scores(x)['compound'] if isinstance(x, str) else 0
    )
    df['sentimento'] = df['sentimento'].fillna(0)

    # Classificação baseada no VADER
    df['sentimento_classificacao'] = df['sentimento'].apply(
        lambda x: 'positivo' if x >= 0.05 else ('negativo' if x <= -0.05 else 'neutro')
    )
    df['target'] = df['sentimento_classificacao'].apply(lambda x: 1 if x == 'positivo' else 0)

    sentimento_summary = df['sentimento_classificacao'].value_counts().to_frame().reset_index()
    sentimento_summary.columns = ['Sentimento', 'Contagem']
    sentimento_file = os.path.join(resultados_dir, f"{transformed_table}_sentimento_summary.csv")
    sentimento_summary.to_csv(sentimento_file, index=False)
    print(f"Resumo de sentimentos salvo em '{sentimento_file}'.")

    salvar_no_mongo(df.to_dict(orient='records'), f"{collection_name}_processed")
    print(f"Dados processados salvos na coleção '{collection_name}_processed' no MongoDB.")

    # Salva tabela processed no DuckDB sempre
    with duckdb_lock:
        conn = duckdb.connect(DUCKDB_FILE)
        conn.register("df_view", df)
        conn.execute(f"CREATE OR REPLACE TABLE {processed_table} AS SELECT * FROM df_view")
        conn.close()
    print(f"Dados tratados salvos na tabela '{processed_table}' no DuckDB.")

    return df

def treinar_ia(df: pd.DataFrame, nome_modelo: str, resultados_dir: str):
    """
    Treina modelos de IA. Sempre gera relatório, mesmo se não houver métricas.
    """
    # Caso não haja dados, gera relatório mínimo
    if df.empty:
        print(f"Nenhum dado para treinar IA em {nome_modelo}. Gerando relatório vazio.")
        # Cria um relatório vazio
        relatorio_path = os.path.join(resultados_dir, f"{nome_modelo}_relatorio.txt")
        with open(relatorio_path, 'w') as relatorio:
            relatorio.write(f"Relatório de Treinamento de IA para {nome_modelo}\n")
            relatorio.write("="*50 + "\n\n")
            relatorio.write("Nenhum dado disponível para treinamento.\n")
        return

    X = df[['sentimento']].fillna(0)
    y = df['target']

    if y.nunique() <= 1:
        print(f"Dados insuficientes para treinar IA em {nome_modelo}, mas gerando relatório.")
        # Mesmo sem métricas, gera relatório informando insuficiência.
        sentimento_summary = df['sentimento_classificacao'].value_counts().to_frame().reset_index()
        sentimento_summary.columns = ['Sentimento', 'Contagem']
        sentimento_file = os.path.join(resultados_dir, f"{nome_modelo}_sentimento_summary.csv")
        sentimento_summary.to_csv(sentimento_file, index=False)

        relatorio_path = os.path.join(resultados_dir, f"{nome_modelo}_relatorio.txt")
        with open(relatorio_path, 'w') as relatorio:
            relatorio.write(f"Relatório de Treinamento de IA para {nome_modelo}\n")
            relatorio.write("="*50 + "\n\n")
            relatorio.write("Dados insuficientes para treinamento.\n")
            relatorio.write("\nDistribuição de Sentimentos:\n")
            relatorio.write(sentimento_summary.to_string(index=False))
            relatorio.write("\nNenhuma métrica foi gerada.\n")
        return

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError as e:
        print(f"Erro ao dividir dados: {e}. Gerando relatório indicando erro.")
        relatorio_path = os.path.join(resultados_dir, f"{nome_modelo}_relatorio.txt")
        with open(relatorio_path, 'w') as relatorio:
            relatorio.write(f"Relatório de Treinamento de IA para {nome_modelo}\n")
            relatorio.write("="*50 + "\n\n")
            relatorio.write(f"Erro ao dividir dados: {e}\n")
        return

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        print(f"Classes insuficientes após split para {nome_modelo}. Gerando relatório mínimo.")
        sentimento_summary = df['sentimento_classificacao'].value_counts().to_frame().reset_index()
        sentimento_summary.columns = ['Sentimento', 'Contagem']
        sentimento_file = os.path.join(resultados_dir, f"{nome_modelo}_sentimento_summary.csv")
        sentimento_summary.to_csv(sentimento_file, index=False)

        relatorio_path = os.path.join(resultados_dir, f"{nome_modelo}_relatorio.txt")
        with open(relatorio_path, 'w') as relatorio:
            relatorio.write(f"Relatório IA {nome_modelo}\n")
            relatorio.write("="*50 + "\n\n")
            relatorio.write("Classes insuficientes após split.\n")
            relatorio.write("Distribuição Sentimentos:\n")
            relatorio.write(sentimento_summary.to_string(index=False))
        return

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
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            metrics_list.append({
                "Algoritmo": nome_algoritmo,
                "Acurácia": acuracia,
                "Precisão": precisao,
                "Recall": recall,
                "F1-Score": f1
            })

            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
            cm_df = pd.DataFrame(cm, index=["Negativo", "Positivo"], columns=["Negativo", "Positivo"])
            cm_file = os.path.join(resultados_dir, f"{nome_modelo}_{nome_algoritmo}_confusion_matrix.csv")
            cm_df.to_csv(cm_file)
            print(f"Matriz de Confusão para {nome_algoritmo} salva em '{cm_file}'.")

        except ValueError as e:
            print(f"Erro ao treinar {nome_algoritmo}: {e}")

    sentimento_summary = df['sentimento_classificacao'].value_counts().to_frame().reset_index()
    sentimento_summary.columns = ['Sentimento', 'Contagem']
    sentimento_file = os.path.join(resultados_dir, f"{nome_modelo}_sentimento_summary.csv")
    sentimento_summary.to_csv(sentimento_file, index=False)
    print(f"Resumo de sentimentos salvo em '{sentimento_file}'.")

    metrics_file = os.path.join(resultados_dir, f"{nome_modelo}_metrics.csv")
    relatorio_path = os.path.join(resultados_dir, f"{nome_modelo}_relatorio.txt")

    if metrics_list:
        metrics_df = pd.DataFrame(metrics_list)
        metrics_df.to_csv(metrics_file, index=False)
        print(f"Métricas salvas em '{metrics_file}'.")

        # Cria relatório detalhado
        with open(relatorio_path, 'w') as relatorio:
            relatorio.write(f"Relatório de Treinamento de IA para {nome_modelo}\n")
            relatorio.write("="*50 + "\n\n")
            relatorio.write("Distribuição de Sentimentos:\n")
            relatorio.write(sentimento_summary.to_string(index=False))
            relatorio.write("\n\nMétricas dos Modelos:\n")
            relatorio.write(metrics_df.to_string(index=False))
            relatorio.write("\n\nObservações:\n")
            relatorio.write("IA analisando sentimento com VADER.\nClasses: Positivo(1), Negativo(0).\n")
        print(f"Relatório detalhado salvo em '{relatorio_path}'.")
    else:
        # Mesmo sem métricas, cria relatório mínimo
        with open(relatorio_path, 'w') as relatorio:
            relatorio.write(f"Relatório IA {nome_modelo}\n")
            relatorio.write("="*50 + "\n\n")
            relatorio.write("Nenhuma métrica coletada (dados insuficientes ou erro no treinamento).\n")
            relatorio.write("Distribuição de Sentimentos:\n")
            relatorio.write(sentimento_summary.to_string(index=False))
        print(f"Relatório mínimo salvo em '{relatorio_path}' (sem métricas).")

# ---------------------------- ASSETS ----------------------------
@asset(
    description="Executa o crawler (economia).",
    kinds={"python"}
)
def crawler_economia() -> None:
    run_spider(NoticiasSpider, "economia_raw", "economia_raw")

@asset(
    description="Executa o crawler (governo).",
    kinds={"python"}
)
def crawler_governo() -> None:
    run_spider(G1Spider, "governo_raw", "governo_raw")

@asset(
    description="Executa DBT após coleta.",
    non_argument_deps={"crawler_economia", "crawler_governo"},
    kinds={"dbt"}
)
def executar_dbt() -> None:
    if not os.path.exists(DBT_PROJECT_PATH):
        raise NotADirectoryError(f"Diretório DBT não existe: {DBT_PROJECT_PATH}")
    subprocess.run(["dbt", "run"], check=True, cwd=DBT_PROJECT_PATH)
    print("Modelos DBT executados com sucesso.")

@asset(
    description="Trata dados de economia.",
    non_argument_deps={"executar_dbt"},
    kinds={"duckdb", "pandas", "mongodb"}
)
def tratar_dados_economia() -> pd.DataFrame:
    return tratar_dados_func("economia_transformed", "economia_processed", RESULTADOS_DIR, "economia_processed")

@asset(
    description="Trata dados de governo.",
    non_argument_deps={"executar_dbt"},
    kinds={"duckdb", "pandas", "mongodb"}
)
def tratar_dados_governo() -> pd.DataFrame:
    return tratar_dados_func("governo_transformed", "governo_processed", RESULTADOS_DIR, "governo_processed")

@asset(
    description="Treina IA economia.",
    non_argument_deps={"tratar_dados_economia"},
    kinds={"python", "scikitlearn"}
)
def treinar_ia_economia() -> None:
    df = carregar_dados_duckdb("economia_processed")
    treinar_ia(df, "economia", RESULTADOS_DIR)

@asset(
    description="Treina IA governo.",
    non_argument_deps={"tratar_dados_governo"},
    kinds={"python", "scikitlearn"}
)
def treinar_ia_governo() -> None:
    df = carregar_dados_duckdb("governo_processed")
    treinar_ia(df, "governo", RESULTADOS_DIR)

@asset(
    description="Verifica dados transformados.",
    non_argument_deps={"tratar_dados_economia", "tratar_dados_governo"},
    kinds={"python"}
)
def verificar_dados_transformados() -> None:
    conn = duckdb.connect(DUCKDB_FILE)
    gov_df = conn.execute("SELECT * FROM governo_transformed LIMIT 5").fetchdf()
    print("Governo transformado (amostra):")
    print(gov_df)
    eco_df = conn.execute("SELECT * FROM economia_transformed LIMIT 5").fetchdf()
    print("Economia transformado (amostra):")
    print(eco_df)
    conn.close()

@asset(
    description="Verifica disponibilidade dados IA.",
    non_argument_deps={"tratar_dados_economia", "tratar_dados_governo"},
    kinds={"python"}
)
def verificar_dados_disponiveis() -> None:
    eco_df = carregar_dados_duckdb("economia_processed")
    gov_df = carregar_dados_duckdb("governo_processed")
    print(f"Economia: {eco_df.shape[0]} registros disponíveis.")
    print(f"Governo: {gov_df.shape[0]} registros disponíveis.")'''


'''
import os
import json
import pandas as pd
from datetime import datetime
from dagster import asset
from scrapy.crawler import CrawlerProcess
from crawler_noticia.economia.economia.spiders.noticia import NoticiasSpider
from crawler_noticia.governo.governo.spiders.noticia import G1Spider
import duckdb
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from filelock import FileLock
import hashlib
import subprocess
from db_mongo.conexao_mongo import salvar_no_mongo

# Diretório base do projeto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Pasta de resultados
RESULTADOS_DIR = os.path.join(BASE_DIR, "projeto_dados", "resultados")
os.makedirs(RESULTADOS_DIR, exist_ok=True)

# Arquivo DuckDB
DUCKDB_FILE = os.path.join(BASE_DIR, "noticias.duckdb")
DBT_PROJECT_PATH = os.path.join(BASE_DIR, "dbt_project")

# Lock para controle concorrente do DuckDB
duckdb_lock = FileLock(os.path.join(BASE_DIR, "duckdb.lock"))

def run_spider(spider, raw_table_name, collection_name):
    """
    Executa spider do Scrapy e salva os resultados no DuckDB e MongoDB.
    """

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_file = f"projeto_dados/data/{raw_table_name}_{timestamp}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    process = CrawlerProcess(settings={
        "FEEDS": {output_file: {"format": "json"}},
        "FEED_EXPORT_ENCODING": "utf-8",
    })
    process.crawl(spider)
    process.start()

    with open(output_file, "r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)

    processed_data = []
    if raw_table_name == "economia_raw":
        # Processa dados de economia
        for item in data:
            for time_key, titles in item.items():
                for title in titles:
                    processed_data.append({
                        "id": hashlib.md5(f"{title}{time_key}".encode()).hexdigest(),
                        "titulo_noticia": title if title else "",
                        "texto": "",  # Se não houver texto detalhado
                        "data_publicacao": None,
                        "time_ago": time_key
                    })
    elif raw_table_name == "governo_raw":
        # Processa dados de governo
        for item in data:
            processed_data.append({
                "id": hashlib.md5(str(item).encode()).hexdigest(),
                "titulo_noticia": item.get("title", "") or "",
                "texto": item.get("body", "") or "",
                "data_publicacao": item.get("data_publicacao", "") or ""
            })

    df = pd.DataFrame(processed_data)

    # Garante coluna id
    if 'id' not in df.columns:
        df['id'] = df.apply(lambda row: hashlib.md5(str(row).encode()).hexdigest(), axis=1)
    if 'data_publicacao' not in df.columns:
        df['data_publicacao'] = pd.NaT

    # Insere no DuckDB
    with duckdb_lock:
        conn = duckdb.connect(DUCKDB_FILE)
        conn.register("df_view", df)
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {raw_table_name}").fetchone()[0] # type: ignore
        except duckdb.BinderException:
            count = 0

        if count > 0:
            # Insere sem duplicatas
            conn.execute(f"""
                INSERT INTO {raw_table_name} 
                SELECT * FROM df_view 
                WHERE id NOT IN (SELECT id FROM {raw_table_name})
            """)
        else:
            conn.execute(f"CREATE TABLE {raw_table_name} AS SELECT * FROM df_view")
        conn.close()

    # Salva brutos no MongoDB
    salvar_no_mongo(processed_data, collection_name)

@asset(description="Executa o crawler para economia", kinds={"python"})
def crawler_economia():
    run_spider(NoticiasSpider, "economia_raw", "economia_raw")

@asset(description="Executa o crawler para governo", kinds={"python"})
def crawler_governo():
    run_spider(G1Spider, "governo_raw", "governo_raw")

@asset(description="Executa modelos DBT", non_argument_deps={"crawler_economia","crawler_governo"}, kinds={"dbt"})
def executar_dbt():
    if not os.path.exists(DBT_PROJECT_PATH):
        raise NotADirectoryError(f"DBT project dir não existe: {DBT_PROJECT_PATH}")
    subprocess.run(["dbt", "run"], check=True, cwd=DBT_PROJECT_PATH)

def carregar_dados_duckdb(table_name):
    with duckdb_lock:
        conn = duckdb.connect(DUCKDB_FILE)
        try:
            df = conn.execute(f"SELECT * FROM {table_name}").fetchdf()
        except duckdb.BinderException:
            df = pd.DataFrame()
        conn.close()
    return df

def preparar_dados_para_mongo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajusta datas e valores nulos para antes de salvar no Mongo.
    Converte datas para string, troca NaT por None.
    """
    # Converte NaT para None
    df = df.where(pd.notnull(df), None)

    # Se houver data_publicacao, converter para string
    if 'data_publicacao' in df.columns:
        # Se houver datas no formato datetime64, converter para string
        if pd.api.types.is_datetime64_any_dtype(df['data_publicacao']):
            df['data_publicacao'] = df['data_publicacao'].dt.strftime('%Y-%m-%d %H:%M:%S')
        else:
            # Se não é datetime, mas sim string/None, já está ok
            pass

    return df

def analisar_sentimento(df: pd.DataFrame) -> pd.DataFrame:
    analyzer = SentimentIntensityAnalyzer()
    df['texto_para_sentimento'] = df.apply(
        lambda row: row['texto'] if isinstance(row['texto'], str) and row['texto'].strip() else row['titulo_noticia'],
        axis=1
    )
    df['sentimento'] = df['texto_para_sentimento'].apply(lambda x: analyzer.polarity_scores(x)['compound'] if isinstance(x, str) else 0)
    df['sentimento'] = df['sentimento'].fillna(0)
    df['sentimento_classificacao'] = df['sentimento'].apply(
        lambda x: 'positivo' if x >= 0.05 else ('negativo' if x <= -0.05 else 'neutro')
    )
    df['target'] = df['sentimento_classificacao'].apply(lambda x: 1 if x == 'positivo' else 0)
    return df

def tratar_dados_func(transformed_table, processed_table, resultados_dir, collection_name):
    df = carregar_dados_duckdb(transformed_table)
    if df.empty:
        print(f"Nenhum dado no {transformed_table}.")
        return pd.DataFrame()

    # Aplica análise de sentimento
    df = analisar_sentimento(df)

    # Salva resumo de sentimentos
    sentimento_summary = df['sentimento_classificacao'].value_counts().to_frame().reset_index()
    sentimento_summary.columns = ['Sentimento', 'Contagem']
    sentimento_file = os.path.join(resultados_dir, f"{transformed_table}_sentimento_summary.csv")
    sentimento_summary.to_csv(sentimento_file, index=False)

    # Prepara df para salvar no Mongo
    df = preparar_dados_para_mongo(df)

    # Salva no MongoDB dados processados
    salvar_no_mongo(df.to_dict(orient='records'), f"{collection_name}_processed")

    # Verifica se há classes suficientes
    if df['target'].nunique() < 2:
        print(f"Dados insuficientes para treinamento em {transformed_table}.")
        # Mesmo sem classes suficientes, ainda salvamos o df no DuckDB
        with duckdb_lock:
            conn = duckdb.connect(DUCKDB_FILE)
            conn.register("df_view", df)
            conn.execute(f"CREATE OR REPLACE TABLE {processed_table} AS SELECT * FROM df_view")
            conn.close()
        return df

    # Salva df processado no DuckDB
    with duckdb_lock:
        conn = duckdb.connect(DUCKDB_FILE)
        conn.register("df_view", df)
        conn.execute(f"CREATE OR REPLACE TABLE {processed_table} AS SELECT * FROM df_view")
        conn.close()
    return df

@asset(description="Trata dados economia", non_argument_deps={"executar_dbt"}, kinds={"duckdb","pandas","mongodb"})
def tratar_dados_economia():
    return tratar_dados_func("economia_transformed","economia_processed",RESULTADOS_DIR,"economia_processed")

@asset(description="Trata dados governo", non_argument_deps={"executar_dbt"}, kinds={"duckdb","pandas","mongodb"})
def tratar_dados_governo():
    return tratar_dados_func("governo_transformed","governo_processed",RESULTADOS_DIR,"governo_processed")

def treinar_ia(df: pd.DataFrame, nome_modelo: str, resultados_dir: str):
    if df.empty:
        print(f"Nenhum dado para treinar IA em {nome_modelo}.")
        return

    # Verifica classes
    if df['target'].nunique() <= 1:
        print(f"Dados insuficientes para treino em {nome_modelo}. Apenas uma classe encontrada.")
        # Gera relatório mínimo
        sentimento_summary = df['sentimento_classificacao'].value_counts().to_frame().reset_index()
        sentimento_summary.columns = ['Sentimento','Contagem']
        relatorio_path = os.path.join(resultados_dir, f"{nome_modelo}_relatorio.txt")

        with open(relatorio_path,'w',encoding='utf-8') as rel:
            rel.write(f"Relatório de Treinamento de IA para {nome_modelo}\n")
            rel.write("="*50+"\n\n")
            rel.write("Dados insuficientes para treinamento (apenas uma classe).\n\n")
            rel.write("Distribuição de Sentimentos:\n")
            rel.write(sentimento_summary.to_string(index=False)+"\n\n")
            rel.write("Nenhuma métrica foi gerada.\n")
            rel.write("Ação sugerida: Coletar mais dados para ter pelo menos duas classes.\n")

        # Salva sentimento_summary
        sentimento_file = os.path.join(resultados_dir, f"{nome_modelo}_sentimento_summary.csv")
        sentimento_summary.to_csv(sentimento_file, index=False)

        return

    # Se há classes suficientes, treinar:
    X = df[['sentimento']].fillna(0)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    if y_train.nunique()<2 or y_test.nunique()<2:
        print(f"Ainda insuficiente após split em {nome_modelo}.")
        return

    algoritmos = {
        "RandomForest": RandomForestClassifier(random_state=42),
        "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000),
        "SVM": SVC(random_state=42)
    }

    metrics_list = []
    for nome_alg, modelo in algoritmos.items():
        try:
            modelo.fit(X_train,y_train)
            y_pred = modelo.predict(X_test)
            acuracia = accuracy_score(y_test,y_pred)
            precisao = precision_score(y_test,y_pred,zero_division=0)
            recall = recall_score(y_test,y_pred,zero_division=0)
            f1 = f1_score(y_test,y_pred,zero_division=0)
            metrics_list.append({"Algoritmo":nome_alg,"Acurácia":acuracia,"Precisão":precisao,"Recall":recall,"F1-Score":f1})
            cm = confusion_matrix(y_test,y_pred,labels=[0,1])
            cm_df = pd.DataFrame(cm,index=["Negativo","Positivo"],columns=["Negativo","Positivo"])
            cm_file = os.path.join(resultados_dir,f"{nome_modelo}_{nome_alg}_confusion_matrix.csv")
            cm_df.to_csv(cm_file,index=False)
        except ValueError as e:
            print(f"Erro ao treinar {nome_alg} em {nome_modelo}: {e}")

    sentimento_summary = df['sentimento_classificacao'].value_counts().to_frame().reset_index()
    sentimento_summary.columns=['Sentimento','Contagem']
    sentimento_file = os.path.join(resultados_dir,f"{nome_modelo}_sentimento_summary.csv")
    sentimento_summary.to_csv(sentimento_file,index=False)

    if metrics_list:
        metrics_df = pd.DataFrame(metrics_list)
        metrics_file = os.path.join(resultados_dir,f"{nome_modelo}_metrics.csv")
        metrics_df.to_csv(metrics_file,index=False)

        # Relatório detalhado
        relatorio_path = os.path.join(resultados_dir,f"{nome_modelo}_relatorio.txt")
        with open(relatorio_path,'w',encoding='utf-8') as rel:
            rel.write(f"Relatório de Treinamento de IA para {nome_modelo}\n")
            rel.write("="*50+"\n\n")
            rel.write("Distribuição de Sentimentos:\n")
            rel.write(sentimento_summary.to_string(index=False)+"\n\n")
            rel.write("Métricas dos Modelos:\n")
            rel.write(metrics_df.to_string(index=False)+"\n\n")
            rel.write("Observações:\nA IA está analisando sentimento com base no VADER.\n")
            rel.write("Classes: Positivo (1), Negativo (0)\n")
            rel.write("Sugere-se analisar matrizes de confusão salvas para detalhes.\n")
    else:
        # Caso não haja métricas (falha em todos os algos)
        relatorio_path = os.path.join(resultados_dir,f"{nome_modelo}_relatorio.txt")
        with open(relatorio_path,'w',encoding='utf-8') as rel:
            rel.write(f"Relatório de Treinamento de IA para {nome_modelo}\n")
            rel.write("="*50+"\n\n")
            rel.write("Nenhuma métrica foi coletada.\n")

@asset(description="Treina IA economia",non_argument_deps={"tratar_dados_economia"}, kinds={"python","scikitlearn"})
def treinar_ia_economia():
    df = carregar_dados_duckdb("economia_processed")
    treinar_ia(df,"economia",RESULTADOS_DIR)

@asset(description="Treina IA governo",non_argument_deps={"tratar_dados_governo"}, kinds={"python","scikitlearn"})
def treinar_ia_governo():
    df = carregar_dados_duckdb("governo_processed")
    treinar_ia(df,"governo",RESULTADOS_DIR)

@asset(description="Verifica dados transformados", non_argument_deps={"tratar_dados_economia","tratar_dados_governo"}, kinds={"python"})
def verificar_dados_transformados():
    conn = duckdb.connect(DUCKDB_FILE)
    gov_df = conn.execute("SELECT * FROM governo_transformed LIMIT 5").fetchdf()
    print("Dados transformados de governo:")
    print(gov_df)
    eco_df = conn.execute("SELECT * FROM economia_transformed LIMIT 5").fetchdf()
    print("Dados transformados de economia:")
    print(eco_df)
    conn.close()

@asset(description="Verifica dados disponíveis p/IA", non_argument_deps={"tratar_dados_economia","tratar_dados_governo"}, kinds={"python"})
def verificar_dados_disponiveis():
    eco_df = carregar_dados_duckdb("economia_processed")
    gov_df = carregar_dados_duckdb("governo_processed")
    print(f"Dados disponíveis Economia: {eco_df.shape[0]} registros.")
    print(f"Dados disponíveis Governo: {gov_df.shape[0]} registros.")'''
    
import os # Importar os
import json # Importar json
import pandas as pd # Importar pandas
from datetime import datetime # Importar datetime do Python 
from dagster import asset # Importar asset do Dagster
from scrapy.crawler import CrawlerProcess # Importar CrawlerProcess do Scrapy
from crawler_noticia.economia.economia.spiders.noticia import NoticiasSpider # Importar spiders 
from crawler_noticia.governo.governo.spiders.noticia import G1Spider # Importar spiders
import duckdb # Importar DuckDB
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer # Importar SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split # Importar função de divisão de dados
from sklearn.ensemble import RandomForestClassifier # Importar modelos
from sklearn.linear_model import LogisticRegression # Importar modelos
from sklearn.svm import SVC # Importar modelos
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix # Importar métricas
from filelock import FileLock # Importar FileLock para controle de concorrência
import hashlib # Importar hashlib para gerar hash de strings
import subprocess # Importar subprocess para executar comandos no terminal

# Importar funções de MongoDB
from db_mongo.conexao_mongo import salvar_no_mongo # Função para salvar dados no MongoDB

# ---------------------------- CONFIG ----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Essa parte é importante para o Dagster encontrar os arquivos e diretórios
RESULTADOS_DIR = os.path.join(BASE_DIR, "projeto_dados", "resultados") # Diretório de resultados
os.makedirs(RESULTADOS_DIR, exist_ok=True) # Cria diretório se não existir
DUCKDB_FILE = os.path.join(BASE_DIR, "noticias.duckdb") # Arquivo DuckDB
DBT_PROJECT_PATH = os.path.join(BASE_DIR, "dbt_project") # Diretório do projeto DBT
duckdb_lock = FileLock(os.path.join(BASE_DIR, "duckdb.lock")) # Lock para controle concorrente do DuckDB

# ---------------------------- FUNÇÕES AUXILIARES ----------------------------
def run_spider(spider, raw_table_name, collection_name):
    """
    Executa spider do Scrapy, salva dados no DuckDB e MongoDB.
    - Clean Code: nomes claros, funções coesas.
    - SOLID (S): Função faz apenas coleta e inserção inicial.
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_file = f"projeto_dados/data/{raw_table_name}_{timestamp}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    process = CrawlerProcess(settings={
        "FEEDS": {output_file: {"format": "json"}},
        "FEED_EXPORT_ENCODING": "utf-8",
    })
    process.crawl(spider)
    process.start()

    with open(output_file, "r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)

    processed_data = []
    if raw_table_name == "economia_raw":
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
        for item in data:
            processed_data.append({
                "id": hashlib.md5(str(item).encode()).hexdigest(),
                "titulo_noticia": item.get("title", ""),
                "texto": item.get("body", ""),
                "data_publicacao": item.get("data_publicacao", ""),
            })

    df = pd.DataFrame(processed_data)

    # Garante que exista a coluna 'id'
    if 'id' not in df.columns:
        df['id'] = df.apply(lambda row: hashlib.md5(str(row).encode()).hexdigest(), axis=1)
    # Garante que exista a coluna 'data_publicacao'
    if 'data_publicacao' not in df.columns:
        df['data_publicacao'] = pd.NaT

    # Uso de FileLock para evitar concorrência no mesmo processo
    with duckdb_lock:
        # Uso do context manager para fechar conexão adequadamente
        with duckdb.connect(DUCKDB_FILE) as conn:
            conn.register("df_view", df)
            try:
                # Captura tanto BinderException quanto CatalogException
                count = conn.execute(f"SELECT COUNT(*) FROM {raw_table_name}").fetchone()[0]
            except (duckdb.BinderException, duckdb.CatalogException):
                count = 0

            if count > 0:
                conn.execute(f"""
                    INSERT INTO {raw_table_name} 
                    SELECT * FROM df_view 
                    WHERE id NOT IN (SELECT id FROM {raw_table_name})
                """)
                print(f"Novos dados apendados na tabela '{raw_table_name}' no DuckDB.")
            else:
                conn.execute(f"CREATE TABLE {raw_table_name} AS SELECT * FROM df_view")
                print(f"Tabela '{raw_table_name}' criada e dados inseridos no DuckDB.")

    print(f"Dados salvos na tabela '{raw_table_name}' no DuckDB.")
    salvar_no_mongo(processed_data, collection_name)
    print(f"Novos dados adicionados à coleção '{collection_name}' no MongoDB.")

def carregar_dados_duckdb(table_name):
    """
    Carrega dados de uma tabela no DuckDB.
    - Clean Code: função simples e clara.
    """
    with duckdb_lock:
        # Uso do context manager para fechar conexão adequadamente
        with duckdb.connect(DUCKDB_FILE) as conn:
            try:
                df = conn.execute(f"SELECT * FROM {table_name}").fetchdf()
            except (duckdb.BinderException, duckdb.CatalogException):
                print(f"Tabela {table_name} não encontrada no DuckDB.")
                df = pd.DataFrame()
    return df

def tratar_dados_func(transformed_table, processed_table, resultados_dir, collection_name):
    """
    Trata dados transformados: análise de sentimento, salva resumo e dados processados.
    - Sempre salva a tabela processed, mesmo sem classes suficientes.
    """
    df = carregar_dados_duckdb(transformed_table)
    if df.empty:
        print(f"Nenhum dado encontrado na tabela {transformed_table}.")
        return pd.DataFrame()

    analyzer = SentimentIntensityAnalyzer()
    df['texto_para_sentimento'] = df.apply(
        lambda row: row['texto'] if isinstance(row['texto'], str) and row['texto'].strip() else row['titulo_noticia'],
        axis=1
    )
    df['sentimento'] = df['texto_para_sentimento'].apply(
        lambda x: analyzer.polarity_scores(x)['compound'] if isinstance(x, str) else 0
    )
    df['sentimento'] = df['sentimento'].fillna(0)

    # Classificação baseada no VADER
    df['sentimento_classificacao'] = df['sentimento'].apply(
        lambda x: 'positivo' if x >= 0.05 else ('negativo' if x <= -0.05 else 'neutro')
    )
    df['target'] = df['sentimento_classificacao'].apply(lambda x: 1 if x == 'positivo' else 0)

    sentimento_summary = df['sentimento_classificacao'].value_counts().to_frame().reset_index()
    sentimento_summary.columns = ['Sentimento', 'Contagem']
    sentimento_file = os.path.join(resultados_dir, f"{transformed_table}_sentimento_summary.csv")
    sentimento_summary.to_csv(sentimento_file, index=False)
    print(f"Resumo de sentimentos salvo em '{sentimento_file}'.")

    salvar_no_mongo(df.to_dict(orient='records'), f"{collection_name}_processed")
    print(f"Dados processados salvos na coleção '{collection_name}_processed' no MongoDB.")

    # Salva tabela processed no DuckDB sempre
    with duckdb_lock:
        with duckdb.connect(DUCKDB_FILE) as conn:
            conn.register("df_view", df)
            conn.execute(f"CREATE OR REPLACE TABLE {processed_table} AS SELECT * FROM df_view")
    print(f"Dados tratados salvos na tabela '{processed_table}' no DuckDB.")

    return df

def treinar_ia(df: pd.DataFrame, nome_modelo: str, resultados_dir: str):
    """
    Treina modelos de IA. Sempre gera relatório, mesmo se não houver métricas.
    """
    # Caso não haja dados, gera relatório mínimo
    if df.empty:
        print(f"Nenhum dado para treinar IA em {nome_modelo}. Gerando relatório vazio.")
        relatorio_path = os.path.join(resultados_dir, f"{nome_modelo}_relatorio.txt")
        with open(relatorio_path, 'w') as relatorio:
            relatorio.write(f"Relatório de Treinamento de IA para {nome_modelo}\n")
            relatorio.write("="*50 + "\n\n")
            relatorio.write("Nenhum dado disponível para treinamento.\n")
        return

    X = df[['sentimento']].fillna(0)
    y = df['target']

    if y.nunique() <= 1:
        print(f"Dados insuficientes para treinar IA em {nome_modelo}, mas gerando relatório.")
        sentimento_summary = df['sentimento_classificacao'].value_counts().to_frame().reset_index()
        sentimento_summary.columns = ['Sentimento', 'Contagem']
        sentimento_file = os.path.join(resultados_dir, f"{nome_modelo}_sentimento_summary.csv")
        sentimento_summary.to_csv(sentimento_file, index=False)

        relatorio_path = os.path.join(resultados_dir, f"{nome_modelo}_relatorio.txt")
        with open(relatorio_path, 'w') as relatorio:
            relatorio.write(f"Relatório de Treinamento de IA para {nome_modelo}\n")
            relatorio.write("="*50 + "\n\n")
            relatorio.write("Dados insuficientes para treinamento.\n")
            relatorio.write("\nDistribuição de Sentimentos:\n")
            relatorio.write(sentimento_summary.to_string(index=False))
            relatorio.write("\nNenhuma métrica foi gerada.\n")
        return

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError as e:
        print(f"Erro ao dividir dados: {e}. Gerando relatório indicando erro.")
        relatorio_path = os.path.join(resultados_dir, f"{nome_modelo}_relatorio.txt")
        with open(relatorio_path, 'w') as relatorio:
            relatorio.write(f"Relatório de Treinamento de IA para {nome_modelo}\n")
            relatorio.write("="*50 + "\n\n")
            relatorio.write(f"Erro ao dividir dados: {e}\n")
        return

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        print(f"Classes insuficientes após split para {nome_modelo}. Gerando relatório mínimo.")
        sentimento_summary = df['sentimento_classificacao'].value_counts().to_frame().reset_index()
        sentimento_summary.columns = ['Sentimento', 'Contagem']
        sentimento_file = os.path.join(resultados_dir, f"{nome_modelo}_sentimento_summary.csv")
        sentimento_summary.to_csv(sentimento_file, index=False)

        relatorio_path = os.path.join(resultados_dir, f"{nome_modelo}_relatorio.txt")
        with open(relatorio_path, 'w') as relatorio:
            relatorio.write(f"Relatório IA {nome_modelo}\n")
            relatorio.write("="*50 + "\n\n")
            relatorio.write("Classes insuficientes após split.\n")
            relatorio.write("Distribuição Sentimentos:\n")
            relatorio.write(sentimento_summary.to_string(index=False))
        return

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
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            metrics_list.append({
                "Algoritmo": nome_algoritmo,
                "Acurácia": acuracia,
                "Precisão": precisao,
                "Recall": recall,
                "F1-Score": f1
            })

            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
            cm_df = pd.DataFrame(cm, index=["Negativo", "Positivo"], columns=["Negativo", "Positivo"])
            cm_file = os.path.join(resultados_dir, f"{nome_modelo}_{nome_algoritmo}_confusion_matrix.csv")
            cm_df.to_csv(cm_file)
            print(f"Matriz de Confusão para {nome_algoritmo} salva em '{cm_file}'.")

        except ValueError as e:
            print(f"Erro ao treinar {nome_algoritmo}: {e}")

    sentimento_summary = df['sentimento_classificacao'].value_counts().to_frame().reset_index()
    sentimento_summary.columns = ['Sentimento', 'Contagem']
    sentimento_file = os.path.join(resultados_dir, f"{nome_modelo}_sentimento_summary.csv")
    sentimento_summary.to_csv(sentimento_file, index=False)
    print(f"Resumo de sentimentos salvo em '{sentimento_file}'.")

    metrics_file = os.path.join(resultados_dir, f"{nome_modelo}_metrics.csv")
    relatorio_path = os.path.join(resultados_dir, f"{nome_modelo}_relatorio.txt")

    if metrics_list:
        metrics_df = pd.DataFrame(metrics_list)
        metrics_df.to_csv(metrics_file, index=False)
        print(f"Métricas salvas em '{metrics_file}'.")

        with open(relatorio_path, 'w') as relatorio:
            relatorio.write(f"Relatório de Treinamento de IA para {nome_modelo}\n")
            relatorio.write("="*50 + "\n\n")
            relatorio.write("Distribuição de Sentimentos:\n")
            relatorio.write(sentimento_summary.to_string(index=False))
            relatorio.write("\n\nMétricas dos Modelos:\n")
            relatorio.write(metrics_df.to_string(index=False))
            relatorio.write("\n\nObservações:\n")
            relatorio.write("IA analisando sentimento com VADER.\nClasses: Positivo(1), Negativo(0).\n")
        print(f"Relatório detalhado salvo em '{relatorio_path}'.")
    else:
        with open(relatorio_path, 'w') as relatorio:
            relatorio.write(f"Relatório IA {nome_modelo}\n")
            relatorio.write("="*50 + "\n\n")
            relatorio.write("Nenhuma métrica coletada (dados insuficientes ou erro no treinamento).\n")
            relatorio.write("Distribuição de Sentimentos:\n")
            relatorio.write(sentimento_summary.to_string(index=False))
        print(f"Relatório mínimo salvo em '{relatorio_path}' (sem métricas).")

# ---------------------------- ASSETS ----------------------------
@asset(
    description="Executa o crawler (economia).",
    kinds={"python"}
)
def crawler_economia() -> None:
    run_spider(NoticiasSpider, "economia_raw", "economia_raw")

@asset(
    description="Executa o crawler (governo).",
    kinds={"python"}
)
def crawler_governo() -> None:
    run_spider(G1Spider, "governo_raw", "governo_raw")

@asset(
    description="Executa DBT após coleta.",
    non_argument_deps={"crawler_economia", "crawler_governo"},
    kinds={"dbt"}
)
def executar_dbt() -> None:
    if not os.path.exists(DBT_PROJECT_PATH):
        raise NotADirectoryError(f"Diretório DBT não existe: {DBT_PROJECT_PATH}")
    subprocess.run(["dbt", "run"], check=True, cwd=DBT_PROJECT_PATH)
    print("Modelos DBT executados com sucesso.")

@asset(
    description="Trata dados de economia.",
    non_argument_deps={"executar_dbt"},
    kinds={"duckdb", "pandas", "mongodb"}
)
def tratar_dados_economia() -> pd.DataFrame:
    return tratar_dados_func("economia_transformed", "economia_processed", RESULTADOS_DIR, "economia_processed")

@asset(
    description="Trata dados de governo.",
    non_argument_deps={"executar_dbt"},
    kinds={"duckdb", "pandas", "mongodb"}
)
def tratar_dados_governo() -> pd.DataFrame:
    return tratar_dados_func("governo_transformed", "governo_processed", RESULTADOS_DIR, "governo_processed")

@asset(
    description="Treina IA economia.",
    non_argument_deps={"tratar_dados_economia"},
    kinds={"python", "scikitlearn"}
)
def treinar_ia_economia() -> None:
    df = carregar_dados_duckdb("economia_processed")
    treinar_ia(df, "economia", RESULTADOS_DIR)

@asset(
    description="Treina IA governo.",
    non_argument_deps={"tratar_dados_governo"},
    kinds={"python", "scikitlearn"}
)
def treinar_ia_governo() -> None:
    df = carregar_dados_duckdb("governo_processed")
    treinar_ia(df, "governo", RESULTADOS_DIR)

@asset(
    description="Verifica dados transformados.",
    non_argument_deps={"tratar_dados_economia", "tratar_dados_governo"},
    kinds={"python"}
)
def verificar_dados_transformados() -> None:
    with duckdb_lock:
        with duckdb.connect(DUCKDB_FILE) as conn:
            gov_df = conn.execute("SELECT * FROM governo_transformed LIMIT 5").fetchdf()
            print("Governo transformado (amostra):")
            print(gov_df)
            eco_df = conn.execute("SELECT * FROM economia_transformed LIMIT 5").fetchdf()
            print("Economia transformado (amostra):")
            print(eco_df)

@asset(
    description="Verifica disponibilidade dados IA.",
    non_argument_deps={"tratar_dados_economia", "tratar_dados_governo"},
    kinds={"python"}
)
def verificar_dados_disponiveis() -> None:
    eco_df = carregar_dados_duckdb("economia_processed")
    gov_df = carregar_dados_duckdb("governo_processed")
    print(f"Economia: {eco_df.shape[0]} registros disponíveis.")
    print(f"Governo: {gov_df.shape[0]} registros disponíveis.")


