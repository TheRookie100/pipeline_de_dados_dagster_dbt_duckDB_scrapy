# --------------------------------- BIBLIOTECAS ---------------------------------

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

# ---------------------------- CONFIG DE DIRETÓRIOS E ARQUIVOS ----------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
RESULTADOS_DIR = os.path.join(BASE_DIR, "projeto_dados", "resultados") 
os.makedirs(RESULTADOS_DIR, exist_ok=True) 
DUCKDB_FILE = os.path.join(BASE_DIR, "noticias.duckdb") 
DBT_PROJECT_PATH = os.path.join(BASE_DIR, "dbt_project") 
duckdb_lock = FileLock(os.path.join(BASE_DIR, "duckdb.lock")) 

# ---------------------------- FUNÇÕES AUXILIARES PARA COLETA E TRATAMENTO DE DADOS ----------------------------

def run_spider(spider, raw_table_name, collection_name):
    """
    Executa spider do Scrapy, salva dados no DuckDB e MongoDB.
    - Clean Code: nomes claros, funções coesas.
    - SOLID (S): Função faz apenas coleta e inserção inicial.
    """
    # Obtém o timestamp atual no formato "YYYYMMDDHHMMSS"
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    
    # Define o nome do arquivo de saída com base no nome da tabela e timestamp
    output_file = f"projeto_dados/data/{raw_table_name}_{timestamp}.json"
    
    # Cria o diretório do arquivo de saída, se não existir
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Configura o processo do Scrapy com as configurações de feed
    process = CrawlerProcess(settings={
        "FEEDS": {output_file: {"format": "json"}},
        "FEED_EXPORT_ENCODING": "utf-8",
    })
    
    # Adiciona a spider ao processo do Scrapy
    process.crawl(spider)
    
    # Inicia o processo do Scrapy e aguarda sua conclusão
    process.start()

    # Abre o arquivo de saída e carrega os dados JSON
    with open(output_file, "r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)

    # Inicializa uma lista para armazenar os dados processados
    processed_data = []
    
    # Processa os dados de acordo com o nome da tabela
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

    # Converte os dados processados em um DataFrame do pandas
    df = pd.DataFrame(processed_data)

    # Garante que exista a coluna 'id'
    if 'id' not in df.columns: # Se a coluna 'id' não existir
        df['id'] = df.apply(lambda row: hashlib.md5(str(row).encode()).hexdigest(), axis=1) 
    
    # Garante que exista a coluna 'data_publicacao'
    if 'data_publicacao' not in df.columns: 
        df['data_publicacao'] = pd.NaT 

    # Uso de FileLock para evitar concorrência no mesmo processo
    with duckdb_lock: 
        # Uso do context manager para fechar conexão adequadamente
        with duckdb.connect(DUCKDB_FILE) as conn:
            # Registra o DataFrame como uma view temporária no DuckDB
            conn.register("df_view", df) 
            try: 
                # Tenta contar o número de registros na tabela existente
                count = conn.execute(f"SELECT COUNT(*) FROM {raw_table_name}").fetchone()[0]
            except (duckdb.BinderException, duckdb.CatalogException): # Se a tabela não existir
                # Se a tabela não existir, define o count como 0
                count = 0

            if count > 0: # Se houver registros na tabela
                # Insere novos dados na tabela existente, evitando duplicatas
                conn.execute(f""" 
                    INSERT INTO {raw_table_name} 
                    SELECT * FROM df_view 
                    WHERE id NOT IN (SELECT id FROM {raw_table_name})
                """) # Insere novos dados na tabela existente
                print(f"Novos dados apendados na tabela '{raw_table_name}' no DuckDB.")
            else: 
                
                conn.execute(f"CREATE TABLE {raw_table_name} AS SELECT * FROM df_view") 
                print(f"Tabela '{raw_table_name}' criada e dados inseridos no DuckDB.") 
 
    # Mensagem de confirmação de salvamento no DuckDB
    print(f"Dados salvos na tabela '{raw_table_name}' no DuckDB.")
    
    # Salva os dados processados no MongoDB
    salvar_no_mongo(processed_data, collection_name)
    
    # Mensagem de confirmação de salvamento no MongoDB
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
    
