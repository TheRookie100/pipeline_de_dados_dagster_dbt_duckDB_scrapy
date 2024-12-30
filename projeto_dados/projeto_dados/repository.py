'''# projeto_dados/projeto_dados/repository.py
from dagster import Definitions, ScheduleDefinition, define_asset_job, resource
from .assets import (
    crawler_economia,
    crawler_governo,
    executar_dbt,
    tratar_dados_economia,
    tratar_dados_governo,
    treinar_ia_economia,
    treinar_ia_governo,
    verificar_dados_transformados,
    verificar_dados_disponiveis,
)
from pymongo import MongoClient

@resource(config_schema={"mongo_uri": str})
def mongo_resource(init_context):
    """
    Recurso para conectar ao MongoDB.
    """
    mongo_uri = init_context.resource_config["mongo_uri"]
    client = MongoClient(mongo_uri)
    db = client['projeto_dados']  # Substitua pelo nome do seu banco de dados
    return db

# Definindo o job que materializa os assets
dagster_pipeline_job = define_asset_job(
    "dagster_pipeline_job",
    selection=[
        crawler_economia,
        crawler_governo,
        executar_dbt,
        tratar_dados_economia,
        tratar_dados_governo,
        treinar_ia_economia,
        treinar_ia_governo,
        verificar_dados_transformados,
        verificar_dados_disponiveis,
    ]
)

# Definindo a ScheduleDefinition para o job
dagster_schedule = ScheduleDefinition(
    job=dagster_pipeline_job,
    cron_schedule="0 */1 * * *",  # Executa a cada hora
)

# Definição do repositório de assets com o recurso MongoDB
defs = Definitions(
    assets=[
        crawler_economia,
        crawler_governo,
        executar_dbt,
        tratar_dados_economia,
        tratar_dados_governo,
        treinar_ia_economia,
        treinar_ia_governo,
        verificar_dados_transformados,
        verificar_dados_disponiveis,
    ],
    resources={
        "mongo": mongo_resource.configured({"mongo_uri": "mongodb://<usuario>:<senha>@<host>:<porta>/<database>"})
    },
    schedules=[dagster_schedule],
)
'''



# projeto_dados/projeto_dados/repository.py
from dagster import Definitions, ScheduleDefinition, define_asset_job, resource
from .assets import (
    crawler_economia,
    crawler_governo,
    executar_dbt,
    tratar_dados_economia,
    tratar_dados_governo,
    treinar_ia_economia,
    treinar_ia_governo,
    verificar_dados_transformados,
    verificar_dados_disponiveis,
)
from pymongo import MongoClient
import duckdb

@resource(config_schema={"mongo_uri": str})
def mongo_resource(init_context):
    """
    Recurso para conectar ao MongoDB.
    """
    mongo_uri = init_context.resource_config["mongo_uri"]
    client = MongoClient(mongo_uri)
    db = client['projeto_dados']  # Substitua pelo nome do seu banco de dados
    return db

@resource
def duckdb_resource():
    """
    Recurso para conectar ao DuckDB.
    """
    DUCKDB_FILE = "C:/Users/guilherme.rezende.PC066/Desktop/Projeto/dagster/projeto_dados/noticias.duckdb"
    conn = duckdb.connect(DUCKDB_FILE)
    try:
        yield conn
    finally:
        conn.close()

# Definindo o job que materializa os assets
dagster_pipeline_job = define_asset_job(
    "dagster_pipeline_job",
    selection=[
        crawler_economia,
        crawler_governo,
        executar_dbt,
        tratar_dados_economia,
        tratar_dados_governo,
        treinar_ia_economia,
        treinar_ia_governo,
        verificar_dados_transformados,
        verificar_dados_disponiveis,
    ]
)

# Definindo a ScheduleDefinition para o job
dagster_schedule = ScheduleDefinition(
    job=dagster_pipeline_job,
    cron_schedule="0 */1 * * *",  # Executa a cada hora
)

# Definição do repositório de assets com os recursos MongoDB e DuckDB
defs = Definitions(
    assets=[
        crawler_economia,
        crawler_governo,
        executar_dbt,
        tratar_dados_economia,
        tratar_dados_governo,
        treinar_ia_economia,
        treinar_ia_governo,
        verificar_dados_transformados,
        verificar_dados_disponiveis,
    ],
    resources={
        "mongo": mongo_resource.configured({"mongo_uri": "mongodb://<usuario>:<senha>@<host>:<porta>/<database>"}),
        "duckdb": duckdb_resource,
    },
    schedules=[dagster_schedule],
)





''''''

'''

# projeto_dados/projeto_dados/repository.py
from dagster import Definitions, ScheduleDefinition, define_asset_job, resource
from .assets import (
    crawler_economia,
    crawler_governo,
    executar_dbt,
    tratar_dados_economia,
    tratar_dados_governo,
    treinar_ia_economia,
    treinar_ia_governo,
    verificar_dados_transformados,
    verificar_dados_disponiveis,
)
from pymongo import MongoClient

@resource(config_schema={"mongo_uri": str})
def mongo_resource(init_context):
    """
    Recurso para conectar ao MongoDB.
    """
    mongo_uri = init_context.resource_config["mongo_uri"]
    client = MongoClient(mongo_uri)
    db = client['projeto_dados']  # Substitua pelo nome do seu banco de dados
    return db

# Definindo o job que materializa os assets
dagster_pipeline_job = define_asset_job(
    "dagster_pipeline_job",
    selection=[
        crawler_economia,
        crawler_governo,
        executar_dbt,
        tratar_dados_economia,
        tratar_dados_governo,
        treinar_ia_economia,
        treinar_ia_governo,
        verificar_dados_transformados,
        verificar_dados_disponiveis,
    ]
)

# Definindo a ScheduleDefinition para o job
dagster_schedule = ScheduleDefinition(
    job=dagster_pipeline_job,
    cron_schedule="0 */1 * * *",  # Executa a cada hora
)

# Definição do repositório de assets com o recurso MongoDB
defs = Definitions(
    assets=[
        crawler_economia,
        crawler_governo,
        executar_dbt,
        tratar_dados_economia,
        tratar_dados_governo,
        treinar_ia_economia,
        treinar_ia_governo,
        verificar_dados_transformados,
        verificar_dados_disponiveis,
    ],
    resources={
        "mongo": mongo_resource.configured({"mongo_uri": "mongodb://<usuario>:<senha>@<host>:<porta>/<database>"})
    },
    schedules=[dagster_schedule],
)
'''