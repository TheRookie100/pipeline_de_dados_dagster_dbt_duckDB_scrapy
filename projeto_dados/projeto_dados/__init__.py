"""from dagster import (
    AssetSelection,
    Definitions,
    ScheduleDefinition,
    define_asset_job,
    load_assets_from_modules,
)

from . import assets

all_assets = load_assets_from_modules([assets])

# Define a job that will materialize the assets
dagster_crawler_agendamento_job = define_asset_job("dagster_crawler_agendamento_job", selection=AssetSelection.all())

# Define a ScheduleDefinition for the job it should run and a cron schedule of how frequently to run it
dagster_crawler_agendamento = ScheduleDefinition(
    job=dagster_crawler_agendamento_job,
    cron_schedule="*/5 * * * *",  # every 20 minutes
)

defs = Definitions(
    assets=all_assets,
    schedules=[dagster_crawler_agendamento],
)

from dagster import (
    AssetSelection,  # Importa a classe AssetSelection do Dagster
    Definitions,  # Importa a classe Definitions do Dagster
    ScheduleDefinition,  # Importa a classe ScheduleDefinition do Dagster
    define_asset_job,  # Importa a função define_asset_job do Dagster
    load_assets_from_modules,  # Importa a função load_assets_from_modules do Dagster
)

from . import assets

all_assets = load_assets_from_modules([assets])

# Define a job that will materialize the assets
dagster_crawler_agendamento_job = define_asset_job(
    "dagster_crawler_agendamento_job", 
    selection=AssetSelection.keys(
        "crawler_economia",
        "crawler_governo",
        "tratar_dados_economia",
        "tratar_dados_governo",
        "treinar_ia_economia",
        "treinar_ia_governo"
    )
)

# Define a ScheduleDefinition for the job it should run and a cron schedule of how frequently to run it
dagster_crawler_agendamento = ScheduleDefinition(
    job=dagster_crawler_agendamento_job,
    cron_schedule="0 * * * *",  # Define a expressão cron para agendar o job a cada 60 minutos
)

defs = Definitions(
    assets=all_assets,
    schedules=[dagster_crawler_agendamento],
)
"""

"""
from dagster import (
    AssetSelection,  # Importa a classe AssetSelection do Dagster
    Definitions,  # Importa a classe Definitions do Dagster
    ScheduleDefinition,  # Importa a classe ScheduleDefinition do Dagster
    define_asset_job,  # Importa a função define_asset_job do Dagster
    load_assets_from_modules,  # Importa a função load_assets_from_modules do Dagster
)

from . import assets

all_assets = load_assets_from_modules([assets])

# Define a job that will materialize the assets
dagster_crawler_agendamento_job = define_asset_job(
    "dagster_crawler_agendamento_job", 
    selection=AssetSelection.keys(
        "crawler_economia",
        "crawler_governo",
        "tratar_dados_economia",
        "tratar_dados_governo",
        "treinar_ia_economia",
        "treinar_ia_governo"
    )
)

# Define a ScheduleDefinition for the job it should run and a cron schedule of how frequently to run it
dagster_crawler_agendamento = ScheduleDefinition(
    job=dagster_crawler_agendamento_job,
    cron_schedule="*/10 * * * *",  # Define a expressão cron para agendar o job a cada 10 minutos
)

defs = Definitions(
    assets=all_assets,
    schedules=[dagster_crawler_agendamento],
)
"""
'''
from dagster import (
    AssetSelection,
    Definitions,
    ScheduleDefinition,
    define_asset_job,
    load_assets_from_modules,
)
from . import assets

# Carregue todos os assets do módulo
all_assets = load_assets_from_modules([assets])

# Defina o job que materializará os assets
dagster_crawler_agendamento_job = define_asset_job(
    "dagster_crawler_agendamento_job",
    selection=AssetSelection.keys(
        "crawler_economia",
        "crawler_governo",
        "tratar_dados_economia",
        "tratar_dados_governo",
        "treinar_ia_economia",
        "treinar_ia_governo"
    )
)

# Defina a ScheduleDefinition para o job e um cronograma de execução
dagster_crawler_agendamento = ScheduleDefinition(
    job=dagster_crawler_agendamento_job,
    cron_schedule="*/10 * * * *",  # Executa o job a cada 10 minutos
)

# Definição do repositório de assets
defs = Definitions(
    assets=all_assets,
    schedules=[dagster_crawler_agendamento],
)
'''



'''

from dagster import (
    AssetSelection,
    Definitions,
    ScheduleDefinition,
    define_asset_job,
    load_assets_from_modules,
)
from . import assets

# Carregue todos os assets do módulo
all_assets = load_assets_from_modules([assets])

# Defina o job que materializará os assets
dagster_pipeline_job = define_asset_job(
    "dagster_pipeline_job",
    selection=AssetSelection.keys(
        "crawler_economia",
        "crawler_governo",
        "executar_dbt",
        "tratar_dados_economia",
        "tratar_dados_governo",
        "treinar_ia_economia",
        "treinar_ia_governo"
    )
)

# Defina a ScheduleDefinition para o job
dagster_schedule = ScheduleDefinition(
    job=dagster_pipeline_job,
    cron_schedule="0 */1 * * *",  # Executa a cada hora
)

# Definição do repositório de assets
defs = Definitions(
    assets=all_assets,
    schedules=[dagster_schedule],
)
'''
'''

# Definições do Dagster
from dagster import (
    AssetSelection,
    Definitions,
    ScheduleDefinition,
    define_asset_job,
    load_assets_from_modules,
)

# Importe o módulo de assets
from . import assets

# Carregue todos os assets do módulo 'assets'
all_assets = load_assets_from_modules([assets])

# Defina o job que materializará os assets
dagster_pipeline_job = define_asset_job(
    "dagster_pipeline_job",
    selection=AssetSelection.keys(
        "crawler_economia",
        "crawler_governo",
        "executar_dbt",
        "tratar_dados_economia",
        "tratar_dados_governo",
        "treinar_ia_economia",
        "treinar_ia_governo"
    )
)

# Defina a ScheduleDefinition para o job
dagster_schedule = ScheduleDefinition(
    job=dagster_pipeline_job,
    cron_schedule="0 */1 * * *",  # Executa a cada hora
)

# Definição do repositório de assets
defs = Definitions(
    assets=all_assets,
    schedules=[dagster_schedule],
)'''



