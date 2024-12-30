# **Data Pipeline de Notícias com Dagster, DBT, DuckDB e Scrapy**

## Visão Geral
Este projeto tem como objetivo **coletar notícias** de diferentes fontes (economia, governo etc.), **armazená-las** em um banco de dados local DuckDB, **transformá-las** com DBT e **orquestrar** todo o fluxo usando o Dagster. Além disso, o Scrapy é utilizado para realizar a captura dos dados de forma automática, e há etapas de análise e treinamento de IA (ex.: análise de sentimento).

## Componentes Principais

1. **Scrapy**  
   - **O que faz:** Extrai dados da web (notícias) por meio de spiders.  
   - **Onde está:** Pastas como `crawler_noticia/economia` e `crawler_noticia/governo` (exemplo).  

2. **DuckDB**  
   - **O que faz:** Banco de dados local, usado para armazenar os dados “raw” e transformados.  
   - **Onde está:** O arquivo principal (por exemplo, `noticias.duckdb`) fica na raiz do projeto.  

3. **DBT (Data Build Tool)**  
   - **O que faz:** Permite organizar e versionar as transformações SQL. Cria as tabelas transformadas no DuckDB (ex.: `economia_transformed`, `governo_transformed`).  
   - **Onde está:** A pasta do projeto DBT pode se chamar `dbt_project`, contendo `models/`, `dbt_project.yml` etc.  

4. **Dagster**  
   - **O que faz:** Orquestra toda a pipeline de dados em etapas (“assets” ou “ops”).  
   - **Onde está:** As definições de assets (coleta, transformação, tratamento de dados etc.) geralmente estão em arquivos como `assets.py`.

## Fluxo Resumido

1. **Coleta (Scrapy)**  
   - Dagster aciona os spiders do Scrapy para coletar notícias.  
   - Os dados coletados são salvos em JSON e, em seguida, inseridos no DuckDB (tabelas “raw”).  

2. **Transformação (DBT)**  
   - O Dagster chama `dbt run`.  
   - O DBT lê as tabelas “raw” e gera as tabelas “transformadas” (limpeza, normalização etc.).  

3. **Tratamento e IA**  
   - Em seguida, Dagster executa funções (ou assets) que leem as tabelas “transformadas” no DuckDB, fazem análise de sentimento (ex.: VADER) e treinam modelos de IA (RandomForest, etc.).  

4. **Armazenamento de Resultados**  
   - O projeto pode salvar informações processadas no DuckDB, em MongoDB e gerar relatórios locais (CSV, TXT).  


## Principais Comandos

1. **Executar o pipeline Dagster**  
   - Você pode iniciar um serviço do Dagster ou rodar via CLI. Ex.:  
     ```bash
     dagster dev
     ```
   - Em seguida, acesse o navegador (geralmente `localhost:3000` ou `localhost:3001`) para visualizar e disparar o pipeline.  

2. **Rodar manualmente o DBT**  
   - Dentro da pasta do DBT, você pode compilar e executar:  
     ```bash
     dbt debug
     dbt deps
     dbt run
     ```  

3. **Executar spiders separadamente (Scrapy)**  
   - Se quiser rodar um spider manualmente (fora do Dagster):  
     ```bash
     scrapy crawl noticia
     ```  

## Requisitos

- **Python 3.9+** (ou versão compatível)  
- **Dagster** (instalado via `pip install dagster`)  
- **Scrapy** (instalado via `pip install scrapy`)  
- **DuckDB** (integrado via `pip install duckdb`)  
- **DBT** (instalado via `pip install dbt-core dbt-duckdb` ou conforme documentação oficial)  

Consulte o arquivo `requirements.txt` (se existir) para mais detalhes.

## Como Contribuir

1. Faça um **fork** do projeto.  
2. Crie uma branch de feature: `git checkout -b minha-feature`.  
3. Realize suas modificações e submeta um **pull request**.  

## Licença

Este projeto pode estar sob licença MIT (ou outra de sua escolha). Verifique se há um arquivo `LICENSE`.  

---

**Contato**  
Para dúvidas ou melhorias, entre em contato com os mantenedores do repositório.
