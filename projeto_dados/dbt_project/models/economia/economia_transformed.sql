-- models/economia/economia_transformed.sql
WITH source AS (
    SELECT
        id,
        titulo_noticia,
        texto,
        data_publicacao,
        time_ago
    FROM {{ source('economia_source', 'economia_raw') }}
)

SELECT
    id,
    titulo_noticia,
    texto,
    data_publicacao,
    time_ago
    -- Adicione transformações adicionais aqui, se necessário
FROM source



