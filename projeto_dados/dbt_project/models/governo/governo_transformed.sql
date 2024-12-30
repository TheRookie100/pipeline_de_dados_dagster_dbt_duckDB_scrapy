-- models/governo/governo_transformed.sql

WITH source AS (
    SELECT
        id,
        titulo_noticia,
        texto,
        data_publicacao
    FROM {{ source('governo_source', 'governo_raw') }}
)

SELECT
    id,
    titulo_noticia,
    texto,
    data_publicacao
    -- Adicione transformações adicionais aqui, se necessário
FROM source


