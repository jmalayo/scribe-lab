CREATE VIEW v_rachas_sin_grounding AS
WITH evaluadas AS (
    SELECT
        creada_en,
        oraciones_sin_respaldo > 0 AS sin_respaldo,
        row_number() OVER (ORDER BY creada_en)
            - row_number() OVER (PARTITION BY oraciones_sin_respaldo > 0 ORDER BY creada_en) AS grupo
    FROM consultas
    WHERE oraciones_sin_respaldo IS NOT NULL
)
SELECT
    min(creada_en) AS inicio,
    max(creada_en) AS fin,
    count(*) AS largo
FROM evaluadas
WHERE sin_respaldo
GROUP BY grupo
ORDER BY inicio
