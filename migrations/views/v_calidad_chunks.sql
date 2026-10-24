CREATE VIEW v_calidad_chunks AS
WITH marcados AS (
    SELECT
        c.documento_id,
        trim(c.texto) = '' AS vacio,
        count(*) OVER (PARTITION BY c.texto) > 1 AS duplicado,
        c.n_tokens + 2 > 128 AS supera_limite
    FROM chunks c
)
SELECT
    d.nombre AS documento,
    count(*) AS chunks,
    count(*) FILTER (WHERE m.vacio) AS vacios,
    count(*) FILTER (WHERE m.duplicado) AS duplicados,
    count(*) FILTER (WHERE m.supera_limite) AS sobre_limite
FROM marcados m
JOIN documentos d ON d.id = m.documento_id
GROUP BY d.nombre
HAVING count(*) FILTER (WHERE m.vacio OR m.duplicado OR m.supera_limite) > 0
