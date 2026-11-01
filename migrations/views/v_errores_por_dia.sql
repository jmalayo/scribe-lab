CREATE VIEW v_errores_por_dia AS
SELECT
    creada_en::date AS dia,
    count(*) AS consultas,
    count(*) FILTER (WHERE estado >= 500) AS errores,
    round(count(*) FILTER (WHERE estado >= 500)::numeric / count(*), 4) AS tasa_error
FROM consultas
GROUP BY creada_en::date
ORDER BY dia
