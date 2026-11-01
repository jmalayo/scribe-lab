CREATE VIEW v_latencia_movil AS
WITH dias AS (
    SELECT DISTINCT creada_en::date AS dia
    FROM consultas
    WHERE estado = 200
)
SELECT
    d.dia,
    count(*) AS consultas_7d,
    percentile_cont(0.5) WITHIN GROUP (ORDER BY c.latencia_total_ms) AS p50_ms,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY c.latencia_total_ms) AS p95_ms
FROM dias d
JOIN consultas c
    ON c.estado = 200
    AND c.creada_en::date BETWEEN d.dia - 6 AND d.dia
GROUP BY d.dia
ORDER BY d.dia
