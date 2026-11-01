CREATE VIEW v_drift_semanal AS
WITH top1 AS (
    SELECT consulta_id, score
    FROM consulta_chunks
    WHERE posicion = 1
)
SELECT
    date_trunc('week', c.creada_en)::date AS semana,
    count(*) AS consultas,
    round(avg(c.abstencion::int), 4) AS tasa_abstencion,
    percentile_cont(0.5) WITHIN GROUP (ORDER BY t.score) AS score_top1_p50,
    round(
        count(*) FILTER (WHERE c.oraciones_sin_respaldo > 0)::numeric
            / nullif(count(c.oraciones_sin_respaldo), 0),
        4
    ) AS tasa_sin_respaldo,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY c.latencia_total_ms) AS latencia_p95_ms
FROM consultas c
LEFT JOIN top1 t ON t.consulta_id = c.id
WHERE c.estado = 200
GROUP BY date_trunc('week', c.creada_en)
ORDER BY semana
