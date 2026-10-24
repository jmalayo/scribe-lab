CREATE VIEW v_peores_preguntas AS
WITH fallos AS (
    SELECT
        c.experimento,
        v.fuente,
        v.pregunta_id,
        p.tipo,
        sum(
            CASE
                WHEN p.tipo = 'sin_respuesta' AND v.veredicto <> 'abstencion' THEN 1
                WHEN p.tipo <> 'sin_respuesta' AND v.veredicto IN ('incorrecta', 'abstencion') THEN 1
                WHEN v.veredicto = 'parcial' THEN 0.5
                ELSE 0
            END
        ) AS puntos_fallo
    FROM veredictos v
    JOIN corridas_eval c ON c.id = v.corrida_id
    JOIN preguntas p ON p.id = v.pregunta_id
    GROUP BY c.experimento, v.fuente, v.pregunta_id, p.tipo
),
rankeadas AS (
    SELECT
        *,
        DENSE_RANK() OVER (PARTITION BY experimento, fuente ORDER BY puntos_fallo DESC) AS puesto
    FROM fallos
    WHERE puntos_fallo > 0
)
SELECT experimento, fuente, puesto, pregunta_id, tipo, puntos_fallo
FROM rankeadas
WHERE puesto <= 3
