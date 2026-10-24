CREATE VIEW v_ultima_corrida AS
SELECT experimento, metodo, pregunta_id, corrida_id, creada_en, gold_vigente, respuesta, latencia_ms, tokens, gold_spans_at_10
FROM (
    SELECT
        c.experimento,
        c.metodo,
        r.pregunta_id,
        r.corrida_id,
        c.creada_en,
        r.gold_hash = p.gold_hash AS gold_vigente,
        r.respuesta,
        r.latencia_ms,
        r.tokens,
        r.gold_spans_at_10,
        ROW_NUMBER() OVER (PARTITION BY c.experimento, c.metodo, r.pregunta_id ORDER BY c.creada_en DESC) AS orden
    FROM resultados_pregunta r
    JOIN corridas_eval c ON c.id = r.corrida_id
    JOIN preguntas p ON p.id = r.pregunta_id
) ultimas
WHERE orden = 1
