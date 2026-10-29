CREATE VIEW v_gold_nunca_recuperado AS
SELECT p.id AS pregunta_id, p.tipo, p.pregunta
FROM preguntas p
WHERE p.tipo <> 'sin_respuesta'
  AND EXISTS (
      SELECT 1
      FROM resultados_pregunta r
      WHERE r.pregunta_id = p.id AND r.gold_hash = p.gold_hash AND r.gold_spans_at_10 IS NOT NULL
  )
  AND NOT EXISTS (
      SELECT 1
      FROM resultados_pregunta r
      WHERE r.pregunta_id = p.id AND r.gold_hash = p.gold_hash AND r.gold_spans_at_10 > 0
  )
