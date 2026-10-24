CREATE VIEW v_metricas_por_metodo AS
SELECT
    c.experimento,
    c.metodo,
    v.fuente,
    count(*) FILTER (WHERE p.tipo <> 'sin_respuesta') AS con_respuesta,
    count(*) FILTER (WHERE p.tipo <> 'sin_respuesta' AND v.veredicto = 'correcta') AS correctas,
    round(avg((v.veredicto = 'correcta')::int) FILTER (WHERE p.tipo = 'simple'), 3) AS tasa_simple,
    round(avg((v.veredicto = 'correcta')::int) FILTER (WHERE p.tipo = 'comparativa_mismo_doc'), 3) AS tasa_mismo_doc,
    round(avg((v.veredicto = 'correcta')::int) FILTER (WHERE p.tipo = 'comparativa_entre_docs'), 3) AS tasa_entre_docs,
    count(*) FILTER (WHERE p.tipo = 'sin_respuesta' AND v.veredicto = 'abstencion') AS abstenciones_correctas,
    count(*) FILTER (WHERE p.tipo <> 'sin_respuesta' AND v.veredicto = 'abstencion') AS abstenciones_falsas,
    round(avg(r.tokens), 1) AS tokens_medios
FROM veredictos v
JOIN corridas_eval c ON c.id = v.corrida_id
JOIN preguntas p ON p.id = v.pregunta_id
JOIN resultados_pregunta r ON r.corrida_id = v.corrida_id AND r.pregunta_id = v.pregunta_id
GROUP BY c.experimento, c.metodo, v.fuente
