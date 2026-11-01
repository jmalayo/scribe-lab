CREATE VIEW v_sesiones AS
SELECT
    s.id AS sesion_id,
    s.api_key_id,
    s.creada_en,
    count(c.id) AS consultas,
    max(c.turno) AS ultimo_turno,
    min(c.creada_en) AS primera_consulta,
    max(c.creada_en) AS ultima_consulta
FROM sesiones s
LEFT JOIN consultas c ON c.sesion_id = s.id
GROUP BY s.id, s.api_key_id, s.creada_en
