CREATE VIEW v_acuerdo_juez_humano AS
WITH pares AS (
    SELECT c.experimento, c.metodo, h.veredicto AS humano, j.veredicto AS juez
    FROM veredictos h
    JOIN veredictos j ON j.corrida_id = h.corrida_id AND j.pregunta_id = h.pregunta_id AND j.fuente = 'juez'
    JOIN corridas_eval c ON c.id = h.corrida_id
    WHERE h.fuente = 'humano'
),
grupos AS (
    SELECT experimento, metodo, humano, juez FROM pares
    UNION ALL
    SELECT experimento, 'todos', humano, juez FROM pares
),
observado AS (
    SELECT experimento, metodo, count(*) AS n, avg((humano = juez)::int) AS acuerdo
    FROM grupos
    GROUP BY experimento, metodo
),
por_clase AS (
    SELECT
        g.experimento,
        g.metodo,
        avg((g.humano = k.clase)::int) * avg((g.juez = k.clase)::int) AS azar
    FROM grupos g
    CROSS JOIN (VALUES ('correcta'), ('parcial'), ('incorrecta'), ('abstencion')) AS k (clase)
    GROUP BY g.experimento, g.metodo, k.clase
),
esperado AS (
    SELECT experimento, metodo, sum(azar) AS azar
    FROM por_clase
    GROUP BY experimento, metodo
)
SELECT
    o.experimento,
    o.metodo,
    o.n,
    round(o.acuerdo, 3) AS acuerdo,
    round((o.acuerdo - e.azar) / NULLIF(1 - e.azar, 0), 3) AS kappa
FROM observado o
JOIN esperado e ON e.experimento = o.experimento AND e.metodo = o.metodo
