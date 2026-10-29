# async-music-fetch-engine

Subproyecto correspondiente a la primera etapa de una solución orientada al análisis de sonido mediante *machine learning* supervisado, consistente en un sistema de descarga física de música para la obtención y almacenamiento de archivos de audio en el formato configurado a partir de URL públicas de YouTube/YouTube Music o de playlists personales del servicio de *streaming* Spotify, mediante procesamiento por lotes y ejecución multihilo para la descarga paralela de múltiples enlaces y la reducción del tiempo de respuesta.

El acceso a cada fuente está sujeto a sus propias restricciones, por lo que no se procesa contenido *premium* o restringido mediante mecanismos de pago, a excepción de Spotify, cuyo funcionamiento requiere una cuenta Premium para la obtención del contenido; el alcance de esta integración está determinado por las limitaciones de la Spotify Web API, con acceso únicamente a las playlists propias del usuario autenticado.

## Requisitos

- Python 3.11+
- [ffmpeg](https://ffmpeg.org/) en el `PATH` (normalización de audio) (opcional)
- Node.js en el `PATH` (yt-dlp necesita un runtime de JS para resolver los challenges del reproductor de YouTube)
- Una aplicación de [Spotify Developer](https://developer.spotify.com/dashboard) (Client ID/Secret) (opcionales)
- Cuenta de **Spotify Premium** *(opcional, requerida para el uso de la Web API en modo desarrollo)*.



## Uso



### Credenciales y cookies

Las variables `SPOTIFY_CLIENT_ID` y `SPOTIFY_CLIENT_SECRET` definidas en el archivo `.env` corresponden a las credenciales de la aplicación registrada en Spotify Dashboard (opcionales si no se desea esta ruta de descarga). Su utilización recae en la generación de tokens mediante OAuth y posteriormente alojadas en DB local para su reutilización.

`POST /auth/spotify/login` acepta opcionalmente su propio `client_id`/`client_secret` en el body, para loguearse con una aplicación de Spotify Developer distinta a la del `.env`. Si no se envían (o se envían vacíos), usa los del `.env` por defecto. Las credenciales usadas en el login quedan guardadas junto al token en `spotify_tokens`, porque el refresh posterior tiene que hacerse con las mismas credenciales con las que se obtuvo — no necesariamente las del `.env`.

Para una ejecución completa del sistema se recomienda utilizar un archivo `cookies.txt` obtenido desde una sesión real de YouTube. Este archivo se utiliza en las llamadas a `yt-dlp` para permitir el acceso a contenido que requiere autenticación y mejorar la estabilidad de las descargas.

Su uso permite evitar que ciertos videos con restricción de edad sean rechazados por YouTube y reduce la probabilidad de bloqueos temporales asociados a múltiples solicitudes automatizadas. Cuando un recurso requiere autenticación y no se dispone de cookies válidas, el sistema lo identifica como `needs_auth` y continúa con el procesamiento sin realizar la descarga.

Para obtenerla:

1. Iniciar sesión en youtube.com con la cuenta que se quiera usar.
2. Abrir la consola de herramientas de desarrollador del navegador (F12) estando en esa página.
3. Pegar y ejecutar ahí el contenido de `music-extract/data/cookies/extract.js` (ya incluido en
  el repositorio), usa la CookieStore API del navegador (disponible en Chrome/Edge, no en  todos) para leer las cookies visibles y armar un archivo en formato Netscape.
4. El navegador descarga automáticamente un archivo `cookies.txt`.
5. Mover ese archivo a `music-extract/data/cookies/cookies-files/cookies.txt`.

El archivo `cookies.txt`, al ser un recurso compartido entre workers de descarga en paralelo, utiliza una copia privada temporal para cada descarga en lugar del archivo original. Esto se debe a que `yt-dlp` reescribe el archivo de cookies al cerrar cada sesión, y compartir el mismo archivo entre threads producía corrupción intermitente antes de aislarlo por descarga.

La sesión expira con el tiempo; no hay renovación automática, así que hay que repetir estos pasos periódicamente.

### Arquitectura de Base de datos

SQLite, con el schema administrado por SQLAlchemy y las migraciones por Alembic. Se crea y actualiza automaticamente al levantar el API. 

Cinco tablas: 

- `jobs` (una fila por pedido de descarga)
- `tracks` (una fila por URL dentro de un job) 
- `attempts` (una fila por cada candidato/cliente probado, con su score y proveedor)
- `retries` (una fila por ronda de reintento manual)
- `spotify_tokens` (la sesión OAuth, si se usó)

La verificación de recursos ya descargados se realiza exclusivamente contra estos registros, sin comprobar la existencia física del archivo en el sistema de almacenamiento. En consecuencia, si un archivo `.mp3` es eliminado manualmente pero su registro permanece en la base de datos, el recurso no se descargará nuevamente.

`jobs.elapsed_time_s` se calcula y se graba una única vez, en el momento en que el job pasa a `completed`/`completed_with_errors` — no en cada consulta. Mientras el job sigue `in_progress`, `view-download-status` lo calcula al vuelo (`ahora - creado`); una vez guardado, ese valor se reutiliza tal cual hasta que un reintento/retomado vuelva a poner el job en curso (lo que lo resetea a `null` para que vuelva a calcularse en vivo).

Ejemplo de flujo con datos reales:

```bash
curl -X POST http://localhost:8000/downloads/download-urls-manual \
  -H "Content-Type: application/json" \
  -d '{"urls": ["https://www.youtube.com/watch?v=JEdxT8yCf98"]}'
# -> {"job_id": [AUTOGENERATED_ID], "type": "manual-urls", "total": 1}

curl http://localhost:8000/stages/view-download-status?job_id=[AUTOGENERATED_ID]
# -> {"status": "completed", "successful": 1, "failed": 0, ...}
```

El track resultante queda con `title: "Mala Pata"`, `status: successful`, y un intento (`provider: yt-dlp`) registrado en `attempts`.

## Configuración (local)

```bash
cd music-extract
python -m venv .venv && source .venv/bin/activate   # o .venv\Scripts\activate en Windows
pip install -r requirements.txt
pip install -e .                 # instala musictools como paquete editable
playwright install chromium      # fallback de navegador para videos geo-restringidos
```

Registrar `http://127.0.0.1:8888/callback` como Redirect URI en la aplicación de Spotify Developer (lo requiere el flujo OAuth de `/auth/spotify/login`)

Alternativa a los comandos de arriba: `.vscode/launch.json` (config "FastAPI: music-extract API") levanta la API con F5 desde VS Code, apuntando `cwd` a `async-music-fetch-engine/music-extract/app`. Requiere el mismo `pip install -e .`/`pip install -r requirements.txt` corridos una vez contra el intérprete que declara ese archivo. No corras esta sesión local al mismo tiempo que el contenedor Docker — ambos apuntan al mismo `db_downloads.sqlite` y al mismo puerto `8888` de callback OAuth, y competirían entre sí.

### Selección y validación del candidato

Opciones de rutas de descargas

- Grupo de URLs sueltas (`download-urls-manual`, para YouTube o Spotify indistintamente)  
- URL de la playlist completa (`download-spotify-playlist`). Para un track de Spotify, los candidatos se buscan en dos fuentes independientes: YouTube Music y búsqueda plana (Youtube) combinadas con un cupo garantizado para cada una.

Cada candidato se evalúa comparando su título, artista y duración con los valores reales del track. Los resultados se ordenan de mayor a menor score, por lo que únicamente se procesa el primer candidato que logra descargarse. Si el score obtenido es igual o superior a 60 (techo mínimo por defecto), el track se clasifica como `successful`; en caso contrario, queda como `incorrect`, aunque la descarga del archivo haya sido exitosa.

- El track **"fable"** de **"a vow"** encontró como candidato **"fable"** del canal **"a vow - Topic"**, con un score de **135.0**. Al superar ampliamente el umbral, se aceptó directamente como `successful`.
- El track **"The Eyes, Chico. They Never Lie"** de **OURGRND** encontró como candidato **"Scarface - The Eyes, Chico, They Never Lie."** del canal **"OneMinute TV"**, con un score aproximado de **40**. Aunque el archivo pudo descargarse correctamente, el score quedó por debajo del umbral, por lo que el resultado final fue `incorrect`.

Un track marcado como `incorrect` indica que ningún candidato de YouTube superó el umbral de similitud requerido; la descarga puede existir, pero no corresponde a una coincidencia confiable.

### Resolución de metadata

La metadata se obtiene directamente desde la Spotify Web API y, al procesar playlists completas, se reutiliza la información obtenida durante el listado de tracks, evitando consultas repetidas. En una prueba con una playlist de 54 tracks, esta optimización redujo el tiempo promedio por canción de ~38-39 segundos (flujo anterior basado en `spotdl`) a ~6.6 segundos.

**Aclaración:** Las canciones obtenidas desde Spotify utilizan el mismo flujo y proveedor de descarga (`yt-dlp`) que las URLs directas de YouTube.

`yt-dlp` es el proveedor utilizado en el primer intento, incluyendo su rotación interna de `player_client`. El proveedor `notube` (automatización de navegador contra notube.im) se utiliza únicamente como fallback: para un candidato de Spotify, solo cuando el motivo de fallo es `geo_restricted`; para una URL directa de YouTube, además del caso `geo_restricted`, también como último recurso si `yt-dlp` agota sus intentos por cualquier otro motivo no descartado (`premium_only`, `needs_auth` y `client_blocked` se omiten sin intentar `notube`).

### Carpeta de descargas

Los archivos se almacenan en `data/music-downloaded/{carpeta}/`, donde cada carpeta corresponde a un `group_name` (descargas manuales) o al nombre de la playlist. Ambos valores se normalizan (minúsculas, sin tildes, espacios reemplazados por `_` y eliminación de caracteres inválidos). Si una descarga manual no tiene `group_name`, se almacena en la carpeta compartida `any-musics/`.  

El nombre final del archivo proviene del video descargado del mejor candidato y no del título oficial de Spotify, por lo que puede no coincidir exactamente con el formato "Artista - Canción".

### Variables configurables

Archivo `app/config/settings.py`

- `THREADS` *(Default: 3)* — cantidad de descargas en paralelo. Aumentarlo reduce tiempo de respuesta, pero también multiplica los pedidos simultáneos a YouTube, con más riesgo del bloqueo por detección de bot. 
- `FORMAT/BITRATE` *(Default: mp3/192k)* — formato y calidad del audio final. 
- `MAX_RETRIES_PER_TRACK` *(Default: 3)* — cuántas rondas de reintento se permiten por track. 
- `ETA_WINDOW` *(Default: 20)* — cuántos tracks recientes se promedian para estimar el tiempo restante.
- `TOP_N/CLIENT_CHAIN_LEN` *(Default: 5/5)* — alimentan cuántos intentos máximos recibe un track (`TOP_N` × `CLIENT_CHAIN_LEN` para Spotify, `CLIENT_CHAIN_LEN` + 1 para YouTube directo).

Archivo `musictools/download/core.py`

- `MIN_SCORE` *(Default: 60)* — el umbral de la sección anterior (Selección y validación del candidato).
- `candidates_max`/`candidates_per_source` *(Default: 15/10, con la mitad del cupo garantizada por fuente)*



### Scripts standalone

`scripts/` tiene entry points de CLI que no dependen del servidor corriendo:

- `download_from_csv.py` — descarga en batch a partir de un CSV (mismo formato que exporta `spotify-api/fetch_playlists.js`), sin pasar por el sistema de jobs/reintentos.
- `normalize_folder.py` — corre la normalización LUFS/peak sobre una carpeta ya descargada. Su salida se guarda en `data/music-normalized/{carpeta}/`, manteniendo el mismo esquema de directorios que la carpeta de origen.
- `check_boundaries.py` — el chequeo de las dos reglas de `musictools` (ver "Arquitectura"), pensado para correr en CI.
- `spotify-api/fetch_playlists.js` — corre en el host (**no** forma parte de la imagen Docker), abre un navegador real y escucha en `127.0.0.1:8888`). Exporta una de tus playlists de Spotify a un CSV:

```bash
cd spotify-api
npm install
node fetch_playlists.js --client-id <id> --client-secret <secret>
```

Abre un navegador para un login OAuth interactivo, lista tus playlists, y escribe un CSV en `playlists-fetched/`, que `scripts/download_from_csv.py` puede consumir después para una corrida de descarga batch, solo por CLI.

> **Nota:** `POST /downloads/download-spotify-playlist` (una vez completado `/auth/spotify/login`) cubre exactamente lo mismo de punta a punta — login, listar tus playlists, traer los tracks — sin ningún CSV intermedio, directo a un job con seguimiento y reintentos. Este script se mantiene como la alternativa batch/CSV para los casos en los que preferís inspeccionar o editar la lista de tracks antes de descargar cualquier cosa.



## Arquitectura

```
music-extract/
├── musictools/               # paquete independiente — no sabe nada de FastAPI ni de la DB.
│   ├── download/              #   core.py orquesta: metadata → candidatos → scoring → descarga
│   └── normalize/              #   normalización LUFS/peak vía ffmpeg
│
├── app/                        # la API. Depende de musictools, nunca al reves.
│   ├── config/                 #   unica fuente del .env y de las rutas del filesystem
│   ├── core/                   #   logica de negocio de jobs/tracks/reintentos
│   ├── infrastructure/
│   │   ├── persistence/        #   modelos SQLAlchemy + migraciones de Alembic
│   │   └── external/           #   conecta musictools + la API de Spotify al sistema de jobs
│   └── api/                    #   routers de FastAPI
│
├── scripts/                    # entry points de CLI standalone (descarga batch, normalizacion)
└── data/                       # estado de runtime: audio descargado, cookies, cache de bloqueados

spotify-api/
└── fetch_playlists.js          # CLI de OAuth interactivo para exportar una playlist a CSV
```

Dos reglas mantienen a `musictools` genuinamente reusable fuera de este proyecto:

1. `musictools` **nunca importa de** `app`**.** Verificado por `scripts/check_boundaries.py`, un
  chequeo basado en grep pensado para correr en CI, no una convención que se pudre en silencio.
2. `musictools` **nunca lee** `.env` **ni resuelve rutas desde** `__file__`**.** Cada función recibe lo
  que necesita como parámetro explícito — quien la llama (`app/` o un script) decide dónde  
   viven las cosas.



## Docker

Correr el proyecto en Docker es **una opción, no un requisito** — la alternativa es correr la API directamente en local con el intérprete del sistema (ver "Configuración (local)" más arriba, o `.vscode/launch.json` para levantarla con F5 desde VS Code). Docker resulta más cómodo para dejar la API corriendo de forma persistente o accesible desde otro dispositivo en la LAN (`0.0.0.0:8000`); en local es más directo para debuggear paso a paso.

```bash
cd async-music-fetch-engine
docker compose up -d --build
```

El audio descargado, el archivo de cookies y la base SQLite se montan como volúmenes (`music-extract/data/`, `music-extract/app/db/`) para que sobrevivan a los rebuilds.

> **`journal_mode`.** `database.py` fuerza `PRAGMA journal_mode=DELETE` en vez del `WAL` por defecto de SQLite — WAL requiere `mmap` sobre un archivo `-shm` que no es confiable a través de un bind mount de Windows en Docker Desktop (pudo causar un `disk I/O error` irrecuperable sin reiniciar la máquina). El costo de concurrencia es aceptable porque las escrituras que importan ya están serializadas con un lock propio en `job_service.py`.

> **OAuth de Spotify — de dónde sale la URL de login según cómo corras la API.** `run_authorization_flow()` (`spotify_oauth.py`) detecta si está corriendo dentro de un contenedor (`/.dockerenv`):
> - **Corriendo en Docker:** no hay navegador disponible dentro del contenedor (headless), así que la URL de autorización se **imprime en los logs** (`docker compose logs -f api` o la consola donde corriste `up`) — hay que copiarla a mano y abrirla en el navegador del host.
> - **Corriendo en local:** se abre el navegador automáticamente (`webbrowser.open`), sin pasos manuales.
>
> En ambos casos, el servidor de callback bindea a `0.0.0.0:8888` (publicado en el compose como `127.0.0.1:8888:8888` — accesible desde el host, no desde la LAN) y el `redirect_uri` es `http://127.0.0.1:8888/callback`, resuelto por el navegador del host — cumple la exigencia de Spotify de usar loopback literal sin que el contenedor deba ser alcanzable por nombre. El token queda persistido en la base SQLite bajo `app/db/` (volumen), así que la autorización interactiva se hace una sola vez y los reinicios posteriores solo refrescan. El `.env` va en `music-extract/.env`, que es donde lo busca tanto el compose como una corrida local.



## Referencia de la API


| Método | Ruta                                                    | Descripción                                                                                                                                                                                     |
| ------ | -------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `POST` | `/downloads/download-urls-manual`                        | Descarga una o varias URLs sueltas. Body: `{urls: [...], group_name?: str}`.                                                                                                                    |
| `POST` | `/downloads/download-spotify-playlist`                    | Descarga una playlist completa de Spotify. Body: `{playlist_name: str}` — acepta un nombre **o** una URL/URI de playlist.                                                                       |
| `GET`  | `/stages/view-download-status` (`?job_id=`)                | Estado agregado: conteos por estado, ETA, última ronda de reintento. Sin `job_id`, devuelve el job más reciente.                                                                                |
| `GET`  | `/stages/view-failed-urls` (`?job_id=`, `?format=csv`)     | Detalle de los tracks fallidos/omitidos de un job. Sin `job_id`, el más reciente.                                                                                                                |
| `GET`  | `/stages/view-all-jobs`                                    | Lista todos los jobs (incluidos los que están en curso).                                                                                                                                        |
| `POST` | `/auth/spotify/login`                                    | Login OAuth interactivo de Spotify, de una sola vez (abre un navegador o imprime la URL en logs, ver "Docker"). Body opcional `{client_id?, client_secret?}` — sin esto, usa el `.env`.          |
| `GET`  | `/auth/spotify/account-status`                                     | `{"autenticado": bool, ...}` — si hay token guardado, incluye además todos los campos de `GET /v1/me` de Spotify (id, display_name, email, product, etc.), no solo el booleano.                 |

**No existe un endpoint `/retry` separado.** Reintentar o retomar un bloque es simplemente volver a llamar a `download-urls-manual`/`download-spotify-playlist` con las mismas URLs (mismo `group_name`) o la misma playlist — el comportamiento se decide solo, según el estado real de los tracks de ese job:
- Si quedan tracks `pending`/`in_progress` (el proceso se cortó a mitad de camino) → se **retoman** tal cual (los `in_progress` se descargan de nuevo desde cero). No cuenta como ronda de reintento.
- Si el bloque ya terminó de correr del todo y solo quedaron `failed`/`skipped` → recién ahí se **reintenta** de verdad, contra el tope de `MAX_RETRIES_PER_TRACK` rondas.




## Licencia

[MIT](LICENSE)