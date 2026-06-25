"""
Pipeline Data Lake — Cadena de Cines
Obligatorio Big Data — UCU

Zonas: Landing → Raw → Curated
Stack: Python + HDFS (via hdfs lib) + pandas + pyarrow (Parquet)

Requisitos:
    pip install hdfs pandas pyarrow pymongo

Supuestos:
    - Hadoop/HDFS corriendo en Docker en localhost:9870 (WebHDFS puerto 9870 o 50070)
    - NameNode accesible en localhost:9000
    - MongoDB corriendo en localhost:27017 con DB 'imdb'
    - CSV de Kaggle en ./data/movies.csv
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from hdfs import InsecureClient
from datetime import datetime
import pymongo
import json
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────

HDFS_URL      = "http://localhost:9870"   # WebHDFS
HDFS_USER     = "hadoop"
MONGO_URI     = "mongodb://localhost:27017/"
MONGO_DB      = "imdb"
MONGO_COL     = "movies"
CSV_PATH      = "./data/movies.csv"

# Rutas en HDFS
LANDING_PATH  = "/datalake/landing"
RAW_PATH      = "/datalake/raw"
CURATED_PATH  = "/datalake/curated"

# ─────────────────────────────────────────────
# CLIENTE HDFS
# ─────────────────────────────────────────────

def get_hdfs_client():
    return InsecureClient(HDFS_URL, user=HDFS_USER)

def init_hdfs_dirs(client):
    """Crea las zonas del Data Lake si no existen."""
    for path in [LANDING_PATH, RAW_PATH, CURATED_PATH,
                 f"{RAW_PATH}/movies", f"{CURATED_PATH}/kpi_roi",
                 f"{CURATED_PATH}/kpi_arranque", f"{CURATED_PATH}/kpi_directores",
                 f"{CURATED_PATH}/kpi_temporada"]:
        client.makedirs(path)
        log.info(f"Directorio HDFS listo: {path}")

# ─────────────────────────────────────────────
# ZONA LANDING — ingesta cruda
# ─────────────────────────────────────────────

def landing_kaggle(client):
    """
    Copia el CSV de Kaggle a HDFS sin modificar.
    Registra metadatos de ingesta como archivo JSON separado.
    """
    log.info("=== LANDING: Kaggle CSV ===")
    filename  = os.path.basename(CSV_PATH)
    hdfs_dest = f"{LANDING_PATH}/{filename}"

    with open(CSV_PATH, "rb") as f:
        client.write(hdfs_dest, f, overwrite=True)
    log.info(f"CSV subido a HDFS: {hdfs_dest}")

    # Metadatos de ingesta
    meta = {
        "fuente":        "kaggle_movies",
        "archivo":       filename,
        "hdfs_path":     hdfs_dest,
        "fecha_ingesta": datetime.utcnow().isoformat(),
        "filas_aprox":   sum(1 for _ in open(CSV_PATH)) - 1,
        "formato":       "csv"
    }
    meta_path = f"{LANDING_PATH}/meta_kaggle.json"
    client.write(meta_path, json.dumps(meta, indent=2).encode(), overwrite=True)
    log.info(f"Metadatos guardados: {meta_path}")
    return meta


def landing_imdb(client):
    """
    Exporta colección IMDb de MongoDB a JSON y sube a HDFS landing.
    """
    log.info("=== LANDING: IMDb MongoDB ===")
    mc   = pymongo.MongoClient(MONGO_URI)
    col  = mc[MONGO_DB][MONGO_COL]
    docs = list(col.find({}, {"_id": 0}))
    log.info(f"Documentos extraídos de MongoDB: {len(docs)}")

    hdfs_dest = f"{LANDING_PATH}/imdb_movies.json"
    client.write(hdfs_dest, json.dumps(docs).encode(), overwrite=True)
    log.info(f"IMDb subido a HDFS: {hdfs_dest}")

    meta = {
        "fuente":        "imdb_mongodb",
        "hdfs_path":     hdfs_dest,
        "fecha_ingesta": datetime.utcnow().isoformat(),
        "documentos":    len(docs),
        "formato":       "json"
    }
    meta_path = f"{LANDING_PATH}/meta_imdb.json"
    client.write(meta_path, json.dumps(meta, indent=2).encode(), overwrite=True)
    log.info(f"Metadatos guardados: {meta_path}")
    return docs

# ─────────────────────────────────────────────
# ZONA RAW — limpieza y normalización
# ─────────────────────────────────────────────

def raw_transform_kaggle(client):
    """
    Lee el CSV desde HDFS landing, limpia y guarda como Parquet en raw.
    Particionado por genre y release_decade.
    """
    log.info("=== RAW: transformación Kaggle ===")

    # Leer CSV desde HDFS
    with client.read(f"{LANDING_PATH}/{os.path.basename(CSV_PATH)}") as f:
        df = pd.read_csv(f)

    log.info(f"Filas originales: {len(df)}")

    # ── Limpieza ──────────────────────────────
    # Renombrar columnas a snake_case estándar
    df.columns = (df.columns
                    .str.strip()
                    .str.lower()
                    .str.replace(" ", "_"))

    # Eliminar duplicados
    df = df.drop_duplicates()

    # Eliminar filas sin título o sin género
    df = df.dropna(subset=["title", "genre"])

    # Columnas financieras: convertir a numérico, nulos → 0
    financial_cols = [
        "production_budget",
        "us_box_office",
        "global_box_office",
        "opening_day_sales",
        "first_week_us_sales"
    ]
    for col in financial_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Ratings a numérico
    for col in ["imdb_rating", "rotten_tomatoes_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Año de estreno limpio
    if "release_year" in df.columns:
        df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")
        df = df.dropna(subset=["release_year"])
        df["release_year"] = df["release_year"].astype(int)
        df["release_decade"] = (df["release_year"] // 10 * 10).astype(str) + "s"

    log.info(f"Filas después de limpieza: {len(df)}")

    # ── Guardar como Parquet ──────────────────
    # Guardamos local temporalmente y luego subimos a HDFS
    local_parquet = "/tmp/movies_raw.parquet"
    df.to_parquet(local_parquet, index=False, engine="pyarrow")

    hdfs_dest = f"{RAW_PATH}/movies/movies.parquet"
    with open(local_parquet, "rb") as f:
        client.write(hdfs_dest, f, overwrite=True)

    log.info(f"Parquet guardado en HDFS: {hdfs_dest}")
    return df


def raw_transform_imdb(client, imdb_docs):
    """
    Normaliza los documentos IMDb y guarda como Parquet en raw.
    """
    log.info("=== RAW: transformación IMDb ===")

    df_imdb = pd.DataFrame(imdb_docs)

    # Seleccionar campos relevantes si existen
    campos = ["title", "year", "genre", "director", "rating",
              "votes", "runtime", "language", "country"]
    campos_presentes = [c for c in campos if c in df_imdb.columns]
    df_imdb = df_imdb[campos_presentes].copy()

    df_imdb = df_imdb.drop_duplicates(subset=["title", "year"] if "year" in df_imdb.columns else ["title"])
    df_imdb = df_imdb.dropna(subset=["title"])

    if "rating" in df_imdb.columns:
        df_imdb["rating"] = pd.to_numeric(df_imdb["rating"], errors="coerce")
    if "year" in df_imdb.columns:
        df_imdb["year"] = pd.to_numeric(df_imdb["year"], errors="coerce").fillna(0).astype(int)

    local_parquet = "/tmp/imdb_raw.parquet"
    df_imdb.to_parquet(local_parquet, index=False, engine="pyarrow")

    hdfs_dest = f"{RAW_PATH}/movies/imdb.parquet"
    with open(local_parquet, "rb") as f:
        client.write(hdfs_dest, f, overwrite=True)

    log.info(f"IMDb Parquet guardado: {hdfs_dest}")
    return df_imdb

# ─────────────────────────────────────────────
# ZONA CURATED — KPIs
# ─────────────────────────────────────────────

def curated_kpi_roi(client, df):
    """
    KPI 1: Revenue por película + ROI por género
    ROI = (global_box_office - production_budget) / production_budget
    """
    log.info("=== CURATED: KPI ROI por género ===")

    df = df[df["production_budget"] > 0].copy()
    df["roi"] = (df["global_box_office"] - df["production_budget"]) / df["production_budget"]

    # Revenue por película
    revenue_peli = (df[["title", "release_year", "genre",
                          "global_box_office", "us_box_office",
                          "production_budget", "roi"]]
                    .sort_values("global_box_office", ascending=False))

    # ROI promedio por género
    roi_genero = (df.groupby("genre")
                    .agg(
                        roi_promedio=("roi", "mean"),
                        roi_mediana=("roi", "median"),
                        peliculas=("title", "count"),
                        revenue_total=("global_box_office", "sum")
                    )
                    .reset_index()
                    .sort_values("roi_promedio", ascending=False))

    _save_parquet(client, revenue_peli, f"{CURATED_PATH}/kpi_roi/revenue_por_pelicula.parquet")
    _save_parquet(client, roi_genero,   f"{CURATED_PATH}/kpi_roi/roi_por_genero.parquet")

    log.info(f"Top 5 géneros por ROI:\n{roi_genero[['genre','roi_promedio']].head()}")
    return roi_genero


def curated_kpi_temporada(client, df):
    """
    KPI 2: Películas con mayores ventas por mes / temporada
    Usa release_date o release_year para aproximar estacionalidad.
    """
    log.info("=== CURATED: KPI ventas por temporada ===")

    # Intentar parsear fecha si existe, sino usar año
    if "release_date" in df.columns:
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
        df["mes"] = df["release_date"].dt.month
    elif "release_year" in df.columns:
        # Sin mes exacto, agrupamos por año y década
        df["mes"] = None

    # Mapa de temporadas
    def temporada(mes):
        if mes in [12, 1, 2]:   return "verano_uy"     # dic-feb (verano Uruguay)
        elif mes in [3, 4, 5]:  return "otoño"
        elif mes in [6, 7, 8]:  return "invierno"
        elif mes in [9, 10, 11]:return "primavera"
        return "desconocida"

    if "mes" in df.columns and df["mes"].notna().any():
        df["temporada"] = df["mes"].apply(lambda m: temporada(m) if pd.notna(m) else "desconocida")
        ventas_temporada = (df.groupby("temporada")
                              .agg(
                                  revenue_total=("global_box_office", "sum"),
                                  revenue_promedio=("global_box_office", "mean"),
                                  peliculas=("title", "count")
                              )
                              .reset_index()
                              .sort_values("revenue_total", ascending=False))
        _save_parquet(client, ventas_temporada, f"{CURATED_PATH}/kpi_temporada/ventas_por_temporada.parquet")

    # Top películas por revenue sin importar temporada
    top_peliculas = (df[["title", "genre", "release_year", "global_box_office"]]
                     .sort_values("global_box_office", ascending=False)
                     .head(100))
    _save_parquet(client, top_peliculas, f"{CURATED_PATH}/kpi_temporada/top_peliculas.parquet")

    log.info("KPI temporada guardado.")
    return top_peliculas


def curated_kpi_directores(client, df):
    """
    KPI 3: Performance de directores — taquilla promedio ponderada
    Score = mean(global_box_office) * log(1 + count_peliculas)
    """
    log.info("=== CURATED: KPI directores ===")

    import numpy as np

    df_dir = df[df["director"].notna() & (df["global_box_office"] > 0)].copy()

    directores = (df_dir.groupby("director")
                        .agg(
                            peliculas=("title", "count"),
                            revenue_promedio=("global_box_office", "mean"),
                            revenue_total=("global_box_office", "sum"),
                            roi_promedio=("roi", "mean") if "roi" in df_dir.columns else ("global_box_office", "mean"),
                            rating_promedio=("imdb_rating", "mean") if "imdb_rating" in df_dir.columns else ("global_box_office", "count")
                        )
                        .reset_index())

    # Score ponderado: penaliza directores con una sola película
    directores["score_ponderado"] = (
        directores["revenue_promedio"] * np.log1p(directores["peliculas"])
    )

    directores = directores.sort_values("score_ponderado", ascending=False)

    _save_parquet(client, directores, f"{CURATED_PATH}/kpi_directores/ranking_directores.parquet")
    log.info(f"Top 5 directores:\n{directores[['director','peliculas','revenue_promedio','score_ponderado']].head()}")
    return directores


def curated_kpi_arranque(client, df):
    """
    KPI 4: Velocidad de arranque — ratio apertura vs total US
    ratio = opening_day_sales / us_box_office
    Alto ratio → explota y cae (blockbuster puro)
    Bajo ratio → tracción sostenida (mantener en cartelera)
    """
    log.info("=== CURATED: KPI velocidad de arranque ===")

    df_arr = df[
        (df["opening_day_sales"] > 0) &
        (df["us_box_office"] > 0)
    ].copy()

    df_arr["ratio_arranque"] = df_arr["opening_day_sales"] / df_arr["us_box_office"]

    def clasificar(r):
        if r >= 0.15:   return "explota_y_cae"
        elif r >= 0.08: return "arranque_normal"
        else:           return "traccion_sostenida"

    df_arr["tipo_arranque"] = df_arr["ratio_arranque"].apply(clasificar)

    arranque = (df_arr[["title", "genre", "release_year",
                         "opening_day_sales", "us_box_office",
                         "ratio_arranque", "tipo_arranque"]]
                .sort_values("ratio_arranque", ascending=False))

    # Resumen por género
    arranque_genero = (df_arr.groupby(["genre", "tipo_arranque"])
                             .agg(peliculas=("title", "count"))
                             .reset_index())

    _save_parquet(client, arranque,        f"{CURATED_PATH}/kpi_arranque/arranque_por_pelicula.parquet")
    _save_parquet(client, arranque_genero, f"{CURATED_PATH}/kpi_arranque/arranque_por_genero.parquet")

    log.info("KPI arranque guardado.")
    return arranque

# ─────────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────────

def _save_parquet(client, df, hdfs_path):
    """Guarda un DataFrame como Parquet en HDFS."""
    local = "/tmp/_temp_output.parquet"
    df.to_parquet(local, index=False, engine="pyarrow")
    with open(local, "rb") as f:
        client.write(hdfs_path, f, overwrite=True)
    log.info(f"Guardado: {hdfs_path} ({len(df)} filas)")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run_pipeline():
    log.info("========================================")
    log.info("  PIPELINE DATA LAKE — CADENA DE CINES  ")
    log.info("========================================")

    client = get_hdfs_client()
    init_hdfs_dirs(client)

    # ── LANDING ──────────────────────────────
    landing_kaggle(client)
    imdb_docs = landing_imdb(client)

    # ── RAW ──────────────────────────────────
    df_kaggle = raw_transform_kaggle(client)
    df_imdb   = raw_transform_imdb(client, imdb_docs)

    # ── CURATED ──────────────────────────────
    curated_kpi_roi(client, df_kaggle)
    curated_kpi_temporada(client, df_kaggle)
    curated_kpi_directores(client, df_kaggle)
    curated_kpi_arranque(client, df_kaggle)

    log.info("========================================")
    log.info("  PIPELINE COMPLETADO EXITOSAMENTE      ")
    log.info("========================================")


if __name__ == "__main__":
    run_pipeline()
