"""
Pipeline Data Lake — Cadena de Cines
Obligatorio Big Data — UCU

Zonas: Landing → Raw → Curated
Stack: PySpark + HDFS + pandas + pyarrow

Diseñado para correr desde JupyterLab del docker-compose del curso.
Puede ejecutarse celda a celda en un notebook .ipynb o como script.

Servicios del compose:
  - namenode:       hdfs://namenode:9000  (WebHDFS en namenode:9870)
  - spark-master:   spark://spark-master:7077
  - jupyterlab:     localhost:8888  ← acá se corre este script

Instrucciones:
  1. Copiá el CSV de Kaggle dentro de la carpeta ./notebooks/ del proyecto
     (está montada en /home/jovyan/work dentro del contenedor)
  2. Abrí JupyterLab en localhost:8888
  3. Abrí una terminal dentro de JupyterLab y corré:
         python work/pipeline_datalake.py
     O bien pegá el contenido en un notebook y ejecutá celda a celda.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType
from hdfs import InsecureClient
from datetime import datetime
import pandas as pd
import numpy as np
import json
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIGURACIÓN — ajustada al docker-compose
# ─────────────────────────────────────────────

# Dentro del contenedor jupyterlab, namenode se resuelve por nombre de servicio
HDFS_NAMENODE  = "hdfs://namenode:9000"
WEBHDFS_URL    = "http://namenode:9870"       # WebHDFS accesible dentro de la red bigdata-net
HDFS_USER      = "root"
SPARK_MASTER   = "spark://spark-master:7077"

# El CSV debe estar en /home/jovyan/work/ (montado desde ./notebooks/)
CSV_PATH       = "/home/jovyan/work/movies.csv"

# Rutas en HDFS
LANDING_PATH   = "/datalake/landing"
RAW_PATH       = "/datalake/raw"
CURATED_PATH   = "/datalake/curated"

# ─────────────────────────────────────────────
# SPARK SESSION
# ─────────────────────────────────────────────

def get_spark():
    spark = (SparkSession.builder
             .appName("DataLake_CadenaCines")
             .master(SPARK_MASTER)
             .config("spark.hadoop.fs.defaultFS", HDFS_NAMENODE)
             # Evita warnings de metastore sin Hive configurado
             .config("spark.sql.warehouse.dir", f"{HDFS_NAMENODE}/user/hive/warehouse")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    log.info(f"Spark session iniciada — master: {SPARK_MASTER}")
    return spark

# ─────────────────────────────────────────────
# CLIENTE HDFS (WebHDFS — para subir archivos)
# ─────────────────────────────────────────────

def get_hdfs_client():
    return InsecureClient(WEBHDFS_URL, user=HDFS_USER)

def init_hdfs_dirs(client):
    """Crea las zonas del Data Lake si no existen."""
    dirs = [
        LANDING_PATH,
        RAW_PATH,
        CURATED_PATH,
        f"{RAW_PATH}/movies",
        f"{CURATED_PATH}/kpi_roi",
        f"{CURATED_PATH}/kpi_arranque",
        f"{CURATED_PATH}/kpi_directores",
        f"{CURATED_PATH}/kpi_temporada",
    ]
    for path in dirs:
        client.makedirs(path)
        log.info(f"Directorio HDFS listo: {path}")

# ─────────────────────────────────────────────
# ZONA LANDING — ingesta cruda sin modificar
# ─────────────────────────────────────────────

def landing_kaggle(client):
    """
    Copia el CSV de Kaggle a HDFS/landing tal cual.
    Genera un JSON de metadatos de ingesta.
    """
    log.info("=== LANDING: Kaggle CSV ===")
    filename  = os.path.basename(CSV_PATH)
    hdfs_dest = f"{LANDING_PATH}/{filename}"

    with open(CSV_PATH, "rb") as f:
        client.write(hdfs_dest, f, overwrite=True)
    log.info(f"CSV subido → {hdfs_dest}")

    meta = {
        "fuente":        "kaggle_movies",
        "archivo":       filename,
        "hdfs_path":     hdfs_dest,
        "fecha_ingesta": datetime.utcnow().isoformat(),
        "filas_aprox":   sum(1 for _ in open(CSV_PATH)) - 1,
        "formato":       "csv",
        "descripcion":   "Dataset sintetico de peliculas con info financiera y ratings"
    }
    _write_meta(client, meta, f"{LANDING_PATH}/meta_kaggle.json")
    return meta


def landing_imdb_csv(client, imdb_csv_path="/home/jovyan/work/imdb_movies.csv"):
    """
    Sube el CSV exportado de IMDb (del curso) a HDFS/landing.
    Si no existe el archivo, loguea un warning y continua.
    (No hay MongoDB en este docker-compose)
    """
    log.info("=== LANDING: IMDb CSV ===")
    if not os.path.exists(imdb_csv_path):
        log.warning(f"IMDb CSV no encontrado en {imdb_csv_path} — se omite esta fuente.")
        log.warning("Para incluirlo: exporta la colección MongoDB del curso a CSV y copiala en ./notebooks/")
        return None

    hdfs_dest = f"{LANDING_PATH}/imdb_movies.csv"
    with open(imdb_csv_path, "rb") as f:
        client.write(hdfs_dest, f, overwrite=True)
    log.info(f"IMDb CSV subido → {hdfs_dest}")

    meta = {
        "fuente":        "imdb_csv",
        "hdfs_path":     hdfs_dest,
        "fecha_ingesta": datetime.utcnow().isoformat(),
        "formato":       "csv",
        "descripcion":   "Dataset IMDb trabajado en el curso (exportado de MongoDB)"
    }
    _write_meta(client, meta, f"{LANDING_PATH}/meta_imdb.json")
    return hdfs_dest

# ─────────────────────────────────────────────
# ZONA RAW — limpieza con PySpark
# ─────────────────────────────────────────────

def raw_transform_kaggle(spark):
    """
    Lee el CSV desde HDFS/landing con PySpark.
    Limpia, normaliza tipos y guarda como Parquet en raw.
    """
    log.info("=== RAW: transformación Kaggle (PySpark) ===")

    csv_hdfs = f"{HDFS_NAMENODE}{LANDING_PATH}/{os.path.basename(CSV_PATH)}"
    df = spark.read.csv(csv_hdfs, header=True, inferSchema=True)
    log.info(f"Filas originales: {df.count()}")

    # Normalizar nombres de columnas a snake_case
    df = df.toDF(*[c.strip().lower().replace(" ", "_") for c in df.columns])

    # Eliminar duplicados
    df = df.dropDuplicates()

    # Eliminar filas sin título o género
    df = df.dropna(subset=["title", "genre"])

    # Columnas financieras a Double, nulos → 0
    financial_cols = [
        "production_budget",
        "us_box_office",
        "global_box_office",
        "opening_day_sales",
        "first_week_us_sales"
    ]
    for col in financial_cols:
        if col in df.columns:
            df = (df.withColumn(col, F.col(col).cast(DoubleType()))
                    .fillna({col: 0.0}))

    # Ratings a Double
    for col in ["imdb_rating", "rotten_tomatoes_score"]:
        if col in df.columns:
            df = df.withColumn(col, F.col(col).cast(DoubleType()))

    # Año limpio + columna decade
    if "release_year" in df.columns:
        df = (df.withColumn("release_year", F.col("release_year").cast(IntegerType()))
                .dropna(subset=["release_year"])
                .withColumn("release_decade",
                            F.concat((F.col("release_year") / 10).cast(IntegerType()) * 10,
                                     F.lit("s"))))

    log.info(f"Filas después de limpieza: {df.count()}")

    # Guardar como Parquet en HDFS/raw
    out_path = f"{HDFS_NAMENODE}{RAW_PATH}/movies/kaggle"
    df.write.mode("overwrite").parquet(out_path)
    log.info(f"Parquet guardado en HDFS: {out_path}")
    return df


def raw_transform_imdb(spark, imdb_hdfs_path):
    """
    Lee el CSV de IMDb desde HDFS/landing y guarda Parquet en raw.
    Solo se ejecuta si el archivo existe.
    """
    if imdb_hdfs_path is None:
        log.warning("IMDb no disponible — se omite transformación RAW de IMDb.")
        return None

    log.info("=== RAW: transformación IMDb (PySpark) ===")
    csv_hdfs = f"{HDFS_NAMENODE}{imdb_hdfs_path}"
    df = spark.read.csv(csv_hdfs, header=True, inferSchema=True)
    df = df.toDF(*[c.strip().lower().replace(" ", "_") for c in df.columns])
    df = df.dropDuplicates().dropna(subset=["title"])

    if "rating" in df.columns:
        df = df.withColumn("rating", F.col("rating").cast(DoubleType()))
    if "year" in df.columns:
        df = df.withColumn("year", F.col("year").cast(IntegerType()))

    out_path = f"{HDFS_NAMENODE}{RAW_PATH}/movies/imdb"
    df.write.mode("overwrite").parquet(out_path)
    log.info(f"IMDb Parquet guardado: {out_path}")
    return df

# ─────────────────────────────────────────────
# ZONA CURATED — KPIs con PySpark
# ─────────────────────────────────────────────

def curated_kpi_roi(spark, df):
    """
    KPI 1: Revenue por película + ROI por género
    ROI = (global_box_office - production_budget) / production_budget
    """
    log.info("=== CURATED: KPI ROI por género ===")

    df = df.filter(F.col("production_budget") > 0)
    df = df.withColumn("roi",
            (F.col("global_box_office") - F.col("production_budget"))
            / F.col("production_budget"))

    # Revenue por película
    revenue_peli = df.select(
        "title", "release_year", "genre",
        "global_box_office", "us_box_office",
        "production_budget", "roi"
    ).orderBy(F.col("global_box_office").desc())

    # ROI promedio por género
    roi_genero = (df.groupBy("genre")
                    .agg(
                        F.mean("roi").alias("roi_promedio"),
                        F.expr("percentile(roi, 0.5)").alias("roi_mediana"),
                        F.count("title").alias("peliculas"),
                        F.sum("global_box_office").alias("revenue_total")
                    )
                    .orderBy(F.col("roi_promedio").desc()))

    revenue_peli.write.mode("overwrite").parquet(
        f"{HDFS_NAMENODE}{CURATED_PATH}/kpi_roi/revenue_por_pelicula")
    roi_genero.write.mode("overwrite").parquet(
        f"{HDFS_NAMENODE}{CURATED_PATH}/kpi_roi/roi_por_genero")

    log.info("KPI ROI guardado.")
    roi_genero.show(5)
    return roi_genero


def curated_kpi_temporada(spark, df):
    """
    KPI 2: Películas con mayores ventas por temporada
    Temporadas adaptadas al hemisferio sur (Uruguay).
    """
    log.info("=== CURATED: KPI ventas por temporada ===")

    if "release_date" in df.columns:
        df = df.withColumn("mes", F.month(F.to_date("release_date")))
    else:
        # Sin fecha exacta: columna nula, se omite agrupación por mes
        df = df.withColumn("mes", F.lit(None).cast(IntegerType()))

    # Temporadas hemisferio sur
    df = df.withColumn("temporada",
        F.when(F.col("mes").isin(12, 1, 2),  "verano_uy")
         .when(F.col("mes").isin(3, 4, 5),   "otono")
         .when(F.col("mes").isin(6, 7, 8),   "invierno")
         .when(F.col("mes").isin(9, 10, 11), "primavera")
         .otherwise("sin_fecha"))

    ventas_temporada = (df.groupBy("temporada")
                          .agg(
                              F.sum("global_box_office").alias("revenue_total"),
                              F.mean("global_box_office").alias("revenue_promedio"),
                              F.count("title").alias("peliculas")
                          )
                          .orderBy(F.col("revenue_total").desc()))

    top_peliculas = (df.select("title", "genre", "release_year", "global_box_office")
                       .orderBy(F.col("global_box_office").desc())
                       .limit(100))

    ventas_temporada.write.mode("overwrite").parquet(
        f"{HDFS_NAMENODE}{CURATED_PATH}/kpi_temporada/ventas_por_temporada")
    top_peliculas.write.mode("overwrite").parquet(
        f"{HDFS_NAMENODE}{CURATED_PATH}/kpi_temporada/top_peliculas")

    log.info("KPI temporada guardado.")
    ventas_temporada.show()
    return top_peliculas


def curated_kpi_directores(spark, df):
    """
    KPI 3: Ranking de directores — taquilla promedio ponderada
    Score = revenue_promedio * ln(1 + cant_peliculas)
    Penaliza directores con una sola película.
    """
    log.info("=== CURATED: KPI directores ===")

    df_dir = df.filter(F.col("director").isNotNull() & (F.col("global_box_office") > 0))

    aggs = [
        F.count("title").alias("peliculas"),
        F.mean("global_box_office").alias("revenue_promedio"),
        F.sum("global_box_office").alias("revenue_total"),
    ]
    if "roi" in df.columns:
        aggs.append(F.mean("roi").alias("roi_promedio"))
    if "imdb_rating" in df.columns:
        aggs.append(F.mean("imdb_rating").alias("rating_promedio"))

    directores = df_dir.groupBy("director").agg(*aggs)

    # Score ponderado
    directores = directores.withColumn(
        "score_ponderado",
        F.col("revenue_promedio") * F.log1p(F.col("peliculas"))
    ).orderBy(F.col("score_ponderado").desc())

    directores.write.mode("overwrite").parquet(
        f"{HDFS_NAMENODE}{CURATED_PATH}/kpi_directores/ranking_directores")

    log.info("KPI directores guardado.")
    directores.show(5)
    return directores


def curated_kpi_arranque(spark, df):
    """
    KPI 4: Velocidad de arranque
    ratio = opening_day_sales / us_box_office
      >= 0.15  → explota_y_cae     (blockbuster de fin de semana)
      >= 0.08  → arranque_normal
      <  0.08  → traccion_sostenida (candidato a mantener en cartelera)
    """
    log.info("=== CURATED: KPI velocidad de arranque ===")

    df_arr = df.filter(
        (F.col("opening_day_sales") > 0) & (F.col("us_box_office") > 0)
    ).withColumn(
        "ratio_arranque",
        F.col("opening_day_sales") / F.col("us_box_office")
    ).withColumn(
        "tipo_arranque",
        F.when(F.col("ratio_arranque") >= 0.15, "explota_y_cae")
         .when(F.col("ratio_arranque") >= 0.08, "arranque_normal")
         .otherwise("traccion_sostenida")
    )

    arranque = df_arr.select(
        "title", "genre", "release_year",
        "opening_day_sales", "us_box_office",
        "ratio_arranque", "tipo_arranque"
    ).orderBy(F.col("ratio_arranque").desc())

    arranque_genero = (df_arr.groupBy("genre", "tipo_arranque")
                             .agg(F.count("title").alias("peliculas"))
                             .orderBy("genre", "tipo_arranque"))

    arranque.write.mode("overwrite").parquet(
        f"{HDFS_NAMENODE}{CURATED_PATH}/kpi_arranque/arranque_por_pelicula")
    arranque_genero.write.mode("overwrite").parquet(
        f"{HDFS_NAMENODE}{CURATED_PATH}/kpi_arranque/arranque_por_genero")

    log.info("KPI arranque guardado.")
    arranque_genero.show()
    return arranque

# ─────────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────────

def _write_meta(client, meta, hdfs_path):
    client.write(hdfs_path, json.dumps(meta, indent=2).encode(), overwrite=True)
    log.info(f"Metadatos guardados: {hdfs_path}")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run_pipeline():
    log.info("========================================")
    log.info("  PIPELINE DATA LAKE — CADENA DE CINES  ")
    log.info("========================================")

    client = get_hdfs_client()
    spark  = get_spark()

    init_hdfs_dirs(client)

    # ── LANDING ──────────────────────────────
    landing_kaggle(client)
    imdb_path = landing_imdb_csv(client)   # opcional, no falla si no está

    # ── RAW ──────────────────────────────────
    df_kaggle = raw_transform_kaggle(spark)
    raw_transform_imdb(spark, imdb_path)   # no-op si IMDb no está disponible

    # ── CURATED ──────────────────────────────
    curated_kpi_roi(spark, df_kaggle)
    curated_kpi_temporada(spark, df_kaggle)
    curated_kpi_directores(spark, df_kaggle)
    curated_kpi_arranque(spark, df_kaggle)

    spark.stop()
    log.info("========================================")
    log.info("  PIPELINE COMPLETADO EXITOSAMENTE      ")
    log.info("========================================")


if __name__ == "__main__":
    run_pipeline()
