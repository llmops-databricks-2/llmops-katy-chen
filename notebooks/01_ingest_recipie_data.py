# Databricks notebook source
# MAGIC %pip install kagglehub --quiet

# COMMAND ----------

from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from pyspark.sql import functions as F
from recipe_curator.config import get_env, load_config 
import kagglehub
import shutil
import os

# COMMAND ----------
# Create Spark session
spark = SparkSession.builder.getOrCreate()

# Load config
env = get_env()
cfg = load_config("../project_config.yml", env)

CATALOG = cfg.catalog
SCHEMA = cfg.schema
VOLUME_NAME = "input_data"

# Create the volume if it doesn't exist
spark.sql(f"""CREATE VOLUME IF NOT EXISTS `{CATALOG}`.`{SCHEMA}`.`{VOLUME_NAME}`""")

logger.info(f"Volume {CATALOG}.{SCHEMA}.{VOLUME_NAME} is ready")

# Convert catalog.schema.volume -> /Volumes/catalog/schema/volume
target_dir = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_NAME}/better_recipes"

# Download latest version (lands in the local kagglehub cache)
path = kagglehub.dataset_download("thedevastator/better-recipes-for-a-better-life")
print("Path to dataset files:", path)

# Copy the downloaded files into the Unity Catalog volume
shutil.copytree(path, target_dir, dirs_exist_ok=True)

for f in os.listdir(target_dir):
    print(f, os.path.getsize(os.path.join(target_dir, f)), "bytes")


# Schema for recipes.csv (columns in file order; the unnamed first column is the row index)
recipes_schema = StructType(
    [
        StructField("row_id", IntegerType(), True),        # unnamed index column (_c0) in the CSV
        StructField("recipe_name", StringType(), True),
        StructField("prep_time", StringType(), True),      # e.g. "30 mins"
        StructField("cook_time", StringType(), True),
        StructField("total_time", StringType(), True),
        StructField("servings", IntegerType(), True),
        StructField("yield", StringType(), True),
        StructField("ingredients", StringType(), True),    # comma-separated ingredient text
        StructField("directions", StringType(), True),
        StructField("rating", DoubleType(), True),         # e.g. 4.8
        StructField("url", StringType(), True),
        StructField("cuisine_path", StringType(), True),   # e.g. "/Desserts/Pies/Apple Pie Recipes/"
        StructField("nutrition", StringType(), True),
        StructField("timing", StringType(), True),
        StructField("img_src", StringType(), True),
    ]
)

input_df = (
    spark.read
    .option("header", "true")
    .option("multiLine", "true")
    .schema(recipes_schema)
    .csv(f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_NAME}/better_recipes/recipes.csv")
)



(
    input_df
    .withColumn("ingestion_timestamp", F.current_timestamp())
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{CATALOG}.{SCHEMA}.recipes_input")
)
