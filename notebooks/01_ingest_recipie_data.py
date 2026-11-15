# Databricks notebook source
# MAGIC %pip install kagglehub --quiet

# COMMAND ----------

import os
import shutil

import kagglehub
from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType

from recipe_curator.config import get_env, load_config

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
        StructField("row_id", IntegerType(), True),  # unnamed index column (_c0) in the CSV
        StructField("recipe_name", StringType(), True),
        StructField("prep_time", StringType(), True),  # e.g. "30 mins"
        StructField("cook_time", StringType(), True),
        StructField("total_time", StringType(), True),
        StructField("servings", IntegerType(), True),
        StructField("yield", StringType(), True),
        StructField("ingredients", StringType(), True),  # comma-separated ingredient text
        StructField("directions", StringType(), True),
        StructField("rating", DoubleType(), True),  # e.g. 4.8
        StructField("url", StringType(), True),
        StructField("cuisine_path", StringType(), True),  # e.g. "/Desserts/Pies/Apple Pie Recipes/"
        StructField("nutrition", StringType(), True),
        StructField("timing", StringType(), True),
        StructField("img_src", StringType(), True),
    ]
)

input_df = (
    spark.read.option("header", "true")
    .option("multiLine", "true")
    .schema(recipes_schema)
    .csv(f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_NAME}/better_recipes/recipes.csv")
)


(
    input_df.withColumn("ingestion_timestamp", F.current_timestamp())
    .write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{CATALOG}.{SCHEMA}.recipes_input")
)

# COMMAND ----------
# Add table and column descriptions (shown in Unity Catalog / Catalog Explorer)
TABLE_NAME = f"{CATALOG}.{SCHEMA}.recipes_input"

table_description = "Raw recipes ingested from the Kaggle dataset 'Better Recipes for a Better Life'."

column_descriptions = {
    "row_id": "Recipie ID",
    "recipe_name": "Name of the recipe.",
    "prep_time": "Preparation time as free text, e.g. '30 mins'.",
    "cook_time": "Cooking time as free text, e.g. '1 hrs 10 mins'.",
    "total_time": "Total time (prep + cook + other) as free text.",
    "servings": "Number of servings the recipe makes.",
    "yield": "Recipe yield as free text, e.g. '1 9-inch pie'.",
    "ingredients": "Comma-separated list of ingredients with quantities.",
    "directions": "Step-by-step cooking instructions.",
    "rating": "Average user rating (0-5).",
    "url": "Source URL of the recipe.",
    "cuisine_path": "Category path of the recipe, e.g. '/Desserts/Pies/Apple Pie Recipes/'.",
    "nutrition": "Nutrition facts per serving as free text.",
    "timing": "Combined timing information as free text.",
    "img_src": "URL of the recipe image.",
    "ingestion_timestamp": "Timestamp when the row was ingested into this table.",
}


def _escape(text: str) -> str:
    """Escape single quotes for use in a SQL string literal."""
    return text.replace("'", "\\'")


spark.sql(f"COMMENT ON TABLE {TABLE_NAME} IS '{_escape(table_description)}'")

for column, description in column_descriptions.items():
    spark.sql(f"ALTER TABLE {TABLE_NAME} ALTER COLUMN `{column}` COMMENT '{_escape(description)}'")

logger.info(f"Added descriptions to {TABLE_NAME} ({len(column_descriptions)} columns)")
