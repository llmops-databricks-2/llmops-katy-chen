# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Chunk and embed recipes
# MAGIC
# MAGIC This notebook turns each recipe into small, searchable text **chunks** and
# MAGIC embeds every chunk, ready to be synced to Lakebase in `03_sync_table`.
# MAGIC
# MAGIC ```text
# MAGIC recipes_input        (01_ingest_recipie_data: one row per recipe)
# MAGIC      │  1. combine ingredients / directions / nutrition / timing into one labeled text
# MAGIC      │  2. split it back into sections, and sections into paragraphs (chunk_recipe)
# MAGIC      │  3. give each chunk a stable chunk_id
# MAGIC      │  4. embed each chunk with ai_query
# MAGIC      ▼
# MAGIC recipes_chunks       (one row per chunk: metadata + text + embedding)   ← this notebook
# MAGIC      ▼
# MAGIC 03_sync_table        (Lakebase Synced Table for pgvector search)
# MAGIC ```
# MAGIC
# MAGIC **Chunking strategy: by section, then by paragraph**
# MAGIC
# MAGIC | Section | Typical chunks per recipe | Why |
# MAGIC |---|---|---|
# MAGIC | `ingredients` | 1 | Stored as one comma-separated line |
# MAGIC | `directions` | 1 per step (up to ~11) | Steps are separated by line breaks |
# MAGIC | `nutrition` | 1 | One line of nutrients |
# MAGIC | `timing` | 1 | Prep / cook / total time, servings, yield |
# MAGIC
# MAGIC In the dev data this gives about 7,300 chunks for about 1,090 recipes (~6.7 per recipe).
# MAGIC Chunks average 90–225 characters per section and are at most 745: well within
# MAGIC the embedding model's input limit.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Setup
# MAGIC
# MAGIC `chunk_recipe` lives in `recipe_curator.data_processor` (the project wheel), so
# MAGIC the splitting logic can be unit-tested outside Databricks.

# COMMAND ----------

import re
from functools import partial

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from recipe_curator.config import get_env, load_config
from recipe_curator.data_processor import chunk_recipe

# COMMAND ----------

env = get_env()
cfg = load_config("../project_config.yml", env)

CATALOG = cfg.catalog
SCHEMA = cfg.schema
EMBEDDING_ENDPOINT = cfg.embedding_endpoint  # databricks-gte-large-en: 1024-dim vectors

SOURCE_TABLE = f"{CATALOG}.{SCHEMA}.recipes_input"
TARGET_TABLE = f"{CATALOG}.{SCHEMA}.recipes_chunks"

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Combine the text columns into one labeled field
# MAGIC
# MAGIC The four text columns are joined into a single `recipe_content` string, each
# MAGIC part prefixed with a header and separated by a blank line:
# MAGIC
# MAGIC ```text
# MAGIC Ingredients: 3 tablespoons butter, 2 pounds Granny Smith apples, ...
# MAGIC
# MAGIC Directions: Heat butter in a large skillet ...
# MAGIC Adjust oven rack to low position ...
# MAGIC
# MAGIC Nutrition: Total Fat 18g 23%, ...
# MAGIC
# MAGIC Timing: Servings: 8, Yield: 6 to 8 - servings
# MAGIC ```
# MAGIC
# MAGIC This looks like a roundabout way to keep the sections apart, but it is on
# MAGIC purpose: it lets `chunk_recipe` work on one document, the way it would on
# MAGIC unstructured text, so we can test the header-based chunking.
# MAGIC
# MAGIC **Null handling:** `concat` returns null when its column is null, and
# MAGIC `concat_ws` skips nulls, so a missing section simply has no header.

# COMMAND ----------

recipes_df = spark.read.table(SOURCE_TABLE)

# Header label -> source column
text_cols = {
    "Ingredients": "ingredients",
    "Directions": "directions",
    "Nutrition": "nutrition",
    "Timing": "timing",
}

recipes_df = (
    recipes_df.withColumn(
        "recipe_content",
        F.concat_ws(
            "\n\n",
            *[F.concat(F.lit(f"{label}: "), F.col(c)) for label, c in text_cols.items()],
        ),
    ).drop(*text_cols.values())  # the content now lives in recipe_content
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Define the chunker
# MAGIC
# MAGIC **`SECTION_HEADER`** finds the section headers. It only matches at the
# MAGIC start of a line (`MULTILINE` + `^`), so a word like "directions" inside a
# MAGIC sentence is not treated as a header. Matching is case-insensitive.
# MAGIC
# MAGIC **`chunk_recipe(SECTION_HEADER, content)`** (see `data_processor.py`):
# MAGIC 1. Normalizes line endings (`\r\n`, `\r` → `\n`).
# MAGIC 2. Cuts the text at each header; the text up to the next header is that section's body.
# MAGIC 3. Splits each body on line breaks into paragraphs and collapses whitespace.
# MAGIC 4. Returns `(section, paragraph_index, text)` for every non-empty paragraph.
# MAGIC
# MAGIC Empty content returns no chunks. Content that does not start with a header
# MAGIC raises a `ValueError`, which fails the job. That can't happen with the
# MAGIC content built in step 1, so it signals a bug, not bad data.
# MAGIC
# MAGIC The UDF returns an **array** of chunks per recipe; step 3 explodes it into
# MAGIC one row per chunk. `partial` binds the regex as the first argument so Spark
# MAGIC only has to pass the content column.

# COMMAND ----------

SECTION_HEADER = re.compile(
    r"^[ \t]*(Ingredients|Directions|Nutrition|Timing)[ \t]*:[ \t]*",
    flags=re.IGNORECASE | re.MULTILINE,
)

chunk_schema = T.ArrayType(
    T.StructType(
        [
            T.StructField("section", T.StringType(), nullable=False),  # lower-case header
            T.StructField("paragraph_index", T.IntegerType(), nullable=False),  # 0-based within section
            T.StructField("text", T.StringType(), nullable=False),  # cleaned paragraph
        ]
    )
)

chunked_recipes_udf = F.udf(
    partial(chunk_recipe, SECTION_HEADER),
    chunk_schema,
)

# Single quotes are doubled so the endpoint name is safe inside the SQL string literal below
escaped_endpoint = EMBEDDING_ENDPOINT.replace("'", "''")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Chunk, assign IDs and embed
# MAGIC
# MAGIC **Output columns**
# MAGIC
# MAGIC | Column | Source | Purpose |
# MAGIC |---|---|---|
# MAGIC | `row_id`, `recipe_name`, `cook_time`, `prep_time`, `total_time`, `servings`, `yield`, `rating` | `recipes_input` | Recipe metadata, repeated on every chunk so search results are self-contained |
# MAGIC | `section`, `paragraph_index`, `text` | `chunk_recipe` | The chunk itself and where it came from |
# MAGIC | `chunk_id` | SHA-256 of `row_id` + `section` + `paragraph_index` + `text` | Primary key for the Lakebase Synced Table |
# MAGIC | `embedding` | `ai_query(embedding endpoint, text)` | `ARRAY<FLOAT>` with 1024 values; becomes a pgvector column in `03_sync_table` |
# MAGIC
# MAGIC **Why hash for `chunk_id`?** The same recipe and text always get the same ID
# MAGIC on every run, so the ID is stable across reruns. Because `text` is part of the
# MAGIC hash, editing a chunk produces a new ID (a delete + insert in the synced table).
# MAGIC
# MAGIC **Why `F.expr` for `ai_query`?** Not every runtime exposes `ai_query` in
# MAGIC `pyspark.sql.functions`, but the SQL function is always available. The cast
# MAGIC turns the result into the `ARRAY<FLOAT>` type the Synced Table expects.
# MAGIC
# MAGIC Spark is lazy: nothing is chunked or embedded until the write in step 4.

# COMMAND ----------

recipe_chunks_df = (
    recipes_df
    # One row per chunk
    .withColumn("chunk", F.explode(chunked_recipes_udf(F.col("recipe_content"))))
    .select(
        "row_id",
        "recipe_name",
        "cook_time",
        "prep_time",
        "total_time",
        "servings",
        "yield",
        "rating",
        F.col("chunk.section").alias("section"),
        F.col("chunk.paragraph_index").alias("paragraph_index"),
        F.col("chunk.text").alias("text"),
    )
    # Stable primary key
    .withColumn(
        "chunk_id",
        F.sha2(F.to_json(F.struct("row_id", "section", "paragraph_index", "text")), 256),
    )
    # One embedding call per chunk
    .withColumn(
        "embedding",
        F.expr(f"CAST(ai_query('{escaped_endpoint}', text) AS ARRAY<FLOAT>)"),
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Write `recipes_chunks`
# MAGIC
# MAGIC The table is fully **overwritten** on every run (`overwriteSchema` allows
# MAGIC column changes). That keeps the notebook simple, but note:
# MAGIC - Every run re-embeds **all** chunks with `ai_query`, which costs time and money.
# MAGIC - `03_sync_table` syncs incrementally through the Change Data Feed. A full
# MAGIC   overwrite marks every row as changed, and a schema change can require a
# MAGIC   full refresh of the Synced Table.
# MAGIC
# MAGIC A `MERGE INTO ... ON chunk_id` that only embeds new chunks would fix both.

# COMMAND ----------

(recipe_chunks_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(TARGET_TABLE))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Check the result
# MAGIC
# MAGIC Chunks per section and their text length. Expect four sections, no null
# MAGIC embeddings, and `chunk_id` count equal to the row count (03 requires it).

# COMMAND ----------

chunks = spark.table(TARGET_TABLE)

display(
    chunks.groupBy("section").agg(
        F.count("*").alias("chunks"),
        F.round(F.avg(F.length("text"))).alias("avg_chars"),
        F.max(F.length("text")).alias("max_chars"),
        F.max("paragraph_index").alias("max_paragraph_index"),
        F.sum(F.col("embedding").isNull().cast("int")).alias("null_embeddings"),
    )
)
display(
    chunks.agg(
        F.count("*").alias("rows"),
        F.countDistinct("chunk_id").alias("distinct_chunk_ids"),
        F.countDistinct("row_id").alias("recipes"),
    )
)
