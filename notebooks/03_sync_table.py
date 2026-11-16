# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Sync recipe chunks to Lakebase
# MAGIC
# MAGIC This notebook copies the embedded recipe chunks from Unity Catalog into a
# MAGIC Lakebase (PostgreSQL) **Synced Table**, so we can search them with pgvector.
# MAGIC
# MAGIC ```text
# MAGIC recipes_input                     (01_ingest_recipie_data)
# MAGIC      │
# MAGIC recipes_chunks                    (02_chunk: text + embedding, UC Delta)
# MAGIC      │  Synced Table, TRIGGERED, embedding -> vector(1024)
# MAGIC      ▼
# MAGIC "<schema>".recipes_chunks_synced  (Lakebase Postgres, read-only)   ← this notebook
# MAGIC      │  GIN full-text index
# MAGIC      ▼
# MAGIC LakebasePgVectorManager.search / hybrid_search
# MAGIC ```
# MAGIC
# MAGIC **Prerequisites**
# MAGIC - `02_chunk` has written `recipes_chunks` with a `chunk_id` and an `embedding` column.
# MAGIC - Your Lakebase project (`lakebase_instance` in `project_config.yml`) exists.
# MAGIC
# MAGIC **Rerunnable:** the first run creates the Synced Table; later runs trigger a sync
# MAGIC that picks up changes to `recipes_chunks` through its Change Data Feed.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Setup
# MAGIC
# MAGIC Load the environment config and create the Spark session and workspace client.
# MAGIC `usage_policy_id` sits at the top level of `project_config.yml` (outside the
# MAGIC `dev`/`acc`/`prd` blocks), so it is not part of `ProjectConfig` and we read it
# MAGIC from the YAML directly.

# COMMAND ----------

import time
from pathlib import Path

import yaml
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.postgres import (
    NewPipelineSpec,
    SyncedTable,
    SyncedTableState,
    SyncedTableSyncedTableSpec,
    SyncedTableSyncedTableSpecPgSpecificType,
    SyncedTableSyncedTableSpecSyncedTableSchedulingPolicy,
    SyncedTableSyncedTableSpecTypeOverride,
)
from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from recipe_curator.config import get_env, load_config
from recipe_curator.vector_search import LakebasePgVectorManager

# COMMAND ----------

CONFIG_PATH = "../project_config.yml"
EMBEDDING_DIM = 1024  # output size of databricks-gte-large-en
POSTGRES_DATABASE = "databricks_postgres"
SYNC_TIMEOUT_MINUTES = 60

spark = SparkSession.builder.getOrCreate()
w = WorkspaceClient()

env = get_env()
cfg = load_config(CONFIG_PATH, env)
usage_policy_id = yaml.safe_load(Path(CONFIG_PATH).read_text()).get("usage_policy_id")

pgvector = LakebasePgVectorManager(config=cfg, workspace=w)

source_table = f"{cfg.catalog}.{cfg.schema}.recipes_chunks"
synced_table = f"{cfg.catalog}.{cfg.schema}.{pgvector.synced_table_name}"
synced_table_name = f"synced_tables/{synced_table}"  # resource name used by the Postgres API

logger.info(f"Environment:        {env}")
logger.info(f"Source table:       {source_table}")
logger.info(f"Synced table:       {synced_table}")
logger.info(f"Postgres table:     {pgvector.table_name}")
logger.info(f"Lakebase project:   {cfg.lakebase_instance}")
logger.info(f"Embedding endpoint: {cfg.embedding_endpoint}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Validate the Delta source
# MAGIC
# MAGIC The Synced Table has strict requirements on its source, and violations surface
# MAGIC late as vague pipeline errors, so we check them up front:
# MAGIC
# MAGIC | Check | Why |
# MAGIC |---|---|
# MAGIC | `chunk_id` exists, non-null, unique | It is the primary key the sync upserts on |
# MAGIC | `embedding` exists | It becomes the pgvector column |
# MAGIC | every embedding has 1024 values | A `vector(1024)` column rejects any other length |
# MAGIC
# MAGIC Null embeddings are allowed (search skips them) but logged as a warning.
# MAGIC
# MAGIC Finally we enable **Change Data Feed**: a TRIGGERED Synced Table reads it to
# MAGIC sync only the rows that changed since the last run.

# COMMAND ----------

source_columns = set(spark.table(source_table).columns)
missing = {"chunk_id", "embedding"} - source_columns
if missing:
    raise ValueError(f"{source_table} is missing columns {missing}. Run 02_chunk first.")

stats = (
    spark.table(source_table)
    .agg(
        F.count("*").alias("rows"),
        F.countDistinct("chunk_id").alias("distinct_ids"),
        F.sum(F.col("chunk_id").isNull().cast("int")).alias("null_ids"),
        F.sum(F.col("embedding").isNull().cast("int")).alias("null_embeddings"),
        F.sum((F.col("embedding").isNotNull() & (F.size("embedding") != EMBEDDING_DIM)).cast("int")).alias(
            "wrong_dim_embeddings"
        ),
    )
    .first()
)
logger.info(f"Source stats: {stats.asDict()}")

if stats.null_ids or stats.distinct_ids != stats.rows:
    raise ValueError(f"chunk_id is not a valid primary key: {stats.asDict()}")
if stats.wrong_dim_embeddings:
    raise ValueError(f"{stats.wrong_dim_embeddings} embeddings do not have {EMBEDDING_DIM} values")
if stats.null_embeddings:
    logger.warning(f"{stats.null_embeddings} chunks have no embedding; vector search skips them")

spark.sql(f"ALTER TABLE {source_table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
logger.info(f"Validated source table and enabled Change Data Feed: {source_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Enable the pgvector extension
# MAGIC
# MAGIC Creating a Synced Table does **not** install Postgres extensions, and the
# MAGIC `vector` type override in step 3 fails without it. `CREATE EXTENSION IF NOT
# MAGIC EXISTS` makes this safe to rerun.

# COMMAND ----------

pgvector.ensure_extension()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Create the Synced Table, or trigger a sync
# MAGIC
# MAGIC **Branch:** we use the same branch `LakebasePgVectorManager` connects to
# MAGIC (the project's first branch), so we always query the branch we synced to.
# MAGIC
# MAGIC **First run** creates the Synced Table with:
# MAGIC
# MAGIC | Setting | Value | Meaning |
# MAGIC |---|---|---|
# MAGIC | `primary_key_columns` | `chunk_id` | Upsert key in Postgres |
# MAGIC | `scheduling_policy` | `TRIGGERED` | Syncs only when we ask (this notebook / a job), not continuously |
# MAGIC | `type_overrides` | `embedding` → `vector(1024)` | Stores `ARRAY<FLOAT>` as a pgvector column instead of `real[]` |
# MAGIC | `new_pipeline_spec` | your catalog/schema + usage policy | Where the sync pipeline keeps checkpoints, and who pays |
# MAGIC | `create_database_objects_if_missing` | `True` | Creates the Postgres schema `"<schema>"` if needed |
# MAGIC
# MAGIC **Later runs:** a Synced Table runs on a Lakeflow pipeline, so starting a pipeline
# MAGIC update is the same as **Sync now** in the UI.

# COMMAND ----------

branch = next(iter(w.postgres.list_branches(parent=f"projects/{cfg.lakebase_instance}"))).name
logger.info(f"Lakebase branch: {branch}")

try:
    existing = w.postgres.get_synced_table(name=synced_table_name)
except NotFound:
    existing = None

if existing is None:
    w.postgres.create_synced_table(
        synced_table_id=synced_table,
        synced_table=SyncedTable(
            spec=SyncedTableSyncedTableSpec(
                source_table_full_name=source_table,
                branch=branch,
                postgres_database=POSTGRES_DATABASE,
                primary_key_columns=["chunk_id"],
                scheduling_policy=SyncedTableSyncedTableSpecSyncedTableSchedulingPolicy.TRIGGERED,
                create_database_objects_if_missing=True,
                new_pipeline_spec=NewPipelineSpec(
                    budget_policy_id=usage_policy_id,
                    storage_catalog=cfg.catalog,
                    storage_schema=cfg.schema,
                ),
                type_overrides=[
                    SyncedTableSyncedTableSpecTypeOverride(
                        column_name="embedding",
                        pg_type=SyncedTableSyncedTableSpecPgSpecificType.PG_SPECIFIC_TYPE_VECTOR,
                        size=EMBEDDING_DIM,
                    )
                ],
            )
        ),
    ).wait()
    logger.info(f"Created Synced Table {synced_table}; initial snapshot is running")
else:
    pipeline_id = existing.status.pipeline_id
    update_id = w.pipelines.start_update(pipeline_id=pipeline_id).update_id
    logger.info(f"Sync started: pipeline {pipeline_id}, update {update_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Wait for the sync to finish
# MAGIC
# MAGIC Both `create_synced_table(...).wait()` and `start_update` return before the
# MAGIC data has landed, so querying right away can return old or no rows. We poll the
# MAGIC Synced Table state every 30 s until it is online with no pending update, and
# MAGIC fail fast if the pipeline fails.

# COMMAND ----------

DONE_STATES = {
    SyncedTableState.SYNCED_TABLE_ONLINE,
    SyncedTableState.SYNCED_TABLE_ONLINE_NO_PENDING_UPDATE,
}
FAILED_STATES = {
    SyncedTableState.SYNCED_TABLE_OFFLINE_FAILED,
    SyncedTableState.SYNCED_TABLE_ONLINE_PIPELINE_FAILED,
}

# Give a just-started update a moment to move the state out of "no pending update"
time.sleep(30)
deadline = time.time() + SYNC_TIMEOUT_MINUTES * 60

while True:
    status = w.postgres.get_synced_table(name=synced_table_name).status
    state = status.detailed_state if status else None
    logger.info(f"Synced Table state: {state} | {status.message if status else ''}")

    if state in DONE_STATES:
        break
    if state in FAILED_STATES:
        raise RuntimeError(f"Synced Table sync failed: {status.message}")
    if time.time() > deadline:
        raise TimeoutError(f"Synced Table still {state} after {SYNC_TIMEOUT_MINUTES} minutes")
    time.sleep(30)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Create indexes
# MAGIC
# MAGIC Synced Tables are read-only, but `CREATE INDEX` is allowed. `ensure_indexes`
# MAGIC creates a GIN full-text index on `recipe_name || text`, which `hybrid_search`
# MAGIC uses. The vector (HNSW) index is disabled in `vector_search.py`, so vector
# MAGIC search scans the table: fine at this size.
# MAGIC
# MAGIC Indexes are idempotent (`IF NOT EXISTS`); rerun this if the Synced Table was
# MAGIC ever deleted and recreated.

# COMMAND ----------

pgvector.ensure_indexes()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Verify the synced data
# MAGIC
# MAGIC - Row counts in Delta and Postgres should match.
# MAGIC - `pg_typeof(embedding)` should be `vector`, confirming the type override worked.

# COMMAND ----------

with pgvector.connection() as conn:
    synced_rows = conn.execute(f"SELECT count(*) FROM {pgvector.table_name}").fetchone()[0]
    sample = conn.execute(
        f"""
        SELECT chunk_id, row_id, recipe_name, rating,
               pg_typeof(embedding) AS embedding_type, left(text, 80) AS text
        FROM {pgvector.table_name}
        LIMIT 5
        """
    ).fetchall()

logger.info(f"Rows: Delta {stats.rows} | Lakebase {synced_rows}")
if synced_rows != stats.rows:
    logger.warning("Row counts differ: check the sync pipeline's event log")

for row in sample:
    logger.info(row)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Smoke test: search
# MAGIC
# MAGIC - `search`: cosine similarity between the query embedding and `embedding`.
# MAGIC - `hybrid_search`: 0.7 × vector score + 0.3 × full-text rank.

# COMMAND ----------

query = "quick vegetarian pasta with garlic"

logger.info(f"Vector search: {query!r}")
for row in pgvector.search(query, num_results=5):
    logger.info(f"  {row['score']:.4f} | {row['recipe_name']} | {row['text'][:80]}")

logger.info(f"Hybrid search: {query!r}")
for row in pgvector.hybrid_search(query, num_results=5):
    logger.info(f"  {row['score']:.4f} | {row['recipe_name']} | {row['text'][:80]}")
