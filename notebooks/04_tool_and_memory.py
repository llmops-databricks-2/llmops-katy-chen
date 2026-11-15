# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Tools and memory
# MAGIC
# MAGIC ```text
# MAGIC user turn ─► load history (session_id) ─► LLM + tools ─► save user + assistant message
# MAGIC                      ▲                                              │
# MAGIC                      └──────────── session_messages (JSONB) ◄───────┘
# MAGIC ```
# MAGIC
# MAGIC | Section | What it shows |
# MAGIC |---|---|
# MAGIC | 1 | Connect to the Lakebase project and create `session_messages` |
# MAGIC | 2 | Save and load messages with `LakebaseMemory` |
# MAGIC | 3 | The agent's tools: search, Unity Catalog SQL functions, Genie space |
# MAGIC | 4 | A recipe agent that uses the tools and remembers the conversation |
# MAGIC | 5 | (Optional) Service principal authentication, as used in production |
# MAGIC
# MAGIC | Tool | Kind | Where it runs | What it answers |
# MAGIC |---|---|---|---|
# MAGIC | `search_recipes` | Python function | In the agent process, queries Lakebase pgvector | "Find me a recipe for …" |
# MAGIC | `parse_duration_minutes` | Unity Catalog SQL function (scalar) | SQL warehouse | Helper: `"1 hrs 15 mins"` → `75` |
# MAGIC | `get_recipe_cooking_time` | Unity Catalog SQL function (table) | SQL warehouse, via managed MCP | "How long does … take?" |
# MAGIC | Genie `recipe-curator-space` | Genie space | SQL warehouse, via managed MCP | "How many …?", "Top 10 …", other aggregate questions |
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Setup

# COMMAND ----------

import asyncio
import json
import os
from uuid import uuid4

import nest_asyncio
from databricks.sdk import WorkspaceClient
from loguru import logger
from openai import OpenAI
from pyspark.sql import SparkSession

from recipe_curator.config import get_env, load_config
from recipe_curator.mcp import ToolInfo, create_mcp_tools
from recipe_curator.memory import LakebaseMemory
from recipe_curator.vector_search import LakebasePgVectorManager

# Databricks notebooks already run an event loop; this lets asyncio.run() work inside it
nest_asyncio.apply()

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
w = WorkspaceClient()

env = get_env()
cfg = load_config("../project_config.yml", env)

CATALOG = cfg.catalog
SCHEMA = cfg.schema
RECIPES_TABLE = f"{CATALOG}.{SCHEMA}.recipes_input"

# The Lakebase project is the same one that holds the pgvector Synced Table (03_sync_table)
LAKEBASE_PROJECT = cfg.lakebase_instance

logger.info(f"Environment:      {env}")
logger.info(f"Lakebase project: {LAKEBASE_PROJECT}")
logger.info(f"LLM endpoint:     {cfg.llm_endpoint}")
logger.info(f"Recipes table:    {RECIPES_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Connect and create the session table
# MAGIC
# MAGIC `LakebaseMemory` finds the project's default branch and its endpoint, then
# MAGIC asks the Databricks SDK for a **short-lived database credential**. No password
# MAGIC is stored anywhere. Without service principal environment variables (section 5)
# MAGIC it connects as myself, with my Databricks user name as Postgres role.
# MAGIC
# MAGIC ```text
# MAGIC Project (lb-llmops-…)
# MAGIC  └── Branch (main)
# MAGIC        ├── Endpoint (R/W compute)  ← host + credential
# MAGIC        └── Database databricks_postgres
# MAGIC              └── public.session_messages
# MAGIC ```
# MAGIC
# MAGIC The table holds one row per message. `session_id` groups messages into a
# MAGIC conversation, `message_data` is the OpenAI-style message as JSONB, and the
# MAGIC index keeps loading one session fast.

# COMMAND ----------

memory = LakebaseMemory(project_id=LAKEBASE_PROJECT, workspace=w)
memory.ensure_schema()
logger.info("session_messages table is ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Save and load messages
# MAGIC
# MAGIC `save_messages` appends to a session; `load_messages` returns the session in
# MAGIC the order it was written. A fresh `session_id` per test keeps runs separate.

# COMMAND ----------

test_session_id = f"memory-test-{uuid4()}"
test_messages = [
    {"role": "user", "content": "I'm looking for an apple dessert."},
    {"role": "assistant", "content": "How about an Apple Pie by Grandma Ople? It's rated 4.8."},
]

memory.save_messages(test_session_id, test_messages)
loaded = memory.load_messages(test_session_id)

assert loaded == test_messages, "Session message round-trip failed"
logger.info(f"Round-trip OK for {test_session_id}:")
for msg in loaded:
    logger.info(f"  {msg['role']}: {msg['content']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. The agent's tools
# MAGIC
# MAGIC ### 3.1 Retrieval tool: `search_recipes`
# MAGIC
# MAGIC Hybrid search (`0.7 × vector similarity + 0.3 × full-text rank`) over the
# MAGIC chunks synced to Lakebase in `03_sync_table`.

# COMMAND ----------

pgvector = LakebasePgVectorManager(config=cfg, workspace=w)


def search_recipes(query: str, num_results: int = 5) -> str:
    """Search recipe chunks with Lakebase pgvector hybrid search."""
    num_results = max(1, min(int(num_results), 20))
    try:
        rows = pgvector.hybrid_search(query=query, num_results=num_results)
    except Exception as e:  # noqa: BLE001 - surface any failure to the LLM
        logger.exception("search_recipes failed")
        return json.dumps({"error": f"Search failed: {e}"})
    results = [
        {
            "recipe_name": row.get("recipe_name"),
            "section": row.get("section"),
            "rating": row.get("rating"),
            "total_time": row.get("total_time"),
            "servings": row.get("servings"),
            "excerpt": (row.get("text") or "")[:300],
        }
        for row in rows
    ]
    return json.dumps(results, indent=2, default=str)


search_recipes_tool = ToolInfo(
    name="search_recipes",
    exec_fn=search_recipes,
    spec={
        "type": "function",
        "function": {
            "name": "search_recipes",
            "description": (
                "Search the recipe collection with hybrid semantic and keyword search. "
                "Use it to find recipes by dish, ingredient, technique or description."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for, e.g. 'easy apple dessert'"},
                    "num_results": {"type": "integer", "description": "Number of chunks, 1-20 (default 5)"},
                },
                "required": ["query"],
            },
        },
    },
)

logger.info(search_recipes("apple dessert with cinnamon", num_results=3))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2 Unity Catalog SQL functions: cooking time
# MAGIC
# MAGIC The time columns in `recipes_input` are free text (`10 mins`, `1 hrs 15 mins`,
# MAGIC `2 days 3 hrs`). LLMs are unreliable at adding up durations like these, so SQL
# MAGIC does it. `COMMENT`s on the functions and parameters become the tool
# MAGIC description the LLM sees, so they are written for the LLM.
# MAGIC

# COMMAND ----------

parse_fn = f"{CATALOG}.{SCHEMA}.parse_duration_minutes"

spark.sql(f"""
CREATE OR REPLACE FUNCTION {parse_fn}(
    duration STRING COMMENT 'Duration text such as "45 mins", "1 hrs 15 mins" or "2 days 3 hrs"'
)
RETURNS INT
COMMENT 'Convert a recipe duration string (days, hrs, mins) to a total number of minutes. Returns NULL if the text has no recognizable duration.'
RETURN
    CASE
        WHEN duration IS NULL OR NOT duration RLIKE '[0-9]+ *(day|hr|hour|min)' THEN NULL
        ELSE
              coalesce(try_cast(regexp_extract(duration, '([0-9]+) *days?', 1) AS INT), 0) * 1440
            + coalesce(try_cast(regexp_extract(duration, '([0-9]+) *(hrs?|hours?)', 1) AS INT), 0) * 60
            + coalesce(try_cast(regexp_extract(duration, '([0-9]+) *(mins?|minutes?)', 1) AS INT), 0)
    END
""")
logger.info(f"Created: {parse_fn}")

display(
    spark.sql(f"""
        SELECT duration, {parse_fn}(duration) AS minutes
        FROM VALUES ('10 mins'), ('1 hrs'), ('1 hrs 15 mins'), ('2 days 3 hrs'), (NULL), ('n/a')
            AS t(duration)
    """)
)
# Expected: 10, 60, 75, 3060, NULL, NULL

# COMMAND ----------

# MAGIC %md
# MAGIC **`get_recipe_cooking_time`** (table function, the agent tool) looks up recipes
# MAGIC by (part of) their name and returns prep, cook, other (resting, chilling…) and
# MAGIC total minutes.

# COMMAND ----------

cooking_time_fn = f"{CATALOG}.{SCHEMA}.get_recipe_cooking_time"

spark.sql(f"""
CREATE OR REPLACE FUNCTION {cooking_time_fn}(
    recipe_query STRING COMMENT 'Recipe name or part of it, e.g. "apple pie". Case-insensitive.'
)
RETURNS TABLE (
    recipe_name     STRING,
    prep_minutes    INT,
    cook_minutes    INT,
    other_minutes   INT,
    total_minutes   INT,
    total_time_text STRING,
    rating          DOUBLE
)
COMMENT 'Get the preparation, cooking, additional and total time in minutes for recipes whose name contains recipe_query. Returns up to 5 best-rated matches. Use it to answer how long a recipe takes.'
RETURN
    WITH matches AS (
        SELECT DISTINCT
            recipe_name,
            {parse_fn}(prep_time)  AS prep_minutes,
            {parse_fn}(cook_time)  AS cook_minutes,
            {parse_fn}(total_time) AS parsed_total,
            total_time,
            rating
        FROM {RECIPES_TABLE}
        WHERE row_id IS NOT NULL
          AND recipe_name ILIKE concat('%', recipe_query, '%')
    )
    SELECT
        recipe_name,
        prep_minutes,
        cook_minutes,
        -- greatest() ignores NULLs, so guard explicitly: unknown total -> unknown other time
        CASE
            WHEN parsed_total IS NOT NULL
            THEN greatest(parsed_total - coalesce(prep_minutes, 0) - coalesce(cook_minutes, 0), 0)
        END AS other_minutes,
        coalesce(parsed_total, prep_minutes + cook_minutes) AS total_minutes,
        total_time AS total_time_text,
        rating
    FROM matches
    ORDER BY rating DESC NULLS LAST, recipe_name
    LIMIT 5
""")
logger.info(f"Created: {cooking_time_fn}")

display(spark.sql(f"SELECT * FROM {cooking_time_fn}('apple pie')"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.3 Expose the UC function to the agent through managed MCP

# COMMAND ----------

uc_function_url = f"{w.config.host}/api/2.0/mcp/functions/{CATALOG}/{SCHEMA}/get_recipe_cooking_time"
uc_tools = asyncio.run(create_mcp_tools(w, [uc_function_url]))
logger.info(f"Discovered UC MCP tools: {[tool.name for tool in uc_tools]}")

cooking_time_tool = next(tool for tool in uc_tools if tool.name.endswith("get_recipe_cooking_time"))
logger.info(cooking_time_tool.exec_fn(recipe_query="apple pie"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.4 Expose the Genie space to the agent through managed MCP
# MAGIC
# MAGIC The Genie space from `resources/genie_space.yml` is served by Databricks
# MAGIC managed MCP at `/api/2.0/mcp/genie/{space_id}`. Genie writes and runs SQL on
# MAGIC `recipes_input`, so it covers counts, rankings and filters across the whole table.
# MAGIC

# COMMAND ----------

GENIE_SPACE_TITLE = "recipe-curator-space"


def find_genie_space_id(title_suffix: str) -> str | None:
    """Return the ID of the Genie space whose title ends with title_suffix, or None."""
    page_token = None
    while True:
        response = w.genie.list_spaces(page_token=page_token)
        for space in response.spaces or []:
            if space.title and space.title.endswith(title_suffix):
                return space.space_id
        page_token = response.next_page_token
        if not page_token:
            return None


try:
    genie_space_id = dbutils.widgets.get("genie_space_id")  # noqa: F821 - provided by Databricks
except Exception:  # noqa: BLE001 - widget not defined in interactive runs
    genie_space_id = None
genie_space_id = genie_space_id or cfg.genie_space_id or find_genie_space_id(GENIE_SPACE_TITLE)

if not genie_space_id:
    raise ValueError(
        f"No Genie space found. Deploy it with `databricks bundle deploy -t {env}` "
        "or set genie_space_id in project_config.yml."
    )
logger.info(f"Genie space ID: {genie_space_id}")

genie_url = f"{w.config.host}/api/2.0/mcp/genie/{genie_space_id}"
genie_tools = asyncio.run(create_mcp_tools(w, [genie_url]))
logger.info(f"Discovered Genie MCP tools: {[tool.name for tool in genie_tools]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. A recipe agent with tools and memory
# MAGIC
# MAGIC Every call to `chat(session_id, message)`:
# MAGIC 1. loads the earlier messages of the session from Lakebase,
# MAGIC 2. runs the usual tool-calling loop on *system prompt + history + new message*,
# MAGIC 3. saves the new user message and the final answer.
# MAGIC
# MAGIC **Only the user message and the final answer are stored**, not the tool calls
# MAGIC and tool results. That keeps the history short (tool results can be large) and
# MAGIC avoids replaying `tool` messages without their matching `tool_calls`. The
# MAGIC answer itself mentions the recipes, which is what a follow-up needs.
# MAGIC
# MAGIC **History is capped** at the last `max_history` messages, so long
# MAGIC conversations don't grow the prompt without limit.

# COMMAND ----------


class MemoryAgent:
    """Tool-calling agent that keeps each conversation in Lakebase."""

    def __init__(
        self,
        llm_endpoint: str,
        system_prompt: str,
        tools: list[ToolInfo],
        memory: LakebaseMemory,
        max_history: int = 20,
    ) -> None:
        self.llm_endpoint = llm_endpoint
        self.system_prompt = system_prompt
        self.tools = {tool.name: tool for tool in tools}
        self.memory = memory
        self.max_history = max_history
        self.client = OpenAI(
            api_key=w.tokens.create(lifetime_seconds=1200).token_value,
            base_url=f"{w.config.host}/ai-gateway/mlflow/v1",
        )

    def chat(self, session_id: str, user_message: str, max_iterations: int = 10) -> str:
        """Answer user_message in the context of the session, then save both to memory."""
        history = self.memory.load_messages(session_id)[-self.max_history :]
        logger.info(f"Session {session_id}: loaded {len(history)} earlier messages")

        messages = [
            {"role": "system", "content": self.system_prompt},
            *history,
            {"role": "user", "content": user_message},
        ]
        answer = self._run_tool_loop(messages, max_iterations)

        self.memory.save_messages(
            session_id,
            [{"role": "user", "content": user_message}, {"role": "assistant", "content": answer}],
        )
        return answer

    def _run_tool_loop(self, messages: list[dict], max_iterations: int) -> str:
        for _ in range(max_iterations):
            response = self.client.chat.completions.create(
                model=self.llm_endpoint,
                messages=messages,
                tools=[tool.spec for tool in self.tools.values()],
            )
            assistant_message = response.choices[0].message
            if not assistant_message.tool_calls:
                return assistant_message.content or ""

            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_message.content,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {"name": call.function.name, "arguments": call.function.arguments},
                        }
                        for call in assistant_message.tool_calls
                    ],
                }
            )
            for call in assistant_message.tool_calls:
                args = json.loads(call.function.arguments)
                logger.info(f"Tool call: {call.function.name}({args})")
                try:
                    result = self.tools[call.function.name].exec_fn(**args)
                except Exception as e:  # noqa: BLE001 - let the LLM see the error
                    result = f"Error: {e}"
                messages.append({"role": "tool", "tool_call_id": call.id, "content": str(result)})
        return "Max iterations reached."


# COMMAND ----------

SYSTEM_PROMPT = (
    f"{cfg.system_prompt} "
    "Use search_recipes to find recipes and cite the recipe names you used. "
    "Use the get_recipe_cooking_time tool for any question about how long a recipe takes; "
    "do not add up times yourself. "
    "Use the Genie tools for questions about the whole recipe collection, such as counts, "
    "rankings, averages or filtering by category, rating or servings. "
    "Use the earlier conversation to resolve follow-up questions such as 'that one' or 'make it vegetarian'. "
    "If no recipe matches, say so."
)

agent = MemoryAgent(
    llm_endpoint=cfg.llm_endpoint,
    system_prompt=SYSTEM_PROMPT,
    tools=[search_recipes_tool, cooking_time_tool, *genie_tools],
    memory=memory,
)
logger.info(f"Agent tools: {list(agent.tools)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### EDA: a follow-up that only works with memory
# MAGIC

# COMMAND ----------

session_id = f"recipe-chat-{uuid4()}"

# search_recipes
logger.info(agent.chat(session_id, "Can you suggest a dessert with apples and cinnamon?"))

# COMMAND ----------

# memory only (the answer is in the previous turn)
logger.info(agent.chat(session_id, "How many servings does the first one make?"))

# COMMAND ----------

# get_recipe_cooking_time, for the recipe from the previous turns
logger.info(agent.chat(session_id, "How many minutes does it take in total?"))

# COMMAND ----------

# search_recipes again, relative to the earlier answer
logger.info(agent.chat(session_id, "Is there something similar that is quicker to make?"))

# COMMAND ----------

# Genie, about the category of what we've been discussing
logger.info(agent.chat(session_id, "How many recipes are there in that category, and what is their average rating?"))

# COMMAND ----------

# The whole conversation, as stored in Lakebase
for msg in memory.load_messages(session_id):
    logger.info(f"{msg['role']:>9}: {msg['content'][:120]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Control: a new session starts empty
# MAGIC
# MAGIC The same follow-up in a **new** session has no history, so the agent should
# MAGIC ask which recipe you mean (or search for something on its own).

# COMMAND ----------

logger.info(agent.chat(f"recipe-chat-{uuid4()}", "How many servings does the first one make?"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. (Optional) Service principal authentication
# MAGIC
# MAGIC A deployed agent (model serving or an
# MAGIC app) runs as a SPN instead. `LakebaseMemory` switches
# MAGIC to SPN auth when these environment variables are set:
# MAGIC
# MAGIC | Variable | Value |
# MAGIC |---|---|
# MAGIC | `LAKEBASE_SP_CLIENT_ID` | SPN application (client) ID, also its Postgres role name |
# MAGIC | `LAKEBASE_SP_CLIENT_SECRET` | SPN OAuth secret |
# MAGIC | `LAKEBASE_SP_HOST` | Workspace URL |
# MAGIC
# MAGIC The `LAKEBASE_SP_*` names (instead of `DATABRICKS_CLIENT_ID`, …) only change
# MAGIC Lakebase auth; the default `WorkspaceClient` used by the tools stays as is.
# MAGIC
# MAGIC **Prerequisites**
# MAGIC - The SPN's client ID and secret are stored as Unity Catalog secrets
# MAGIC   `databricks_client_id` and `databricks_client_secret` in your schema, and
# MAGIC   you have `READ SECRET` on them.
# MAGIC - The SPN has a Postgres role in the Lakebase project.
# MAGIC - The SPN may write to `session_messages`. You created the table in section 1,
# MAGIC   so you own it; grant access as its owner, for example:
# MAGIC   `GRANT SELECT, INSERT ON session_messages TO "<client-id>";`
# MAGIC   `GRANT USAGE ON SEQUENCE session_messages_id_seq TO "<client-id>";`
# MAGIC
# MAGIC Set `RUN_SPN_TEST = True` once these are in place.

# COMMAND ----------

RUN_SPN_TEST = False

# COMMAND ----------

SPN_ENV_VARS = ["LAKEBASE_SP_CLIENT_ID", "LAKEBASE_SP_CLIENT_SECRET", "LAKEBASE_SP_HOST"]

if RUN_SPN_TEST:
    os.environ["LAKEBASE_SP_CLIENT_ID"] = dbutils.secrets.get(  # noqa: F821 - provided by Databricks
        catalog=cfg.catalog, schema=cfg.schema, key="databricks_client_id"
    )
    os.environ["LAKEBASE_SP_CLIENT_SECRET"] = dbutils.secrets.get(  # noqa: F821
        catalog=cfg.catalog, schema=cfg.schema, key="databricks_client_secret"
    )
    os.environ["LAKEBASE_SP_HOST"] = w.config.host

    try:
        # No workspace client passed: the memory builds its own SPN client from the env vars
        spn_memory = LakebaseMemory(project_id=LAKEBASE_PROJECT)

        spn_session_id = f"spn-test-{uuid4()}"
        spn_messages = [
            {"role": "user", "content": "Find me a quick pasta recipe."},
            {"role": "assistant", "content": "Here is a 20-minute garlic butter pasta..."},
        ]
        spn_memory.save_messages(spn_session_id, spn_messages)
        assert spn_memory.load_messages(spn_session_id) == spn_messages, "SPN message round-trip failed"
        logger.info(f"SPN round-trip OK for {spn_session_id}")
    finally:
        # Remove the SPN credentials so later cells in this notebook use your own identity again
        for var in SPN_ENV_VARS:
            os.environ.pop(var, None)
else:
    logger.info("Skipping SPN test (RUN_SPN_TEST = False)")
