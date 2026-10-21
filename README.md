# llmops-katy-chen

## Use case
### 1. User and problem

This is a recipe assistant for supermarket customers planning
everyday meals. It would help them find a suitable recipe on the supermarket's
website without browsing many recipes manually. Customers could describe the
ingredients they already have, their available cooking time, and their preferences,
then refine the suggestions through follow-up questions.

### 2. Data

The project will use the public Kaggle dataset
[Better Recipes for a Better Life](https://www.kaggle.com/datasets/thedevastator/better-recipes-for-a-better-life),
which is already referenced in the recipe-ingestion notebook. No confidential
company or customer data will be used.

- **Recipe documents:** recipe names, ingredient lists, and preparation directions
  will provide the searchable content for retrieving relevant recipes.
- **Structured data:** cooking and preparation times, servings, ratings, cuisine
  categories, source URLs, and nutrition information where available will support
  filtering and displaying results. Missing or incomplete values will be treated
  as unknown rather than inferred as facts.
- **Possible later additions:** a small mock product catalogue with ingredient
  prices could support shopping suggestions and illustrative budget estimates.
  These would not represent live supermarket prices or availability.

### 3. Example interactions

Customers would use a chat interface to describe a meal and refine the results:

- "Find a high-protein dinner for four people that takes no more than 30 minutes."
- "I have chickpeas, tomatoes, and spinach. Which dinners can I make with them?"
- "Show me a vegetarian alternative from those suggestions."
- "Show me the ingredients and preparation steps for the second recipe."


### 4. Scope

The smallest useful course version will be a standalone chat prototype covering
around 100 dinner recipes with a stated total cooking time of 30 minutes or less.
It will retrieve recipes from this curated collection, use available serving and
ingredient information to narrow suggestions, and answer follow-up questions using
the retrieved recipe content. It will not invent recipes or claim a match when
the required information is missing.


## Course repository

Your repository for the LLMOps course on Databricks. Everything you need in
the course workspace was provisioned for you before you got this repo, and every id is already in
`project_config.yml`: you configure nothing.

## What was provisioned for you

| What | Value |
|---|---|
| Your schema in development, which you manage | `llmops_dev.katy_chen` |
| Your schema in acceptance, which you read | `llmops_acc.katy_chen` |
| Your schema in production, which you read | `llmops_prd.katy_chen` |
| Suggested volume name (create during the course) | `arxiv_files`, such as `llmops_dev.katy_chen.arxiv_files` |
| SQL warehouse id | `b11cb5f20e908516` |
| Usage policy id | `c7e24808-d4b5-373a-a34c-bbe50a5f5d3f` |
| MLflow experiments | Dev: `/llmops/katy_chen-dev`; acc: `/llmops/katy_chen-acc`; prd: `/llmops/katy_chen-prd` |
| Chat model service | `course_ops.gateway.chat` |
| Embedding model service | `course_ops.gateway.embed` |

Both settings name model services of the course’s AI Gateway, with a monthly budget per student.
Chat routes to GPT-OSS-120B; embeddings route to GTE Large. Call both through the gateway
with the `databricks-openai` package:

```python
from databricks_openai import DatabricksOpenAI

client = DatabricksOpenAI(use_ai_gateway=True)
client.chat.completions.create(
    model="course_ops.gateway.chat",
    messages=[...],
    reasoning_effort="low",
    max_tokens=1024,
)
client.embeddings.create(model="course_ops.gateway.embed", input="A paper abstract")
```

The `databricks-openai` package is not included in this repo’s dependencies. The same APIs
are available through the installed `openai` client with the workspace base URL
`/ai-gateway/mlflow/v1` and Databricks authentication.

`embedding_endpoint` is a gateway model service despite the existing field name. Generate
vectors through its embeddings API before storing them in pgvector. It is not a managed
Vector Search index’s `embedding_model_endpoint_name`.

## Lakebase

The `lakebase_instance` value in `project_config.yml` identifies your Lakebase project.
All three environments share it; your Unity Catalog schemas remain separate.
Onboarding does not create an application database or install pgvector. Create the database
and extensions you need during the course. A Vector Search endpoint is not provisioned.

For a dedicated student project, you and your three deployment service principals have project
management access and OAuth Postgres roles with database administration privileges. The project
carries your usage policy and student cost tag. Its initial compute range is 0.5–2 CU, with
suspension after five idle minutes. Creating extra branches or computes can increase its cost.

## What is in the repo



- `src/recipe_curator/`: contains `config.py`, which loads `project_config.yml`, and
  `data_processor.py`, which downloads arXiv PDFs, parses them with `ai_parse_document`, and
  extracts cleaned chunks into Unity Catalog tables.
- `notebooks/`: `00_hello.py` is the bundle's hello-world task;
  `01_arxiv_data_ingestion.py` fetches recent AI/ML paper metadata from arXiv and writes it to
  the target schema's `arxiv_papers` table. `lecture/lecture_1/` contains the lecture notebooks.
- `lecture_1/`, `lecture_2/` and so on, when you add them: copy a lecture's folder from the course
  code as it is, to the root of this repo, where its notebooks find `project_config.yml` one level up.
  Commit it with the hooks installed (below): the first commit stops once, after the hooks have
  formatted the notebooks and fixed what ruff can, and goes through when you add their changes and
  commit again. Ruff rules are configured in `pyproject.toml`.
- `databricks.yml` and `resources/`: the bundle, with three targets, `dev` (your sandbox, in
  development mode, the default), `acc` and `prd`. It builds the package as a wheel with `uv build`
  and defines two jobs: `hello`, which installs the wheel and writes a row to the target schema's
  `hello` table, and `arxiv-data-pipeline`, which runs `01_arxiv_data_ingestion.py` daily at 06:00
  Europe/Amsterdam. Both jobs use serverless notebook tasks. Job schedules are paused by default;
  the pipeline job is defined in the bundle but is not deployed by course onboarding. Deployment
  workflows are introduced in a later lecture.
- `pyproject.toml` and `version.txt`: the package and its version, bumped as the lectures ask.
- `tests/`: your tests; `test_config.py` checks that `project_config.yml` loads.
- `.github/workflows/ci.yml`: on every pull request and push to `main`, installs the package with
  uv and runs the pre-commit hooks and the tests.
- `.github/CODEOWNERS`: who reviews a pull request to `main`, which needs one approving review.

## Getting started

```bash
uv sync --extra dev
uvx pre-commit install
databricks bundle validate -t dev
```