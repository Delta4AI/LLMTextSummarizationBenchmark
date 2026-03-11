# ---------------------------------------------------------
# Benchmark visualization and z-score computation script
# LLM Text Summarization Benchmark
# ---------------------------------------------------------

# ------------------- Libraries --------------------------
packages <- c(
  "readr",
  "dplyr",
  "tidyr",
  "tibble",
  "ggplot2",
  "scales",
  "ComplexHeatmap",
  "circlize",
  "grid",
  "readxl",
  "openxlsx",
  "here"
)

installed <- packages %in% rownames(installed.packages())

if (any(!installed)) {
  install.packages(packages[!installed])
}

# Load all packages
invisible(lapply(packages, library, character.only = TRUE))

GH_HASH <- "1362b291718b57188a7909f08de26da760a0b9346d52111c97671d97d713af38"

Outcome <- here("Output", "llm_summarization_benchmark", GH_HASH)
if (!dir.exists(Outcome)) dir.create(Outcome, recursive = TRUE)

# ------------------- Import data ------------------------


heatmap_data <- read_excel(
  here("Output", "llm_summarization_benchmark", GH_HASH, "heatmap_data.xlsx"),
  col_types = c(
    "text", "numeric", "numeric",
    "numeric", "numeric", "numeric",
    "numeric", "numeric", "numeric",
    "numeric", "numeric", "numeric", "numeric", "text", "text"
  )
)

model_groups   <- read_excel(here("Resources", "model_groups.xlsx"))
model_families <- read_excel(here("Resources", "model_families.xlsx"))
Methods_ID     <- read_excel(here("Resources", "Methods_ID.xlsx"))

# Join IDs
heatmap_data <- left_join(Methods_ID, heatmap_data, by = "model_name")

# Rename metrics (no FactCC anymore)
heatmap_data <- heatmap_data %>%
  rename(
    "ROUGE1"            = rouge1_mean,
    "ROUGE2"            = rouge2_mean,
    "ROUGEL"            = rougeL_mean,
    "METEOR"            = meteor_mean,
    "BLEU"              = bleu_mean,
    "RoBERTa"           = roberta_f1_mean,
    "DeBERTa"           = deberta_f1_mean,
    `all-mpnet-base-v2` = content_coverage_mean,
    "AlignScore"        = alignscore_mean,
    "SummaC"            = summac_mean,
    "MiniCheck-ft5"     = minicheck_ft5_mean,
    "MiniCheck-7b"      = minicheck_7b_mean
  ) %>%
  dplyr::select(
    method,
    ROUGE1, ROUGE2, ROUGEL, METEOR, BLEU,
    RoBERTa, DeBERTa, `all-mpnet-base-v2`,
    AlignScore, SummaC, `MiniCheck-ft5`, `MiniCheck-7b`,
    model_group, model_family
  )

# 1) Category averages (4 factual metrics)
heatmap_data <- heatmap_data %>%
  mutate(
    lexical_avg  = (ROUGE1 + ROUGE2 + ROUGEL + METEOR + BLEU) / 5,
    semantic_avg = (RoBERTa + DeBERTa + `all-mpnet-base-v2`) / 3,
    factual_avg  = (AlignScore + SummaC + `MiniCheck-ft5` + `MiniCheck-7b`) / 4
  )

# 2) Z-scores across models
heatmap_data <- heatmap_data %>%
  mutate(
    z_lex  = (lexical_avg  - mean(lexical_avg,  na.rm = TRUE)) /
      sd(lexical_avg,  na.rm = TRUE),
    z_sem  = (semantic_avg - mean(semantic_avg, na.rm = TRUE)) /
      sd(semantic_avg, na.rm = TRUE),
    z_fact = (factual_avg  - mean(factual_avg,  na.rm = TRUE)) /
      sd(factual_avg,  na.rm = TRUE)
  )

# 3) Overall performance score = mean of z-scores
heatmap_data <- heatmap_data %>%
  mutate(
    performance_score = (z_lex + z_sem + z_fact) / 3
  )

# 4) Category and overall ranks (higher = better → rank(-score))
heatmap_data <- heatmap_data %>%
  mutate(
    `Lexical Rank`     = rank(-z_lex,             ties.method = "min"),
    `Semantic Rank`    = rank(-z_sem,             ties.method = "min"),
    `Factual Rank`     = rank(-z_fact,            ties.method = "min"),
    `Performance Rank` = rank(-performance_score, ties.method = "min")
  )

# 5) Full per-metric ranks (no FactCC)
heatmap_data_full <- heatmap_data %>%
  mutate(
    `ROUGE-1_rank`           = rank(-ROUGE1,             ties.method = "min"),
    `ROUGE-2_rank`           = rank(-ROUGE2,             ties.method = "min"),
    `ROUGE-L_rank`           = rank(-ROUGEL,             ties.method = "min"),
    METEOR_rank              = rank(-METEOR,             ties.method = "min"),
    BLEU_rank                = rank(-BLEU,               ties.method = "min"),
    RoBERTa_rank             = rank(-RoBERTa,            ties.method = "min"),
    DeBERTa_rank             = rank(-DeBERTa,            ties.method = "min"),
    `all-mpnet-base-v2_rank` = rank(-`all-mpnet-base-v2`,ties.method = "min"),
    AlignScore_rank          = rank(-AlignScore,         ties.method = "min"),
    SummaC_rank              = rank(-SummaC,             ties.method = "min"),
    `MiniCheck-ft5_rank`     = rank(-`MiniCheck-ft5`,    ties.method = "min"),
    `MiniCheck-7b_rank`      = rank(-`MiniCheck-7b`,     ties.method = "min")
  )

# 6) Write outputs
write.xlsx(heatmap_data,      paste0(Outcome, "Compute_zscore_based.xlsx"))
write.xlsx(heatmap_data_full, paste0(Outcome, "Compute_zscore_based_full_ranks.xlsx"))

