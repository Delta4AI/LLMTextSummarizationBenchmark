# ---------------------------------------------------------
# Human Evaluation - Violin Plots with Individual Points
# LLM Text Summarization Benchmark
# Generates 2x2 subplot of coherence, fluency, relevance,
# consistency ratings by model
# ---------------------------------------------------------

packages <- c("jsonlite", "dplyr", "tidyr", "ggplot2", "patchwork", "here")

installed <- packages %in% rownames(installed.packages())
if (any(!installed)) {
  install.packages(packages[!installed], repos = "https://cloud.r-project.org")
}

invisible(lapply(packages, library, character.only = TRUE))


# config
GH_HASH <- "1362b291718b57188a7909f08de26da760a0b9346d52111c97671d97d713af38"
base_dir <- here("Output", "llm_summarization_benchmark", GH_HASH)
eval_dir <- file.path(base_dir, "human_evaluations")
Outcome <- base_dir

eval_files <- c(
  file.path(eval_dir, "evaluation_1.json"),
  file.path(eval_dir, "evaluation_2.json"),
  file.path(eval_dir, "evaluation_3.json"),
  file.path(eval_dir, "evaluation_4.json"),
  file.path(eval_dir, "evaluation_5.json"),
  file.path(eval_dir, "evaluation_6.json"),
  file.path(eval_dir, "evaluation_7.json"),
  file.path(eval_dir, "evaluation_8.json")
)

model_labels <- c(
  "huggingface_google/bigbird-pegasus-large-pubmed" = "bigbird-pegasus",
  "huggingface_csebuetnlp/mT5_multilingual_XLSum"   = "mT5",
  "ollama_mistral-small3.2:24b"                      = "mistral-small-3.2:24B",
  "mistral_mistral-small-2506"                       = "mistral-small-2506"
)

# Colors aligned with the paper's category scheme:
#   - bigbird-pegasus & mT5 are Domain-specific EDMs → blue tones
#     (matching the steel-blue used for that category in the boxplot/heatmap)
#   - mistral-small-3.2:24B & mistral-small-2506 are General-purpose LLMs → orange/amber tones
#     (matching the orange used for that category in the boxplot/heatmap)
#   Within each pair, a darker and lighter variant provides distinction.
model_colors <- c(
  "bigbird-pegasus"        = "#4575B4",
  "mT5"                    = "#FEE090",
  "mistral-small-3.2:24B"  = "#A6D96A",
  "mistral-small-2506"     = "#D73027"
)

dimensions <- c("coherence", "fluency", "relevance", "consistency")

# load and merge assessments
load_assessments <- function(filepaths) {
  all_assessments <- list()
  for (fp in filepaths) {
    data <- fromJSON(fp)
    all_assessments <- c(all_assessments, list(data$assessments))
  }
  bind_rows(all_assessments)
}

assessments <- load_assessments(eval_files)

assessments_filtered <- assessments %>%
  filter(model %in% names(model_labels)) %>%
  mutate(
    model_label = model_labels[model],
    coherence   = ratings$coherence,
    fluency     = ratings$fluency,
    relevance   = ratings$relevance,
    consistency = ratings$consistency
  )

ratings_long <- assessments_filtered %>%
  select(paper_id, model_label, all_of(dimensions)) %>%
  pivot_longer(
    cols = all_of(dimensions),
    names_to = "dimension",
    values_to = "rating"
  )

ratings_long$model_label <- factor(
  ratings_long$model_label,
  levels = c("bigbird-pegasus", "mT5",
             "mistral-small-3.2:24B", "mistral-small-2506")
)

ratings_long$dimension <- factor(
  ratings_long$dimension,
  levels = dimensions
)

# compute means for annotation
summary_stats <- ratings_long %>%
  group_by(model_label, dimension) %>%
  summarise(
    mean_rating = mean(rating, na.rm = TRUE),
    .groups = "drop"
  )


plot_dimension <- function(data, summary, dim_name) {
  df <- data %>% filter(dimension == dim_name)
  df_summary <- summary %>% filter(dimension == dim_name)

  ggplot(df, aes(x = model_label, y = rating, fill = model_label)) +
    geom_violin(
      trim = TRUE, alpha = 1.0,
      color = "gray30", linewidth = 0.4,
      scale = "width"
    ) +
    geom_boxplot(
      width = 0.12, outlier.shape = NA,
      color = "gray20", fill = "white", alpha = 0.8
    ) +
    geom_text(
      data = df_summary,
      aes(x = model_label, y = 5.2, label = sprintf("%.2f", mean_rating)),
      size = 3.5, fontface = "bold", color = "black"
    ) +
    scale_fill_manual(values = model_colors) +
    scale_y_continuous(
      breaks = 1:5,
      expand = c(0, 0)
    ) +
    coord_cartesian(ylim = c(0.5, 5.5)) +
    labs(
      title = tools::toTitleCase(dim_name),
      x = NULL,
      y = "Rating"
    ) +
    theme_minimal(base_size = 14) +
    theme(
      axis.text.x        = element_text(face = "bold", size = 10, color = "black"),
      axis.text.y         = element_text(size = 10, color = "black"),
      axis.title          = element_text(face = "bold"),
      plot.title          = element_text(face = "bold", size = 14, hjust = 0.5),
      panel.grid.major.x  = element_blank(),
      panel.grid.minor    = element_blank(),
      panel.border        = element_rect(color = "black", fill = NA, linewidth = 0.8),
      legend.position     = "none"
    )
}

p_coherence   <- plot_dimension(ratings_long, summary_stats, "coherence")
p_fluency     <- plot_dimension(ratings_long, summary_stats, "fluency")
p_relevance   <- plot_dimension(ratings_long, summary_stats, "relevance")
p_consistency <- plot_dimension(ratings_long, summary_stats, "consistency")

combined <- (p_coherence | p_fluency) / (p_relevance | p_consistency) +
  plot_annotation(
    title = "Human Evaluation Ratings by Model",
    theme = theme(
      plot.title = element_text(face = "bold", size = 16, hjust = 0.5)
    )
  )

ggsave(
  filename = file.path(Outcome, "human_evaluation_violin.pdf"),
  plot = combined,
  width = 14,
  height = 10
)

cat("Saved:", file.path(Outcome, "human_evaluation_violin.pdf"), "\n")