# ---------------------------------------------------------
# Human Evaluation - Grouped Bar Charts
# LLM Text Summarization Benchmark
# Generates 2x2 subplot of coherence, fluency, relevance,
# consistency ratings by model (mean ± SEM)
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
  file.path(eval_dir, "evaluation_PPE_after80_2026-03-24.json"),
  file.path(eval_dir, "evaluation_KKI_final.json")
)

model_labels <- c(
  "huggingface_google/bigbird-pegasus-large-pubmed" = "BigBird-Pegasus",
  "huggingface_csebuetnlp/mT5_multilingual_XLSum"   = "mT5-XLSum",
  "ollama_mistral-small3.2:24b"                      = "Mistral-Small-3.2 (24B)",
  "mistral_mistral-small-2506"                       = "Mistral-Small-2506"
)

model_colors <- c(
  "BigBird-Pegasus"          = "#4C72B0",
  "mT5-XLSum"                = "#DD8452",
  "Mistral-Small-3.2 (24B)"  = "#55A868",
  "Mistral-Small-2506"       = "#C44E52"
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

# compute mean and SEM per model per dimension
summary_stats <- ratings_long %>%
  group_by(model_label, dimension) %>%
  summarise(
    mean_rating = mean(rating, na.rm = TRUE),
    sem         = sd(rating, na.rm = TRUE) / sqrt(n()),
    n           = n(),
    .groups     = "drop"
  )

summary_stats$model_label <- factor(
  summary_stats$model_label,
  levels = c("BigBird-Pegasus", "mT5-XLSum",
             "Mistral-Small-3.2 (24B)", "Mistral-Small-2506")
)

summary_stats$dimension <- factor(
  summary_stats$dimension,
  levels = dimensions
)



plot_dimension <- function(data, dim_name) {
  df <- data %>% filter(dimension == dim_name)

  ggplot(df, aes(x = model_label, y = mean_rating, fill = model_label)) +
    geom_col(width = 0.6, color = "white", linewidth = 0.4) +
    geom_errorbar(
      aes(ymin = mean_rating - sem, ymax = mean_rating + sem),
      width = 0.15, linewidth = 0.6, color = "gray30"
    ) +
    geom_text(
      aes(y = mean_rating + sem + 0.08, label = sprintf("%.2f", mean_rating)),
      size = 3.5, fontface = "bold", color = "black"
    ) +
    scale_fill_manual(values = model_colors) +
    scale_y_continuous(
      breaks = 0:5,
      expand = c(0, 0)
    ) +
    coord_cartesian(ylim = c(0, 5.4)) +
    labs(
      title = tools::toTitleCase(dim_name),
      x = NULL,
      y = "Mean Rating"
    ) +
    theme_minimal(base_size = 14) +
    theme(
      axis.text.x       = element_text(face = "bold", size = 10, color = "black"),
      axis.text.y        = element_text(size = 10, color = "black"),
      axis.title         = element_text(face = "bold"),
      plot.title         = element_text(face = "bold", size = 14, hjust = 0.5),
      panel.grid.major.x = element_blank(),
      panel.grid.minor   = element_blank(),
      panel.border       = element_rect(color = "black", fill = NA, linewidth = 0.8),
      legend.position    = "none"
    )
}

p_coherence   <- plot_dimension(summary_stats, "coherence")
p_fluency     <- plot_dimension(summary_stats, "fluency")
p_relevance   <- plot_dimension(summary_stats, "relevance")
p_consistency <- plot_dimension(summary_stats, "consistency")

combined <- (p_coherence | p_fluency) / (p_relevance | p_consistency) +
  plot_annotation(
    title = "Human Evaluation Ratings by Model",
    theme = theme(
      plot.title = element_text(face = "bold", size = 16, hjust = 0.5)
    )
  )

ggsave(
  filename = file.path(Outcome, "human_evaluation_combined_plot.pdf"),
  plot = combined,
  width = 14,
  height = 10
)

cat("Saved:", file.path(Outcome, "human_evaluation_combined_plot.pdf"), "\n")