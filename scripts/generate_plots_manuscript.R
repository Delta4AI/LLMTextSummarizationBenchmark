# ---------------------------------------------------------
# Benchmark visualization script
# LLM Text Summarization Benchmark
# Generates heatmaps and statistical comparison plots
# ---------------------------------------------------------


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

invisible(lapply(packages, library, character.only = TRUE))


#### load data for the heatmap and following plots ###


Outcome <- here("Output", "Plots_benchmark_data")

if (!dir.exists(Outcome)) {
  dir.create(Outcome, recursive = TRUE)
}

GH_HASH <- "1362b291718b57188a7909f08de26da760a0b9346d52111c97671d97d713af38"

heatmap_data <- read_excel(
  here("Output", "llm_summarization_benchmark", GH_HASH, "Compute_zscore_based.xlsx")
)



games_howell_pvalues_matrix_categories <- read_csv(
  here("Output", "llm_summarization_benchmark", GH_HASH, "games_howell_pvalues_matrix_categories.csv")
)

games_howell_pvalues_matrix_families <- read_csv(
  here("Output", "llm_summarization_benchmark", GH_HASH, "games_howell_pvalues_matrix_families.csv")
)

metric_correlation_spearman <- read_csv(
  here("Output", "llm_summarization_benchmark", GH_HASH, "metric_correlation_spearman.csv")
)

### Figure S1 ####

metric_category <- c(
  `ROUGE-1` = "Lexical",
  `ROUGE-2` = "Lexical",
  `ROUGE-L` = "Lexical",
  METEOR = "Lexical",
  BLEU = "Lexical",
  RoBERTa = "Semantic",
  DeBERTa = "Semantic",
  `all-mpnet-base-v2` = "Semantic",
  AlignScore = "Factual",
  SummaC = "Factual",
  `MiniCheck-ft5` = "Factual",
  `MiniCheck-7b` = "Factual"
)

metric_colors <- c(
  Lexical  = "#D77E61", 
  Semantic = "#91A4A3", 
  Factual  = "#E4E8E8"  
)

metric_category <- factor(
  metric_category,
  levels = c("Lexical", "Semantic", "Factual")
)


heatmap_data_rank <- heatmap_data

heatmap_data_rank <- heatmap_data_rank %>%
  arrange(`Performance Rank`)

heatmap_data_rank <- heatmap_data_rank %>%
  mutate(
    `ROUGE-1` = rank(-ROUGE1, ties.method = "min"),
    `ROUGE-2` = rank(-ROUGE2, ties.method = "min"),
    `ROUGE-L` = rank(-ROUGEL, ties.method = "min"),
    METEOR = rank(-METEOR, ties.method = "min"),
    BLEU = rank(-BLEU, ties.method = "min"),
    RoBERTa = rank(-RoBERTa, ties.method = "min"),
    DeBERTa = rank(-DeBERTa, ties.method = "min"),
    `all-mpnet-base-v2` = rank(-`all-mpnet-base-v2`, ties.method = "min"),
    AlignScore = rank(-AlignScore, ties.method = "min"),
    SummaC = rank(-SummaC, ties.method = "min"),
    `MiniCheck-ft5` = rank(-`MiniCheck-ft5`, ties.method = "min"),
    `MiniCheck-7b` = rank(-`MiniCheck-7b`, ties.method = "min")
  )


# Score matrix
score_matrix <- heatmap_data_rank %>%
  dplyr::select(
    method,
    `ROUGE-1`, `ROUGE-2`, `ROUGE-L`, METEOR, BLEU,
    RoBERTa, DeBERTa, `all-mpnet-base-v2`,
    AlignScore, SummaC, `MiniCheck-ft5`, `MiniCheck-7b`
  ) %>%
  column_to_rownames("method") %>%
  as.matrix()


median_val <- median(score_matrix, na.rm = TRUE)
min_val <- min(score_matrix, na.rm = TRUE)
mid_val <- 5
max_val <- max(score_matrix, na.rm = TRUE)

col_fun <- colorRamp2(
  c(min_val, mid_val, max_val),
  c("#403C53", "#E4E3EA", "#C33D35")
)


Families <- c(
  "Word-frequency" = "#5B8A8A",      
  "BART"           = "#E78551",      
  "Pegasus"        = "#D45769",      
  "T5"             = "#6A9E5F",      
  "LED"            = "#A67853",      
  "Qwen"           = "#B39DDB",     
  "Apertus"        = "#9575A8",      
  "Gemma"          = "#E6B84D",      
  "Granite"        = "#7BB3D9",      
  "Llama"          = "#EE9AA0",      
  "Mistral"        = "#7A7A7A",      
  "Phi"            = "#F4A460",      
  "DeepSeek"       = "#B5A094",      
  "GPT"            = "#4CA387",      
  "Claude"         = "#C69B6D"       
)


Groups <- c(
  "Traditional Models"        = "#4575B4",  
  "General-purpose EDMs"       = "#C6E2F0",  
  "Domain-specific EDMs"       = "#74ADD1",  
  "General-purpose SLMs"       = "#FEE090",  
  "General-purpose LLMs"       = "#FDAE61", 
  "Reasoning-oriented SLMs"    = "#F46D43",  
  "Reasoning-oriented LLMs"    = "#D73027",  
  "Domain-specific SLMs"       = "#A6D96A",  
  "Domain-specific LLMs"       = "#1A9850"   
)


heatmap_data_rank$model_family <- factor(
  heatmap_data_rank$model_family,
  levels = names(Families)
)

heatmap_data_rank$model_group <- factor(
  heatmap_data_rank$model_group,
  levels = names(Groups)
)


row_ha <- rowAnnotation(
  Families = heatmap_data_rank$model_family,
  Categories = heatmap_data_rank$model_group,
  col = list(
    Families = Families,
    Categories = Groups
  ),
  annotation_name_gp = gpar(fontsize = 10, fontface = "bold"),
  
  show_annotation_name = TRUE,
  annotation_legend_param = list(
    Families = list(border = TRUE),
    Categories = list(border = TRUE)
  ),
  
  gp = gpar(col = "black", lwd = 0.5)  
)

top_ha <- HeatmapAnnotation(
  MetricType = metric_category,
  col = list(MetricType = metric_colors),
  annotation_name_gp = gpar(fontsize = 10, fontface = "bold"),
  simple_anno_size = unit(2.5, "mm"),
  border = TRUE,  
  annotation_legend_param = list(
    MetricType = list(
      border = TRUE 
    )
  )
)



top_n <- 3


Figure_S1 <- Heatmap(
  score_matrix,
  name = "Rank",
  col = col_fun,
  cluster_rows = FALSE,
  cluster_columns = FALSE,
  row_dend_side = "right",
  row_names_side = "left",
  row_dend_width = unit(3, "cm"),
  show_row_names = TRUE,
  show_column_names = TRUE,
  row_names_gp = gpar(fontsize = 9, fontface = "bold"),
  column_names_gp = gpar(fontsize = 10, fontface = "bold"),
  left_annotation = row_ha,
  top_annotation = top_ha,    # ← ADD THIS
  rect_gp = gpar(col = "black", lwd = 0.5),
  column_title = "Models performance across metrics",
  column_title_gp = gpar(fontsize = 14, fontface = "bold", hjust = 1),
  cell_fun = function(j, i, x, y, width, height, fill) {
    value <- score_matrix[i, j]
    text_col <- if (value <= 3) "white" else "black"
    text_face <- if (value %in% sort(score_matrix[, j])[1:top_n]) "bold" else "plain"
    grid.text(value, x, y, gp = gpar(fontsize = 9, col = text_col, fontface = text_face))
  },
  heatmap_legend_param = list(
    direction = "vertical",
    at = c(max_val, min_val),
    labels = c(max_val, min_val)
  )
)


draw(Figure_S1, heatmap_legend_side = "right", annotation_legend_side = "right")

pdf_file <- file.path(Outcome, "Figure_S1.pdf")
pdf(file = pdf_file, width = 10, height = 11.69)  # A4 in inches
draw(Figure_S1, heatmap_legend_side = "right", annotation_legend_side = "right")
dev.off()



### Figure 3 ####

heatmap_data <- heatmap_data %>%
  arrange(`Performance Rank`)

# Score matrix
score_matrix <- heatmap_data %>%
  dplyr::select(method, `Performance Rank`, `Lexical Rank`, `Semantic Rank`, `Factual Rank`) %>%
  column_to_rownames("method") %>%
  as.matrix()


median_val <- median(score_matrix, na.rm = TRUE)
min_val <- min(score_matrix, na.rm = TRUE)
mid_val <- 5
max_val <- max(score_matrix, na.rm = TRUE)

col_fun <- colorRamp2(
  c(min_val, mid_val, max_val),
  c("#403C53", "#E4E3EA", "#C33D35")
)



Families <- c(
  "Word-frequency" = "#5B8A8A",      
  "BART"           = "#E78551",      
  "Pegasus"        = "#D45769",      
  "T5"             = "#6A9E5F",      
  "LED"            = "#A67853",      
  "Qwen"           = "#B39DDB",      
  "Apertus"        = "#9575A8",      
  "Gemma"          = "#E6B84D",      
  "Granite"        = "#7BB3D9",      
  "Llama"          = "#EE9AA0",      
  "Mistral"        = "#7A7A7A",      
  "Phi"            = "#F4A460",      
  "DeepSeek"       = "#B5A094",      
  "GPT"            = "#4CA387",      
  "Claude"         = "#C69B6D"       
)


Groups <- c(
  "Traditional Models"        = "#4575B4",  
  "General-purpose EDMs"       = "#C6E2F0",  
  "Domain-specific EDMs"       = "#74ADD1",  
  "General-purpose SLMs"       = "#FEE090",  
  "General-purpose LLMs"       = "#FDAE61",  
  "Reasoning-oriented SLMs"    = "#F46D43",  
  "Reasoning-oriented LLMs"    = "#D73027",  
  "Domain-specific SLMs"       = "#A6D96A",  
  "Domain-specific LLMs"       = "#1A9850"  
)


heatmap_data$model_family <- factor(
  heatmap_data$model_family,
  levels = names(Families)
)

heatmap_data$model_group <- factor(
  heatmap_data$model_group,
  levels = names(Groups)
)

row_ha <- rowAnnotation(
  Families = heatmap_data$model_family,
  Categories = heatmap_data$model_group,
  col = list(
    Families = Families,
    Categories = Groups
  ),
  annotation_name_gp = gpar(fontsize = 10, fontface = "bold"),
  
  show_annotation_name = TRUE,
  annotation_legend_param = list(
    Families = list(border = TRUE),
    Categories = list(border = TRUE)
  ),
  
  gp = gpar(col = "black", lwd = 0.5)  
)



top_n <- 3

Figure_3 <- Heatmap(
  score_matrix,
  name = "Rank",
  col = col_fun,
  cluster_rows = FALSE,
  cluster_columns = FALSE,
  row_dend_side = "right",
  row_names_side = "left",
  row_dend_width = unit(3, "cm"),
  show_row_names = TRUE,
  show_column_names = TRUE,
  row_names_gp = gpar(fontsize = 9, fontface = "bold"),
  column_names_gp = gpar(fontsize = 10, fontface = "bold"),
  left_annotation = row_ha,
  rect_gp = gpar(col = "black", lwd = 0.5),
  column_title = "Ranking of summarization models",
  column_title_gp = gpar(fontsize = 14, fontface = "bold", hjust = 1),
  cell_fun = function(j, i, x, y, width, height, fill) {
    value <- score_matrix[i, j]
    text_col <- if (value <= 3) "white" else "black"
    text_face <- if (value %in% sort(score_matrix[, j])[1:top_n]) "bold" else "plain"
    grid.text(value, x, y, gp = gpar(fontsize = 9, col = text_col, fontface = text_face))
  },
  
  heatmap_legend_param = list(
    direction = "vertical",
    at = c(max_val, min_val),
    labels = c(max_val, min_val)
  )
)

draw(Figure_3, heatmap_legend_side = "right", annotation_legend_side = "bottom")

pdf_file <- file.path(Outcome, "Figure_3.pdf")
pdf(file = pdf_file, width = 7, height = 11.69)  # A4 in inches
draw(Figure_3, heatmap_legend_side = "right", annotation_legend_side = "bottom")
dev.off()




#### Figure 4a ####

library(dplyr)
library(ggplot2)


min_rank <- min(heatmap_data_rank$performance_score)
max_rank <- max(heatmap_data_rank$performance_score)

#min max normalization for plotting purpose

heatmap_data_norm <- heatmap_data %>%
  mutate(normalized_composite = (performance_score - min_rank) / (max_rank - min_rank)) 

categories_means <- heatmap_data_norm %>%
  rowwise() %>%
  mutate(mean_score = normalized_composite) %>%  # use normalized performance rank
  ungroup()

group_order <- categories_means %>%
  group_by(model_group) %>%
  summarise(group_mean = mean(mean_score), .groups = "drop") %>%
  arrange(desc(group_mean)) %>%
  pull(model_group)

categories_means$model_group <- factor(categories_means$model_group, levels = group_order)


labels_stats <- categories_means %>%
  group_by(model_group) %>%
  summarise(
    n_models     = n(),
    lowest_model = method[which.min(mean_score)],
    lowest_value = min(mean_score),
    highest_model = method[which.max(mean_score)],
    highest_value = max(mean_score),
    .groups = "drop"
  ) %>%
  # Split long model names into two lines, add '-' and shift only the second line
  mutate(
    lowest_model = ifelse(lowest_model == "mT5_multilingual_XLSum",
                          "mT5_multi-\n   lingual-XLSum",  # '-' added
                          lowest_model),
    highest_model = ifelse(highest_model == "led_large_16384_arxiv_summarization",
                           "led_large_16384-\n   arxiv_summarization",
                           highest_model),
    # Add nudge_y only for bigbird
    nudge_y_lowest = ifelse(lowest_model == "mT5_multi-\n   lingual-XLSum", -0.25, 0),
    nudge_y_highest = ifelse(highest_model == "led_large_16384-\n   arxiv_summarization", -0.15, 0)
  )

nudge <- 0.02

Figure_4a <- ggplot(categories_means, aes(x = mean_score, y = model_group, fill = model_group)) +
  
  
  stat_summary(
    fun.data = function(x) {
      r <- quantile(x, probs = c(0, 0.25, 0.75, 1))
      data.frame(
        ymin   = r[1],
        lower  = r[2],
        middle = mean(x),
        upper  = r[3],
        ymax   = r[4]
      )
    },
    geom = "boxplot",
    width = 0.4,
    color = "gray40",
    alpha = 0.8,
    outlier.shape = NA
  ) +
  
  
  geom_point(
    size = 1,
    color = "black",
    alpha = 0.9,
    show.legend = FALSE
  ) +
  
  geom_text(
    data = labels_stats,
    aes(
      x = lowest_value, 
      y = model_group, 
      label = lowest_model
    ),
    color = "black",
    fontface = "italic",
    vjust = 0,
    hjust = 1,       
    nudge_x = -nudge,
    nudge_y = labels_stats$nudge_y_lowest,
    size = 6
  ) +
  
  geom_text(
    data = subset(labels_stats, n_models > 1),
    aes(
      x = highest_value, 
      y = model_group, 
      label = highest_model
    ),
    color = "black",
    fontface = "italic",
    vjust = 0,
    hjust = 0,       
    nudge_x = nudge,
    nudge_y = subset(labels_stats, n_models > 1)$nudge_y_highest,
    size = 6
  ) +
  
  scale_fill_manual(values = Groups, breaks = names(Groups)) +
  
  scale_x_continuous(
    breaks = c(0, 0.25, 0.5, 0.75, 1),
    limits = c(-0.15, 1.2)
  ) +
  
  labs(x = "Overall Performance Score" , y = "Categories", fill = "Categories") +
  ggtitle("Overall performance of Model Categories") +
  
  theme_minimal(base_size = 18) +
  theme(
    axis.text.y = element_text(face = "bold", size = 16, color = "black"),
    axis.text.x = element_text(size = 16, color = "black"),
    axis.title = element_text(face = "bold"),
    panel.grid.major.x = element_line(color = "gray70"),
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position = "none",
    plot.title = element_text(face = "bold", size = 18, hjust = 0.5),
    panel.border = element_rect(color = "black", fill = NA, size = 1)
  )

ggsave(
  filename = file.path(Outcome, "Figure_4a.pdf"),
  plot = Figure_4a,
  width = 18,
  height = 10
)

#### Figure 4c ####


family_means <- heatmap_data_norm %>%
  rowwise() %>%
  mutate(mean_score = normalized_composite) %>%
  ungroup()


family_order <- family_means %>%
  group_by(model_family) %>%
  summarise(family_mean = mean(mean_score), .groups = "drop") %>%
  arrange(desc(family_mean)) %>%
  pull(model_family)

family_means$model_family <- factor(family_means$model_family, levels = family_order)

labels_stats <- family_means %>%
  group_by(model_family) %>%
  summarise(
    n_models     = n(),
    lowest_model = method[which.min(mean_score)],
    lowest_value = min(mean_score),
    highest_model = method[which.max(mean_score)],
    highest_value = max(mean_score),
    .groups = "drop"
  ) %>%
  # Only apply special formatting for bigbird-pegasus-large-pubmed
  mutate(
    lowest_model = ifelse(
      lowest_model == "bigbird-pegasus-large-pubmed",
      "bigbird-pegasus-\n   large-pubmed",
      lowest_model
    ),
    highest_model = ifelse(
      highest_model == "bigbird-pegasus-large-pubmed",
      "bigbird-pegasus-\n   large-pubmed",
      highest_model
    ),
    nudge_y_lowest = ifelse(lowest_model == "bigbird-pegasus-\n   large-pubmed", -0.35, 0),
    nudge_y_highest = ifelse(highest_model == "bigbird-pegasus-\n   large-pubmed", -0.15, 0)
  )

nudge <- 0.02

Figure_4c <- ggplot(family_means, aes(x = mean_score, y = model_family, fill = model_family)) +
  
  stat_summary(
    fun.data = function(x) {
      r <- quantile(x, probs = c(0, 0.25, 0.75, 1))
      data.frame(
        ymin   = r[1],
        lower  = r[2],
        middle = mean(x),
        upper  = r[3],
        ymax   = r[4]
      )
    },
    geom = "boxplot",
    width = 0.4,
    color = "gray40",
    alpha = 0.8,
    outlier.shape = NA
  ) +
  
  geom_point(
    size = 1,
    color = "black",
    alpha = 0.9,
    show.legend = FALSE
  ) +
  
  geom_text(
    data = labels_stats,
    aes(x = lowest_value, y = model_family, label = lowest_model),
    color = "black",
    fontface = "italic",
    vjust = 0,
    hjust = 1,
    nudge_x = -nudge,
    nudge_y = labels_stats$nudge_y_lowest,
    size = 6
  ) +
  
  geom_text(
    data = subset(labels_stats, n_models > 1),
    aes(x = highest_value, y = model_family, label = highest_model),
    color = "black",
    fontface = "italic",
    vjust = 0,
    hjust = 0,
    nudge_x = nudge,
    nudge_y = subset(labels_stats, n_models > 1)$nudge_y_highest,
    size = 6
  ) +
  
  scale_fill_manual(values = Families, breaks = names(Families)) +
  
  scale_x_continuous(
    breaks = c(0, 0.25, 0.5, 0.75, 1),
    limits = c(-0.15, 1.1)
  ) +
  
  labs(
    x = "Overall Performance Score",
    y = "Families",
    fill = "Family"
  ) +
  ggtitle("Overall Performance of Model Families") +
  
  # Theme
  theme_minimal(base_size = 18) +
  theme(
    axis.text.y = element_text(face = "bold", size = 16, color = "black"),
    axis.text.x = element_text(size = 16, color = "black"),
    axis.title  = element_text(face = "bold"),
    panel.grid.major.x = element_line(color = "gray70"),
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position = "none",
    plot.title = element_text(face = "bold", size = 18, hjust = 0.5),
    panel.border = element_rect(color = "black", fill = NA, size = 1)
  )

ggsave(
  filename = file.path(Outcome, "Figure_4c.pdf"),
  plot = Figure_4c,
  width = 24,
  height = 12
)



#### Figure 4b #####


gh_df <- games_howell_pvalues_matrix_categories

# Convert wide → long format
gh_long <- gh_df %>%
  pivot_longer(
    cols = -Categories,
    names_to = "Comparison",
    values_to = "p_value"
  )

gh_long$Categories <- factor(gh_long$Categories, levels = gh_df$Categories)
gh_long$Comparison <- factor(gh_long$Comparison, levels = rev(names(gh_df)[-1]))

cat_order <- gh_df$Categories
comp_order <- names(gh_df)[-1]

gh_long <- gh_long %>%
  mutate(
    cat_index = match(Categories, cat_order),
    comp_index = match(as.character(Comparison), comp_order),
    keep_tile = (cat_index <= comp_index)  # Upper triangle + diagonal
  ) %>%
  filter(keep_tile)  # **Filter only upper triangle**

format_pval <- function(p) {
  ifelse(p < 0.001, "<0.001", sprintf("%.3g", p))
}

Figure_4b <- ggplot(gh_long, aes(x = Comparison, y = Categories, fill = p_value)) +
  geom_tile(color = "black", size = 0.7) +
  
  geom_text(aes(
    label = format_pval(p_value),
    color = ifelse(p_value > 0.5, "white", "black")  # p > 0.05 → white
  ),
  size = 6, fontface = "bold"
  ) +
  scale_color_identity() +
  
  scale_fill_gradientn(
    colours = c("#F2F2F2", "#7F7F7F", "#FEF0F0", "#660000"),   # grey → white → red
    values = scales::rescale(c(0, 0.05, 0.051, 1)),            # sharp transition at 0.05
    limits = c(0, 1),
    name = "p-value",
    breaks = c(0, 0.05, 1),
    labels = c("0", "0.05", "1"),
    guide = guide_colorbar(
      ticks = TRUE,
      ticks.colour = "black",
      frame.colour = "black",
      barheight = unit(8, "cm"),
      nbin = 100,
      label.position = "right"
    )
  ) +
  
  labs(
    title = "Games–Howell Post-hoc Test across Model Categories",
    x = NULL,
    y = NULL
  ) +
  
  theme_minimal(base_size = 18) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, face = "bold", colour = "black"),
    axis.text.y = element_text(face = "bold", colour = "black"),
    plot.title = element_text(face = "bold", size = 18, hjust = 0.5, colour = "#222222"),
    panel.grid = element_blank(),
    panel.background = element_rect(fill = "white", color = NA),
    plot.background = element_rect(fill = "white", color = NA),
    legend.position = "right",
    legend.title = element_text(face = "bold")
  )

ggsave(
  filename = file.path(Outcome, "Figure_4b.pdf"),
  plot = Figure_4b,
  width = 13,
  height = 9
)


###Figure 4d #####


gh_df <- games_howell_pvalues_matrix_families

# Convert wide → long format
gh_long <- gh_df %>%
  pivot_longer(
    cols = -Families,
    names_to = "Comparison",
    values_to = "p_value"
  )

gh_long$Families <- factor(gh_long$Families, levels = gh_df$Families)
gh_long$Comparison <- factor(gh_long$Comparison, levels = rev(names(gh_df)[-1]))

cat_order <- gh_df$Families
comp_order <- names(gh_df)[-1]

gh_long <- gh_long %>%
  mutate(
    cat_index = match(Families, cat_order),
    comp_index = match(as.character(Comparison), comp_order),
    keep_tile = (cat_index <= comp_index)  # Upper triangle + diagonal
  ) %>%
  filter(keep_tile)  # **Filter only upper triangle**

format_pval <- function(p) {
  ifelse(p < 0.001, "<0.001", sprintf("%.3g", p))
}

Figure_4d <- ggplot(gh_long, aes(x = Comparison, y = Families, fill = p_value)) +
  geom_tile(color = "black", size = 0.7) +
  
  geom_text(aes(
    label = format_pval(p_value),
    color = ifelse(p_value > 0.5, "white", "black")  
  ),
  size = 6, fontface = "bold"
  ) +
  scale_color_identity() +
  
  scale_fill_gradientn(
    colours = c("#F2F2F2", "#7F7F7F", "#FEF0F0", "#660000"),  
    values = scales::rescale(c(0, 0.05, 0.051, 1)),            
    limits = c(0, 1),
    name = "p-value",
    breaks = c(0, 0.05, 1),
    labels = c("0", "0.05", "1"),
    guide = guide_colorbar(
      ticks = TRUE,
      ticks.colour = "black",
      frame.colour = "black",
      barheight = unit(8, "cm"),
      nbin = 100,
      label.position = "right"
    )
  ) +
  
  labs(
    title = "Games–Howell Post-hoc Test across Model Families",
    x = NULL,
    y = NULL
  ) +
  
  theme_minimal(base_size = 18) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, face = "bold", colour = "black"),
    axis.text.y = element_text(face = "bold", colour = "black"),
    plot.title = element_text(face = "bold", size = 18, hjust = 0.5, colour = "#222222"),
    panel.grid = element_blank(),
    panel.background = element_rect(fill = "white", color = NA),
    plot.background = element_rect(fill = "white", color = NA),
    legend.position = "right",
    legend.title = element_text(face = "bold")
  )

ggsave(
  filename = file.path(Outcome, "Figure_4d.pdf"),
  plot = Figure_4d,
  width = 16,
  height = 10
)

##### Figure 2 #####


metric_category <- c(
  `ROUGE-1`           = "Lexical",
  `ROUGE-2`           = "Lexical",
  `ROUGE-L`           = "Lexical",
  METEOR              = "Lexical",
  BLEU                = "Lexical",
  RoBERTa             = "Semantic",
  DeBERTa             = "Semantic",
  `all-mpnet-base-v2` = "Semantic",
  AlignScore          = "Factual",
  SummaC              = "Factual",
  `MiniCheck-FT5`     = "Factual",
  `MiniCheck-7B`      = "Factual"
)

category_colors <- c(
  Lexical  = "#D77E61",
  Semantic = "#91A4A3",
  Factual  = "#E4E8E8"
)

# Convert wide → long format
corr_long <- metric_correlation_spearman %>%
  pivot_longer(cols = -Metrics, names_to = "Comparison", values_to = "correlation")

metrics_order <- metric_correlation_spearman$Metrics
corr_long$Metrics    <- factor(corr_long$Metrics,    levels = metrics_order)
corr_long$Comparison <- factor(corr_long$Comparison, levels = metrics_order)

corr_long <- corr_long %>%
  mutate(
    metrics_index = match(Metrics, metrics_order),
    comp_index    = match(as.character(Comparison), metrics_order),
    keep_tile     = (metrics_index >= comp_index)
  ) %>%
  filter(keep_tile)

x_cat <- data.frame(
  Comparison = factor(metrics_order, levels = metrics_order),
  category   = metric_category[metrics_order],
  y          = -0.3
)

y_cat <- data.frame(
  Metrics  = factor(metrics_order, levels = metrics_order),
  category = metric_category[metrics_order],
  x        = -0.3
) %>%
  # index that matches the FLIPPED y-axis (rev(metrics_order))
  mutate(
    Metrics_rev_idx = as.numeric(factor(Metrics, levels = rev(metrics_order)))
  )

x_cat_label <- x_cat %>%
  group_by(category) %>%
  summarise(x_center = mean(as.numeric(Comparison)), .groups = "drop")

y_cat_label <- y_cat %>%
  group_by(category) %>%
  summarise(y_center = mean(Metrics_rev_idx), .groups = "drop")

min_corr <- min(corr_long$correlation, na.rm = TRUE)  # ~ -0.56

Figure_2 <- ggplot() +
  geom_tile(
    data = corr_long,
    aes(x = Comparison, y = Metrics, fill = correlation),
    color = "black", size = 0.7
  ) +
  geom_text(
    data = corr_long,
    aes(
      x = Comparison, y = Metrics,
      label = sprintf("%.2f", correlation),
      color = ifelse(correlation > 0.69, "white", "black")
    ),
    size = 5, fontface = "bold"
  ) +
  scale_color_identity() +
  scale_fill_gradientn(
    colours = c("#91A4A3", "#FFFFFF", "#660000"),   
    values  = scales::rescale(c(min_corr, 0, 1)),   
    limits  = c(min_corr, 1),
    name    = expression(Spearman~rho),
    breaks  = c(min_corr, 0, 0.5, 0.8, 1),
    labels  = c(
      sprintf("%.2f", min_corr), "0", "0.5", "0.8", "1"
    ),
    guide   = guide_colorbar(barheight = unit(8, "cm"))
  ) + 
  # X-axis category bar (top)
  geom_tile(
    data = x_cat,
    aes(x = Comparison, y = -0.3),
    inherit.aes = FALSE,
    height = 0.3,
    fill = NA,
    color = NA
  ) +
  geom_rect(
    data = x_cat,
    inherit.aes = FALSE,
    aes(
      xmin = as.numeric(Comparison) - 0.5,
      xmax = as.numeric(Comparison) + 0.5,
      ymin = -0.3 - 0.25,
      ymax = -0.3 + 0.25
    ),
    fill  = category_colors[x_cat$category],
    color = NA
  ) +
  geom_text(
    data = x_cat_label,
    aes(x = x_center, y = -0.3, label = category),
    inherit.aes = FALSE,
    fontface = "bold",
    size = 5   
  ) +
  
  # Y-axis category bar (left) – use Metrics_rev_idx to align with flipped y
  geom_rect(
    data = y_cat,
    inherit.aes = FALSE,
    aes(
      xmin = -0.3 - 0.2,
      xmax = -0.3 + 0.2,
      ymin = Metrics_rev_idx - 0.5,
      ymax = Metrics_rev_idx + 0.5
    ),
    fill  = category_colors[y_cat$category],
    color = NA
  ) +
  geom_text(
    data = y_cat_label,
    aes(x = -0.3, y = y_center, label = category),
    angle = 90,
    inherit.aes = FALSE,
    fontface = "bold",
    size = 5   
  ) +
  
  labs(
    title = "Spearman Correlation between Metrics",
    x = NULL, y = NULL
  ) +
  theme_minimal(base_size = 18) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, face = "bold", colour = "black"),
    axis.text.y = element_text(face = "bold", colour = "black"),
    plot.title  = element_text(face = "bold", size = 18, hjust = 0.5, colour = "#222222"),
    panel.grid  = element_blank(),
    panel.background = element_rect(fill = "white", color = NA),
    plot.background  = element_rect(fill = "white", color = NA),
    legend.position  = "right",
    legend.title     = element_text(face = "bold")
  ) +
  scale_y_discrete(limits = rev(metrics_order))

Figure_2

ggsave(
  filename = file.path(Outcome, "Figure_2.pdf"),
  plot = Figure_2,
  width = 13,
  height = 9
)
