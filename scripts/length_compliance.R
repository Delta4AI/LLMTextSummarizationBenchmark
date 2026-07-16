# ---------------------------------------------------------
# Summary Length Compliance - Stacked Bar (Supplementary)
# LLM Text Summarization Benchmark
# Per-method breakdown of summaries that fall within the
# target word range, are too short, or are too long.
# ---------------------------------------------------------

packages <- c("readr", "dplyr", "tidyr", "ggplot2", "here")

# Install into a writable personal library so this works without admin rights
# (the system library under Program Files is read-only on most Windows setups).
user_lib <- Sys.getenv("R_LIBS_USER")
if (user_lib == "" || is.na(user_lib)) {
  user_lib <- file.path(path.expand("~"), "R-libs")
}
if (!dir.exists(user_lib)) dir.create(user_lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(user_lib, .libPaths()))

installed <- packages %in% rownames(installed.packages())
if (any(!installed)) {
  install.packages(packages[!installed], lib = user_lib, repos = "https://cloud.r-project.org")
}

invisible(lapply(packages, library, character.only = TRUE))


# config
GH_HASH <- "1362b291718b57188a7909f08de26da760a0b9346d52111c97671d97d713af38"
base_dir <- here("Output", "llm_summarization_benchmark", GH_HASH)
Outcome <- base_dir

# Per-method length compliance, exported from the benchmark visualizer.
# This CSV is written into the SAME directory as length_analysis.html, i.e.
# benchmark.hashed_and_dated_output_dir. That dir may carry a date suffix and
# therefore differ from the violin script's hash-only path above - verify
# against the path the export prints, and hardcode an absolute path if needed.
# Expected columns: method, within_bounds_pct, too_short_pct, too_long_pct (each 0-100)
DATA_PATH <- file.path(base_dir, "length_compliance.csv")

MIN_WORDS <- 15
MAX_WORDS <- 100

# Compliance colors from the manuscript's RdYlBu/RdYlGn family, chosen to stay
# colorblind-distinguishable: within = green, too short = blue, too long = amber.
# For a traffic-light look instead, set "Too Long" to "#D73027" (red).
compliance_colors <- c(
  "Within Bounds" = "#1A9750",
  "Too Short"     = "#4575B4",
  "Too Long"      = "#FDAE61"
)


# Shorten method names to match the convention used in the other figures:
# drop the provider prefix and the org path, keep the model id; drop the
# date stamps on Anthropic/OpenAI models but keep Mistral's version numbers.
shorten_method <- function(name) {
  x <- name
  x <- sub("^local:", "", x)
  x <- sub("^huggingface:[a-z]+_", "", x)            # huggingface:chat_ / :completion_ / :conversational_
  x <- sub("^huggingface_", "", x)
  x <- sub("^(ollama|openai|mistral|anthropic)_", "", x)
  x <- sub("^.*/", "", x)                            # drop org path (everything up to last "/")
  x <- sub("-[0-9]{8}$", "", x)                      # drop 8-digit date  (claude-*-20250514)
  x <- sub("-[0-9]{4}-[0-9]{2}-[0-9]{2}$", "", x)    # drop YYYY-MM-DD     (gpt-5-*-2025-08-07)
  x <- sub("^Apertus-8B-Instruct-2509$", "Apertus-8B-Instruct", x)  # lone 4-digit stamp to drop
  x
}


# load + reshape
length_data <- read_csv(DATA_PATH, show_col_types = FALSE) %>%
  filter(method != "mistral_mistral-medium-2508") %>%   # stray run, not part of the documented 62
  mutate(method_label = shorten_method(method))

# order methods by compliance (most compliant ends up on top)
method_order <- length_data %>%
  arrange(within_bounds_pct) %>%
  pull(method_label)

length_long <- length_data %>%
  select(method_label, within_bounds_pct, too_short_pct, too_long_pct) %>%
  pivot_longer(
    cols      = c(within_bounds_pct, too_short_pct, too_long_pct),
    names_to  = "category",
    values_to = "percentage"
  ) %>%
  mutate(
    category = recode(category,
      within_bounds_pct = "Within Bounds",
      too_short_pct     = "Too Short",
      too_long_pct      = "Too Long"
    ),
    category     = factor(category, levels = c("Within Bounds", "Too Short", "Too Long")),
    method_label = factor(method_label, levels = method_order)
  )

# within-bounds value as a clean number column down the right edge
wb_labels <- length_data %>%
  mutate(method_label = factor(method_label, levels = method_order),
         label = sprintf("%.0f%%", within_bounds_pct))


# plot
n_methods <- nrow(length_data)

p <- ggplot(length_long, aes(x = percentage, y = method_label, fill = category)) +
  geom_col(
    position  = position_stack(reverse = TRUE),  # anchor "Within Bounds" at x = 0
    width     = 0.72,
    color     = "gray25",
    linewidth = 0.15
  ) +
  geom_text(
    data = wb_labels,
    aes(x = 102, y = method_label, label = label),
    inherit.aes = FALSE,
    hjust = 0, size = 2.6, fontface = "bold", color = "gray20"
  ) +
  scale_fill_manual(values = compliance_colors, name = NULL) +
  scale_x_continuous(
    breaks = seq(0, 100, 25),
    labels = function(x) paste0(x, "%"),
    expand = expansion(mult = c(0, 0))
  ) +
  coord_cartesian(xlim = c(0, 100), clip = "off") +
  labs(
    x = "Share of summaries",
    y = NULL
  ) +
  theme_minimal(base_size = 14) +
  theme(
    axis.text.x        = element_text(size = 9, color = "black"),
    axis.text.y        = element_text(face = "bold", size = 7, color = "black"),
    axis.title.x       = element_text(face = "bold", size = 11),
    panel.grid.major.y = element_blank(),
    panel.grid.minor   = element_blank(),
    panel.grid.major.x = element_line(color = "gray85", linewidth = 0.3),
    panel.border       = element_rect(color = "black", fill = NA, linewidth = 0.8),
    legend.position    = "top",
    legend.text        = element_text(size = 10),
    plot.margin        = margin(t = 8, r = 34, b = 8, l = 8)  # room for the right-edge % column
  )

# save vector (PDF) + raster (PNG) from the same plot
fig_height <- 2.2 + 0.17 * n_methods   # auto-scales to the method count

ggsave(
  filename = file.path(Outcome, "length_compliance.pdf"),
  plot     = p,
  width    = 7,
  height   = fig_height
)

ggsave(
  filename = file.path(Outcome, "length_compliance.png"),
  plot     = p,
  width    = 7,
  height   = fig_height,
  dpi      = 300,
  bg       = "white"
)

cat("Saved:", file.path(Outcome, "length_compliance.pdf"), "\n")
cat("Saved:", file.path(Outcome, "length_compliance.png"), "\n")