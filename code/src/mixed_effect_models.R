library(lmerTest)
library(lme4)
library(dplyr)
library(tidyr)
library(emmeans)
library(pbkrtest)
emm_options(lmerTest.limit = 20000)
emm_options(pbkrtest.limit = 20000)


rating_analysis <- function(df_path,
                            save_dir,
                            save_txt = TRUE,
                            custom_fa = FALSE,
                            n_components = 3) {
  if (custom_fa == TRUE) {
    category <- sprintf("%d_components", n_components)
  }
  else {
    category <- "body-heart-mind"
  }
  save_results_path <- sprintf("%s/%s_results.txt", save_dir, category)
  # Load data frame & column items
  df <- read.csv(df_path)

  # Convert portrayal to categorical factor
  df$portrayal <- factor(df$condition, levels=c("Baseline", "Mechanistic", "Functional", "Intentional"))
  # Fit a mixed-effects regression model
  if (save_txt) {
    sink(file = save_results_path)
  }
  model <- lmer(rating ~ portrayal * category + (1 | pid), data = df)
  cat("\n\n", "Model Summary:", "\n")
  print(summary(model))

  # Post-Hoc Tests
  # Estimated Marginal Means Model
  cat("\n\n", "EMMeans Analysis for portrayals:", "\n")
  print(emmeans(model, list(pairwise ~ portrayal), adjust = "tukey"))
  cat("\n\n", "EMMeans Analysis for portrayals marginalized over category:", "\n")
  print(emmeans(model, list(pairwise ~ portrayal | category), adjust = "tukey"))

  # EMMeans for category
  cat("\n", "EMMeans for category", "\n")
  print(emmeans(model, list(pairwise ~ category), adjust = "tukey"))

  # Baseline model without interaction
  no_interaction_model <- lmer(rating ~ portrayal + category + (1 | pid), data = df)

  # Baseline model with category only
  category_model <- lmer(rating ~ category + (1 | pid), data = df)

  # Null model without the portrayal
  null_model <- lmer(rating ~ (1 | pid), data = df)

  # Nested model comparison
  cat("ANOVA with category model", "\n")
  print(anova(null_model, category_model, no_interaction_model, model))

  if (save_txt) {
    sink(file = NULL)
  }
}

# # Variables
# exp_name <- "pilot_v3_11142024"
# category <- "weisman"
# # exclude_k <- 40
# min_time <- "1.3333"
# manual_exclusions <- 0
# categoryings <- "absolute_study4"
# save_txt <- TRUE
# # Parameters for custom factor analysis categoryings
# custom_fa <- TRUE
# n_components <- 3

# # Don't modify after here
# save_dir <- sprintf("data/%s/min_time_%s/manual_%d/categoryings_%s", exp_name, min_time, manual_exclusions, categoryings)
# # If using custom factor analysis, correct path
# if (custom_fa == TRUE) {
#   category <- sprintf("%d_components", n_components)
#   save_dir <- sprintf("%s/decomposition/factor_analysis/%d_components/data_fa", save_dir, n_components)
# }
# # save_dir <- "debug"
# save_results_path <- sprintf("%s/R_results/%s_results.txt", save_dir, category)


# # Below here should not change
# if (custom_fa == TRUE) {
#   df_path <- sprintf("%s/R_csvs/factor_analysis.csv", save_dir)
# } else {
#   df_path <- sprintf("%s/R_csvs/%s.csv", save_dir, category)
# }

# # items_path <- sprintf("%s/R_csvs/%s_items.txt", save_dir, category)

# # Load data frame & column items
# df <- read.csv(df_path)
# # cols <- readLines(items_path)

# # Convert portrayal to categorical factor
# df$portrayal <- factor(df$portrayal, levels=c("Baseline", "Mechanistic", "Functional", "Intentional"))

# # Means for each portrayal using raw data (can print as sanity checks)
# # mean(apply(df[df$portrayal == "Baseline", cols], 1, mean, na.rm=FALSE))
# # mean(apply(df[df$portrayal == "Mechanistic", cols], 1, mean, na.rm=FALSE))
# # mean(apply(df[df$portrayal == "Functional", cols], 1, mean, na.rm=FALSE))
# # mean(apply(df[df$portrayal == "Intentional", cols], 1, mean, na.rm=FALSE))

# # Fit a mixed-effects regression model
# if (save_txt) {
#   sink(file = save_results_path)
# }
# model <- lmer(rating ~ portrayal * category + (1 | pid), data = df)
# cat("\n\n", "Model Summary:", "\n")
# summary(model)

# # Post-Hoc Tests
# # Usually need to do some sort of adjustment
# # TODO: search how pvalues are adjusted
# # This is the line that we would report values from

# # Estimated Marginal Means Model
# cat("\n\n", "EMMeans Analysis for portrayals:", "\n")
# emmeans(model, list(pairwise ~ portrayal), adjust = "tukey")
# cat("\n\n", "EMMeans Analysis for portrayals marginalized over category:", "\n")
# emmeans(model, list(pairwise ~ portrayal | category), adjust = "tukey")

# # EMMeans for category
# cat("\n", "EMMeans for category", "\n")
# emmeans(model, list(pairwise ~ category), adjust = "tukey")

# # Baseline model without interaction
# no_interaction_model <- lmer(rating ~ portrayal + category + (1 | pid), data = df)
# # cat("\n\n", "No Interaction Model Summary:", "\n")
# # summary(no_interaction_model)
# # Baseline model portrayal
# portrayal_model <- lmer(rating ~ portrayal + (1 | pid), data = df)

# # Baseline model with category only
# category_model <- lmer(rating ~ category + (1 | pid), data = df)

# # Null model without the portrayal
# null_model <- lmer(rating ~ (1 | pid), data = df)

# # Nested model comparison
# # cat("ANOVA with portrayal model", "\n")
# # anova(null_model, portrayal_model, no_interaction_model, model)

# cat("ANOVA with category model", "\n")
# anova(null_model, category_model, no_interaction_model, model)

# if (save_txt) {
#   sink(file = NULL)
# }
