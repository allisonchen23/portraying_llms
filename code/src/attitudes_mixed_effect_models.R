library(lmerTest)
library(lme4)
library(dplyr)
library(tidyr)
library(emmeans)

# Variables
attitude_analysis <- function(attitude,
                              df_path,
                              save_dir,
                              save_txt = TRUE) {
  # Load data frame & column items
  df <- read.csv(df_path)

  # Make file to save results 
  save_results_path <- sprintf("%s/%s_results.txt", save_dir, attitude)

  # Convert condition to categorical factor
  df$condition <- factor(df$condition, levels=c("Baseline", "Mechanistic", "Functional", "Intentional"))

  # Add column for participant ID
  df$pid <- paste0("pid", seq_len(nrow(df)))
  # Rename attitude column to "attitude"
  names(df)[names(df) == attitude] <- "attitude"

  if (save_txt) {
    sink(file = save_results_path)
  }

  # Fit a linear regression model
  model <- lm(attitude ~ 1 + condition, data=df)

  cat("---***---","\n\n", attitude, "analysis: ", "\n")
  cat("\n\n", "Model Summary:", "\n")
  print(summary(model))

  # Post-Hoc Tests
  # This is the line that we would report values from
  cat("\n\n", "EMMeans Analysis:", "\n")
  # Estimated Marginal Means Model
  print(emmeans(model, list(pairwise ~ condition), adjust = "tukey"))

  # Baseline model without the condition
  baseline_model <- lm(attitude ~ 1, data = df)
  cat("\n\n", "Anova Analysis:", "\n")
  print(anova(baseline_model, model))

  # Redirect outputs back to console
  if (save_txt) {
    sink(file = NULL)
  }


}
# attitude <- "se_use"
# exp_name <- "pilot_v3_11142024"
# min_time <- "1.3333"
# manual_exclusions <- 0

# # Below here should not change
# save_dir <- sprintf("data/%s/min_time_%s/manual_%d/groupings_absolute_study4/additional_attitudes", exp_name, min_time, manual_exclusions)
# save_results_path <- sprintf("%s/R_results/%s_results.txt", save_dir, attitude)
# save_txt <- TRUE
# df_path <- sprintf("%s/R_csvs/%s.csv", save_dir, attitude)

# # Load data frame & column items
# df <- read.csv(df_path)

# # Convert condition to categorical factor
# df$condition <- factor(df$condition, levels=c("Baseline", "Mechanistic", "Functional", "Intentional"))

# # Add column for participant ID
# df$pid <- paste0("pid", seq_len(nrow(df)))
# # Rename attitude column to "attitude"
# names(df)[names(df) == attitude] <- "attitude"

# if (save_txt) {
#   sink(file = save_results_path)
# }

# # Fit a linear regression model
# model <- lm(attitude ~ 1 + condition, data=df)

# cat("---***---","\n\n", attitude, "analysis: ", "\n")
# cat("\n\n", "Model Summary:", "\n")
# summary(model)

# # Post-Hoc Tests
# # Usually need to do some sort of adjustment
# # This is the line that we would report values from
# cat("\n\n", "EMMeans Analysis:", "\n")
# # Estimated Marginal Means Model
# emmeans(model, list(pairwise ~ condition), adjust = "tukey")

# # Baseline model without the condition
# baseline_model <- lm(attitude ~ 1, data=df)
# cat("\n\n", "Anova Analysis:", "\n")
# anova(baseline_model, model)

# # Redirect outputs back to console
# if (save_txt) {
#   sink(file = NULL)
# }
