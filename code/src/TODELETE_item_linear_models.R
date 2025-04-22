# library(lmerTest)
# library(lme4)
# library(dplyr)
# library(tidyr)
# library(emmeans)
# library(pbkrtest)
# emm_options(lmerTest.limit = 20000)
# emm_options(pbkrtest.limit = 20000)

# # Set local temp dir

# tempdir <- "debug/tmp"
# Sys.setenv(TMPDIR=tempdir)

# # Variables
# exp_name <- "pilot_v3_11142024"
# category <- "colombatto"
# min_time <- "1.3333"
# manual_exclusions <- 0
# groupings <- "absolute_study4"
# save_txt <- TRUE
# # Don't modify after here
# save_dir <- sprintf("data/%s/min_time_%s/manual_%d/groupings_%s", exp_name, min_time, manual_exclusions, groupings)
# # save_dir <- "debug"
# save_results_path <- sprintf("%s/R_results/item_level/results.txt", save_dir)

# # Below here should not change
# df_path <- sprintf("%s/R_csvs/%s.csv", save_dir, category)

# # Load data frame & column items
# df <- read.csv(df_path)
# cols <- unique(df$item)

# # Convert condition to categorical factor
# df$condition <- factor(df$condition, levels=c("Baseline", "Mechanistic", "Functional", "Intentional"))


# if (save_txt) {
#   sink(file = save_results_path)
# }

# model <- lmer(rating ~ condition * item + (1 | pid), data = df)
# cat("\n\n", "Model Summary:", "\n")
# summary(model)

# # Post-Hoc Tests
# cat("\n\n", "EMMeans Analysis for conditions:", "\n")
# emmeans(model, list(pairwise ~ condition), adjust = "tukey")
# cat("\n\n", "EMMeans Analysis for conditions marginalized over item:", "\n")
# emmeans(model, list(pairwise ~ condition | item), adjust = "tukey")

# # Baseline models
# no_interaction_model <- lmer(rating ~ condition + item + (1 | pid), data = df)
# null_model <- lmer(rating ~ condition + (1 | pid), data = df)

# cat("ANOVA null -> no interaction -> interaction", "\n")

# anova(null_model, no_interaction_model, model)

# # Redirect outputs back to console
# if (save_txt) {
#   sink(file = NULL)
# }