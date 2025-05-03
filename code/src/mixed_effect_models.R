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
                            n_components = NULL,
                            overwrite = FALSE) {
  if (!is.null(n_components)) {
    category <- sprintf("%d_components", n_components)
  }
  else {
    category <- "body-heart-mind" # Categories from Weisman (2017)
  }
  if (!is.null(save_dir)) {
    save_results_path <- sprintf("%s/%s_results.txt", save_dir, category)
    if (file.exists(save_results_path) && !overwrite) {
      print(sprintf("File exists at %s and not overwriting.", save_results_path))
      return()
    }
    cat("Saving results to", save_results_path, "\n")
  }
  # Load data frame & column items
  df <- read.csv(df_path)

  # Convert portrayal to categorical factor
  df$portrayal <- factor(df$condition, levels=c("NoVideo", "Machines", "Tools", "Companions"))

  if (save_txt) {
    sink(file = save_results_path)
  }

  # Fit a mixed-effects regression model
  model <- lmer(rating ~ portrayal * category + (1 | pid), data = df)
  cat("\n\n", "Model Summary:", "\n")
  print(summary(model))

  # Post-Hoc Tests
  # Estimated Marginal Means Model
  cat("\n\n", "EMMeans Analysis for portrayals:", "\n")
  emm <-emmeans(model, list(pairwise ~ portrayal), adjust = "tukey")
  print(summary(emm))
  emm_means <- emm[[1]]

  cat("\n", "Mean of baseline, machine, and tool portrayal conditions", "\n")
  print(summary(contrast(emm_means, method = list("Non-companion aggregate" = c(
    NoVideo = 1/3,
    Machines = 1/3,
    Tools = 1/3,
    Companions = 0
  ))), infer = c(TRUE, FALSE)))

  cat("\nNon-companion vs companion differences\n")
  print(summary(contrast(emm_means, method = list("Companion - Noncompanion" = c(
    NoVideo = -1/3,
    Machines = -1/3,
    Tools = -1/3,
    Companions = 1
  ))), infer = c(TRUE, TRUE)))

  cat("\n\n", "EMMeans Analysis for portrayal marginalized over category:", "\n")
  emm_marginalized <- emmeans(model, list(pairwise ~ portrayal | category), adjust = "tukey")
  print(summary(emm_marginalized))

  cat("\n", "Mean of baseline, machine, and tool portrayal conditions marginalized by category", "\n")
  emm_marginalized_means <- emm_marginalized[[1]]
  print(summary(contrast(emm_marginalized_means, method = list("Non-companion aggregate" = c(
    NoVideo = 1 / 3,
    Machines = 1 / 3,
    Tools = 1 / 3,
    Companions = 0
  )), by = "category"), infer = c(TRUE, FALSE)))

  cat("\nNon-companion vs companion differences marginalized by category\n")
  print(summary(contrast(emm_marginalized_means, method = list("Companion - Noncompanion" = c(
    NoVideo = -1 / 3,
    Machines = -1 / 3,
    Tools = -1 / 3,
    Companions = 1
  )), by = "category"), infer = c(TRUE, TRUE)))

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

  # -------------------###-------------------
  # Add column for aggregating video conditions
  df <- df %>% mutate(video = case_when(
    condition == "NoVideo" ~ factor("No video"),
    condition == "Machines" ~ factor("Video"),
    condition == "Tools" ~ factor("Video"),
    condition == "Companions" ~ factor("Video")))

  cat("\n--------------###--------------\n",
      "EMMeans Analysis for video vs no video conditions:", "\n")

  video_model <- lmer(rating ~ video * category + (1 | pid), data = df)
  cat("\n\n", "Model Summary:", "\n")
  print(summary(video_model))

  cat("\n\n", "EMMeans Analysis for video:", "\n")
  video_emm <- emmeans(video_model, list(pairwise ~ video), adjust = "tukey")
  print(summary(video_emm))

  # Baseline model without interaction
  no_interaction_video_model <- lmer(rating ~ video + category + (1 | pid), data = df)

  # Baseline model with category only
  category_model <- lmer(rating ~ category + (1 | pid), data = df)

  # Null model without the condition
  null_model <- lmer(rating ~ (1 | pid), data = df)

  cat("ANOVA with category model", "\n")
  print(anova(null_model, category_model, no_interaction_video_model, video_model))

  if (save_txt) {
    sink(file = NULL)
  }
}

# item_level_rating_analysis <- function(df_path,
#                                        save_dir,
#                                        save_txt = TRUE,
#                                        overwrite = FALSE) {
#   if (!is.null(save_dir)) {
#     save_results_path <- sprintf("%s/results.txt", save_dir)
#     if (file.exists(save_results_path) && !overwrite) {
#       print(sprintf("File exists at %s and not overwriting.", save_results_path))
#       return()
#     }
#     cat("Saving results to", save_results_path, "\n")
#   }
#   df <- read.csv(df_path)
#   # cols <- unique(df$item)

#   # Convert condition to categorical factor called portrayal
#   df$portrayal <- factor(df$condition, levels=c("NoVideo", "Machines", "Tools", "Companions"))

#   if (save_txt) {
#     sink(file = save_results_path)
#   }

#   # Create model using just item and portrayal
#   model <- lmer(rating ~ portrayal * item + (1 | pid), data = df)
#   cat("\n\n", "Model Summary:", "\n")
#   print(summary(model))

#   # Post-Hoc Tests
#   cat("\n\n", "EMMeans Analysis for portrayal:", "\n")
#   print(emmeans(model, list(pairwise ~ portrayal), adjust = "tukey"))
#   cat("\n\n", "EMMeans Analysis for portrayal marginalized over item:", "\n")
#   print(emmeans(model, list(pairwise ~ portrayal | item), adjust = "tukey"))

#   # Baseline models
#   no_interaction_model <- lmer(rating ~ portrayal + item + (1 | pid), data = df)
#   no_item_model <- lmer(rating ~ portrayal + (1 | pid), data = df)
#   null_model <- lmer(rating ~ (1 | pid), data=df)
#   cat("ANOVA null -> no interaction -> interaction", "\n")
#   print(anova(null_model, no_item_model, no_interaction_model, model))

#   # Redirect outputs back to console
#   if (save_txt) {
#     sink(file = NULL)
#   }
# }

attitude_analysis <- function(attitude,
                              df_path,
                              save_dir,
                              save_txt = TRUE,
                              overwrite = FALSE) {
  # Load data frame & column items
  df <- read.csv(df_path)

  # Make file to save results
  if (!is.null(save_dir)) {
    save_results_path <- sprintf("%s/%s_results.txt", save_dir, attitude)
    if (file.exists(save_results_path) && !overwrite) {
      print(sprintf("File exists at %s and not overwriting.", save_results_path))
      return()
    }
    cat("Saving results to", save_results_path, "\n")
  }

  # Convert portrayal to categorical factor
  condition_map <- c("Baseline" = "NoVideo", "Mechanistic" = "Machines", "Functional" = "Tools", "Intentional" = "Companions")
  df$portrayal <- condition_map[df$condition]
  df$portrayal <- factor(df$portrayal, levels=c("NoVideo", "Machines", "Tools", "Companions"))

  # Add column for participant ID
  df$pid <- paste0("pid", seq_len(nrow(df)))
  # Rename attitude column to "attitude"
  names(df)[names(df) == attitude] <- "attitude"

  if (save_txt) {
    sink(file = save_results_path)
  }

  # Fit a linear regression model
  model <- lm(attitude ~ 1 + portrayal, data=df)

  cat("---***---","\n\n", attitude, "analysis: ", "\n")
  cat("\n\n", "Model Summary:", "\n")
  print(summary(model))

  # Post-Hoc Tests
  # This is the line that we would report values from
  cat("\n\n", "EMMeans Analysis:", "\n")
  #   Estimated Marginal Means Model
  emm <- emmeans(model, list(pairwise ~ portrayal), adjust = "tukey")
  emm_means <- emm[[1]]
  print(summary(emm))

  # Baseline model without the portrayal
  baseline_model <- lm(attitude ~ 1, data = df)
  cat("\n\n", "Anova Analysis:", "\n")
  print(anova(baseline_model, model))


  # Pairwise comparison: Companions vs Non-companion aggregate
  cat("\n--------------###--------------\n",
    "Non-companion aggregate", "\n")
  print(summary(contrast(emm_means, method = list("Non-companion aggregate" = c(
    NoVideo = 1/3,
    Machines = 1/3,
    Tools = 1/3,
    Companions = 0
  )), infer = c(TRUE, FALSE))))

  cat("\nNon-companion vs companion means\n")
  print(summary(contrast(emm_means, method = list(
    "Companion - Non-companion" = c(
      NoVideo = -1/3,
      Machines = -1/3,
      Tools = -1/3,
      Companions = 1
    )), infer = c(TRUE, TRUE)))) # infer = (show_CIs, show_pvals)

  # Add video vs no video column in dataframe
  df <- df %>%
    mutate(video = case_when(
      portrayal == "NoVideo" ~ factor("No video"),
      portrayal == "Machines" ~ factor("Video"),
      portrayal == "Tools" ~ factor("Video"),
      portrayal == "Companions" ~ factor("Video")))
  cat("\n--------------###--------------\n",
      "EMMeans Analysis for video vs no video conditions:", "\n")

  video_model <- lm(attitude ~ 1 + video, data=df)
  cat("\n", "Model Summary:", "\n")
  print(summary(video_model))

  cat("\n", "EMMeans Analysis for video:", "\n")
  emm <- emmeans(video_model, list(pairwise ~ video), adjust = "tukey")
  print(summary(emm))

  cat("\n", "Anova Analysis for video:", "\n")
  print(anova(baseline_model, video_model))

  # Redirect outputs back to console
  if (save_txt) {
    sink(file = NULL)
  }
}

mentioned_items_analysis <- function(df_path,
                                     fa_df_path,
                                     save_dir,
                                     save_txt = TRUE) {
  # Load data frame & column items
  df <- read.csv(df_path)

  # Make file to save results
  save_results_path <- sprintf("%s/results.txt", save_dir)
  cat("Saving results to", save_results_path, "\n")
  # Convert portrayal to categorical factor
  df$portrayal <- factor(df$condition, levels=c("NoVideo", "Machines", "Tools", "Companions"))
  df$category <- factor(df$category, levels = c("unmentioned", "mentioned"))

  if (save_txt) {
    sink(file = save_results_path)
  }

  # Read fa_df_path & add column for factor groupings
  fa_df <- read.csv(fa_df_path)
  fa_df <- fa_df %>% rename(factor = category)
  # Rename condition -> portrayal
  names(fa_df)[names(fa_df) == "condition"] <- "portrayal"
  # Add factor labels to main df
  df <- left_join(df, fa_df, by = c("portrayal", "pid", "item", "rating"))

  cat("\n--------------###--------------\n",
    "Perform ANOVA only on items that were not mentioned", "\n")

  # Only keep items that were unmentioned
  unmentioned_df <- df[df$category == "unmentioned", ]

  unmentioned_model <- lmer(rating ~ portrayal * factor + (1 | pid), data = unmentioned_df)
  cat("\n", "Model Summary:", "\n")
  print(summary(unmentioned_model))

  # Baseline model without interaction
  unmentioned_no_interaction_model <- lmer(rating ~ portrayal + factor + (1 | pid), data = unmentioned_df)

  # Baseline model with item only
  unmentioned_item_model <- lmer(rating ~ factor + (1 | pid), data = unmentioned_df)

  # Baseline model with condition only
  unmentioned_condition_model <- lmer(rating ~ portrayal + (1 | pid), data = unmentioned_df)

  # Null model without the condition
  unmentioned_null_model <- lmer(rating ~ (1 | pid), data = unmentioned_df)

  # Nested model comparison
  cat("\n[Unmentioned] ANOVA with item model", "\n")
  print(anova(unmentioned_null_model, unmentioned_item_model,
    unmentioned_no_interaction_model, unmentioned_model))

  cat("\n\n", "[Unmentioned] EMMeans Analysis for portrayal:", "\n")
  print(emmeans(unmentioned_model, list(pairwise ~ portrayal), adjust = "tukey"))

  if (save_txt) {
    sink(file = NULL)
  }
}