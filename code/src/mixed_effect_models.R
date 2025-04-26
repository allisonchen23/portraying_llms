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
    category <- "body-heart-mind"
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

item_level_rating_analysis <- function(df_path,
                                       save_dir,
                                       save_txt = TRUE,
                                       overwrite = FALSE) {
  if (!is.null(save_dir)) {
    save_results_path <- sprintf("%s/results.txt", save_dir)
    if (file.exists(save_results_path) && !overwrite) {
      print(sprintf("File exists at %s and not overwriting.", save_results_path))
      return()
    }
    cat("Saving results to", save_results_path, "\n")
  }
  df <- read.csv(df_path)
  # cols <- unique(df$item)

  # Convert condition to categorical factor called portrayal
  df$portrayal <- factor(df$condition, levels=c("Baseline", "Mechanistic", "Functional", "Intentional"))

  if (save_txt) {
    sink(file = save_results_path)
  }

  # Create model using just item and portrayal
  model <- lmer(rating ~ portrayal * item + (1 | pid), data = df)
  cat("\n\n", "Model Summary:", "\n")
  print(summary(model))

  # Post-Hoc Tests
  cat("\n\n", "EMMeans Analysis for portrayal:", "\n")
  print(emmeans(model, list(pairwise ~ portrayal), adjust = "tukey"))
  cat("\n\n", "EMMeans Analysis for portrayal marginalized over item:", "\n")
  print(emmeans(model, list(pairwise ~ portrayal | item), adjust = "tukey"))

  # Baseline models
  no_interaction_model <- lmer(rating ~ portrayal + item + (1 | pid), data = df)
  no_item_model <- lmer(rating ~ portrayal + (1 | pid), data = df)
  null_model <- lmer(rating ~ (1 | pid), data=df)
  cat("ANOVA null -> no interaction -> interaction", "\n")
  print(anova(null_model, no_item_model, no_interaction_model, model))

  # Redirect outputs back to console
  if (save_txt) {
    sink(file = NULL)
  }
}

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
  df$portrayal <- factor(df$condition, levels=c("Baseline", "Mechanistic", "Functional", "Intentional"))

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
  # Estimated Marginal Means Model
  print(emmeans(model, list(pairwise ~ portrayal), adjust = "tukey"))

  # Baseline model without the portrayal
  baseline_model <- lm(attitude ~ 1, data = df)
  cat("\n\n", "Anova Analysis:", "\n")
  print(anova(baseline_model, model))

  # Redirect outputs back to console
  if (save_txt) {
    sink(file = NULL)
  }
}

mentioned_items_analysis <- function(df_path,
                                     save_dir,
                                     save_txt = TRUE) {
  # Load data frame & column items
  df <- read.csv(df_path)

  # Make file to save results
  save_results_path <- sprintf("%s/results.txt", save_dir)
  cat("Saving results to", save_results_path, "\n")
  # Convert portrayal to categorical factor
  df$portrayal <- factor(df$condition, levels=c("Baseline", "Mechanistic", "Functional", "Intentional"))
  df$category <- factor(df$category, levels = c("unmentioned", "mentioned"))

  # Select only baseline and intentional conditions
  df <- df[df$portrayal %in% c("Baseline", "Intentional"), ]

  if (save_txt) {
    sink(file = save_results_path)
  }

  # Define mixed effects model
  model <- lmer(rating ~ portrayal * category + (1 | pid), data = df)
  cat("\n\n", "Model Summary:", "\n")
  print(summary(model))

  # Estimated Marginal Means Model
  cat("\n\n", "EMMeans Analysis for portrayal:", "\n")
  print(emmeans(model, list(pairwise ~ portrayal), adjust = "tukey"))
  cat("\n\n", "EMMeans Analysis for portrayal marginalized over category:", "\n")
  print(emmeans(model, list(pairwise ~ portrayal | category), adjust = "tukey"))
  # EMMeans for category
  cat("\n", "EMMeans for category", "\n")
  print(emmeans(model, list(pairwise ~ category), adjust = "tukey"))
  # EMMeans for interaction
  # cat("\n\n", "EMMeans Analysis for interaction:", "\n")
  # emmeans(model, list(pairwise ~ portrayal * category), adjust = "tukey")

  # Baseline model without interaction
  no_interaction_model <- lmer(rating ~ portrayal + category + (1 | pid), data = df)

  # Baseline model with category only
  category_model <- lmer(rating ~ category + (1 | pid), data = df)

  # Null model without the portrayal condition
  null_model <- lmer(rating ~ (1 | pid), data = df)

  # Nested model comparison

  cat("ANOVA with category model", "\n")
  print(anova(null_model, category_model, no_interaction_model, model))

  if (save_txt) {
    sink(file = NULL)
  }
}