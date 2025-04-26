library(dplyr)
library(psych)
library(jsonlite)

# Utils
ensure_dir <- function(dirname) {
  if (!dir.exists(dirname)) {
    dir.create(dirname, recursive = TRUE) # recursive = TRUE creates parent directories if needed
    cat("Directory created:", dirname, "\n")
  } else {
    cat("Directory already exists:", dirname, "\n")
  }
}

read_rating_df <- function(rating_df_path) {
  # Read and normalize rating_df
  # Process rating_df for K-fold analysis
  rating_df <- read.csv(rating_df_path)
  items <- readLines("../data/files/mental_capacity_items_R.txt")

  rating_df <- rating_df[, items]
  stopifnot(dim(rating_df) == c(470, 40))

  normalized_ratings <- normalize_data(rating_df)
  return(normalized_ratings)
}

normalize_data <- function(df) {
  # """
  # df : n_participants x n_items array
  # """
  # Calculate factor scores based on modified loadings
  mus <- colMeans(df)
  sigmas <- apply(df, 2, sd) # calculate `sd` over all columns (2 is for columns)

  scaled_df = scale(df, mus, sigmas)
  return(scaled_df)
}

compute_factor_scores <- function(data, loadings) {
  # """
  # Arg(s):
  #     data : unnormalized n_participants x n_items
  #     loadings : n_items x n_factors matrix

  # Utilize Thurstone method of computing factor scores
  #     * https://personality-project.org/r/psych/help/factor.scores.html
  #     * https://pmc.ncbi.nlm.nih.gov/articles/PMC3773873/ (Eq1)

  # """
  data <- normalize_data(data) # n_participants x n_items

  lambda_inv <- solve(cor(data)) # n_items x n_items
  weights <- lambda_inv %*% loadings # n_items x n_factors
  factor_scores <- data %*% weights # n_participants x n_factors
  predictions <- factor_scores %*% t(loadings)

  # Ensure that predictions and data have the same dimensions
  stopifnot(dim(predictions) == dim(data))

  mse <- mean((predictions - data) ^ 2)

  return(list(factor_scores = factor_scores, predictions = predictions))

}

k_fold <- function(rating_df, k, max_n_factors,
                   rotate, fm, scores, seed,
                   save_dir, overwrite) {

  if (!is.null(save_dir)) {
    kfold_metrics_save_path <- sprintf("%s/kfold_metrics.csv", save_dir)
    if (file.exists(kfold_metrics_save_path) && !overwrite) {
      print(sprintf("File exists at %s and not overwriting.", kfold_metrics_save_path))
      return(read.csv(kfold_metrics_save_path))
    }
  }
  fold_size <- ceiling(nrow(rating_df) / k)
  print(sprintf("With %d folds and %d rows, %d rows per fold", k, nrow(rating_df), fold_size))
  # shuffle data
  set.seed(seed)
  shuffled_idxs <- sample(nrow(rating_df))

  metric_df <- data.frame()

  # Iterate through folds
  for (k_idx in seq(1:k)) {
    test_idxs <- shuffled_idxs[(((k_idx - 1) * fold_size) + 1):(k_idx * fold_size)]
    test <- rating_df[test_idxs, ]
    # Exclude test idxs
    train <- rating_df[-test_idxs, ]
    for (n_factors in seq(1:max_n_factors)) {
      fa_results <- fa(r = train,
                       nfactors = n_factors,
                       rotate = rotate,
                       fm = fm,
                       scores = scores)
      if (n_factors == 1) {
        variance_explained <- fa_results$Vaccounted[2, n_factors]
      } else {
        variance_explained <- fa_results$Vaccounted[3, n_factors]
      }

      # Use factor loadings to predict on test set
      loadings <- fa_results$loadings
      scores_result <- compute_factor_scores(
        data = test,
        loadings = loadings
      )
      predictions <- scores_result$predictions

      normalized_test <- normalize_data(test)

      # Calculate fold"s MSE and BIC
      mse <- mean((normalized_test - predictions) ^ 2)

      metric_df <- bind_rows(metric_df,
                             data.frame(fold = k_idx,
                                       n_factors = n_factors,
                                       bic = fa_results[["BIC"]],
                                       mse = mse,
                                       variance_explained = variance_explained))
    }
  }

  if (!is.null(save_dir)) {
    write.csv(metric_df, kfold_metrics_save_path, row.names = FALSE)
    print(sprintf("Saving metric_df to %s", kfold_metrics_save_path))
  }
  return(metric_df)
}

data_driven_fa <- function(rating_df, n_factors, rotate, fm, scores,
                           save_dir = NULL, overwrite = FALSE) {
  if (!is.null(save_dir)) {
    loadings_save_path <- sprintf("%s/loadings.csv", save_dir)
    factor_scores_save_path <- sprintf("%s/factor_scores_T.csv", save_dir)
    predictions_save_path <- sprintf("%s/predictions_T.csv", save_dir)
    metrics_save_path <- sprintf("%s/metrics.json", save_dir)
    if (file.exists(loadings_save_path) &&
          file.exists(factor_scores_save_path) &&
          file.exists(predictions_save_path) &&
          file.exists(metrics_save_path) &&
          !overwrite) {
      cat("Files already exist in :", save_dir, "\n")

      return(list(
        loadings = read.csv(loadings_save_path),
        factor_scores = read.csv(factor_scores_save_path),
        predictions = read.csv(predictions_save_path),
        metrics = fromJSON(metrics_save_path)
      ))
    }
  }
  fa_results <- fa(
    r = rating_df,
    rotate = rotate,
    fm = fm,
    scores = scores,
    nfactors = n_factors
  )

  normalized_ratings <- normalize_data(rating_df)
  predictions <- fa_results$scores %*% t(fa_results$loadings)
  bic <- fa_results[["BIC"]]
  mse <- mean((normalized_ratings - predictions) ^2)
  metrics <- list(
    bic = bic,
    mse = mse,
    variance = fa_results$Vaccounted,
    var_explained = fa_results$Vaccounted[3, n_factors]
  )
  # Save to files
  if (!is.null(save_dir)) {
    write.csv(fa_results$loadings, loadings_save_path)
    write.csv(fa_results$scores, factor_scores_save_path)
    write.csv(predictions, predictions_save_path)
    write_json(metrics, metrics_save_path)
    print(sprintf("Saved files in %s", save_dir))
  }

  return(list(
    loadings = fa_results$loadings,
    factor_scores = fa_results$scores,
    predictions = predictions,
    metrics = metrics
  ))
}