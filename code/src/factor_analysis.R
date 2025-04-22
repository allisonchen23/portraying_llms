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

# k_fold_mse <- function(rating_df, k, max_n_factors,
#                        rotate, fm, scores, seed,
#                        save_dir, overwrite, save_loadings = FALSE) {

#   if (!is.null(save_dir)) {
#     metrics_save_path <- sprintf("%s/k_fold_overall_metrics.json", save_dir)
#     fold_metrics_save_path <- sprintf("%s/fold_metrics.csv", save_dir)
#     if (file.exists(metrics_save_path) && 
#       file.exists(fold_metrics_save_path) && !overwrite) {
#       print(sprintf("File exists at %s", metrics_save_path))

#       return(list(
#         metrics = fromJSON(metrics_save_path), 
#         fold_metrics = read.csv(fold_metrics_save_path)))
#     }
#   }
#   fold_size <- ceiling(nrow(rating_df) / k)
#   print(sprintf("With %d folds and %d rows, %d rows per fold", k, nrow(rating_df), fold_size))
#   # shuffle data
#   set.seed(seed)
#   shuffled_idxs <- sample(nrow(rating_df))

#   # Initialize empty lists
#   all_test <- list()
#   all_predictions <- list()
#   metric_df <- data.frame()

#   # Iterate through folds
#   for (k_idx in seq(1:k)) {
#     test_idxs <- shuffled_idxs[(((k_idx - 1) * fold_size) + 1):(k_idx * fold_size)]
#     test <- rating_df[test_idxs, ]
#     # Exclude test idxs
#     train <- rating_df[-test_idxs, ]
#     fa_results <- fa(r = train,
#                      nfactors = n_factors,
#                      rotate = rotate,
#                      fm = fm,
#                      scores = scores)

#     # Get factor scores and predictions
#     loadings <- fa_results$loadings
#     scores_result <- compute_factor_scores(
#       data = test,
#       loadings = loadings
#     )
#     predictions <- scores_result$predictions

#     normalized_test <- normalize_data(test)

#     # Calculate fold"s MSE and BIC
#     bic <- fa_results[["BIC"]]
#     mse <- mean((normalized_test - predictions) ^ 2)
#     metric_df <- bind_rows(
#       metric_df,
#       data.frame(fold = k_idx,
#                  bic = bic,
#                  mse = mse))

#     # Add to lists
#     all_test[[k_idx]] <- normalized_test
#     all_predictions[[k_idx]] <- predictions

#     # Save loadings
#     if (save_loadings) {
#       loading_save_path <- sprintf("%s/loadings_fold%d.csv", save_dir, k_idx)
#       print(sprintf("Saving loadings to %s", loading_save_path))
#       write.csv(loadings, loading_save_path)
#     }
#   }

#   # Concatenate along rows
#   all_test <- do.call(rbind, all_test)
#   all_predictions <- do.call(rbind, all_predictions)

#   mse <- mean((all_test - all_predictions) ^2)

#   metrics <- list(
#     mse = mse
#   )
#   if (!is.null(save_dir)) {
#     write(toJSON(metrics), metrics_save_path)
#     print(sprintf("Saving metrics to %s", metrics_save_path))

#     write.csv(metric_df, fold_metrics_save_path)
#     print(sprintf("Saving fold-level metrics to %s", fold_metrics_save_path))
#   }
#   return(list(metrics = metrics, fold_metrics = metric_df))
# }

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