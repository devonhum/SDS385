### SDS 385 - Statistical models for big data ###

### Exercise 1 - Linear Regression ###

library(microbenchmark)
library(Matrix)

## inversion method

# the matrix X contains our explanatory variables in y = Xb + e

inversion_solver <- function (n_obs, n_vars) {
  # create a random matrix for demonstration
  X <- rnorm(n_obs * n_vars, mean = 0, sd = 1)
  X <- matrix(X, nrow = n_obs, ncol = n_vars)
  
  # generate the vector y to be regressed on X
  y <- rnorm(n_obs, mean = 0, sd = 1)
  
  # solve the least squares problem using the inversion method 
  b <- solve(t(X) %*% X) %*% t(X) %*% y
  
  return (b)
}

## QR method

qr_solver <- function (n_obs, n_vars) {
  # generate a random feature matrix X
  X <- rnorm(n_obs * n_vars, mean = 0, sd = 1)
  X <- matrix(X, nrow = n_obs, ncol = n_vars)
  
  # generate an observation vector y
  y <- rnorm(n_obs, mean = 0, sd = 1)
  y <- matrix(y, nrow = n_obs, ncol = n_vars)
  
  # solve the least squares problem using QR decomposition of X
  b <- qr.solve(X, y)
  return (b)
}

## Dealing with sparse matrices in R

inversion_sparse <- function (n_obs, n_vars, sparsity) {
  # define a random feature matrix X with sparsity of 95%
  X <- rnorm(n_obs * n_vars, mean = 0, sd = 1)
  X <- matrix(X, nrow = n_obs, ncol = n_vars)
  mask <- matrix(rbinom(n_obs * n_vars, 1, sparsity), nrow = n_obs)
  X <- X * mask
  X <- Matrix(X, sparse = T) # converts to sparse format
  
  # generate random observations in a vector y
  y <- rnorm(n_obs, mean = 1, sd = 1)
  
  # solve the least squares problem taking advantage of the sparse matrix format
  inv_mat <- solve(t(X) %*% X, sparse = T)
  b <- inv_mat %*% t(X) %*% y
  return (b)
}

## benchmarking the dense matrix solvers on increasing number of variables
var100 <- microbenchmark(inversion_solver(200, 100), qr_solver(200, 100), times = 10); var100
var1000 <- microbenchmark(inversion_solver(2000, 1000), qr_solver(2000, 1000), times = 10); var1000
var2000 <- microbenchmark(inversion_solver(5000, 2000), qr_solver(5000, 200), times = 10); var2000
var5000 <- microbenchmark(inversion_solver(2000, 5000), qr_solver(2000, 5000), times = 10); var5000

## benchmarking the sparse matrix solvers on increasing levels of sparsity and dimension
#sp05 <- microbenchmark(inversion_sparse(150, 50), inversion_solver(150, 50), times = 10); sp05
