### SDS 385 - Statistical models for big data ###

### Exercise 1 - Linear Regression ###

library(microbenchmark)
library(Matrix)

## inversion method

# the matrix X contains our explanatory variables in y = Xb + e

inversion_solver <- function (X, y) {
  b <- solve(t(X) %*% X) %*% t(X) %*% y # solve the least squares problem using the inversion method 
  return (b)
}

## QR method

qr_solver <- function (X, y) {
  b <- qr.solve(X, y)  # solve the least squares problem using QR decomposition of X
  return (b)
}

## Dealing with sparse matrices in R
# Begin with a dense matrix, and convert to sparse by masking with 0s.
# Then we will demonstrate how to handle such a matrix for illustration.
inversion_sparse <- function (X, y, sparsity) {
  mask <- matrix(rbinom(N*P, 1, sparsity), nrow = N)
  X <- mask * X # generate the sparsity
  X <- Matrix(X, sparse = T) # converts to sparse format
  inv_mat <- solve(t(X) %*% X, sparse = T) # solve the least squares problem taking advantage of the sparse matrix format
  b <- inv_mat %*% t(X) %*% y
  return (b)
}

# Examples 
N <- 5000 # note that N must be > P for the matrix to be non-singular
P <- 1000
X <- rnorm(N*P, mean = 0, sd = 1)
X <- matrix(X, nrow = N, ncol = P)
y <- rnorm(N, mean = 0, sd = 1)
# define global sparse matrices for benchmarking
mask <- matrix(rbinom(N*P, 1, 0.1), nrow = N)
X_sp <- mask * X

## benchmarking the dense matrix solvers on increasing number of variables
var200 <- microbenchmark(inversion_solver(X, y), qr_solver(X, y), times = 10); var200
var500 <- microbenchmark(inversion_solver(X, y), qr_solver(X, y), times = 10); var500
var1000 <- microbenchmark(inversion_solver(X, y), qr_solver(X, y), times = 10); var1000
var5000 <- microbenchmark(inversion_solver(X, y), qr_solver(X, y), times = 10); var5000

## benchmarking the sparse matrix solvers on increasing levels of sparsity
sp_0.05 <- microbenchmark(inversion_sparse(X, y, 0.05), inversion_solver(X_sp, y), times = 10); sp_0.05
sp_0.1 <- microbenchmark(inversion_sparse(X, y, 0.1), inversion_solver(X_sp, y), times = 10); sp_0.1
