### SDS 385 - Exercise 1 ####

### Logistic Regression ###

library(microbenchmark)
library(Matrix)

# define the link function that makes 
sigmoid <- function (z) {
  g <- 1 / (1 + exp(-z))
  return(g)
}

# function to compute the likelihood of some beta
likelihood <- function (beta, X, y, m) {
  m <- nrow(X)
  g <- sigmoid(X %*% beta)
  J <- -(1/m) * (t(y) %*% log(g)) + (t(1 - y) %*% log(1 - g))
  return(J)
}

# function to compute the gradient of the logistic regression likelihood model
grad <- function (beta, X, y) {
  gradient <- t(sigmoid(X %*% beta) - y) %*% X
  return (gradient)
}


# gradient descent algorithm to minimize the objective function
gradient_descent <- function (alpha, iterations, beta, X, y) {
  beta_cols <- ncol(X)
  beta <- matrix(0, nrow = beta_cols)
  conv_diag <- rep(NA, iterations)
  for (i in 1:iterations) {
    beta <- beta - matrix(alpha * grad(beta, X, y),nrow = beta_cols)
    conv_diag[i] <- likelihood(beta, X, y, 1)
  }
  return (beta)
}

# function to compute the hessian matrix of the logistic function for X and beta
hessian <- function (beta, X, m) {
  w<-as.vector(1 / (1 + exp(-X %*% beta)))
  D<-diag(m * w *(1 - w))
  hes<-t(X) %*% D %*% X
  return(hes)
}

# Newton's method of minimizing the objective
Newton_solver <- function (beta, X, y, m) {
  log_likelihood <- c(likelihood(beta, X, y, m))
  step <- 0
  conv <- 100
  while (step < 100 & conv > 0.001) {
    step <- step + 1
    inv_hessian <- solve(hessian(beta, X, m))
    beta <- beta - inv_hessian %*% t(grad(beta, X, y))
    log_likelihood <- c(log_likelihood, likelihood(beta, X, y, m))
  }
  return (c("beta", beta, "step", step, "log Likelihood", log_likelihood))
}


## Examples 


data <- read.csv(url("https://raw.githubusercontent.com/jgscott/SDS385/master/data/wdbc.csv"))
data$intercept <- rep(1, nrow(data)) #add column of ones for intercept offset

# visualize data
plot(data$X17.99, data$X10.38, col = as.factor(data$M), xlab = "X17.99", ylab = "X10.38")

#s et predictor and response variables
y <- as.matrix(data$M)
y <- y == "M"
X <- as.matrix(data[,3:12])
X <- cbind(rep(1, nrow(X)), X) #add 1s to X, to deal with the intercept term
init_beta <- rep(0, nrow(t(X)))

# calculate optimal beta via gradient descent
beta_opt_grad <- gradient_descent(0.001, 100, init_beta, X, y)

# use Newton's method to optimize beta
newton_results <- Newton_solver(init_beta, X, y, 1)

# benchmarking
bench1 <- microbenchmark(gradient_descent(0.001, 100, init_beta, X, y), Newton_solver(init_beta, X, y, 1))

