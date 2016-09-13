library(Matrix)

# I added spaces throughout the document where I find them appropriate or needed
# I renamed some variables to shorten them while retaining their informative title
# I realigned some function definitions that spilled over to a second line if aesthetically preferable to keeping it on a single line after shortening variable names
# Hope this helps!

negloglikelihood <- function(m, y, X, beta){ 
  total <- 0
  N <- length(y)
  for (i in 1:N){
    total <- total + (m[i] - y[i]) * t(X[i,]) %*% beta + m[i] * log(1 + exp(-t(X[i,]) %*% beta))
  }
  return (total)
}

gradient_negloglik <- function(m, y, X, beta){
  w <- 1/(1 + exp(-X %*% beta))
  S <- m * w - y
  grad <- t(X) %*% S
  return(grad)
}

#this function is very long and convoluted. I think it is possible to write a more modular set of functions and then wrap them into a for/while loop. While this solves the problem, it seems unnecessarily complicated.
gradient_descent <- function(m, y, X, beta0, alpha, 
                             maxiter, acc_obj, acc_beta){
  #initialize variables
  negloglik <- numeric(maxiter)
  negloglik[1] <- negloglikelihood(m, y, X, beta0)
  gradient <- gradient_negloglik(m, y, X, beta0)
  diff_beta <- acc_beta + 1
  diff_obj <- acc_obj + 1
  i <- 1
  
  while(!(i == maxiter) &&
        ((acc_beta < diff_beta)||
        (acc_obj < diff_obj))){
    i <- i + 1
    beta1 <- beta0 - alpha * gradient
    diff_beta <- sum(abs(beta0 - beta1))
    beta0 <- beta1
    negloglik[i] <- negloglikelihood(m, y, X, beta0)
    diff_obj_fun <- negloglik[i-1] - negloglik[i]
    gradient <- gradient_negloglik(m, y, X, beta0)
  }
  return(list(betahat = beta0, negloglik = negloglik, step = i))
}

data_wdbc <- read.csv("./wdbc.csv", header = FALSE)
X <- as.matrix(cbind(rep(1,569), scale(data_wdbc[,3:12])))
y <- data_wdbc[,2]
y <- as.numeric(y == "M")
m <- rep(1,569)

beta0 <- rep(0,11)
stepsize <- 0.02
maxiter <- 10000
acc_obj <- 0.001 #really long variable names (this and acc_beta). Simplified as an example. 
acc_beta <- 0.001
result_grad_desc <- gradient_descent(m, y, X, beta0, alpha, 
                                     maxiter, acc_obj, acc_beta) #try to keep these aligned if you have to separate lines

result_grad_desc$betahat
plot(result_grad_desc$negloglik[1:result_grad_desc$step],
     main = "Negative Log-likelihood at each step",
     xlab = "Step",ylab = "negloglik",type = "l",log = "xy")

hessian_negloglik <- function(m, y, X, beta){
  w <- as.vector(1/(1 + exp(-X %*% beta)))
  D <- diag(m * w * (1 - w))
  hes<-t(X) %*% D %*% X
  return(hes)
}


newton_descent <- function(m, y, X, beta0, maxiter, acc_obj, acc_beta){
  #initialize variables
  negloglik <- numeric(maxiter)
  negloglik[1] <- negloglikelihood(m, y, X, beta0)
  gradient <- gradient_negloglik(m, y, X, beta0)
  hessian <- hessian_negloglik(m, y, X, beta0)
  diff_beta <- acc_beta +1
  diff_obj <- acc_obj + 1
  i <- 1
  
  while(!(i == maxiter)&&
        ((acc_beta < diff_beta)||
        (acc_obj < diff_obj))){
    i <- i + 1
    beta1 <- beta0 - solve(hessian, gradient)
    diff_beta <- sum(abs(beta0 - beta1))
    beta0 <- beta1
    negloglik[i] <- negloglikelihood(m, y, X, beta0)
    diff_obj <- negloglik[i-1] - negloglik[i]
    gradient <- gradient_negloglik(m, y, X, beta0)
    hessian <- hessian_negloglik(m, y, X, beta0)
  }
  return(list(betahat = beta0, negloglik = negloglik, step = i))
}

result_newton_desc <- newton_descent(m, y, X, beta0, maxiter, acc_obj, acc_beta)

result_newton_desc$betahat

plot(result_newton_desc$negloglik[1:result_newton_desc$step],
     main = "Negative Log-likelihood at each step",
     xlab = "Step",ylab = "negloglik",type = "l")
