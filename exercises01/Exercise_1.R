# SDS385 Exercise 1 ----------------


library(microbenchmark)
library(Matrix)

# Definitions of functions ---------------

#inversion method
norm_regression = function(y, X, W){
  beta_hat = solve(t(X) %*% W %*% X) %*% t(X) %*% W %*% y
  return(beta_hat)
}

#singular value decomposition (SVD)
svd_regression = function(y, X, W){
  sigma = diag(svd(X)$d)
  U = svd(X)$u
  V = svd(X)$v
  beta = V %*% solve(sigma) %*% t(U) %*% y
  return(beta)
}

#sparse matrix operations - solving Ax = b in a sparse A
qr_regression_sparse = function(y, X){
  Q = qr.Q(qr(X))
  R = qrR(qr(X))
  return(solve(R) %*% t(Q) %*% y)
}

#sigmoid function (w_i(beta))
sigmoid = function(z){
  g = 1 / (1 + exp(-z))
  return(g)
}

#likelihood function
likelihood = function(beta, X, y){
  m = nrow(X)
  g = sigmoid(beta %*% t(X))
  J = -(1/m) * (y %*% t(log(g))) + ((1 - y) %*% t(log(1 - g)))
  return(J)
}

#gradient
grad = function(y, X, beta){
  gradient = ((sigmoid(beta %*% t(X))) - Y) %*% X
  return (gradient)
}


#gradient descent algorithm 
gradient_descent = function(alpha, iterations){
  beta = rep(0, nrow(t(X)))
  for (i in 1:iterations){
    beta = beta - alpha * grad(Y, X, beta)
    conv_diag <<- c(conv_diag, likelihood(beta, X, y))
  }
  return (beta)
}

#convert character data to binary 
as_binary = function(Y){
  for (i in 1:length(Y)){
    if (Y[i] == "B"){
      Y[i] = 0
    } else{
      Y[i] = 1
      }
  }
  return (as.numeric(Y))
}

#Newton's method


# Examples and implementation -------------------

#read in and format data
data = read.csv("wdbc.csv")
data$intercept <- rep(1, nrow(data)) #add column of ones for intercept offset

#visualize data
plot(data$X17.99, data$X10.38, col = as.factor(data$M), xlab = "X17.99", ylab = "X10.38")

#set predictor and response variables
y = as.matrix(data$M)
y = as_binary(Y)
X = as.matrix(data[, 3:12])
X = cbind(rep(1, nrow(X)), X) #add 1s to X, to deal with the intercept term
init_beta = rep(0, nrow(t(X)))
conv_diag = c()


#calculate optimal beta via gradient descent
beta_opt = gradient_descent(0.001, 100)

