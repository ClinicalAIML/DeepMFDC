library(fda)
library(plyr)
library(refund)
library(FADPclust)


################################################################################
## semiVS functions
################################################################################
getP <- function(data, y){
  fdata <- data.frame(y=y)
  fdata$fd <- data
  # Fit a functional logistic regression model
  fit.lf <- pfr(y ~ lf(fd, k=20, presmooth="bspline", presmooth.opts=list(nbasis=20)), family="binomial", data=fdata)
  p_val <- summary(fit.lf)$s.pv
  return(p_val)
}


# Semi-supervised Variable Selection With Cross-Validation
semiVS_cv <- function(vlist, data, R = 10, test_pct = 0.4, pve = 0.99){
  cat("vlist", vlist, "\n")
  
  n.var = length(vlist)
  p_chisq <- numeric(R)
  for (f in 1:R) {
    # get index for each fold and split training and test data
    set.seed(sample.seed[f])
    index <- sample(c(TRUE,FALSE), dim(data[[1]])[[1]], replace=TRUE, prob=c(1-test_pct,test_pct))
    data.training <- lapply(vlist, function(x) data[[x]][index, ])
    data.test <- lapply(vlist, function(x) data[[x]][!index, ])
    y.training <- y[index]
    y.test <- y[!index]
    
    ########### Training: FADP on pre-screened fd
    # convert to fd
    fd.training=list()
    for (j in 1:n.var) {
      knots=x
      nbasis=20
      mybasis <- create.bspline.basis(c(min(x),max(x)), nbasis)
      Lfdobj=2
      lambda=10^(-5)   
      myfdPar=fdPar(mybasis, Lfdobj, lambda)
      fd.training[[j]]=smooth.basis(x, t(data.training[[j]]), myfdPar)$fd
    }
    # FADPclust and save centers
    fadp = FADPclust(fdata=fd.training, cluster=cluster, method = "FADP1")
    fadp.center = fadp$center
    
    ########### Test: Step 1 - convert to fd and univariate FPCA
    # merge in the center curves
    data.test.new <- lapply(seq_len(n.var), function(x) 
      rbind(data.training[[x]][fadp.center,],data.test[[x]]))
    # convert to fd
    fd.test=list()
    for (j in 1:n.var) {
      knots=x
      nbasis=20
      mybasis <- create.bspline.basis(c(min(x),max(x)), nbasis)
      Lfdobj=2
      lambda=10^(-5)   
      myfdPar=fdPar(mybasis, Lfdobj, lambda)
      fd.test[[j]]=smooth.basis(x, t(data.test.new[[j]]), myfdPar)$fd
    }
    # univariate FPCA
    pca = lapply(fd.test, function(x) pca.fd(x, nharm=20))
    score = list()
    for (v in 1:n.var){
      prop = pca[[v]]$varprop
      for (num in 1:length(prop)) {
        s = sum(prop[1:num])
        if(s >= pve){break}
      }
      score[[v]] = pca[[v]]$scores[,1:num]
    }
    score <- matrix(unlist(score), nrow = nrow(score[[1]]), byrow = TRUE)
    
    ########### Test: Step 2 - distance to each center and assign the nearest neighbor
    distance <- dist(score, method = "euclidean", upper = TRUE)
    distmat <- as.matrix(distance)[,c(1:length(fadp.center))]
    test.assign <- sapply(seq_len(dim(data.test.new[[1]])[1]), 
                          function(x) order(distmat[x,])[1])
    all.assign <- c(fadp$clust, test.assign[-c(1:length(fadp.center))])

    ########### Test: Step 3 - chi-square test of outcome
    test.chisq <- chisq.test(all.assign, y)
    
    ########### Output p value of chi-square test 
    output <- c(paste(t(vlist), collapse = " "), test.chisq$p.value)
    
  }
  
  return(output)
  
}


################################################################################
## Real case -- N=10000
################################################################################

load(file='FDC_labels_10000.Rdata')

################# Outcome: ICU admission
y = as.numeric(sampleds$ICU_admission) - 1
xlist <- list(as.matrix(fda.map[,3:130]), as.matrix(fda.fio2[,3:130]), as.matrix(fda.temp[,3:130]), 
              as.matrix(fda.hr[,3:130]), as.matrix(fda.spo2[,3:130]), as.matrix(fda.mac[,3:130]))

################# Pre-screen with univ functional regression p<0.05
n.x = length(xlist)
pvals <- sapply(seq_len(n.x), function(x) getP(xlist[[x]], y=y))
pvals_sig <- seq_len(n.x)[pvals<0.10]
n.x.pre <- length(pvals_sig)

########### Decide M from m0 to n.x.pre
m0 = 2
cluster = 4
x=c(1:128)

for (m in m0:n.x.pre) {
  cat("iteration", m, "\n")
  Mlist <- combn(pvals_sig, m, simplify = TRUE) 
  set.seed(m); sample.seed <- sample(1:1e5,100,replace = F)
  median_p <- apply(Mlist, 2, function(x) semiVS_cv(x, data=xlist, R=10))
  
  res <- as.data.frame(t(median_p))
  res$n_var = m
  
  res.m <- if(m == 2) res else rbind(res.m, res)
  
}

semiVS_surgery <- as.data.frame(rbind(res.2, res.3, as.matrix(res.4), as.matrix(res.5), t(median_p)))
