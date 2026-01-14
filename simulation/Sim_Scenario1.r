library(foreach)
library(doParallel)
library(reticulate)
library(flexclust)
library(fda)
library(fda.usc)
library(funHDDC)
library(Funclustering)
library(FADPclust)
library(cluster)




cores <- 10
cl <- makeCluster(cores)
registerDoParallel(cl, cores=cores)


data_case1 <- foreach(run=1:100) %dopar%
  {
    library(clusterGeneration)
    
    n<-250;cla=2;n.curve=n*cla
    x=seq(0,8*pi,by=pi/12)
    dot=length(x)
    p=5
    eps=1.5
    sepVal.list=seq(-0.05, 0.01, length = p)
    my_coef=list()
    sample=list()
    for (i in p) { set.seed(i); sample[[i]] <- sample(1:1e5,10000,replace = F) }
    for (i in 1:p) {
      set.seed(sample[[i]][run])
      generate <- genRandomClust(
        numClust = 2,
        sepVal = sepVal.list[i],
        numNonNoisy = 5,
        numNoisy = 10,
        numOutlier = 0,       
        clustszind = 3,      
        clustSizes = c(n, n),
        rangeVar = c(1, 5),
        lambdaLow = 1, 
        ratioLambda = 100,
        covMethod = "eigen"
      )
      data_temp <- cbind(generate[[4]]$test_1,generate[[3]]$test_1)
      data_temp <- data_temp[order(data_temp[,1]), ] 
      my_coef[[i]] <- data_temp[, -1] +2
    }
    truecluster<-data_temp[,1]

    func1 <- function(t) sin(t/2) / sqrt(2*pi)
    func2 <- function(t) cos(t/2) / sqrt(2*pi)
    func3 <- function(t) sin(t) / sqrt(2*pi)
    func4 <- function(t) cos(t) / sqrt(2*pi)
    
    data<-array(0,dim=c(n.curve,dot,p))
    err=matrix(rnorm(n.curve*dot,0,eps),n.curve,dot)
    
    for (j in 1:p) {
      set.seed(12345+j)
      err=matrix(rnorm(n.curve*dot,0,eps),n.curve,dot)
      for (i in 1:n.curve) {
        data[i,,j]=matrix(func1(x)*my_coef[[j]][i,1]+
                            func2(x)*my_coef[[j]][i,2]+
                            func3(x)*my_coef[[j]][i,3]+
                            func4(x)*my_coef[[j]][i,4]+
                            (1/sqrt(4*pi))*my_coef[[j]][i,5]+err[i,],
                          byrow = T,nrow = 1,ncol = dot)
      }
    }
    
    data
  }




stopImplicitCluster()
stopCluster(cl)

save(data_case1, file = "data/Sim_Scenario1.RData")




#######################
### Plot Scenario 1 ###
#######################

n<-250;cla=2;n.curve=n*cla
x=seq(0,8*pi,by=pi/12)
dot=length(x)
p=5
knots=x
nbasis=dot
mybasis=create.bspline.basis(range(x), nbasis)
Lfdobj=2
lambda=1e-5   
myfdPar=fdPar(mybasis, Lfdobj, lambda)

data <- data_case1[[1]]
datafd=list()
for (i in 1:p) {
  datafd[[i]]=smooth.basis(x,t(data[,,i]),myfdPar)$fd
}
par(mfrow=c(3,2), mar=c(3,3,3,1), mgp=c(3,1,0))
plot(datafd[[1]], col=truecluster)
plot(datafd[[2]], col=truecluster)
plot(datafd[[3]], col=truecluster)
plot(datafd[[4]], col=truecluster)
plot(datafd[[5]], col=truecluster)



#######################
###  Other methods  ###
#######################

for(r in 1:100) 
{
  cat("iteration", r, "\n")
  
  ds <- data_case1[[r]]
  n=500;cla=2;n.curve=n*cla
  truecluster<-c(rep(1,n),rep(2,n))
  x=seq(0,8*pi,by=pi/12)
  dot=length(x)
  p=5
  
  #################functional data structure
  knots=x
  nbasis=dot
  mybasis=create.bspline.basis(range(x), nbasis)
  Lfdobj=2
  lambda=1e-5   
  myfdPar=fdPar(mybasis, Lfdobj, lambda)
  
  datafd=list()
  for (i in 1:p) {
    datafd[[i]]=smooth.basis(x,t(ds[,,i]),myfdPar)$fd
  }
  
  #################Other clustering methods
  n.cl=2:4
  result=data.frame(iteriation=run, 
                    ADP_ARI=NA,ADP_RI=NA,ADP_Cdiff=NA, 
                    funHDDC_ARI=NA,funHDDC_RI=NA,funHDDC_Cdiff=NA,
                    funclust_ARI=NA,funclust_RI=NA,funclust_Cdiff=NA)
  ################# FADPclust
  ans1 = FADPclust(fdata=datafd, cluster=n.cl,method = "FADP1")
  result$ADP_ARI = comPart(ans1[["clust"]],truecluster)[1]
  result$ADP_RI = comPart(ans1[["clust"]],truecluster)[2]
  result$ADP_Cdiff <- abs(ans1$nclust - cla)
  ################# funHDDC
  ans2 = funHDDC(datafd,K=n.cl,model="ABQkDk",init="kmeans",threshold=0.5)
  result$funHDDC_ARI = comPart(ans2[["class"]],truecluster)[1]
  result$funHDDC_RI = comPart(ans2[["class"]],truecluster)[2]
  result$funHDDC_Cdiff <- abs(ans2$K - cla)
  ################# funclust
  clust.temp=list()
  bic.funclust=list()
  for (k in 1:length(n.cl)) {
    ans3.temp=funclust(datafd, K=n.cl[k], thd=0.2, epsilon=1e-3, nbInit=2, nbIterInit=5, nbIteration=20, increaseDimension=TRUE)
    clust.temp[[k]]=ans3.temp$cls
    bic.funclust[[k]]=ans3.temp$bic
  }
  loc3 <- which.min(bic.funclust)
  selectK <- length(unique(clust.temp[[loc3]]))
  result$funclust_ARI=comPart(clust.temp[[loc3]],truecluster)[1]
  result$funclust_RI=comPart(clust.temp[[loc3]],truecluster)[2]
  result$funclust_Cdiff <- abs(length(unique(clust.temp[[loc3]])) - cla)

  case1_result <- if(run == 1) result else rbind(case1_result, result)
}


summary(case1_result)
write.csv(case1_result, file="scenario1_results.csv", row.names = F)

