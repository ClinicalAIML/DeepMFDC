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


data_case3 <- foreach(run=1:100) %dopar%
  {
    n=500;cla=2;n.curve=n*cla
    truecluster<-c(rep(1,n),rep(2,n))
    cluster.temp=seq(max(2,cla-2),cla+2,by=1)
    rangeval<-c(0,1)
    x=seq(0,1,by=0.05)
    dot=length(x)

    ############
    ###Cutoff###
    ############
    set.seed(111);sample1<-sample(1:1e5,10000,replace = F)
    set.seed(222);sample2<-sample(1:1e5,10000,replace = F)
    set.seed(333);sample3<-sample(1:1e5,10000,replace = F)
    set.seed(444);sample4<-sample(1:1e5,10000,replace = F)
    set.seed(555);sample5<-sample(1:1e5,10000,replace = F)
    set.seed(666);sample6<-sample(1:1e5,10000,replace = F)
    
    t0=seq(0.1,0.5,by=0.05)
    p=length(t0)
    data=array(0,dim=c(n.curve,dot,p))
    for (k in 1:p) {
      set.seed(sample1[run]);a_data=rnorm(n=2*n,mean=3,sd=0.5)
      set.seed(sample2[run]);b_data=rnorm(n=2*n,mean=2,sd=0.25)
      set.seed(sample3[run]);c1_data=rnorm(n=n,mean=0,sd=0.25)
      set.seed(sample4[run]);c2_data=rnorm(n=n,mean=4,sd=0.25)
      set.seed(sample5[run]);d_data=rnorm(n=2*n,mean=1/(2*t0[k]),sd=0.5)
      data.temp=matrix(0,n.curve,dot)
      for (i in 1:n) {
        a=a_data[i]
        b=b_data[i]
        c1=c1_data[i]
        d=d_data[i]
        for (j in 1:length(x)) {
          data.temp[i,j]=(b*sin(d*pi*x[j])+a)*(a-4*x[j])+c1
        }
      }
      cutoff=which.max(which(x<t0[k]))
      for (i in (n+1):(2*n)) {
        a=a_data[i]
        b=b_data[i]
        c2=c2_data[i-n]
        d=d_data[i]
        for (j in 1:cutoff) {
          data.temp[i,j]=(b*sin(d*pi*x[j])+a)*(a-4*x[j])+c2
        }
        for (j in (cutoff+1):length(x)) {
          data.temp[i,j]=(b*sin(d*pi*x[j])+a)*(a-4*(t0[k]-x[j]))-2*c2*(x[j]-1)
        }
      }
      data[,,k]=data.temp
    }
    
    data
  }




stopImplicitCluster()
stopCluster(cl)

save(data_case1, file = "data/Sim_Scenario3.RData")




#######################
### Plot Scenario 3 ###
#######################

n=500;cla=2;n.curve=n*cla
truecluster<-c(rep(1,n),rep(2,n))
x=seq(0,1,by=0.05)
dot=length(x)
p=9

knots=x
nbasis=dot
mybasis=create.bspline.basis(range(x), nbasis)
Lfdobj=2
lambda=1e-5   
myfdPar=fdPar(mybasis, Lfdobj, lambda)

data <- data_case3[[1]]
datafd=list()
for (i in 1:p) {
  datafd[[i]]=smooth.basis(x,t(data[,,i]),myfdPar)$fd
}
par(mfrow=c(3,3), mar=c(3,3,3,1), mgp=c(3,1,0))
plot(datafd[[1]], col=truecluster)
plot(datafd[[2]], col=truecluster)
plot(datafd[[3]], col=truecluster)
plot(datafd[[4]], col=truecluster)
plot(datafd[[5]], col=truecluster)
plot(datafd[[6]], col=truecluster)
plot(datafd[[7]], col=truecluster)
plot(datafd[[8]], col=truecluster)
plot(datafd[[9]], col=truecluster)



#######################
###  Other methods  ###
#######################

for(r in 1:100) 
{
  cat("iteration", r, "\n")
  
  ds <- data_case3[[r]]
  n=500;cla=2;n.curve=n*cla
  truecluster<-c(rep(1,n),rep(2,n))
  x=seq(0,1,by=0.05)
  dot=length(x)
  p=9
  
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

  case3_result <- if(run == 1) result else rbind(case3_result, result)
}


summary(case3_result)
write.csv(case3_result, file="scenario3_results.csv", row.names = F)

