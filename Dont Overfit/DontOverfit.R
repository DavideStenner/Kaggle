rm(list=ls())
library(readr)
library(knockoff)
library(xgboost)
library(doParallel)
library(MLmetrics)
library(randomForest)
train <- read_csv("./Dont Overfit/train.csv")[,-1]

Target<-as.numeric(train$target)
TrainData<-as.matrix(train[,-1])
p<-ncol(TrainData)

selected<-1:p
X<-TrainData[,selected]
test <- read_csv("Dont Overfit/test.csv")

XTest<-as.matrix(test)[,-1]
X<-data.frame(X)

XTest<-data.frame(XTest)
fit<-preProcess(rbind(X,XTest))

X1<-predict(fit,newdata=X)
XTest1<-predict(fit,newdata = XTest)

X1<-as.matrix(X1)
XTest1<-as.matrix(XTest1)

fdr<-.25

cl <- makeCluster(30)
registerDoParallel(cl)

W=foreach(i=1:100,.combine = "rbind",.packages = c("knockoff","glmnet"))%dopar%{
  p<-ncol(X1)
  mu = rep(0,p)
  rho = 0.10
  Sigma = toeplitz(rho^(0:(p-1)))
  
  X_k = create.gaussian(X1, mu, Sigma)
  Vec<-rep(0,p)
  Score<-stat.glmnet_coefdiff(X1, X_k, Target, nfolds=5, family="binomial",cores = 30,type.measure="auc")
  Thresh<-knockoff.threshold(Score, fdr=fdr, offset=0)
  Vec[which(Score>=Thresh)]<-1
  return(Vec)
}
stopCluster(cl)

ScoreL<-stat.glmnet_coefdiff(X1, X_k, Target, nfolds=5, family="binomial",cores = 30,type.measure="auc")

thresL = knockoff.threshold(ScoreL, fdr=fdr, offset=0)
SelL = which(ScoreL >= thresL)


Sel<-which(colSums(W)>0)

X2=X1[,Sel]
XTest2<-XTest1[,Sel]

Y<-Target

# L=list()
# library(foreach)

# cl <- makeCluster(30)
# registerDoParallel(cl)

# L<-foreach(i=1:500,.packages = c("caret","glmnet","MLmetrics"))%dopar%{
#   alpha<-runif(n=1,0,1)
#   family<-"binomial"
#   W<-runif(n=1,0.4,.7)
#   Weight<-c(W,1-W)/(table(Target)/(length(Target)))
#   PesiGLM<-Target
#   PesiGLM[which(PesiGLM==1)]<-Weight[2]
#   PesiGLM[which(PesiGLM==0)]<-Weight[1]
#   fit<-cv.glmnet(x=X2,y=Y,alpha=alpha,standardize = F,weight=PesiGLM,family="binomial")
#   pred<-predict(object=fit,newx=XTest2,type="response")
#   return(list(PredictionEnet=pred,ScoreEnet=1/min(fit$cvm)))
# }

# stopCluster(cl)

# C<-L[[1]]$PredictionEnet*L[[1]]$ScoreEnet
# FScore<-L[[1]]$ScoreEnet
# A<-L[[1]]$ScoreEnet
# for(i in 2:length(L)){
#   C<-C+L[[i]]$PredictionEnet*L[[i]]$ScoreEnet
#   FScore<-FScore+L[[i]]$ScoreEnet
#   A<-c(A,L[[i]]$ScoreEnet)
# }
# predF<-C/FScore


# cv.glmnet(x=X2,y=Y,alpha=alpha,standardize = F,weight=PesiGLM,family="binomial")


fit<-glm(Pred ~.,data=data.frame(X2,Pred=as.factor(Target)),family=binomial("logit"))

predglm<-predict(object=fit,newdata=data.frame(XTest2),type="response")

Pred<-as.factor(Target)
levels(Pred)<-c("No","Yes")

df_test <- data.frame(id=test[,'id'], target=predF)
names(df_test) = c("id","target")
# submission
write.csv(df_test, 'submission.csv', row.names = FALSE)
##############################################################################################

