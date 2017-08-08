## Feature extraction using soundecology package 
# calculation of indices used in ecology research

library(soundecology)
library(tuneR)
library(ineq)
library(vegan)
library(parallel)
library(seewave)
library(pracma)
library(oce)
library(parallel)

# load .wav data files from set a
setwd("~/Data Mining/Semester Project/heartbeat-sounds")



mydir=list.files("set_a", recursive=TRUE) 

dataOut<-matrix(nrow=length(mydir),ncol=6)

colnames(dataOut)<-c("class","ACI","NDSI","Bio","ADI","AEI")
dataOut[,1]<-substr(mydir,start=1,stop=4)
setwd("~/Data Mining/Semester Project/heartbeat-sounds/set_a")



# calculate ACI for each file
for ( i in 168:length(mydir)){
  soundFile<-readWave(mydir[i])
  aci<-acoustic_complexity(soundFile,max_freq = 195)
  dataOut[i,2]<-aci$AciTotAll_left
  
  
}


# calculate NDSI for each file
for (i in 120:length(mydir)){
  soundFile<-readWave(mydir[i])
  ndsi<-ndsi(soundFile )
  dataOut[i,3]<-ndsi$ndsi_left
  
}

# calculate bioacoustic index for each file
for (i in 1:length(mydir)){
  soundFile<-readWave(mydir[i])
  bio<-bioacoustic_index(soundFile,max_freq = 195)
  dataOut[i,4]<-bio$left_area
  
}

# calculate acoustic diversity index for each file
for (i in 1:length(mydir)){
  soundFile<-readWave(mydir[i])
  ad<-acoustic_diversity(soundFile,max_freq = 195,freq_step = 10)
  dataOut[i,5]<-ad$adi_left
  
}


# calculate acoustic evenness index for each file
for (i in 1:length(mydir)){
  soundFile<-readWave(mydir[i])
  ae<-acoustic_evenness(soundFile,max_freq = 195,freq_step = 10)
  dataOut[i,6]<-ae$aei_left
  
}

dataOut1<-cbind(mydir,dataOut)
setwd("~/Data Mining/Semester Project/heartbeat-sounds")

write.csv(dataOut1,file='featureExtraction_ecologyIndex.csv')
