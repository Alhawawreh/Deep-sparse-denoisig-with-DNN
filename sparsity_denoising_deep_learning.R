
library(tensorflow)
library(keras)
library(FSelector)
library(class)
library(kernlab)
library(dplyr)
library(data.table)
library(caret)
library(ggplot2)
library(ranger)
library(e1071)
library(keras)
set.seed(24)  
########################Read Dataset https://sites.google.com/a/uah.edu/tommy-morris-uah/ics-data-sets
gas_data<-read.csv("d:\\gas_dataset.csv", header=TRUE)

#################shuffel data 
gas_data<-gas_data[sample(nrow(gas_data)),]


#######################seperate class and change its type 
class<-gas_data$class
class<-as.factor(class)

################normaliziation 
data_new=as.data.frame(scale(gas_data[,1:18], center = TRUE, scale = TRUE))
normalized_original_data<-cbind(data_new,class)

#####################dividing data to three set 
train_set1<-normalized_original_data[1:48511,]
train_set2<-normalized_original_data[48512:77617,]
test_set<-normalized_original_data[77618:97019,]
################performing mask to the set 1
orginal_train_set<-train_set1
with_missing_noise<-train_set1[,1:18]

noisy_data<-with_missing_noise
dd=dim(noisy_data)
nna=20/100 #overall
new_noisy_data<-noisy_data
new_noisy_data[matrix(rbinom(prod(dd), size=1,prob=nna)==1,nrow=dd[1])]<-0
summary(new_noisy_data)

####################### convert data groups/sets to matrix to model 
noisy_train_set1<-new_noisy_data[,1:18]%>%as.matrix()
new_orginal_train_set1<-orginal_train_set1[,1:18]%>%as.matrix()




#################################################create model based on denoising autoencoder + sparsity in the encoder sub-network
model2 <- keras_model_sequential()
model2%>%
  layer_dense(units=15,activation="tanh", input_shape=c(18),activity_regularizer=regularizer_l1(10e-7))%>%
  layer_dense(units=9,activation="tanh",activity_regularizer=regularizer_l1(10e-7))%>%
  layer_dense(units=7,activation="tanh",name="bottleneck")%>%
  layer_dense(units=9,activation="tanh")%>%
  #layer_dense(units=1,activation="tanh")%>%
  layer_dense(units=15,activation="tanh")%>%
    layer_dense(units=18,activation="tanh")

epochs = 500

rms=optimizer_rmsprop()

early_stopping=callback_early_stopping(patience = 50, mode="auto")

model2%>%compile(loss='mean_squared_logarithmic_error',optimizer=rms, metric="accuracy")


history<-model2%>%fit(noisy_train_set1, new_orginal_train_set1,
                      epochs = epochs, 
                      batch_size =128,
                      suffle=FALSE,
                      validation_split=0.1,
                      callbacks = c(
                        callback_early_stopping(monitor = "val_loss", min_delta = 0.1, patience = 50, mode = "auto")))

###################### we will take this model and add new classifier layer
save_model_hdf5(model2, 'my_model.h5')
model <- load_model_hdf5('my_model.h5')
model3<-model2

pop_layer(model)     # to remove the last layers 
pop_layer(model)
pop_layer(model)

summary(model)

####################################### creatig the new supervised model 
input<-model$input
h1<-model$output
output<- h1%>%layer_dense(units=1,activation="sigmoid")
new_model2<-keras_model(inputs = input,outputs = output)

summary(new_model2)

##################################convert the training and testing data 

n_train_data<-train_set2[,1:18]%>%as.matrix()
n_test_data<-test_set[,1:18]%>%as.matrix()

####################################convert class to numerical 
y_train_vec <- ifelse(pull(train_set2, class) == "normal", 1, 0)
y_test_vec  <- ifelse(pull(test_set, class) == "normal", 1, 0)


###################################3training the  classifier


new_model2 %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = "accuracy")

history <- new_model2 %>% fit(n_train_data, y_train_vec, 
                              epochs = 500, batch_size = 128,
                              suffle=FALSE,
                              callbacks = callback_early_stopping(patience = 20, monitor = 'acc'),
                              validation_split=0.1,  
                              allbacks = c( callback_early_stopping(monitor = "val_loss", min_delta = 0.1, patience = 50, mode = "auto")))



##########################prediction 
pred <- mynew_model %>% predict(n_test_data, batch_size = 128)
Y_pred = round(pred)
########################### Confusion matrix
CM = table(Y_pred, y_test_vec)

#############################3 evaluate the model
evals <- mynew_model %>% evaluate(n_test_data, y_test_vec, batch_size = 1)

accuracy = evals[2][[1]]* 100
CM
table(test_set$result2)
accuracy


############################roc for binary 
library(ROCR)

library(pROC)
plot(roc(as.numeric(test_set$class), pred, direction="<"),
     col="red", lwd=4, main="20% Masking-Gas pipeline Dataset")



