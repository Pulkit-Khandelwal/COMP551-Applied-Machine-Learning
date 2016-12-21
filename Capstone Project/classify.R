if (!file.exists("interpolated_data.csv")) {
	if (!file.exists("preprocess.R")) {
		print("Can't access clean data")
		quit(save = "no", status = 0, runLast = FALSE)
	}
	source("preprocess.R")
}

D <- read.csv("interpolated_data.csv")


# UNSUPERVISED
library(stats)
cl <- kmeans(scale(D[,c("vlat","vlong","vel","lat")]),4,iter.max=20,nstart=5)
D$unsup <- cl$cluster

# SUPERVISED
library(e1071)
# Manually label one individial
d <- D[D$id == "91916A",]
d$lab <- "mate"
d$lab[d$lat <= 60] <- "mnorth"
d$lab[d$lat <= 60 & d$day >= 150] <- "msouth"
d$lab[d$lat <= 22] <- "nest"
d$lab <- as.factor(d$lab)

# Predict for others
D$sup <- predict(svm(lab ~ ., d[,c("vlong","vlat","lab")]), D)

# Write to file
write.csv(D, "classified_data.csv", row.names=FALSE)




# Unsuccessful RNN stuff

#library(rnn)

#A <- data.frame(d$vlat,d$vlong,model.matrix(~lab-1,d))
#A <- split(A, (seq(nrow(A))-1) %/% 2000)
#A <- lapply(A[1:(length(A)-1)], simplify2array)
#A <- aperm(simplify2array(A), c(3,1,2))

#X <- A[,,1:2]
#Y <- A[,,3:6]

#trainr(Y, X, learningrate=0.01, learningrate_decay = 1, momentum = 0,
#	   hidden_dim = c(50,50), network_type = "lstm", numepochs = 3,
#	   sigmoid = "tanh", use_bias = TRUE,
#	   batch_size = 1, seq_to_seq_unsync = FALSE, update_rule = "sgd",
#	   epoch_function = c(epoch_print, epoch_annealing), loss_function = loss_L1)
