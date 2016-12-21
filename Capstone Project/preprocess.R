D <- read.csv('Temperature/Temp_tagged_data.csv')
library(geosphere)

# Get rid of screwy data
D <- D[D$individual.local.identifier != "",]

# Add posix timestamp
time <- as.numeric(as.POSIXct(D$timestamp))
day <- as.numeric(strftime(D$timestamp, format="%j"))

# For the sake of brevity
id <- D$individual.local.identifier
lat <- D$location.lat
long <- D$location.long
temp <- D$ECMWF.Interim.Full.Daily.SFC.Temperature..2.m.above.Ground.
temp <- temp - 273.15

# Create new data frame
cleaned <- data.frame(id, time, day, lat, long, temp)

# Add columns for velocity
cleaned$vel <- 0
cleaned$vlat <- 0
cleaned$vlong <- 0
cleaned$dt <- 0
keep <- rep(TRUE,nrow(cleaned))
id.vals <- levels(id)
for (i in id.vals) {
	w <- i == id
	if (sum(w) < 2) {
		keep[w] <- FALSE
		next
	}
	loc <- cleaned[w, c("long","lat")]
	long <- cleaned[w,"long"]
	lat <- cleaned[w,"lat"]
	n <- length(lat)
	t <- time[w]
	dt <- (t[2:n] - t[1:n-1]) / (60*60)
	dist <- distCosine(loc[2:n,], loc[1:n-1,])
	
	# If there are any large gaps, remove it
	keep[w] <- max(dt) < 2000
	
	# Lateral distance
	l1 <- lat[1:n-1]
	l2 <- lat[2:n]
	dlat <- sign(l2-l1) * distCosine(data.frame(0,l1), data.frame(0,l2))
	
	# Longitudinal distance
	l1 <- long[1:n-1]
	l2 <- long[2:n]
	dlong <- sign(l2-l1) * distCosine(data.frame(l1,0), data.frame(l2,0))
	
	# Velocities in km/hr
	cleaned$vel[w] <- c(dist/dt, 0)
	cleaned$vlat[w] <- c(dlat/dt, 0)
	cleaned$vlong[w] <- c(dlong/dt, 0)
	cleaned$dt[w] <- c(dt, 0)
}
cleaned <- cleaned[keep,]

write.csv(cleaned, "raw_data.csv", row.names=FALSE)

# Get rid of data before the birds were released
#cleaned <- cleaned[cleaned$time > 1.258e9,]
cleaned <- cleaned[cleaned$time > 1.265e9,]
write.csv(cleaned, "processed_data.csv", row.names=FALSE)


# Now interpolate the data (to have constant time steps)
library(zoo)
interpolated <- data.frame(id=character(), time=integer(), day=integer(),
						   lat=double(), long=double(), vel=double(),
						   vlat=double(), vlong=double(), temp=double())

# Helper function for conversion to data frame
z2df <- function(x, index.name="time") {
	stopifnot(is.zoo(x))
	xn <- if(is.null(dim(x))) deparse(substitute(x)) else colnames(x)
	setNames(data.frame(index(x), x, row.names=NULL), c(index.name,xn))
}

# Shorthand
time <- cleaned$time
id <- cleaned$id
step <- 2*60*60

for (i in id.vals) {
	w <- i == id
	
	# Ignore individuals without enough data
	if (sum(w) < 2) next
	if (max(time[w]) - min(time[w]) < step) next
	
	sub <- cleaned[w,c("lat","long")]
	sub2 <- cleaned[w,c("day","temp")]
	
	z <- zoo(sub, order.by=time[w])
	z2 <- zoo(sub2, order.by=time[w])
	
	# The constant timesteps
	g <- seq(min(time[w]),max(time[w]),by=step)
	
	# Interpolate
	sub <- z2df(na.spline(z, xout=g, method="monoH.FC", ties=mean))
	sub2 <- z2df(na.approx(z2, xout=g))
	
	sub$day <- sub2$day
	sub$temp <- sub2$temp
	
	n <- nrow(sub)
	lat1 <- sub$lat[1:n-1]
	lat2 <- sub$lat[2:n]
	
	lng1 <- sub$long[1:n-1]
	lng2 <- sub$long[2:n]
	
	dlat <- sign(lat2-lat1) * distCosine(data.frame(0,lat1), data.frame(0,lat2))
	dlng <- sign(lng2-lng1) * distCosine(data.frame(lng1,0), data.frame(lng2,0))
	dtot <- distCosine(data.frame(lng1,lat1), data.frame(lng2,lat2))
	
	dlat[is.na(dlat)] = 0
	dlng[is.na(dlng)] = 0
	dtot[is.na(dtot)] = 0
	
	sub$vlat <- c(dlat/(1000*step), 0)
	sub$vlong <- c(dlng/(1000*step), 0)
	sub$vel <- c(dtot/(1000*step), 0)
	
	sub$id <- i
	sub$time <- g
	interpolated <- rbind(interpolated, sub)
}

write.csv(interpolated, "interpolated_data.csv", row.names=FALSE)




