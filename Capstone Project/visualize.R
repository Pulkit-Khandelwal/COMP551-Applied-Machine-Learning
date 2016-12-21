normalize <- function(x) (x - min(x)) / (max(x) - min(x))
to.unif <- function(x) pnorm(-(x - mean(x)) / sd(x), lower=FALSE)

if (!file.exists("interpolated_data.csv")) {
	if (!file.exists("preprocess.R")) {
		print("Can't access clean data")
		quit(save = "no", status = 0, runLast = FALSE)
	}
	source("preprocess.R")
}

if (!file.exists("classified_data.csv")) {
	if (!file.exists("classify.R")) {
		print("Can't access classified data")
		quit(save = "no", status = 0, runLast = FALSE)
	}
	source("classify.R")
}

interp <- read.csv("interpolated_data.csv")
orig <- read.csv("processed_data.csv")
classified <- read.csv("classified_data.csv")
library(rworldmap)

# For bounds
xlim <- c(min(interp$long),max(interp$long))
ylim <- c(min(interp$lat),max(interp$lat))
map <- getMap(resolution="li")

# Original and Interpolated Location
orig <- read.csv("processed_data.csv")
png("Original.png", width=10, height=16, res=300, units='in')
par(bg = 'white', fg = 'gray')
plot(map, col='gray', xlim=xlim, ylim=ylim)
points(orig$long, orig$lat, col="black", pch=20, cex=0.5)
dev.off()

png("Interpolated.png", width=10, height=16, res=300, units='in')
par(bg = 'white', fg = 'gray')
plot(map, col='gray', xlim=xlim, ylim=ylim)
points(interp$long, interp$lat, col="black", pch=20, cex=0.5)
dev.off()

# Classified Map
unsup <- c("#E69F00", "#56B4E9", "#009E73", "#D55E00")[classified$unsup]
png("MapClassifiedUnsup.png", width=10, height=16, res=300, units='in')
par(bg = 'white', fg = 'gray')
plot(map, col='gray', xlim=xlim, ylim=ylim)
points(classified$long, classified$lat, col=unsup, pch=20, cex=0.5)
dev.off()

sup <- c("#009E73", "#E69F00", "#D55E00", "#56B4E9")[as.integer(classified$sup)]
png("MapClassifiedSup.png", width=10, height=16, res=300, units='in')
par(bg = 'white', fg = 'gray')
plot(map, col='gray', xlim=xlim, ylim=ylim)
points(classified$long, classified$lat, col=sup, pch=20, cex=0.5)
dev.off()

# Velocity Over Time
library(ggplot2)
library(reshape2)
d = interp[,c("id","time","vlat","vlong")]
colnames(d) <- c("ID","Time","Meridional","Zonal")
d = melt(d, id=c("ID","Time"))
colnames(d) <- c("ID","Time","Var","Velocity")
d$Time <- as.Date(as.POSIXct(d$Time, origin="1970-01-01"))
par(bg = NA, fg = 'black')
plt <- ggplot(data=d, aes(x=Time, y=Velocity, group=ID)) +
	geom_segment(aes(x=Time, y=0, xend=Time, yend=Velocity), size=0.1) +
	facet_grid(Var ~ .) +
	scale_x_date(date_labels="%b %Y", date_breaks="3 months") +
	theme(panel.grid.major=element_blank(), panel.grid.minor=element_blank())
ggsave("Velocity.png", width=16, height=5, dpi=300)

# Location Over Time
d = interp[,c("id","time","lat","long")]
colnames(d) <- c("ID","Time","Latitude","Longitude")
d = melt(d, id=c("ID","Time"))
colnames(d) <- c("ID","Time","Var","Degrees")
d$Time <- as.Date(as.POSIXct(d$Time, origin="1970-01-01"))
par(bg = NA, fg = 'black')
plt <- ggplot(data=d, aes(x=Time, y=Degrees, group=ID)) +
	geom_line(size=0.4) +
	facet_grid(Var ~ .) +
	scale_x_date(date_labels="%b %Y", date_breaks="3 months") +
	theme(panel.grid.major=element_blank(), panel.grid.minor=element_blank())
ggsave("Location.png", width=16, height=5, dpi=300)


# Classification Over Time
d <- classified[,c("id","time","day","lat","vlat","unsup","sup")]
d$unsup <- c("#E69F00", "#56B4E9", "#009E73", "#D55E00")[classified$unsup]
d$sup <- c("#009E73", "#E69F00", "#D55E00", "#56B4E9")[as.integer(classified$sup)]

colnames(d) <- c("ID","Time","Day","Latitude","Velocity","Unsupervised","Supervised")
d = melt(d, id=c("ID","Time","Day","Latitude","Velocity"))
colnames(d) <- c("ID","Time","Day","Latitude","Velocity","Method","col")
d$Time <- as.Date(as.POSIXct(d$Time, origin="1970-01-01"))
par(bg = NA, fg = 'black')
plt <- ggplot(data=d, aes(x=Time, y=Latitude, group=ID)) +
	geom_line(size=0.6, col=d$col) +
	facet_grid(Method ~ ., scales="free_y") +
	scale_x_date(date_labels="%b %Y", date_breaks="6 months") +
	theme(panel.grid.major=element_blank(), panel.grid.minor=element_blank())
ggsave("ClassifiedLocation.png", width=16, height=5, dpi=300)

plt <- ggplot(data=d, aes(x=Time, y=Velocity, group=ID)) +
	geom_segment(aes(x=Time, y=0, xend=Time, yend=Velocity), size=0.1, col=d$col) +
	facet_grid(Method ~ ., scales="free_y") +
	scale_x_date(date_labels="%b %Y", date_breaks="6 months") +
	theme(panel.grid.major=element_blank(), panel.grid.minor=element_blank())
ggsave("ClassifiedVelocity.png", width=16, height=5, dpi=300)

plt <- ggplot(data=d, aes(x=Day, y=Latitude, group=ID)) +
	geom_point(size=0.6, stroke=0, col=d$col) +
	facet_grid(Method ~ ., scales="free_y") +
	theme(panel.grid.major=element_blank(), panel.grid.minor=element_blank())
ggsave("ClassifiedYearlyLocation.png", width=16, height=5, dpi=300)

plt <- ggplot(data=d, aes(x=Day, y=Velocity, group=ID)) +
	geom_segment(aes(x=Day, y=0, xend=Day, yend=Velocity), size=0.1, col=d$col) +
	facet_grid(Method ~ ., scales="free_y") +
	theme(panel.grid.major=element_blank(), panel.grid.minor=element_blank())
ggsave("ClassifiedYearlyVelocity.png", width=16, height=5, dpi=300)
