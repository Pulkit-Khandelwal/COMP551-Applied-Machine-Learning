# comp551-project4

## Dependencies

- matplotlib, basemap for the fancy maps. Be aware, installing basemap is a pain.

- geopy for computing distances from 2 coordinates with respect to ellipsoid.

- I used geographiclib, to compute angle on ellipsoid between two locations. (function WGS84.Inverse())

http://geographiclib.sourceforge.net/html/classGeographicLib_1_1Geodesic.html

## Potentially relevant links

- dataset reference page : https://www.datarepository.movebank.org/handle/10255/move.494

- Examples use of matplotlib and BaseMap to plot lattitude/longitude information on territory map : http://matplotlib.org/basemap/users/examples.html

- Nice Deep Belief Network implementation : https://github.com/lucastheis/deepbelief/tree/master/code

- https://plot.ly/ A nice API for plotting sexy graphs in Pyhton
- Plot maps in R and Pyhton http://blog.kaggle.com/2016/11/30/seventeen-ways-to-map-data-in-kaggle-kernels/ 

## Potential questions / Things to do with the data
/ To have a look at

- How is the dataset distributed over time? By a quick look there seems to be more early data then late data, that may limit the possibilities of what we can do.

- Do we know what the seasonnal movement pattern looks like? Could we characterize it? If its recurrent between general area we could try predicting paths between them.

- Is there a corellation between temperature and {path, path deviation, migration period, ending point?}?

- Assuming there is no major sampling bias (which might be a lot), we could maybe learn a distribution of location conditionaly to some features (like moment of the year and temperature distribution). But that might also be unfeasable with the data.

- Is there some individual id or flock id with every entry? Its seems like there should be.

- We could fit a mixture of gaussians on the data and say something meaningful about it.

- We could learn a latent distribution from all features and then measure how likely is the test data.

- Make a logistic regression with time as another axis and then try to extrapolate.

