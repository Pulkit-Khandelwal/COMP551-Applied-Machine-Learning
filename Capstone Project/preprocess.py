
#column index = ['event-id', 'visible', 'timestamp', 'location-long', 'location-lat', 'argos:altitude', 'argos:best-level', 'argos:calcul-freq', 'argos:iq', 'argos:lat1', 'argos:lat2', 'argos:lc', 'argos:lon1', 'argos:lon2', 'argos:nb-mes', 'argos:nb-mes-120', 'argos:nopc', 'argos:pass-duration', 'argos:sensor-1', 'argos:sensor-2', 'argos:sensor-3', 'argos:sensor-4', 'argos:valid-location-algorithm', 'sensor-type', 'individual-taxon-canonical-name', 'tag-local-identifier', 'individual-local-identifier', 'study-name']
def load():
    """
    Loads the dataset.
    - 84162 entries in the dataset
    - Between 2006 and 2015
    """
    import csv
#    with open("Navigation","rU") as f :
    with open("../data/Navigation2","rU") as f :
        c = csv.reader(f)
        c = list(c)

    column_indexes = c[0]
    return column_indexes, c[1:]

def split_train_valid(data, train=0.7, shuffle=False):
    """ """
    n_examples = len(data)
    if shuffle :
        from random import shuffle
        shuffle(data)

    split1 = int(train*n_examples)
    print(split1)
    return data[:split1], data[split1:]



def get_paths():
    from preprocess import load
    column_indexes, data = load()
    import time, datetime
    import itertools
    from operator import itemgetter

    long_index, lat_index = column_indexes.index('location-long'), column_indexes.index('location-lat')
    tmp_index = column_indexes.index('timestamp')
    local_tag_index = column_indexes.index('tag-local-identifier')

    groups = itertools.groupby(data, key=lambda x: x[local_tag_index])

    i = 0

    path_by_tag_id = {}

    for k, g in groups:
        path_by_tag_id[k] = []
        #    lats, longs = [], []
        #    tmpstamps = []
        for w in g:
            lon, lat = w[long_index], w[lat_index]
            tmp = w[tmp_index]
            tmp = time.mktime(datetime.datetime.strptime(tmp, "%Y-%m-%d %H:%M:%S.000").timetuple())

            if lon == '' or lat == '':
                i += 1
                continue

            path_by_tag_id[k].append((float(lon), float(lat), int(tmp)))

        # Sorting each path time wise
        for k in path_by_tag_id.keys():
            path_by_tag_id[k] = sorted(path_by_tag_id[k], key=itemgetter(2))

    return path_by_tag_id


from geographiclib.geodesic import Geodesic
from geopy.distance import vincenty
import numpy as np
from datetime import datetime
from math import floor

def coord_dist(lat1, lon1, lat2, lon2):
    return vincenty((lat1, lon1), (lat2, lon2)).meters


def coord_angle(lat1, lon1, lat2, lon2, radiant=False):
    """Returns either in gradiant or in degree,minute,second depending of boolean variable radiant"""

    result = Geodesic.WGS84.Inverse(lat1, lon1, lat2, lon2)
    # options : ['lat1', 'a12', 's12', 'lat2', 'azi2', 'azi1', 'lon1', 'lon2']
    # see http://geographiclib.sourceforge.net/html/classGeographicLib_1_1Geodesic.html#ad7e59a242125a35a95c96cdb20573081

    if radiant :
        # Cast to radiant and return
        return np.array([result['azi1']*np.pi/180])
    else :
        # Decomposition to degree/minute/seconds
        angle = result['azi1']
        negative = angle < 0

        degree = floor(angle)
        r = angle - degree
        m = r*60.0
        minutes = floor(m)
        s = (m-minutes)*60
#        seconds = floor(s)

        return np.array([degree,minutes, s])

def encode_input(tmstmp, lat, lon, timestamp_option='week'):
    if timestamp_option == 'week':
        week, hour = timestamp_to_week_and_hour(tmstmp)
        enc_week = np.zeros(shape=(53,))
        enc_hour = np.zeros(shape=(24,))
        enc_week[week], enc_hour[hour] = 1,1
        return np.concatenate(([lat, lon],enc_week,enc_hour))
    elif timestamp_option == "raw":
        enc_tmp = [tmstmp] # That definitly needs normalisation before use.
        return np.concatenate((enc_tmp, [lat, lon]))
    else:
        raise Exception()

def timestamp_to_week_and_hour(tmpstmp):
    """  return a pair where first element is the week in [0,52] and the other is the hour in [0,23]   """
    date = datetime.fromtimestamp(tmpstmp)
    week = date.isocalendar()[1] - 1
    return week, date.hour

def tags_by_category():
    """returns a mapping category to tag. where category is encoded in integers between 0 and 6."""
    import csv
    with open("../data/Navigation-reference-data.csv", "rU") as f:
#    with open("Navigation-reference-data.csv", "rU") as f:
        c = csv.reader(f)
        c = list(c)

    column_indexes = c[0]
    category_index = column_indexes.index("manipulation-comments")
    deployment_index = column_indexes.index("deployment-id")

    to_ignore_index = column_indexes.index("deployment-comments")

    category_to_tag = {i:[] for i in range(7)}

    for line in c[1:]:
        comments, tag = line[to_ignore_index], line[deployment_index]
        tag = tag.split('-')[0]
        if comments != "":
            print("Skipped tag "+tag+" because of comment "+comments)
            continue

        category = line[category_index]
        if category == "not translocated; control":
            category_to_tag[0].append(tag)
        elif category == "translocated to Kazan; control":
            category_to_tag[1].append(tag)
        elif category == "translocated to Heligoland; control" :
            category_to_tag[2].append(tag)
        elif category == "translocated to Heligoland; TRIGEM":
            category_to_tag[3].append(tag)
        elif category == "translocated to Kazan; TRIGEM":
            category_to_tag[4].append(tag)
        elif category == "translocated to Heligoland; OLFAC" :
            category_to_tag[5].append(tag)
        elif category == "translocated to Kazan; OLFAC":
            category_to_tag[6].append(tag)
        else :
            raise Exception("Unknowned category +'"+category+"'")

#    print(map(lambda i: len(category_to_tag[i]),range(7)))
    return category_to_tag

def get_group(by_tag):

    X,groups,i = [], [],0
    for tag, input_list in by_tag.items():
        print(i)
        groups += [i for _ in range(len(input_list))]
        X += input_list
        i += 1

    return np.array(X), np.array(groups)

def get_dataset(threshold,keep_only_control=True):
    """for unsupervised

    keep_only_control : if true, keeps only control group
    """
    paths = get_paths()
    if keep_only_control:
        tag_category_dict = tags_by_category()
        paths = {i:j for i,j in filter(lambda x : x[0] in tag_category_dict[0], paths.items())}

    ex_by_tag = {}
    for tag, p in paths.items():
        path_length = len(p)
        if path_length <= threshold:
            continue
        enc_path = [encode_input(x[2], x[0], x[1]) for x in p]
        ex_by_tag[tag] = enc_path

    X, groups = get_group(ex_by_tag)

    return X, groups

##

def get_filtered_dataset(window_size, keep_sets=[0],output_radiant=False):
    """ This version allows you to decide which part of the data to keep in respect to the 7 differents categories (see func above). all elements in keep_sets must be in [0,1,2,3,4,5,6]. """

    paths_by_tag_id = get_paths()
    tag_category_dict = tags_by_category()

  #  print("valid tag", tag_category_dict[0])
  #  valid_tags = tag_category_dict[0]
#    valid_tags = list(map(lambda x: x.split('-')[0], tag_category_dict[0]))

    keep_tags = []
    for k in keep_sets:
        keep_tags += tag_category_dict[k]
#        keep_tags += list(map(lambda x: x.split('-')[0], tag_category_dict[k]))
    print("num entry before filtering " + str(len(paths_by_tag_id.values())))
    paths_by_tag_id = {i:j for i,j in filter(lambda x : x[0] in keep_tags, paths_by_tag_id.items())}
    print("num entry after filtering "+str(len(paths_by_tag_id.values())))
    return extract_features(paths_by_tag_id, window_size,radiant=output_radiant)

def get_dataset_timeseries(window_size):
    paths = get_paths()
    return extract_features(paths,window_size)

def extract_features(paths, window_size, anomaly_detect=False, radiant=False):
    """ This returns a dictionnary {tag_id : X,Y} where X and Y, are same size lists of respectively inputs and outputs.
    X is a sequence of $window_size length data points expressed as a one-hot encoded week indicator, a one-hot encoded hour indicator and lattitude and longitude. Y is a sequence pf the angle of displacement between [-pi,pi] (where 0 is east) and the speed of displacement (in meter/second).


    anomaly_detect : if True, looks at the average speed for every displacement. if more than 90 m/s, the examples is skipped.
    """

    ex_by_tag = {}
    for tag, p in paths.items():
        path_length = len(p)
        if path_length <= window_size:
            continue
        enc_path = [encode_input(x[2], x[0], x[1]) for x in p]

        X, Y = [], []
        for i in range(path_length - (window_size + 1)):
            x = np.concatenate(enc_path[i:i + window_size])
    #        print("x",x)
            # Calculating y
            lat1, lon1, tmp1 = p[i + window_size]
            lat2, lon2, tmp2 = p[i + window_size + 1]
            norm = coord_dist(lat1, lon1, lat2, lon2)  # in meters
            timedelta = tmp2 - tmp1  # in seconds
            speed = norm / timedelta
            if timedelta == 0 or speed > 8:
                continue
            direction = coord_angle(lat1, lon1, lat2, lon2,radiant=radiant)  # in radian
            print("norm,timedelta,speed,direction:",norm, timedelta, speed, direction)
            y = np.concatenate(([speed],direction))
            #y = np.array((speed, direction))
            if anomaly_detect == True:
                if speed > 90:
                    continue

            X.append(x)
            Y.append(y)

        X, Y = np.stack(X), np.stack(Y)
        ex_by_tag[tag] = (X, Y)

    return ex_by_tag

def get_outlier_detection_set(split=True):
    """  X : an numpy representing the features """
    paths = get_paths()
    if split :
        X,Y = extract_features2(paths)
        examples = zip(X,Y)
        print(len(X), len(Y))
        train, test = split_train_valid(examples, shuffle=True)
        X_train, Y_train = map(np.array,zip(*train))
        X_test, Y_test = map(np.array,zip(*test))
        return X_train, Y_train, X_test, Y_test
    else :
        return extract_features2(paths)


def extract_features2(paths):
    """X is histogram of displacement vector , Y is binary output that identify whether the bird has been surgically treated or not. """
    tag_by_cat = tags_by_category()
    length_threshold = 5
    X, Y = [], []

    skipped = 0
    not_skipped = 0
    number_of_considered_birds = 0
    for cat in (1,2,3,4,5,6):

        for tag in tag_by_cat[cat]:
            try :
                p = paths[tag]
            except KeyError:
                print('skipped referenced tag '+str(tag))
                # This is a sketchy fix but I have valid reasons
                # Sorrysorrysorrysorry
                continue
#        for tag in filter(lambda x: x in category_by_tag[i] , paths.keys()):
            path_length = len(p)
            if path_length <= length_threshold:
                continue
            number_of_considered_birds += 1

            tmp_list = []
            for i in range(path_length - 1):
                # Calculating y
                lat1, lon1, tmp1 = p[i]
                lat2, lon2, tmp2 = p[i + 1]
                norm = coord_dist(lat1, lon1, lat2, lon2)  # in meters
                timedelta = tmp2 - tmp1  # in seconds
                speed = norm / timedelta
                if timedelta == 0 or speed > 12:
                    if speed > 7 :
                        skipped += 1
                    continue
                else :
                    not_skipped +=1
                direction = coord_angle(lat1, lon1, lat2, lon2, radiant=True)  # in radian
                print("norm,timedelta,speed,bearing:",norm, timedelta, speed, direction)
                y = np.concatenate(([speed], direction))
                tmp_list.append(y)

            tmp_list = np.stack(tmp_list)

            X.append(tmp_list)
            if cat in (1,2,3,4):
                Y.append(0)
            else :
                Y.append(1)

    print("number_of_considered_birds")
    print(number_of_considered_birds)

    print('vector skipped')
    print(skipped,not_skipped)

    Y = np.stack(Y)

    X = histogram_preprocess(X)

    return X,Y

from operator import itemgetter

def histogram_preprocess(X,num_bins=10):
    print(X)
    min_list, max_list = [min(map(min,[map(itemgetter(i),x) for x in X])) for i in (0, 1)], [max(map(max,[map(itemgetter(i),x) for x in X])) for i in (0, 1)]
    print("min_list", min_list)
    print("maxlist", max_list)

    xedges, yedges = [np.linspace(min_list[i],max_list[i], num=num_bins) for i in (0,1)]
    X2 = []
    for examples in X :
        x2 = np.zeros(shape=(num_bins+1,num_bins+1))
        for speed, bearing in examples :
            i = 0
            for x in xedges:
                if speed < x :
                    break
                i +=1

            j = 0
            for y in yedges:
                if bearing < y :
                    break
                j += 1
            x2[i][j] += 1

        sum = np.sum(x2)
        print(x2)
        x2 = 1.0/sum * x2
        print(x2)
        X2.append(x2)

    X2 = np.stack(X2)
    print(X2)
    return X2
####


def load_preprocessed_data():
    """with temperature """
    import csv
    from itertools import groupby
    with open('./processed_data.csv','r') as f:
        c = csv.reader(f)
        c = list(c)
        column_indexes = c[0]
        id_index = column_indexes.index('id')

        by_tag = {i:list(j) for i,j in groupby(c[1:],key=lambda x:x[id_index])}

    for tag, data in by_tag.items():

        print(tag)
        print(data)
        map(lambda x : x.pop(id_index),data)
        map(lambda x : x.pop(0),data)
        data = map(lambda x : list(map(float,x)),data)
        by_tag[tag] = np.array(data)

    return by_tag


def main_load_and_plot():
    """ Loads the dataset and project it on the region """
    from mpl_toolkits.basemap import Basemap
    from matplotlib import pyplot as plt
    import matplotlib.cm as cm
    import time, datetime

    column_indexes, data = load()
    long_index, lat_index = column_indexes.index('location-long'), column_indexes.index('location-lat')
    tmp_index = column_indexes.index('timestamp')

    lats, longs, i = [], [], 0
    tmpstamps = []
    for w in data:
        lon, lat = w[long_index], w[lat_index]
        tmp = w[tmp_index]
        if lon == '' or lat == '':
            i += 1
            continue

        tmp = time.mktime(datetime.datetime.strptime(tmp, "%Y-%m-%d %H:%M:%S.000").timetuple())

        lats.append(float(lat))
        longs.append(float(lon))
        tmpstamps.append(int(tmp))
    print("Skipped " + str(i) + " data point.")


    # projection='ortho', projection='mill'
    m = Basemap(projection='mill', llcrnrlon=-10, llcrnrlat=2, urcrnrlon=70, urcrnrlat=70,lon_0=30, lat_0=35, resolution='l')
    x1, y1 = m(longs,lats)
    m.scatter(x1,y1,s=30,c=tmpstamps,marker="o",cmap=cm.cool,alpha=0.7)
    m.drawmapboundary(fill_color='black') # fill to edge
    m.drawcountries()
    m.fillcontinents(color='white',lake_color='black',zorder=0)
    plt.colorbar()
    plt.show()

#if __name__ == "__main__":
#    main_load_and_plot()


if __name__ == "__main__":
#    data = get_outlier_detection_set()
#    print(data[0].shape, data[1].shape)
#    dataset = get_dataset(2)
#    dataset = get_filtered_dataset(1)
#    data = load_preprocessed_data()
#    for tag, data in data.items():
#        print(tag)
#        print(data.shape)
    pass