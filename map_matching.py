import csv
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import math
import sys
import random
import numpy as np

# dir='/Users/anastasiamontgomery/Documents/EECS395/probe_data_map_matching/'
# f1='Partition6467LinkData.csv'
# f2='Partition6467ProbePoints.csv'
#
# linkdata=pd.read_csv(dir+f1)
# probedata=pd.read_csv(dir+f2)

# Read in link data and sort by latitude to make search function easier
# def reformat(dir,f1,f2):
#    linkdata=pd.read_csv(dir+f1)
#    probedata=pd.read_csv(dir+f2)
#    startpoint=[linkdata['shapeInfo'][x].split('|')[0] for x in range(len(linkdata))]
#    startpointLat=[startpoint[x].split('/')[0] for x in range(len(startpoint))]
#    startpointLon=[startpoint[x].split('/')[1] for x in range(len(startpoint))]
#    startpointAlt=[startpoint[x].split('/')[2] for x in range(len(startpoint))]
#    linkdata['startpointLat']=startpointLat; linkdata['startpointLon']= startpointLon; linkdata['startpointAlt']= startpointAlt
#    endpoint=[linkdata['shapeInfo'][x].split('|')[0] for x in range(len(linkdata))]
#    endpointLat=[endpoint[x].split('/')[0] for x in range(len(endpoint))]
#    endpointLon=[endpoint[x].split('/')[1] for x in range(len(endpoint))]
#    endpointAlt=[endpoint[x].split('/')[2] for x in range(len(endpoint))]
#    linkdata['endpointLat']= endpointLat; linkdata['endpointLon']= endpointLon; linkdata['endpointAlt']= endpointAlt
#    linkdataSorted=linkdata.sort_values('startpointLat').reset_index()
#    probedataSorted= probedata.sort_values('latitude').reset_index()
#    return linkdataSorted, probedataSorted


#------- this search function is driving me crazy i think im looping too much

def listComprehensionSearch():
   import time; import pandas as pd; import numpy as np
   from math import sqrt

   dir='/Users/anastasiamontgomery/Documents/EECS395/probe_data_map_matching/'
   f1='Partition6467LinkData.csv'
   f2='Partition6467ProbePoints.csv'

   linkdata=pd.read_csv(dir+f1,header=None)
   probedata=pd.read_csv(dir+f2, header=None)

   a=np.array(linkdata[14])
   a_spl=[a[i].split('|') for i in range(len(a))]

   # format data
   a_spl_final=[[(float(a_spl[t][z].split('/')[0]), float(a_spl[t][z].split('/')[1])) for z in range(len(a_spl[t]))] for t in range(len(a_spl))]

   nodelinks_xy=[[(float(a_spl[t][z].split('/')[0]),float(a_spl[t][z].split('/')[1])) for z in range(len(a_spl[t]))] for t in range(len(a_spl))]
   #a_spl_final_x=[[float(a_spl[t][z].split('/')[0]) for z in range(len(a_spl[t]))] for t in range(len(a_spl))]
   #a_spl_final_y=[[float(a_spl[t][z].split('/')[1]) for z in range(len(a_spl[t]))] for t in range(len(a_spl))]

   start=time.time()
   linkmatch=[];linkdistance=[]
   for t in range(len(probedata)):
       start=time.time()
       #get min distance of point to any node on link
       d=[min((np.subtract((probedata[3][t],probedata[4][t]),nodelinks_xy[l]).ravel()[::2]**2+np.subtract((probedata[3][t],probedata[4][t]),nodelinks_xy[l]).ravel()[1::2]**2)**.5) for l in range(len(nodelinks_xy))]
       #take min distance
       linkmatch.append(linkdata[0][d.index(min(d))])
       linkdistance.append(min(d))
       print(t)
       print(str(start-time.time()))

#---------------------- really doesn't fit in this code but here u guys go. it's super slow rn because of line 61

#------------------------- the first way I wrote the code, it does like 100 loops per 5 seconds 

def matchProbeToLink(dirin,f1):
   linkdata=pd.read_csv(dirin+f1,header=None)
   startpoint=[linkdata[14][x].split('|')[0] for x in range(len(linkdata))]
   startpoint_broken=[(startpoint[x].split('/'),startpoint[x].split('/')[1]) for x in range(len(startpoint))]
   from math import sqrt
   #from time import time
   linkdata=pd.read_csv(dirin+f1,header=None)
   probedata=pd.read_csv(dirin+f2,header=None)
   a=np.array(linkdata[14])
   a_spl=[a[i].split('|') for i in range(len(a))]
   min_idx_tmp.append =[]; min_idx=[]; dist_to_Node=[]
   # format data
   a_spl_final=[([(float(a_spl[t][z].split('/')[0]), float(a_spl[t][z].split('/')[1])) for z in range(len(a_spl[t]))]) for t in range(len(a_spl))]
   #make np array
   print('Done parsing Linkdata')
   a_final_np=np.array(a_final)
   # go through probes
   for t in range(len(probedata)):
      dxdy= a_final_np- (probedata[3][t],probedata[4][t])
      d= ((np.array(pd.DataFrame(dxdy)[0])**2+np.array(pd.DataFrame(dxdy)[1])**2)**(.5)).tolist()
      d_sorted=d.copy()
      min_idx_tmp.append(d.index(min(d_sorted))) # taking too long?
      d_sorted_list.remove(min(d_sorted_list))
      min_idx_tmp.append(d.index(min(d_sorted)))
      d_sorted_list.remove(min(d_sorted_list))
      min_idx_tmp.append(d.index(min(d_sorted)))
      min_idx.append(min_idx_tmp)
      #print(str(t))
      if t == int(len(probedata)/4):
          print('25% complete getting distances')
      if t == int(3*len(probedata)/4):
          print('75% complete getting distances')
   #final return & writeout
   min_idx,a_final=matchProbeToLink(dirin,f1)
   min_idx_df=pd.DataFrame(min_idx)
   a_final_df=pd.DataFrame(a_final)
   return min_idx,a_final

#------------------------- the first way I wrote the code, it does like 100 loops per 5 seconds 


class ProbeDataPoint:
    def __init__(self, sampleID, dateTime, sourceCode, lat, long, altitude, speed, heading):
        self.sampleID = sampleID
        self.dateTime = dateTime
        self.sourceCode = sourceCode
        self.lat = float(lat)
        self.long = float(long)
        self.altitude = altitude
        self.speed = speed
        self.heading = heading

        ### AFTER MATCHING WITH LINK
        self.linkPVID = ""
        self.direction = ""
        self.distFromRef = ""
        self.distFromLink = ""

        ### TEMPORARY
        self.distFromRefLat = ""
        self.distFromRefLong = ""
        self.distFromLinkLat = ""
        self.distFromLinkLong = ""
    def __str__(self):
        return "Probe ID: " + str(self.sampleID) + "\n" + "\tDateTime: " + str(self.dateTime) + "\n" + "\tSource Code: " + str(self.sourceCode) + "\n" + "\tLatitude: " + str(self.lat) + "\n" + "\tLongitude: " + str(self.long) + "\n" + "\tAltitude: " + str(self.altitude) + "\n" + "\tSpeed: " + str(self.speed) + "\n" + "\tHeading: " + str(self.heading)

class LinkData:
    def __init__(self, linkPVID, refNodeID, nrefNodeID, length, functionalClass, directionOfTravel, speedCategory, fromRefSpeedLimit, toRefSpeedLimit, fromRefNumLanes, toRefNumLanes, multiDigitized, urban, timeZone, shapeInfo, curvatureInfo, slopeInfo):
        self.linkPVID = linkPVID
        self.refNodeID = refNodeID
        self.nrefNodeID = nrefNodeID
        self.length = length
        self.functionalClass = functionalClass
        self.directionOfTravel = directionOfTravel
        self.speedCategory = speedCategory
        self.fromRefSpeedLimit = fromRefSpeedLimit
        self.toRefSpeedLimit = toRefSpeedLimit
        self.fromRefNumLanes = fromRefNumLanes
        self.toRefNumLanes = toRefNumLanes
        self.multiDigitized = multiDigitized
        self.urban = urban
        self.timeZone = timeZone
        self.shapeInfo = create_link_data_points(shapeInfo)
        self.curvatureInfo = curvatureInfo
        self.slopeInfo = slopeInfo
        self.minLat = min(self.shapeInfo, key=lambda l: l.lat).lat
        self.maxLat = max(self.shapeInfo, key=lambda l: l.lat).lat
        self.minLong = min(self.shapeInfo, key=lambda l: l.long).long
        self.maxLong = max(self.shapeInfo, key=lambda l: l.long).long
    def __str__(self):
        shapeInfo = "["
        for point in self.shapeInfo[:-1]:
            shapeInfo += str(point) + "\n"
        shapeInfo += "\t\t\t" + str(self.shapeInfo[-1]) + "]"
        return "Link PVID: " + str(self.linkPVID) + "\n" + "\tLength: " + str(self.length) + "\n" + "\tShapeInfo: " + shapeInfo

class LinkDataPoint:
    def __init__(self, lat, long):
        self.lat = float(lat)
        self.long = float(long)
    def __str__(self):
        return "Latitude: " + str(self.lat) + ", Longitude: " + str(self.long)

def create_data(probe_data_file, link_data_file):
    probe_data = {}
    link_data = []
    with open(probe_data_file) as probe_csvfile:
        reader = csv.reader(probe_csvfile)
        for row in reader:
            #TODO: PROBES WITH SAME ID ARE NOT YET SEPARATED BY TRIP
            if (str(row[0]) not in probe_data):
                probe_data[str(row[0])] = [[]]
            batch_size = 10
            if (len(probe_data[str(row[0])][-1]) < batch_size):
                probe_data[str(row[0])][-1].append(ProbeDataPoint(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7]))
            else:
                probe_data[str(row[0])].append([ProbeDataPoint(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7])])
    with open(link_data_file) as link_csvfile:
        reader = csv.reader(link_csvfile)
        for row in reader:
            link_data.append(LinkData(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11], row[12], row[13], row[14], row[15], row[16]))
    link_data.sort(key=lambda x: x.shapeInfo[0].lat, reverse=True)
    return probe_data, link_data

def map_match(probe_data, link_data):
    print("START MAP MATCHING\n")
    matched_probes = []
    probe_index = 0
    total_probe_ids = len(probe_data)

    with open('Partition6467MatchedPoints.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(["sampleID", "dateTime", "sourceCode", "Latitude", "Longitude", "Altitude", "Speed", "Heading", "linkPVID", "direction", "distFromRefLat", "distFromRefLong", "distFromLinkLat", "distFromLinkLong"])
        for probe_id in probe_data:
            batches = probe_data[probe_id]
            for batch in batches:
                links = {}
                link_counts = {}
                for probe in batch:
                    closestLink = None
                    closestLinkPoint = None
                    closestLinkPointDistance = math.inf
                    for link in link_data:
                        if (probe.lat < link.minLat or probe.lat > link.maxLat or probe.long < link.minLong or probe.long > link.maxLong):
                            continue
                        for linkPoint in link.shapeInfo:
                            distance = math.sqrt((linkPoint.long - probe.long)**2 + (linkPoint.lat - probe.lat)**2)
                            if (distance < closestLinkPointDistance):
                                closestLinkPointDistance = distance
                                closestLinkPoint = linkPoint
                                closestLink = link
                    if (closestLink == None): closestLink = link_data[random.randint(0, len(link_data)-1)]
                    if (closestLink.linkPVID not in link_counts):
                        link_counts[closestLink.linkPVID] = 0
                    link_counts[closestLink.linkPVID] += 1
                    if (closestLink.linkPVID not in links):
                        links[closestLink.linkPVID] = closestLink
                    probe_index+=1

                    sys.stdout.write('\r')
                    sys.stdout.write('[ ' + str(probe_index) + "/" + str(total_probe_ids) + ' ]')

                best_link = ""
                best_count = 0
                for linkPVID in link_counts:
                    if (link_counts[linkPVID] > best_count):
                        best_count = link_counts[linkPVID]
                        best_link = linkPVID
                for p in batch:
                    p.linkPVID = best_link
                    p.direction = links[best_link].directionOfTravel
                    refNode = None
                    nonRefNode = None
                    if (len(links[best_link].shapeInfo) > 0):
                        refNode = links[best_link].shapeInfo[0]
                        nonRefNode = links[best_link].shapeInfo[-1]
                    p.distFromRef = -1
                    if (refNode != None):
                        p.distFromRefLat = abs(refNode.lat - p.lat)
                        p.distFromRefLong = abs(refNode.long - p.long)
                        #p.distFromRef = math.sqrt((refNode.long - p.long)**2 + (refNode.lat - p.lat)**2)
                    p.distFromLink = -1
                    if (refNode != None and nonRefNode != None):
                        perp_point = []
                        if (nonRefNode.lat - refNode.lat == 0):
                            perp_point = [nonRefNode.lat, p.long]
                        elif (nonRefNode.long - refNode.long == 0):
                            perp_point = [p.lat, nonRefNode.long]
                        else:
                            lineSlope = (nonRefNode.long - refNode.long) / (nonRefNode.lat - refNode.lat)
                            constant = (lineSlope * nonRefNode.lat) - nonRefNode.long
                            perpLineSlope = float(1.0/lineSlope) * -1.0
                            perpConstant = (perpLineSlope * p.lat) - p.long
                            a = np.array([[lineSlope, -1],[perpLineSlope,-1]])
                            b = np.array([constant, perpConstant])
                            perp_point = np.linalg.solve(a,b)
                        p.distFromLinkLat = abs(perp_point[0] - p.lat)
                        p.distFromLinkLong = abs(perp_point[1] - p.long)

                        #p.distFromLink = abs((lineSlope * p.lat + 1 * p.long + constant)) / (math.sqrt(lineSlope * lineSlope + 1 * 1))
                    writer.writerow([p.sampleID, p.dateTime, p.sourceCode, str(p.lat), str(p.long), p.altitude, p.speed, p.heading, p.linkPVID, p.direction, str(p.distFromRefLat), str(p.distFromRefLong), str(p.distFromLinkLat), str(p.distFromLinkLong)])

    print("MAP MATCHING FINISHED")


### SAVING AND LOADING PROBE AND LINK DATA (NOTE: NOT THAT MUCH FASTER THAN JUST CREATING THE DATA SETS AGAIN)###
def save_data(probe_data, link_data):
    probe_filehandler = open('./saved_probe_data.txt', 'wb')
    link_filehandler = open('./saved_link_data.txt', 'wb')
    pickle.dump(probe_data, probe_filehandler)
    pickle.dump(link_data, link_filehandler)

def load_data():
    probe_filehandler = open('./saved_probe_data.txt', 'rb')
    link_filehandler = open('./saved_link_data.txt', 'rb')
    return pickle.load(probe_filehandler), pickle.load(link_filehandler)

def create_link_data_points(shapeInfo):
    parsedPoints = []
    points = shapeInfo.split("|")
    for point in points:
        coordinates = point.split("/")
        parsedPoints.append(LinkDataPoint(coordinates[0], coordinates[1]))
    return parsedPoints

if __name__ == '__main__':
    # if (len(sys.argv) < 3):
    #     print("Please supply the probe points data and the link data. Usage: python3 map_matching.py [probe_data.csv] [link_data.csv]")
    #     exit(0)

    ### FIRST TIME RUNNING ###
    (probe_data, link_data) = create_data("./probe_data_map_matching/Partition6467ProbePoints.csv", "./probe_data_map_matching/Partition6467LinkData.csv")
    map_match(probe_data, link_data)
    #save_data(probe_data, link_data)

    ### LOADING SAVED PROBE AND LINK DATA ###
    #(probe_data, link_data) = load_data()
    #print(len(probe_data), len(link_data))
