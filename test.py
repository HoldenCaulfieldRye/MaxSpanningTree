from numpy import *
from Library import ReadFile
from Skeleton import JPT

noVariables, noRoots, noStates, noDataPoints, datain = ReadFile("HepatitisC.txt")
theData = array(datain)

from Skeleton import MutualInformation
from Skeleton import DependencyMatrix

jPT = JPT(theData, 3, 8, noStates)
