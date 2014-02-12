from numpy import *
from Library import *
from Skeleton import *

set_printoptions(precision=2, suppress=True)

noVariables, noRoots, noStates, noDataPoints, datain = ReadFile("HepatitisC.txt")
theData = array(datain)

from Skeleton import MutualInformation
from Skeleton import DependencyMatrix

jP = JPT(theData, 3, 8, noStates)

print "MutualInformation(jP) =", MutualInformation(jP)

depMatrix = DependencyMatrix(theData, noVariables, noStates)
print depMatrix

print "DependencyList(depMatrix) =", DependencyList(depMatrix)

################################

print 'inspect that weight dict is correct'
weights = createWeightDict(depList)
print depList
print weights
print ''

print 'testing that edges are well sorted'


print 'testing that edges are well sorted'
print 'testing that edges are well sorted'
print 'testing that edges are well sorted'
