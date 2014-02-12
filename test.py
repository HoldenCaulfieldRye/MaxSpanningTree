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

depList = DependencyList(depMatrix)
print "DependencyList(depMatrix) =", depList

################################

vertices = range(noVariables)

print 'inspect that createWeightDict() is correct by comparing the following:'
weights = createWeightDict(depList)
print 'depList', depList
print 'weights', weights
print ''

print 'inspect that weightSort() is correct by making sure the following is in descending order:'
edges = weightSort(weights)
print edges

treeEdges = []
print 'testing that countComponents is correct'
try:
    if countComponents(vertices, treeEdges) != len(vertices):
        print 'Error: countComponents(vertices, []) should be len(vertices)'
    elif countComponents([1,2,3], [[1,2],[2,3]]) != 1:
        print 'Error: countComponents([1,2,3], [[1,2],[2,3]]) should be 1'
    elif countComponents([1,2,3], [[2,3]]) != 2:
        print 'Error: countComponents([1,2,3], [[2,3]]) should be 2'
    elif countComponents([1,2,3], [[1,5],[7,4],[2,3]]) != 1:
        print 'Error: countComponents([1,2,3], [[1,2],[2,3]]) should be 1'
    else:
        print 'countComponents seems to be working fine'
        print ''
        print 'but have you inspected that createWeightDict and weightSort are correct?'
except:
    print 'damn, code for countComponents doesn\'t compile'
