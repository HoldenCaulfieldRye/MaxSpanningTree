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

print ''
print ''
print ''

################################

vertices = range(noVariables)

print 'testing createWeightDict()...'
weights = createWeightDict(depList, noVariables)
print 'depList', depList
print 'weights'
for key in weights.keys():
    print key, weights[key]
print ''

print 'inspect that weightSort() is correct by making sure the following is in descending order:'
edges = weightSort(weights)
print edges

treeEdges = []
print 'testing that countComponents is correct'
if countComponents([1,2,3], [[1,2],[2,3]]) != 1:
    print 'Error: countComponents([1,2,3], [[1,2],[2,3]]) should be 1, but it\'s %i' % (countComponents([1,2,3], [[1,2],[2,3]]))
elif countComponents([1,2,3], [[2,3]]) != 2:
    print 'Error: countComponents([1,2,3], [[2,3]]) should be 2, but it\'s %i' % (countComponents([1,2,3], [[2,3]]))
elif countComponents([1,2,3,4,5,7], [[1,5],[7,4],[2,3]]) != 3:
    print 'Error: countComponents([1,2,3,4,5,7], [[1,2],[2,3]]) should be 3, but it\'s %i' % (countComponents([1,2,3,4,5,7], [[1,5],[7,4],[2,3]]))
else:
    print 'countComponents tests passed successfully'

print 'moving on the Hepatitis graph, there are %i components' % (countComponents(vertices, treeEdges))
