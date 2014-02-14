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

test0, test1, test2, test3 = countComponents([1,2,3], [[1,2],[2,3]]), countComponents([1,2,3], [[2,3]]), countComponents([1,2,3,4,5,7], [[1,5],[7,4],[2,3]]), countComponents([0,1,2,3,4,5,6,7], [[5, 4], [4, 3], [7, 0]]) - countComponents([0,1,2,3,4,5,6,7], [[5, 4], [4, 3], [7, 0], [7, 1]])

if test0 != 1:
    print 'Error: countComponents([1,2,3], [[1,2],[2,3]]) should be 1, but it\'s', test0

elif test1 != 2:
    print 'Error: countComponents([1,2,3], [[2,3]]) should be 2, but it\'s', test1

elif test2 != 3:
    print 'Error: countComponents([1,2,3,4,5,7], [[1,2],[2,3]]) should be 3, but it\'s', test2

elif test3 != 1:
    print 'Error: - countComponents([0,1,2,3,4,5,6,7], [[5, 4], [4, 3], [7, 0], [7, 1]]) + countComponents([0,1,2,3,4,5,6,7], [[5, 4], [4, 3], [7, 0]]) should be 1, and it\'s', test3
    
else:
    print 'countComponents tests passed successfully'


# #####################################################

# print 'moving on to finding maxweight tree'

# tree = SpanningTreeAlgorithm(depList, noVariables)

# print 'tree found:', tree
