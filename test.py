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



# TEST countComponents()
# ======================

print 'testing that countComponents is correct'

test0 = countComponents([1,2,3], [[1,2],[2,3]])
test1 = countComponents([1,2,3], [[2,3]])
test2 = countComponents([1,2,3,4,5,7], [[1,5],[7,4],[2,3]])
test3 = countComponents([0,1,2,3,4,5,6,7], [[5, 4], [4, 3], [7, 0]]) - countComponents([0,1,2,3,4,5,6,7], [[5, 4], [4, 3], [7, 0], [7, 1]])
test4 = countComponents([0,1,2,3,4,5,6,7], [(5.0, 4.0), (4.0, 3.0), (7.0, 0.0), (7.0, 1.0), (6.0, 1.0)]) - countComponents([0,1,2,3,4,5,6,7], [(5.0, 4.0), (4.0, 3.0), (7.0, 0.0), (7.0, 1.0), (6.0, 1.0), (4.0, 1.0)])



# case 1: uComponent==-1, vComponent==-2

# case 2: uComponent==-1, vComponent==k

# case 3: uComponent==k, vComponent==-2

# case 4: uComponent==k, vComponent==k

# case 5: uComponent==k, vComponent == j != k

# case 6: case 1 followed by case 1

# case 7: case 1 followed by case 2

# case 8: case 1 followed by case 3

# case 9: case 1 followed by 4

# case 10: 1 followed by 5

# case : 2, 1

# case : 2, 2

# case : 2, 3

# case : 2, 4

# case : 2, 5

# case : 3, 1

# case : 3, 2

# case : 3, 3

# case : 3, 4

# case : 3, 5

# case : 4, 1

# case : 4, 2

# case : 4, 3

# case : 4, 4

# case : 4, 5

# case : 5, 1

# case : 5, 2

# case : 5, 3

# case : 5, 4

# case : 5, 5




if test0 != 1:
    print 'Error: countComponents([1,2,3], [[1,2],[2,3]]) should be 1, but it\'s', test0

elif test1 != 2:
    print 'Error: countComponents([1,2,3], [[2,3]]) should be 2, but it\'s', test1

elif test2 != 3:
    print 'Error: countComponents([1,2,3,4,5,7], [[1,2],[2,3]]) should be 3, but it\'s', test2

elif test3 != 1:
    print 'Error: - countComponents([0,1,2,3,4,5,6,7], [[5, 4], [4, 3], [7, 0], [7, 1]]) + countComponents([0,1,2,3,4,5,6,7], [[5, 4], [4, 3], [7, 0]]) should be 1, and it\'s', test3

elif test4 != 1:
    print 'Error: adding  (6.0, 1.0) to [(5.0, 4.0), (4.0, 3.0), (7.0, 0.0), (7.0, 1.0), (6.0, 1.0), (4.0, 1.0)] should decrease number of components by 1, but function it does so by', test4
else:
    print 'countComponents tests passed successfully'


#####################################################

# print 'moving on to finding maxweight tree'

# tree = SpanningTreeAlgorithm(depList, noVariables)

# print 'tree found:', tree
