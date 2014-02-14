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


# TEST validEdges()
# ===================
print ''
print ''
print 'testing validEdges()...'

# case 1: every vertex has an edge (valid)
testa = validEdges([1,2,3], [[1,2], [1,3]])

# case 2: some vertices have no edges (valid)
testb = validEdges([1,2,3], [[1,2]])

# case 3: some edges have no vertices (invalid)
testc = validEdges([1,2,3], [[1,2], [3,5]])

if not testa:
    print 'Error: validEdges([1,2,3], [[1,2], [1,3]]) should return true'

elif not testb:
    print 'Error: validEdges([1,2,3], [[1,2]]) should return true'

elif testc:
    print 'Error: validEdges([1,2,3], [[1,2], [3,5]]) should return false'

else:
    print 'validEdges() tests passed successfully'


# TEST countComponents()
# ======================

print ''
print ''
print 'testing that countComponents is correct'


test = countComponents([], [])
test = countComponents([], [])
test = countComponents([], [])
test = countComponents([], [])
test = countComponents([], [])
test = countComponents([], [])
test = countComponents([], [])
test = countComponents([], [])
test = countComponents([], [])
test = countComponents([], [])
test = countComponents([], [])
test = countComponents([], [])
test = countComponents([], [])
test = countComponents([], [])
test = countComponents([], [])
test = countComponents([], []) 


# check that component merge opportunity gets detected

# check that when merge opportunity detected, components are merged

# check that after merging, obsolete component is deleted

# check that 'useless' edge doesn't trigger a merge

# check that 'useless' edge doesn't delete components


#####################################################

# print 'moving on to finding maxweight tree'

# tree = SpanningTreeAlgorithm(depList, noVariables)

# print 'tree found:', tree
