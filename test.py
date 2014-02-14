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
print '\n\ntesting validEdges()...'

# case 1: every vertex has an edge (valid)
testa = validEdges([1,2,3], [[1,2], [1,3]])

# case 2: some vertices have no edges (valid)
testb = validEdges([1,2,3], [[1,2]])

# case 3: some edges have no vertices (invalid)
testc = validEdges([1,2,3], [[1,2], [3,5]])

# case 4: invalid vertex set
testd = validEdges([1,2,3,2], [[1,2], [2,3]])

if not testa:
    print 'Error: validEdges([1,2,3], [[1,2], [1,3]]) should return true'
elif not testb:
    print 'Error: validEdges([1,2,3], [[1,2]]) should return true'
elif testc:
    print 'Error: validEdges([1,2,3], [[1,2], [3,5]]) should return false'
elif testd:
    print 'Error: validEdges([1,2,3,2], [[1,2], [2,3]]) should return false'
else:
    print 'validEdges() tests passed successfully'


    
# TEST initialiseComponents()
# ===========================
print '\n\ntesting initialiseComponents()...'

# check that components initialise correctly
testa = initialiseComponents([0,1,2,3])
testb = initialiseComponents([5,1,9])
testc = initialiseComponents([])

if testa != OrderedDict([(0, [0]),(1, [1]),(2, [2]),(3, [3])]):
    print 'Error:'
elif testb != OrderedDict([(5, [5]), (1, [1]), (9, [9])]):
    print 'Error:'
elif testc != OrderedDict():
    print 'Error:'
else:
    print 'initialiseComponents() tests passed successfully'
    

# TEST evaluateEdge()
# ===================
print '\n\ntesting evaluateEdge()...'

# check that component merge opportunity gets detected
testa = evaluateEdge(OrderedDict([(0, [0]), (1, [1])]), [1,0])
testb = evaluateEdge(OrderedDict([(0, [0,1]), (1, [2,4,5])]), [1,4])

# check that 'useless' edge doesn't trigger a merge
testc = evaluateEdge(OrderedDict([(2, [0,1])]), [1,0])
testd = evaluateEdge(OrderedDict([(0, [5,8,2]), (3, [4,7])]), [8,2])

if testa[0] != 'connects components':
    print 'Error'
elif testb[0] != 'connects components':
    print 'Error'
elif testc[0] != 'edge fully contained in component':
    print 'Error'
elif testd[0] != 'edge fully contained in component':
    print 'Error'
else:
    print 'evaluateEdge() tests passed successfully'
    


# TEST merge()
# ============
print '\n\ntesting merge()...'

# check that merging occurs correctly
# check that obsolete component deletion works correctly
test = merge(OrderedDict([(0, [5,8,2]), (3, [4,7])]), 0, 4)

if test != [5,8,2,4,7]:
    print 'Error: components did not merge correctly'
elif 3 in test.keys():
    print 'Error: obsolete component was not deleted'
else:
    print 'merge() tests passed successfully'



# TEST countComponents()
# ======================

# case 1: no edges
testa = countComponents([1,2,3,4], [])

# case 2: fully connected graph
testb = countComponents([1,2,3,4], [[1,2],[1,3],[4,3],[3,2],[2,4]])

# case 3: fully connected tree
testc = countComponents([1,2,3,4], [[1,2],[1,3],[4,3]])

# case 4: graph and lonely vertices
testd = countComponents([1,2,3,4], [[1,2],[2,3],[3,1]])

# case 5: tree and lonely vertices
teste = countComponents([1,2,3,4], [[1,2]])

if testa != 4:
    print 'Error: '
elif testb != 1:
    print 'Error: '
elif testc != 1:
    print 'Error: '
elif testd != 2:
    print 'Error: '
elif teste != 3:
    print 'Error: '
else:
    print 'countComponents() tests passed successfully'


    
#####################################################

print '\n\nmoving on to finding maxweight tree'

tree = SpanningTreeAlgorithm(depList, noVariables)

print 'tree found:', tree
