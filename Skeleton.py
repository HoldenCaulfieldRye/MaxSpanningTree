#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Coursework in Python 
from Library import *
from numpy import *
import operator
#
# Coursework 1 begins here
#

# def add(x,y): return x+y
# def div(x,y): return x/y
# def count(seq, value):
    

# Function to compute the prior distribution of the variable root from datain
# notice this assumes observations were randomly sampled from distribution!
# datain has a column per variable/attribute, a row per observation/tuple
# noStates is a dictionary holding #states for each variable
def Prior(theData, root, noStates):
    prior = zeros((noStates[root]), float)
# Coursework 1 task 1 should be inserted here
    priorColumn = [theData[k][0] for k in range(len(theData))]
    for i in range(4):
        for j in range(len(priorColumn)):
            if priorColumn[j]==i:
                prior[i]+=1
    prior /= len(theData)
    return prior


# Function to compute a Conditional Prob Table with parent node varP and
# child node varC from the data array
# It is assumed that the states are designated by consecutive integers
# starting with 0
def CPT(theData, varC, varP, noStates):
    cPT = zeros((noStates[varC], noStates[varP]), float )
# Coursework 1 task 2 should be inserte4d here
# I suppose we want P(varC|varP) and since noStates[varC] is #rows, then each
# column is a pdf
    freqP = zeros(noStates[varP], int)
    for i in range(len(theData)):
        freqP[theData[i][varP]] += 1
        cPT[theData[i][varC]][theData[i][varP]] += 1
    cPT /= freqP
   # end of coursework 1 task 2
    return cPT


# Function to calculate the joint probability table of two variables in the data set
def JPT(theData, varRow, varCol, noStates):
    jPT = zeros((noStates[varRow], noStates[varCol]), float )
# Coursework 1 task 3 should be inserted here 
    for k in range(len(theData)):           # observations
        jPT[theData[k][varRow]][theData[k][varCol]] += 1
    jPT/=sum(jPT)
# end of coursework 1 task 3
    return jPT


# Function to convert a joint probability table to a conditional probability table
def JPT2CPT(aJPT):
# Coursework 1 task 4 should be inserted here
    cPT = zeros([shape(aJPT)[0], shape(aJPT)[1]], float)
    for j in range():
        cPT[:,j] /= sum(cPT[:,j])
# coursework 1 takes 4 ends here
    return cPT


# Function to query a naive Bayesian network
# naiveBayes is the network; represented as a list of tables. there is 1 entry per node
# (in numeric order) giving the associated cPT: [prior, cpt1, cpt2, cpt3, cpt4, cpt5]
# theQuery is a set of states for each non-root node
# return value should be the pdf for root conditional on queried states
# we are in naive bayes, so every non-root node is child of root
def Query(theQuery, naiveBayes): 
    rootPdf = ones((naiveBayes[0].shape[0]), float)
# Coursework 1 task 5 should be inserted here
    # assume cpti has 1 root state per row, 1 i state per column
    colList = []
    for i in theQuery:
        rootPdf *= naiveBayes[i+1][:,theQuery[i]]
    rootPdf /= sum(rootPdf)
# end of coursework 1 task 5
    return rootPdf

############################################################################
#                 End of Coursework 1                                       
############################################################################


############################################################################
#                 Coursework 2 begins here                                  
############################################################################

# Calculate the mutual information from the joint probability table of two
# variables
def MutualInformation(jP):
    mi=0.0
    # Coursework 2 task 1 should be inserted here
    # jP must be square matrix for KL to be defined right?
    mD1 = zeros(len(jP), float)
    mD2 = zeros(len(jP[0]), float)

    for i in range(len(mD1)):
        mD1[i] = sum(jP[i])

    for i in range(len(mD2)):
        mD2[i] = sum(jP[:,i])

    for i in range(len(mD1)):
        for j in range(len(mD2)):
            if 0 in (mD1[i], mD2[j], jP[i][j]): continue
            mi += jP[i][j] * (log(jP[i][j]/(mD1[i]*mD2[j]))/log(2))
    # end of coursework 2 task 1
    return mi


# Construct a dependency matrix for all the variables
def DependencyMatrix(theData, noVariables, noStates):
    MIMatrix = zeros((noVariables, noVariables))
# Coursework 2 task 2 should be inserted here
    for var1 in range(noVariables):
        for var2 in range(noVariables):
            if var1==var2: continue
            jPT = JPT(theData, var1, var2, noStates)
            MIMatrix[var1][var2] = MutualInformation(jPT)
# end of coursework 2 task 2
    return MIMatrix


# Function to compute an ordered list of dependencies 
def DependencyList(depMatrix):
    depList=[]
# Coursework 2 task 3 should be inserted here
    depList = [[depMatrix[i][j], i, j] for i in range(len(depMatrix)) for j in range(len(depMatrix[0]))]
    depList2 = sorted(depList, reverse=True)
# end of coursework 2 task 3
    return array(depList2)
#
# Functions implementing the spanning tree algorithm
# Coursework 2 task 4

def SpanningTreeAlgorithm(depList, noVariables):
    # WARNING: is this the correct form for spanningTree?
    spanningTree = []
    vertices = range(noVariables)
    weights = createWeightDict(depList, noVariables)
    edges = weightSort(weights)
    treeEdges = []
    spanningTree = [vertices, treeEdges]
    count = countComponents(vertices, treeEdges)

    print 'initialised vertices, treeEdges to', vertices, treeEdges
    
    while count > 0:
        if edges==[]:
            print 'not enough edges to span a tree!'
            break
        print 'updated treeEdges from'
        print treeEdges 
        treeEdges.append(edges[0])
        del edges[0]
        print 'to'
        print treeEdges
        if count - countComponents(vertices, treeEdges) == 1:
            print 'adding that edge linked 2 components!'
            count -= 1
            print ''
        elif count == countComponents(vertices, treeEdges):
            del treeEdges[-1]
            print 'that edge wouldn\'t link components, getting rid of it'
            print ''
        else:
            print 'Error: adding edge', treeEdge[-1], 'to spanningTree changed the number of components by', count - countComponents(vertices, treeEdges),'; that\'s absurd'
            break

    if [vertices, treeEdges] != spanningTree:
        print "Error: spanningTree isn't a pointer; treeEdges has been modified but this hasn't updated spanningTree"
        
    return array(spanningTree)
    

def createWeightDict(depList, noVariables):
    weights = dict(((depList[i][1], depList[i][2]), depList[i][0]) for i in range(len(depList)))
    for i in range(noVariables):
        for j in range(i): del weights[(j,i)] # remove duplicate edges
        del weights[(i,i)]                    # remove loop edges

    return weights


def weightSort(weights):
    try:
        return sorted(weights.keys(), key = lambda edge: -1*weights[edge])
    except:
        return weights


def countComponents(vertices, treeEdges):
    print ''
    print ''
    print ''
    print 'hello, counting components in', vertices, treeEdges
    components = {0:[]}
    count = len(vertices)
    uComponent, vComponent = -1, -2

    # check no invalid edges
    verticesInEdges = [edge[0] for edge in treeEdges]
    verticesInEdges += [edge[1] for edge in treeEdges]
    for vertex in verticesInEdges:
        if vertex not in vertices:
            print 'Error: edges in treeEdges not covered by given vertices'
            return -1
    
    for [u,v] in treeEdges:
        print '#components = ', count
        print 'evaluating [%i, %i]' % (u, v) 
        for num in components.keys():
            if u in components[num]:
                assert(uComponent==-1)
                print u, 'is in', components[num]
                uComponent = num
            if v in components[num]:
                assert(vComponent==-2)
                print v, 'is in', components[num]
                vComponent = num

                if uComponent == vComponent: #continue
            print '%i, %i do both belong to component[%i]: %s' % (u, v, uComponent, components[uComponent])
            print 'so #components stays the same at', count
        elif uComponent == -1 and vComponent ==-2:
            print '%i, %i do not belong to any component: %s' % (u, v, components)
            components[len(components.keys())] = []
            components[len(components.keys())-1].append(u)
            components[len(components.keys())-1].append(v)
            print 'so created a new component: ', components
            print 'so #components decremented from', count
            count -= 1
            print 'to', count
        elif uComponent != -1 and vComponent == -2:
            print '%i belongs to component[%i]' % (u, uComponent)
            components[uComponent].append(v)
            print 'so adding %i to it too: components: %s' % (v, components)
            print 'so #components decremented from', count
            count -= 1
            print 'to', count
        elif uComponent == -1 and vComponent != -2:
            print '%i belongs to component[%i]' % (v, vComponent)
            components[vComponent].append(u)
            print 'so adding %i to it too: components: %s' % (u, components)
            print 'so #components decremented from', count
            count -= 1
            print 'to', count
            
        uComponent, vComponent = -1, -2

    return count

#
# End of coursework 2
#
# Coursework 3 begins here
#
# Function to compute a CPT with multiple parents from he data set
# it is assumed that the states are designated by consecutive integers
# starting with 0
def CPT_2(theData, child, parent1, parent2, noStates):
    cPT = zeros([noStates[child],noStates[parent1],noStates[parent2]], float )
# Coursework 3 task 1 should be inserted here
   

# End of Coursework 3 task 1           
    return cPT
#
# Definition of a Bayesian Network
def ExampleBayesianNetwork(theData, noStates):
    arcList = [[0],[1],[2,0],[3,2,1],[4,3],[5,3]]
    cpt0 = Prior(theData, 0, noStates)
    cpt1 = Prior(theData, 1, noStates)
    cpt2 = CPT(theData, 2, 0, noStates)
    cpt3 = CPT_2(theData, 3, 2, 1, noStates)
    cpt4 = CPT(theData, 4, 3, noStates)
    cpt5 = CPT(theData, 5, 3, noStates)
    cptList = [cpt0, cpt1, cpt2, cpt3, cpt4, cpt5]
    return arcList, cptList
# Coursework 3 task 2 begins here

# end of coursework 3 task 2
#
# Function to calculate the MDL size of a Bayesian Network
def MDLSize(arcList, cptList, noDataPoints, noStates):
    mdlSize = 0.0
# Coursework 3 task 3 begins here


# Coursework 3 task 3 ends here 
    return mdlSize 
#
# Function to calculate the joint probability of a single data point in a
# Network
def JointProbability(dataPoint, arcList, cptList):
    jP = 1.0
# Coursework 3 task 4 begins here


# Coursework 3 task 4 ends here 
    return jP
#
# Function to calculate the MDLAccuracy from a data set
def MDLAccuracy(theData, arcList, cptList):
    mdlAccuracy=0
# Coursework 3 task 5 begins here


# Coursework 3 task 5 ends here 
    return mdlAccuracy
#
# End of coursework 2
#
# Coursework 3 begins here
#
def Mean(theData):
    realData = theData.astype(float)
    noVariables=theData.shape[1] 
    mean = []
    # Coursework 4 task 1 begins here



    # Coursework 4 task 1 ends here
    return array(mean)


def Covariance(theData):
    realData = theData.astype(float)
    noVariables=theData.shape[1] 
    covar = zeros((noVariables, noVariables), float)
    # Coursework 4 task 2 begins here


    # Coursework 4 task 2 ends here
    return covar
def CreateEigenfaceFiles(theBasis):
    adummystatement = 0 #delete this when you do the coursework
    # Coursework 4 task 3 begins here

    # Coursework 4 task 3 ends here

def ProjectFace(theBasis, theMean, theFaceImage):
    magnitudes = []
    # Coursework 4 task 4 begins here

    # Coursework 4 task 4 ends here
    return array(magnitudes)

def CreatePartialReconstructions(aBasis, aMean, componentMags):
    adummystatement = 0  #delete this when you do the coursework
    # Coursework 4 task 5 begins here

    # Coursework 4 task 5 ends here

def PrincipalComponents(theData):
    orthoPhi = []
    # Coursework 4 task 3 begins here
    # The first part is almost identical to the above Covariance function, but because the
    # data has so many variables you need to use the Kohonen Lowe method described in lecture 15
    # The output should be a list of the principal components normalised and sorted in descending 
    # order of their eignevalues magnitudes

    
    # Coursework 4 task 6 ends here
    return array(orthoPhi)


# main program part for Coursework 2
#
noVariables, noRoots, noStates, noDataPoints, datain = ReadFile("HepatitisC.txt")
theData = array(datain)
AppendString("results.txt","Coursework One Results by dfg")
AppendString("results.txt","") #blank line
AppendString("results.txt","The prior probability of node 0")
prior = Prior(theData, 0, noStates)
AppendList("results.txt", prior)
#
# continue as described
#
#


