#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Coursework in Python 
from IDAPICourseworkLibrary import *
from numpy import *
import math
#
# Coursework 1 begins here
#
# Function to compute the prior distribution of the variable root from the data set
def Prior(theData, root, noStates):
    prior = zeros((noStates[root]), float )
# Coursework 1 task 1 should be inserted here
    for y in range(len(prior)):
        for x in range(len(theData)):
            if (y == theData[x][root]):
                prior[y] += 1
        prior[y] /= len(theData)
# end of coursework 1 task 1
    return prior
# Function to compute a CPT with parent node varP and xchild node varC from the data array
# it is assumed that the states are designated by consecutive integers starting with 0
def CPT(theData, varC, varP, noStates):
    cPT = zeros((noStates[varC], noStates[varP]), float )
# Coursework 1 task 2 should be inserted here
    for x in range(len(cPT)):
        for y in range(len(cPT[0])):
            for z in range(len(theData)):
                if (theData[z][varC] == x and theData[z][varP] == y):
                    cPT[x][y] += 1
                    
    for x in range(len(cPT[0])):
        total = cPT.sum(0)[x]
        if (total > 0):
            for y in range(len(cPT)):
                cPT[y][x] /= total
# end of coursework 1 task 2
    return cPT
# Function to calculate the joint probability table of two variables in the data set
def JPT(theData, varRow, varCol, noStates):
    jPT = zeros((noStates[varRow], noStates[varCol]), float )
#Coursework 1 task 3 should be inserted here 
    for x in range(len(jPT)):
        for y in range(len(jPT[0])):
            for z in range(len(theData)):
                #print "x == {}, y == {}, z == {}".format(x,y,z)
                #print "varRow == {}, varCol == {}".format(varRow, varCol)
                if (theData[z][varRow] == x and theData[z][varCol] == y):
                    jPT[x][y] += 1
            jPT[x][y] /= len(theData)
# end of coursework 1 task 3
    return jPT
#
# Function to convert a joint probability table to a conditional probability table
def JPT2CPT(aJPT):
#Coursework 1 task 4 should be inserted here 
    total = aJPT.sum(0)
    for x in range(len(aJPT)):
        for y in range(len(aJPT[0])):
            aJPT[x][y] /= total[y]
# coursework 1 taks 4 ends here
    return aJPT

#
# Function to query a naive Bayesian network
def Query(theQuery, naiveBayes): 
    rootPdf = zeros((naiveBayes[0].shape[0]), float)
    for x in range(len(rootPdf)):
        rootPdf[x] += naiveBayes[0][x]
        for y in range(len(theQuery)):
            rootPdf[x] *= naiveBayes[y+1][theQuery[y]][x]
# end of coursework 1 task 5
    return rootPdf/sum(rootPdf)
#
# End of Coursework 1
#
# Coursework 2 begins here
#
# Calculate the mutual information from the joint probability table of two variables
def MutualInformation(jP):
    mi=0.0
# Coursework 2 task 1 should be inserted here
    pA = []
    pB = []
    for x in range(len(jP)):
        pA.append(sum(jP[x]))

    for x in range(len(jP[0])):
        col_sum = 0
        for y in range(len(jP)):
            col_sum += jP[y][x]
        pB.append(col_sum)

    for x in range(len(jP)):
        for y in range(len(jP[0])):
            if (jP[x][y] > 0): 
                mi += jP[x][y] * math.log(jP[x][y]/(pA[x]*pB[y]),2)

    # end of coursework 2 task 1
    return mi
#
# construct a dependency matrix for all the variables
def DependencyMatrix(theData, noVariables, noStates):
    MIMatrix = zeros((noVariables,noVariables))
# Coursework 2 task 2 should be inserted here
    for x in range(noVariables):
        for y in range(noVariables):
            jPT = JPT(theData, x, y, noStates)
            MIMatrix[x][y] = MutualInformation(jPT)
# end of coursework 2 task 2
    return MIMatrix
# Function to compute an ordered list of dependencies 
def DependencyList(depMatrix):
    depList=[]
# Coursework 2 task 3 should be inserted here
    depList2=[]
    for x in range(len(depMatrix)-1):
        for y in range(x+1,len(depMatrix[0])):
            if (x != y):
                depList.append(depMatrix[x][y])
                depList.append(x)
                depList.append(y)
                depList2.append(depList)
                depList = []
    
    depList2 = sorted(depList2, key=lambda depList2: depList2[0], reverse=True) 
# end of coursework 2 task 3
    return array(depList2)
#
# Functions implementing the spanning tree algorithm
# Coursework 2 task 4

def LoopCheck(node ,buffer, dictionary):
    if not node:
        pass
    if(False not in [x in buffer for x in node]):
        pass
    else:
        for i in node:
            if i in buffer:
                continue
            else:
                buffer.append(i)
                LoopCheck(dictionary[i], buffer, dictionary)
    

def SpanningTreeAlgorithm(depList, noVariables):
    spanningTree = []
    spanSet = {}
    node1Set = []
    node1_1Set = []
    node2Set = []
    node2_2Set = []
    temp = []
    node1Found = False
    node2Found = False
    
    for i in range(noVariables):
        spanSet[i] = []
        
    for _, node1, node2 in depList:
        for i in spanSet[node1]:
            node1Set.append(i)
        for i in spanSet[node2]:
            node2Set.append(i)
           
        node1FullSet = []
        node2FullSet = []
        
        LoopCheck(node1Set, node1FullSet, spanSet)
        LoopCheck(node2Set, node2FullSet, spanSet)
        
        for node in node1FullSet:
            if node2 == node:
                node2Found = True
                break
            
        for node in node2FullSet:
            if node1 == node:
                node1Found = True
                break
        node1Set = []
        node2Set = []
        if node1Found == False and node2Found == False:
            spanSet[node1].append(node2)
            spanSet[node2].append(node1)
        node1Found = False
        node2Found = False

                    
    found = False
    print spanSet
    for node1 in spanSet:
        for node2 in spanSet[node1]:
            for dep in depList:
                for i in range(len(spanningTree)):
                    if set(dep) ==  set(spanningTree[i]):
                        found = True
                if ((node1 == dep[1] and node2 == dep[2]) or (node1 == dep[2] and node2 == dep[1]) and found == False):
                    spanningTree.append(dep)
                found = False
    
    spanningTree = sorted(spanningTree, key=lambda spanningTree: spanningTree[0], reverse=True)

    return array(spanningTree)
#
# End of coursework 2
#
# Coursework 3 begins here
#
# Function to compute a CPT with multiple parents from he data set
# it is assumed that the states are designated by consecutive integers starting with 0
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
# Function to calculate the joint probability of a single data point in a Network
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
# Coursework 4 begins here
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

def main():
    #
    # main program part for Coursework 1
    #
    noVariables, noRoots, noStates, noDataPoints, datain = ReadFile("Neurones.txt")
    theData = array(datain)
    AppendString("results.txt","Coursework One Results by Alexandre Dalyac, Hamza Haider and Kiryl Trembovolski")
    AppendString("results.txt","") #blank line
    
    AppendString("results.txt","The prior probability of node 0")
    prior = Prior(theData, 0, noStates)
    AppendList("results.txt", prior)
    AppendString("results.txt","")

    cPT = CPT(theData, 2, 0, noStates)
    AppendString("results.txt", "The conditional probability matrix P (2|0) calculated from the data:")
    AppendArray("results.txt", cPT)
    AppendString("results.txt","")

    AppendString("results.txt", "The joint probability matrix P (2&0) calculated from the data:")
    jPT = JPT(theData, 2, 0, noStates)
    AppendArray("results.txt", jPT)
    AppendString("results.txt","")

    AppendString("results.txt", "The conditional probability matrix P (2|0) calculated from the joint probability matrix P (2&0):")
    jPT2cPT = JPT2CPT(jPT)
    AppendArray("results.txt", jPT2cPT)

    cpt1 = CPT(theData, 1, 0, noStates)
    cpt2 = CPT(theData, 2, 0, noStates)
    cpt3 = CPT(theData, 3, 0, noStates)
    cpt4 = CPT(theData, 4, 0, noStates)
    cpt5 = CPT(theData, 5, 0, noStates)

    AppendString("results.txt", "Results of query [4,0,0,0,5] on the naive network:")
    query = Query([4, 0, 0, 0, 5],[prior, cpt1, cpt2, cpt3, cpt4, cpt5])
    AppendList("results.txt", query)
    AppendString("results.txt","")

    AppendString("results.txt", "Results of query [6,5,2,2,5] on the naive network:")
    query = Query([6, 5, 2, 5, 5],[prior, cpt1, cpt2, cpt3, cpt4, cpt5])
    AppendList("results.txt", query)

    #jP = [[5,2,0,4],[7,0,0,1],[3,12,0,0],[0,4,6,0],[4,2,5,1],[6,7,3,2]]
    
    #mi = MutualInformation(jP)

    #print "mi ========== {}".format(mi)

    
    cPT_2 = []
    cPT_2 = CPT_2(theData, 0, 1, 2, noStates)
    #for row in cPT_2:
        #print row
    
    noVariables, noRoots, noStates, noDataPoints, datain = ReadFile("HepatitisC.txt")
    theData = array(datain)
    MI = array([])
    MI = DependencyMatrix(theData, noVariables, noStates)
    set_printoptions(precision=3)
    set_printoptions(suppress=True)
    print DependencyList(MI)
    print(MI)
    print ""
    print ""
    print SpanningTreeAlgorithm(DependencyList(MI), noVariables)
    
if __name__ == "__main__":
    main()

