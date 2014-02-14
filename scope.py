from numpy import *
from Library import *
from Skeleton import *


def countComponents(vertices, treeEdges):
    components = {0:[]}
    count = vertices[:]
    uComponent, vComponent = -1, -2

    # check no invalid edges
    verticesInEdges = [edge[0] for edge in treeEdges]
    verticesInEdges += [edge[1] for edge in treeEdges]
    for vertex in verticesInEdges:
        if vertex not in vertices:
            # print 'Error: edges in treeEdges not covered by given vertices'
            return -1
    
    for [u,v] in treeEdges:
        # print '#components = ', len(count)
        # print 'evaluating [%i, %i]' % (u, v) 
        for num in components.keys():
            if u in components[num]:
                assert(uComponent==-1)
                # print u, 'is in', components[num]
                uComponent = num
            if v in components[num]:
                assert(vComponent==-2)
                # print v, 'is in', components[num]
                vComponent = num

        if uComponent == vComponent: continue
            # print '%i, %i do both belong to %s' % (u, v, components[uComponent])
            # print 'so #components stays the same at', len(count)
        elif uComponent == -1 and vComponent ==-2:
            # print '%i, %i do not belong to any component: %s' % (u, v, components)
            components[len(count)-len(vertices)] = []
            components[len(count)-len(vertices)].append(u)
            components[len(count)-len(vertices)].append(v)
            # print 'so created a new component: ', components
            # print 'so #components decremented from', len(count)
            count.pop()
            # print 'to', len(count)
        elif uComponent != -1 and vComponent == -2:
            # print '%i belongs to component[%i]' % (u, uComponent)
            components[uComponent].append(v)
            # print 'so adding %i to it too: components: %s' % (v, components)
            # print 'so #components decremented from', len(count)
            count.pop()
            # print 'to', len(count)
        elif uComponent == -1 and vComponent != -2:
            # print '%i belongs to component[%i]' % (v, vComponent)
            components[vComponent].append(u)
            # print 'so adding %i to it too: components: %s' % (u, components)
            # print 'so #components decremented from', len(count)
            count.pop()
            # print 'to', len(count)
        uComponent, vComponent = -1, -2
        
    return len(count)
