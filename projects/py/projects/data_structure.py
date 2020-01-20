import numpy as np

#- A linked list
class Node:
    def __init__(self,val):
        self.val=val
        self.next=None #- no pointier at instance

    def next(self,val):
        return val

    def traverse(self):
        node = self # start from the head node
        while node != None:
            #print node.val #  node value
            node = node.next # next node

