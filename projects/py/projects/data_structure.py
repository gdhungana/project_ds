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

    def remove_duplicates(self):
        els = []
        node = self
        previous = None
        while node != None:
            if node.val in els:
                previous.next = node.next
            else:
                els.append(node.val)
            previous = node
            node = node.next
    def kth_to_last(self,k):
        if k < 0:
            return None
        p1 = self
        p2 = self
        i = -1
        while p1 != None:
            p1 = p1.next
            if i < k:
                i += 1
            else:
                p2 = p2.next
        if i == k:
            return p2.val
        else:
            return None

    def delete_node(self):
        node = self
        if node == None or node.next == None:
            return False
        node.val = node.next.val
        node.next = node.next.next
        return True

def sum_linked_lists_backword(p1, p2):
    carry_over = 0
    head = Node(0)
    pointer = head
    digit = 0
    # until both linked lists exist, sum elements
    while p1 != None and p2 != None:
        sum_ = p1.val + p2.val + carry_over
        pointer.next = Node(sum_ % 10)
        pointer = pointer.next
        carry_over = sum_ / 10
        p1 = p1.next
        p2 = p2.next

    if p1 == None:
        while p2 != None:
            sum_ = p2.val + carry_over
            pointer.next = Node(sum_ % 10)
            pointer = pointer.next
            carry_over = sum_ / 10
            p2 = p2.next

    if p2 == None:
        while p1 != None:
            sum_ = p1.val + carry_over
            pointer.next = Node(sum_ % 10)
            pointer = pointer.next
            carry_over = sum_ / 10
            p1 = p1.next
    if carry_over > 0:
        pointer.next = Node(carry_over)
    return head.next



class DoubleNode:
     def __init__(self, val):
        self.val = data
        self.next = None
        self.prev = None


#- Stack/Queue


