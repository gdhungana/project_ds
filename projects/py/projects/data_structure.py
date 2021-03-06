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


#- Stack - using push and pop
class Stack():

    #Constructor
    def __init__(self,list):
      # __ means private (our __stack attribute is private)
      self.__stack = []
      for value in list:
        self.__stack.append(value)
        
    def push(self,value):
      print(" < PUSH < " + str(value))
      self.__stack.append(value)
      
    def pop(self):
      if len(self.__stack)>0:
        index = len(self.__stack) - 1
        print(" > POP > " + str(self.__stack[index]))
        return self.__stack.pop(index) #pop() Returns the value of the item that has been removed
      else:
        print("Stack is empty.")
        return False
      
    def output(self):
      st = ""
      for value in self.__stack:
        st = st + " > " + str(value) 
      print(st)  

#- Queue - enqueue/dequeue
class Queue():

    #Constructor
    def __init__(self,list):
      # __ means private (our __queue attribute is private)
      self.__queue = []
      for value in list:
        self.__queue.append(value)

    def enqueue(self,value):
      print(" < Enqueue < " + str(value))
      self.__queue.append(value)

    def dequeue(self):
      if len(self.__queue)>0:
        print(" > Dequeue > " + str(self.__queue[0]))
        return self.__queue.pop(0) #pop() Returns the value of the item that has been removed
      else:
        print("Queue is empty.")
        return False

    def output(self):
      st = ""
      for value in self.__queue:
        st = st + " > " + str(value)
      print(st)

