# import json
# import sys
import random
# import math
# import time
import networkx as nx
# import matplotlib.pyplot as plt
# from collections import defaultdict

#Queue 类是一个基本的队列实现
class Queue():
    # Constructor creates a list
    def __init__(self):
        self.queue = list()

    # Adding elements to queue
    def enqueue(self, data): # 用于将元素 data 添加到队列的前端，如果该元素不在队列中的话，以避免重复添加。
        # Checking to avoid duplicate entry (not mandatory)
        if data not in self.queue:
            self.queue.insert(0, data)
            return True
        return False

    # Removing the last element from the queue
    def dequeue(self):
        if len(self.queue) > 0:
            return self.queue.pop()
        else:
            # plt.show()
            exit()

    # Getting the size of the queue
    def size(self):
        return len(self.queue)

    # printing the elements of the queue
    def printQueue(self):
        return self.queue


class Snowball():

    def __init__(self):
        #Snowball 类有一个构造函数，初始化一个新的空图 G1
        self.G1 = nx.Graph()
#snowball 方法执行雪球采样，接受一个图 G、采样大小 size 和每次扩展的邻居数 k 作为参数。初始化一个队列 q，节点列表 list_nodes，并设置计数器 m 为 k
    def snowball(self, G, size, k):
        q = Queue()
        list_nodes = list(G.nodes())
        m = k
        dictt = set() #使用集合 dictt 来存储已访问的节点。
        while(m):
            id = random.sample(list(G.nodes()), 1)[0]
            q.enqueue(id)
            m = m - 1
        # print(q.printQueue()) 在循环中，随机选择一个节点 id 加入队列 q，重复 k 次
        while(len(self.G1.nodes()) <= size):
            if(q.size() > 0):
                id = q.dequeue()
                self.G1.add_node(id)
                if(id not in dictt):
                    dictt.add(id)
                    list_neighbors = list(G.neighbors(id))
                    #如果邻居数超过 k，则将前 k 个邻居加入队列，并在 G1 中添加与 id 的边
                    if(len(list_neighbors) > k):
                        for x in list_neighbors[:k]:
                            q.enqueue(x)
                            self.G1.add_edge(id, x)
                            #如果邻居数不超过 k 但大于 0，则将所有邻居加入队列，并在 G1 中添加相应的边
                    elif(len(list_neighbors) <= k and len(list_neighbors) > 0):
                        for x in list_neighbors:
                            q.enqueue(x)
                            self.G1.add_edge(id, x)
                else:
                    continue
            else:
                initial_nodes = random.sample(list(G.nodes()) and list(dictt), k)
                no_of_nodes = len(initial_nodes)
                for id in initial_nodes:
                    q.enqueue(id)
        return self.G1  #返回采样得到的图 G1。这个图是通过雪球采样方法从原图 G 中得到的子图。
