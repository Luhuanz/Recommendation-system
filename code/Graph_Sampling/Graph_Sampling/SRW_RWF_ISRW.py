import random
# import time
# import datetime
# import io
# import array, re, itertools
import numpy as np
import networkx as nx
# import matplotlib.pyplot as plt
# from itertools import groupby


class SRW_RWF_ISRW:

    def __init__(self):
        self.growth_size = 2 #设置图的生长大小为 2，这可能是在每个检查点后图应增加的最小节点数。
        self.T = 100    # number of iterations # 迭代次数
        # with a probability (1-fly_back_prob) select a neighbor node
        # with a probability fly_back_prob go back to the initial vertex
        self.fly_back_prob = 0.15 #置回到初始顶点的概率为0.15，在随机游走中，以15%的概率从当前节点跳回初始节点。
#随机游走采样 Simple Random Walk
    def random_walk_sampling_simple(self, complete_graph, nodes_to_sample):
        complete_graph = nx.convert_node_labels_to_integers(complete_graph, 0, 'default', True) #将图中的节点标签转换为整数。
        # giving unique id to every node same as built-in function id
        for n, data in complete_graph.nodes(data=True):
            complete_graph.nodes[n]['id'] = n #在循环中，给每个节点赋予一个唯一的 id 属性，等于它的节点编号。

        nr_nodes = len(complete_graph.nodes()) #获取图中的节点总数
        upper_bound_nr_nodes_to_sample = nodes_to_sample #是需要采样的节点数
        index_of_first_random_node = random.randint(0, nr_nodes - 1) #选择一个随机节点作为起始节点，并将其添加到sampled_graph中
        sampled_graph = nx.Graph()

        sampled_graph.add_node(complete_graph.nodes[index_of_first_random_node]['id'])
        #通过随机游走过程采样节点，直到采样的节点数达到所需的数量
        #在每次迭代中，随机选择当前节点的一个邻居，并将其添加到 sampled_graph 中
        iteration = 1
        edges_before_t_iter = 0
        curr_node = index_of_first_random_node
        while sampled_graph.number_of_nodes() != upper_bound_nr_nodes_to_sample:
            edges = [n for n in complete_graph.neighbors(curr_node)]
            index_of_edge = random.randint(0, len(edges) - 1)
            chosen_node = edges[index_of_edge]
            sampled_graph.add_node(chosen_node)
            sampled_graph.add_edge(curr_node, chosen_node)
            curr_node = chosen_node
            iteration = iteration + 1
    #每当迭代次数达到 self.T 的倍数时，检查自上次检查以来图是否增长了至少 self.growth_size 个节点。如果没有，选择一个新的随机节点继续随机游走过程
            if iteration % self.T == 0:
                if ((sampled_graph.number_of_edges() - edges_before_t_iter) < self.growth_size):
                    curr_node = random.randint(0, nr_nodes - 1)
                edges_before_t_iter = sampled_graph.number_of_edges()
        return sampled_graph

    #在随机游走过程中引入了“回飞”概念，允许从当前节点回到起始节点 Random Walk with Fly-back
    # 它接受一个完整图 complete_graph、采样节点数 nodes_to_sample 和回飞概率 fly_back_prob 作为参数
    def random_walk_sampling_with_fly_back(self, complete_graph, nodes_to_sample, fly_back_prob):
        complete_graph = nx.convert_node_labels_to_integers(complete_graph, 0, 'default', True)
        # giving unique id to every node same as built-in function id
        for n, data in complete_graph.nodes(data=True):
            complete_graph.nodes[n]['id'] = n #将图中的节点标签转换为整数，并为每个节点分配一个唯一的 id，与其索引相同
        # 初始化一些变量和采样图 sampled_graph，并从完整图中随机选择一个节点作为起始点。
        nr_nodes = len(complete_graph.nodes())
        upper_bound_nr_nodes_to_sample = nodes_to_sample
        index_of_first_random_node = random.randint(0, nr_nodes - 1)
        sampled_graph = nx.Graph()
        sampled_graph.add_node(complete_graph.nodes[index_of_first_random_node]['id'])
        #开始迭代过程，直到采样图中的节点数达到所需的数量。
        iteration = 1
        edges_before_t_iter = 0
        curr_node = index_of_first_random_node
        while sampled_graph.number_of_nodes() != upper_bound_nr_nodes_to_sample:
            edges = [n for n in complete_graph.neighbors(curr_node)]
            index_of_edge = random.randint(0, len(edges) - 1)
            chosen_node = edges[index_of_edge]
            sampled_graph.add_node(chosen_node)
            sampled_graph.add_edge(curr_node, chosen_node)
            #使用随机选择来决定是继续前进到新的邻居节点还是根据 fly_back_prob 概率回到先前的节点
            choice = np.random.choice(['prev', 'neigh'], 1, p=[fly_back_prob, 1 - fly_back_prob])
            if choice == 'neigh':
                curr_node = chosen_node
            iteration = iteration + 1
    #每经过 self.T 次迭代，检查采样图的生长是否达到预期。如果没有，选择一个新的随机节点作为当前节点，并打印信息提示正在选择另一个随机节点以继续随机游走。
            if iteration % self.T == 0:
                if ((sampled_graph.number_of_edges() - edges_before_t_iter) < self.growth_size):
                    curr_node = random.randint(0, nr_nodes - 1)
                    print("Choosing another random node to continue random walk ")
                edges_before_t_iter = sampled_graph.number_of_edges()

        return sampled_graph #返回构建的采样图
#Induced Subgraph Random Walk
    def random_walk_induced_graph_sampling(self, complete_graph, nodes_to_sample):
        #定义一个名为 random_walk_induced_graph_sampling 的方法，它接收完整图 complete_graph 和需要采样的节点数 nodes_to_sample 作为参数。
        complete_graph = nx.convert_node_labels_to_integers(complete_graph, 0, 'default', True)
        # giving unique id to every node same as built-in function id
        for n, data in complete_graph.nodes(data=True):
            complete_graph.nodes[n]['id'] = n
        #初始化一些变量：获取图中的总节点数 nr_nodes，设置采样节点的上限 upper_bound_nr_nodes_to_sample，并随机选择一个起始节点。
        nr_nodes = len(complete_graph.nodes())
        upper_bound_nr_nodes_to_sample = nodes_to_sample
        index_of_first_random_node = random.randint(0, nr_nodes - 1)
        #初始化 Sampled_nodes 集合，包含随机选择的起始节点的 id
        Sampled_nodes = set([complete_graph.nodes[index_of_first_random_node]['id']])
        #开始迭代过程，直到采样节点的数量达到所需数量。
        iteration = 1
        nodes_before_t_iter = 0
        curr_node = index_of_first_random_node
        while len(Sampled_nodes) != upper_bound_nr_nodes_to_sample:
            edges = [n for n in complete_graph.neighbors(curr_node)]
            index_of_edge = random.randint(0, len(edges) - 1)
            chosen_node = edges[index_of_edge]
            Sampled_nodes.add(complete_graph.nodes[chosen_node]['id'])
            curr_node = chosen_node
            iteration = iteration + 1
        #每经过 self.T 次迭代，检查自上次检查点以来是否有足够多的新节点被添加到采样中。如果没有，随机选择一个新的节点作为当前节点，以确保采样过程继续进行。
            if iteration % self.T == 0:
                if ((len(Sampled_nodes) - nodes_before_t_iter) < self.growth_size):
                    curr_node = random.randint(0, nr_nodes - 1)
                nodes_before_t_iter = len(Sampled_nodes)

        sampled_graph = complete_graph.subgraph(Sampled_nodes)

        return sampled_graph    #构建的诱导子图
