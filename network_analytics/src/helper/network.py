# Filter warnings
import warnings
warnings.filterwarnings("ignore")

# Data manipulation and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.spatial.distance as dist
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity

# Network modules
import networkx as nx
from community import community_louvain

class Network:
    """
    A class for network analytics
    """

    def __init__(self):
        """
        Define parameters
        """

    def graph(self, data, from_, to_, weight):
        """
        Create a Networkx graph

        Parameters
        ________________
        data:  pandas dataframe
        from_: from which nodes
        to_: to which nodes
        weight: weight of connection

        Returns
        ________________
        Networkx graph
        """
        # Create a Networkx graph
        G = nx.from_pandas_edgelist(data, from_, to_, edge_attr = [weight])
        return  G

    def bigraph(self, data, from_, to_, weight):
        """
        Create a Networkx bipartite graph

        Parameters
        ________________
        data:  pandas dataframe
        from_: from which nodes
        to_: to which nodes
        weight: weight of connection

        Returns
        ________________
        Networkx bipartite graph and split node sets
        """
        # Splits the nodes of the bipartite network into two parts
        nodes_set1 =  data[from_].unique()
        nodes_set2 =  data[to_].unique()

        # Create a graph
        G = nx.from_pandas_edgelist(data, from_, to_, edge_attr = [weight])
        return  G, nodes_set1, nodes_set2

    def projected_bigraph(self, G, nodes):
        """
        Bipartite projection of G onto one of its node sets.

        Parameters
        ________________
        G: Networkx bipartite graph
        nodes: Nodes to project onto (the “bottom” nodes)

        Returns
        ________________
        A weighted unipartite graph
        """
        pG = nx.bipartite.collaboration_weighted_projected_graph(G, nodes)
        return pG

    def similarity_mtx(self, biadj_mtx):
        """
        Convert a biadjacency matrix to a similarity matrix,
        based on the distance measure and the rows of the data

        Parameters
        ________________
        biadj_mtx: biadjacency matrix

        Returns
        ________________
        A similarity matrix (i.e. weighted unipartite matrix)
        """
        # Pearson and cosine similarities
        pearson_mtx = biadj_mtx.T.corr()
        cosine_mtx = cosine_similarity(biadj_mtx)

        # Remove the self nodes (i.e. diagonal elements)
        pearson_mtx = pearson_mtx * (1-np.eye(biadj_mtx.shape[0]))
        cosine_mtx = cosine_mtx * (1-np.eye(biadj_mtx.shape[0]))

        # Convert to pandas dataframe
        pearson_mtx = pd.DataFrame(pearson_mtx, index = biadj_mtx.index,
                                   columns = biadj_mtx.index)
        cosine_mtx = pd.DataFrame(cosine_mtx, index = biadj_mtx.index,
                                   columns = biadj_mtx.index)
        return pearson_mtx, cosine_mtx

    def similarity_net(self, sim_mtx, density = None):
        """
        Convert a similarity matrix to a sliced similarity network

        Parameters
        ________________
        sim_mtx: similarity matrix
        density: density of the network

        Returns
        ________________
        A sliced similarity network
        """
        # Convert matrix to columns
        stacked = sim_mtx.stack()
        count = int(sim_mtx.shape[0] * (sim_mtx.shape[0] - 1) * density)
        stacked = stacked.sort_values(ascending=False)[:count]

        # Convert matrix to unipartite graph
        edges = stacked.reset_index()
        edges.columns = ["source", "target", "weight"]
        network = nx.from_pandas_edgelist(edges, *edges.columns)

        # Some nodes may be isolated; they have no incident edges
        network.add_nodes_from(sim_mtx.columns)
        return network

    def net_info(self, G):
        """
        Print network information

        Parameters
        ________________
        G: Networkx graph

        Returns
        ________________
        Print graph information
        """
        # Network modularity and community detection
        partitions = community_louvain.best_partition(G)
        net_mod = community_louvain.modularity(partitions, G)

        # Print information
        print("Connected:", nx.is_connected(G))
        print("Mean clustering coefficient:", round(nx.average_clustering(G),4))
        print("Density:", round(nx.density(G),4))
        print("Modularity:", round(net_mod,4))
        print("Communities:", len(set(partitions.values())))
        print("Isolates Nodes:", len(list(nx.isolates(G))))
        print('-' * 35)
        print(nx.info(G))

    def binet_info(self, G, nodeset1 = None, nodeset2 = None):
        """
        Print network information

        Parameters
        ________________
        G: Bipartite Networkx graph

        Returns
        ________________
        Print bigraph information
        """
        print("Total papers:", len(nodeset1))
        print("Total authors:", len(nodeset2))
        print("Average authors per paper:", round(len(G.edges())/len(nodeset1),4))
        print("Average papers per author:", round(len(G.edges())/len(nodeset2),4))
        print('-' * 35)
        print("Bipartite network:", nx.is_bipartite(G))
        print("Connected:", nx.is_connected(G))
        print("Clustering coefficient:", round(nx.bipartite.robins_alexander_clustering(G),4))
        print("Density:", round( len(G.edges())/(len(nodeset1)*len(nodeset2)),6))
        print('-' * 35)
        print(nx.info(G))

    def degree_dist(self, G):
        """
        Compute the degree distribution of nodes in a unipartite graph

        Parameters
        ________________
        G: Networkx graph

        Returns
        ________________
        pandas dataframe
        """
        # Extract the node degree and create a pandas dataframe
        node_dgr = dict(G.degree())
        dgr_pdf = pd.DataFrame ( {'nodes': list(node_dgr.keys()),'dgr':  list(node_dgr.values())} )

        # Calculate the degree probability and create a pandas dataframe
        dgr_proba_ = dgr_pdf.groupby('dgr').count()/len(dgr_pdf)
        dgr_proba_pdf = pd.DataFrame({'dgr': list(dgr_proba_.index), 'dgr_proba': list(dgr_proba_.nodes)})
        return dgr_pdf, dgr_proba_pdf

    def bidegree_dist(self, G, nodes_):
        """
        Compute the degree distribution of one set of nodes in a bipartite graph

        Parameters
        ________________
        G: Networkx bipartite graph
        nodes_: A set of one node

        Returns
        ________________
        pandas dataframe
        """
        # Extract the node degree and create a pandas dataframe
        node_dgr = dict((nx.bipartite.degrees(G,nodes_))[1])
        node_dgr_pdf = pd.DataFrame ( {'nodes': list(node_dgr.keys()), 'dgr':  list(node_dgr.values())} )

        # Calculate the degree probability and create a pandas dataframe
        dgr_proba_ = node_dgr_pdf.groupby('dgr').count()/len(node_dgr_pdf)
        dgr_proba_pdf = pd.DataFrame({'dgr': list(dgr_proba_.index), 'dgr_proba': list(dgr_proba_.nodes)})

        # Left join
        # total_pdf = node_dgr_pdf.merge(dgr_proba_pdf, how = 'left', on = 'dgr')
        return node_dgr_pdf, dgr_proba_pdf

    def bidgr_centrality(self, G, nodes_):
        """
        Compute the degree centrality of the nodes in a bipartite graph

        Parameters
        ________________
        G: Networkx bipartite graph
        nodes_: A set of one node

        Returns
        ________________
        pandas dataframes
        """
        # Compute the degree centrality
        dgr_centr = nx.bipartite.degree_centrality(G, nodes_)

        # Create a dataframe of nodes and their degree centrality
        author_list = []
        paper_list = []
        for key, value in dgr_centr.items():
            if len(key) > 15:
                author_list.append((key, value))
            else:
                paper_list.append((key, value))
        author_pdf = pd.DataFrame(author_list, columns = ['paper', 'paper_dgr_centrality'])
        paper_pdf = pd.DataFrame(paper_list, columns = ['author', 'author_dgr_centrality'])
        return author_pdf, paper_pdf

    def plot_dgr_dist(self, data, xlab = None, label = None, title = None):
        """
        Plot degree distribution of unipartite graph

        Parameters
        ________________
        data: pandas dataframe containing degree & degree probability

        Returns
        ________________
        matplotlib line plot
        """

        # Figure layout
        plt.rcParams.update({'font.size': 15})
        plt.subplots(figsize = (10,7))

        # Plot degree distribution
        plt.loglog(data.dgr, data.dgr_proba, linestyle = 'None', color = 'b', 
        marker = "o", markersize=5, label = label)
        plt.title(title)
        plt.xlabel(xlab)
        plt.ylabel(r'probability $p(k)$')
        plt.legend(loc = 'best')
#         plt.savefig('../images/collabo_dgr.png')
        plt.show()

    def plot_graph(self, G, title = None):
        """
        Plot network and community detection

        Parameters
        ________________
        G: Networkx graph

        Returns
        ________________
        Networkx graph
        """
        # Set plot size
        plt.rcParams.update({'font.size': 15})
        plt.subplots(figsize=(10,6))

        # Community detection
        partitions = community_louvain.best_partition(G)
        values = [partitions.get(node) for node in G.nodes()]

        # Change node_color = 'values' for community detection
        nx.draw_networkx(G, pos = nx.spring_layout(G),node_size=40,
                         cmap = plt.get_cmap("Set2"), font_size=15, with_labels = False,
                         node_color= values, edge_color = 'gray')
        plt.title(label = title)
        plt.axis("off")
        plt.show()

    def plot_bigraph(self, G, nodes_set1, nodes_set2, title = None):
        """
        Plot bipartite network using networkx

        Parameters
        ________________
        G: bipartite network
        node1: first set of nodes
        node2: second set of nodes

        Returns
        ________________
        Networkx bipartite graph
        """
        # Plot size
        plt.rcParams.update({'font.size': 15})
        plt.subplots(figsize=(12,8))

        # Color list
        color_list = []
        for node in G.nodes():
            if node in nodes_set1:
                color_list.append('r')
            else:
                color_list.append('g')

        # # Bipartite layout
        # nx.draw_networkx(G, pos = nx.bipartite_layout(G, nodes_set1, align = 'horizontal'),
        #                  ax = ax1, node_size=40,cmap = plt.get_cmap("viridis"),font_size=15,
        #                  node_color= color_list, with_labels = False, edge_color = 'gray')

        # Sping layout
        nx.draw_networkx(G, pos = nx.spring_layout(G), node_size=40,
                         cmap = plt.get_cmap("viridis"),font_size=15,
                         node_color= color_list, with_labels = False,
                         edge_color = 'gray')
        plt.title(label = title)
        plt.axis("off")
        plt.show()
