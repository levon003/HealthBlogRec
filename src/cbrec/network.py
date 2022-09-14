
import networkx as nx
from collections import defaultdict

class UserGraph:
    def __init__(self, config):
        #self.nxg = nx.DiGraph()
        #self.nxg.add_nodes_from(user_ids)
        self.wccg = WccGraph()
        
        self.outbound_edge_dict = {} # map of user_id -> set(user_id), fast access to existing edges
        self.inbound_edge_dict = {} # currently used exclusively for computing in-degree; could be replaced with a count
    
    def add_edge(self, from_user_id, to_user_id):
        if from_user_id not in self.outbound_edge_dict:
            self.outbound_edge_dict[from_user_id] = set()
        self.outbound_edge_dict[from_user_id].add(to_user_id)
        if to_user_id not in self.inbound_edge_dict:
            self.inbound_edge_dict[to_user_id] = set()
        self.inbound_edge_dict[to_user_id].add(from_user_id)
        #self.nxg.add_edge(from_user_id, to_user_id)
        self.wccg.add_edge(from_user_id, to_user_id)
    
    def get_edge_targets_for_user_id(self, user_id):
        """
        Returns the set of user_ids for which the edges user_id -> set(user_id) exist.
        """
        if user_id in self.outbound_edge_dict:
            return self.outbound_edge_dict[user_id]
        else:
            return set()
        
    def get_outdegree(self, user_id):
        if user_id in self.outbound_edge_dict:
            return len(self.outbound_edge_dict[user_id])
        else:
            return 0
        
    def get_indegree(self, user_id):
        if user_id in self.inbound_edge_dict:
            return len(self.inbound_edge_dict[user_id])
        else:
            return 0
        
    def get_component_size(self, user_id):
        return self.wccg.get_component_size(user_id)
    
    def are_weakly_connected(self, user_id1, user_id2):
        return self.wccg.are_weakly_connected(user_id1, user_id2)
    
    def compute_is_friend_of_friend(self, user_id1, user_id2):
        if self.get_outdegree(user_id1) == 0 or self.get_outdegree(user_id2) == 0: 
            # if there are zero outbound edges from one of the nodes, they can't be strongly connected
            return False
        return self.are_fof_connected(user_id1, user_id2) and self.are_fof_connected(user_id2, user_id1)

    def are_fof_connected(self, source, target):
        # must be a direct connection from either source -> target, or from source -> neighbor -> target
        if source not in self.outbound_edge_dict:
            # source has no outdegree
            return False
        if target in self.outbound_edge_dict[source]:
            return True
        for neighbor in self.outbound_edge_dict[source]:
            if neighbor in self.outbound_edge_dict and target in self.outbound_edge_dict[neighbor]:
                return True
        return False
    
    def is_reciprocal(self, source, target):
        """
        A reciprocal interaction is one where the target has already interacted with the source.
        """
        return target in self.outbound_edge_dict and source in self.outbound_edge_dict[target]


class WccGraph:
    """
    WccGraph (Weakly-connected Component Graph) is a graph implementation that exclusively tracks information about weak connectedness.  
    
    We do this by mapping individual nodes to components. At initialization, every node is in its own component.
    
    As edges are added, we check to see if the components are different; if they are, we combine the components.
    
    Adding intra-component edges is O(1).
    Adding inter-component edges is O(1), but requires copying n ids.
    Checking if two nodes are weakly connected is O(1).
    
    :node_uids: An optional set of node_uids to initialize in the network
    """
    def __init__(self, node_uids=None):
        self.node_dict = {}  # maps node_uid to component_uid
        self.component_dict = {}  # maps component_uid to a set of node_uids
        self.component_uid_counter = 0
        self.edge_count = 0
        if node_uids is not None:
            for node_uid in node_uids:
                self.component_uid_counter += 1
                component_uid = self.component_uid_counter
                self.node_dict[node_uid] = component_uid
                self.component_dict[component_uid] = set((node_uid,))
        
    def _ensure_node_exists(self, node_uid):
        """
        If the given node_uid doesn't exist, it is added along with a new component.
        """
        if node_uid not in self.node_dict:
            self.component_uid_counter += 1
            component_uid = self.component_uid_counter
            self.node_dict[node_uid] = component_uid
            self.component_dict[component_uid] = set((node_uid,))
        
    def add_edge(self, from_node_uid, to_node_uid):
        self._ensure_node_exists(from_node_uid)
        self._ensure_node_exists(to_node_uid)
        self.edge_count += 1
        from_component_uid = self.node_dict[from_node_uid]
        to_component_uid = self.node_dict[to_node_uid]
        if from_component_uid == to_component_uid:
            # these nodes are already weakly connected
            is_intra_component_edge = True
            from_component_size, to_component_size = 0, 0
        else:  # two different components are being merged with this edge
            is_intra_component_edge = False
            from_component_nodes = self.component_dict[from_component_uid]
            to_component_nodes = self.component_dict[to_component_uid]
            from_component_size = len(from_component_nodes)
            to_component_size = len(to_component_nodes)
            
            if from_component_size >= to_component_size:
                # merge To component into From component, deleting the To component
                from_component_nodes.update(to_component_nodes)
                del self.component_dict[to_component_uid]
                for node_uid in to_component_nodes:
                    # update the merged in component ids
                    self.node_dict[node_uid] = from_component_uid
            else:
                # merge From component into To component, deleting the From component
                to_component_nodes.update(from_component_nodes)
                del self.component_dict[from_component_uid]
                for node_uid in from_component_nodes:
                    # update the merged in component ids
                    self.node_dict[node_uid] = to_component_uid
        return is_intra_component_edge, from_component_size, to_component_size
    
    def are_weakly_connected(self, user_id1, user_id2):
        if user_id1 not in self.node_dict or user_id2 not in self.node_dict:
            return False
        # two nodes are weakly connected if they exist in the same WCC
        return self.node_dict[user_id1] == self.node_dict[user_id2]
    
    def get_component_size(self, user_id):
        if user_id not in self.node_dict:
            return 1
        component_uid =  self.node_dict[user_id]
        return len(self.component_dict[component_uid])