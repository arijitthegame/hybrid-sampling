import heapq
import pickle

## Implementation of a HuffmanTree for a heirarchical softmax. 
## The nodes are the words in the vocab. We store the node along with the probabilities. 
#WIP

class Node:
    def __init__(self, token, freq, nhid):
        self.nhid = nhid
        self.vec = torch.randn(nhid, requires_grad=True, dtype=torch.float)
        self.token = token
        self.freq = freq
        self.left = None
        self.right = None
        
    def __lt__(self, other):
        return self.freq < other.freq
    
    def __gt__(self, other):
        return self.freq > other.freq
    
    def __eq__(self, other):
        if(other == None):
            return False
        if(not isinstance(other, Node)):
            return False
        return self.freq == other.freq

class HuffmanTree:
    def __init__(self):
        self.heap = []
        self.codes = {}
        self.reverse_mapping = {}
        self.root = None
        
    def make_heap(self, frequency):
        for key in frequency:
            node = Node(key, frequency[key])
            heapq.heappush(self.heap, node)
            
    def merge_nodes(self):
        while(len(self.heap)>1):
            node1 = heapq.heappop(self.heap)
            node2 = heapq.heappop(self.heap)
            
            merged = Node(None, node1.freq + node2.freq)
            merged.left = node1
            merged.right = node2
            heapq.heappush(self.heap, merged)
            
    def make_codes_helper(self, root, current_code):
        if(root==None):
            return
        if(root.token != None):
            self.codes[root.token] = current_code
            self.reverse_mapping[current_code] = root.token
            return
        
        self.make_codes_helper(root.left, current_code + "0")
        self.make_codes_helper(root.right, current_code + "1")
        
    def make_codes(self):
        root = heapq.heappop(self.heap)
        self.root = root
        current_code = ""
        self.make_codes_helper(root, current_code)


