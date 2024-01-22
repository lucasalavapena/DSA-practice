from math import ceil, log2
from collections import defaultdict

class DSU:
    def __init__(self, N):
        self.p = list(range(N)) # parent
        self.counts = [0] * N # i think rather than counts, ranks are also sometimes used

    def find(self, x):
        if self.p[x] != x:
            self.p[x] = self.find(self.p[x])
        return self.p[x]

    def union(self, x, y):
        xr = self.find(x)
        yr = self.find(y)
        if xr == yr: return False
        # add to the larger one
        elif self.counts[xr] < self.counts[yr]:
            self.p[xr] = yr
            self.counts[yr] += self.counts[xr]
            # could set self.counts[xr] = 0 and same below
        else:
            self.p[yr] = xr
            self.counts[xr] += self.counts[yr]


class SparseTable:
    # O(N lg N)
    def __init__(self, arr):
        N = len(arr)
        B = ceil(log2(N)) + 1
        self.table = [[float("inf")] * B for i in range(N)]        
        self.bin_log = [0] * (N +1) # precomputing log2
        for i in range(2, N+1):
            self.bin_log[i] = self.bin_log[i//2]+1

        for i, a in enumerate(arr):
            self.table[i][0] = a

        for k in range(1, B):
            i = 0
            while i + (1 << k) - 1 < N:
                self.table[i][k] = min(self.table[i][k-1], self.table[i+(1<<(k-1))][k-1])
                i += 1
    # O(1)
    def query(self, l, r):
        length = r - l + 1
        k = self.bin_log[length] 
        return min(self.table[l][k], self.table[r-(1<<k)+1][k])
    
# Note this is the sum version, you could easily change it or make it more generic
# But not it only works for cumulative properties like addition but not for max
class BIT:
    def __init__(self, n: int):
        self.sums = [0] * (n + 1) # is 1 indexed

    # O(lg N)
    def update(self, i: int, delta: int) -> None:
        while i < len(self.sums):
            self.sums[i] += delta
            i += i & (-i) # add least sig bit

    # O(lg N)
    def query(self, i: int) -> int:
        res = 0
        while i > 0:
            res += self.sums[i]
            i -= i & (-i) # sub least sig bit
        return res
    


class TrieNode:
    def __init__(self):
        self.children = defaultdict(TrieNode)
        self.is_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        curr = self.root
        for letter in word:
            curr = curr.children[letter]
        curr.is_word = True
         
    def search(self, word: str) -> bool:
        curr = self.root
        for letter in word:
            curr = curr.children.get(letter)
            if curr is None:
                return False        
        return curr.is_word
    
    def startsWith(self, prefix: str) -> bool:
        curr = self.root
        for letter in prefix:
            curr = curr.children.get(letter)
            if curr is None:
                return False        
        return True

