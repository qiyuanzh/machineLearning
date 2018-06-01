if len(ops)==0:
            return m*n
        ll=[[0 for x in range(m)] for y in range(n)]
        
        for i in range(len(ops)):
            for q in range(ops[i][0]):
                for w in range(ops[i][1]):
                    ll[q][w]+=1
        total=[]
        for i in ll:
            total+=i
        m=max(total)
        return total.count(m)
        
       
class Solution:
    def maxCount(self, m, n, ops):
        """
        :type m: int
        :type n: int
        :type ops: List[List[int]]
        :rtype: int
        """
        if not ops: #this is so smart to see if ops is empty
            return m*n
        ms, ns=zip(*ops)
        return min(ms)*min(ns) 
        
 class Solution:
    def maxCount(self, m, n, ops):
        """
        :type m: int
        :type n: int
        :type ops: List[List[int]]
        :rtype: int
        """
        if not ops:
            return m*n
        min_row=ops[0][0]
        min_col=ops[0][1]
        for op in ops:
            min_row=min(min_row,op[0])
            min_col=min(min_col,op[1])
        return min_row*min_col 
