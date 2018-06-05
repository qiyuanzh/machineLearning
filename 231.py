class Solution:
    def isPowerOfTwo(self, n):
        """
        :type n: int
        :rtype: bool
        """
        if n<1:
            return False
        while not n%2:      #if can be divided by 2
            n=n/2
        return n==1 
