class Solution:
    def titleToNumber(self, s):
        """
        :type s: str
        :rtype: int
        """
        n=len(s)
        temp=0
        for i, x in enumerate(s):#enumerate add index to the loop value
            temp+=(ord(x)-ord('A')+1)*26**(n-i-1)#** is the exponential
        
        return temp
