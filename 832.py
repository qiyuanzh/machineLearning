class Solution:
    def flipAndInvertImage(self, A):
        """
        :type A: List[List[int]]
        :rtype: List[List[int]]
        """
        result=[]
        for first in A:
            first_ss=first[::-1]
            temp=[]
            for intra in first_ss:
                if intra==0:
                    temp.append(1)
                else:
                    temp.append(0)
            result.append(temp)
        return result
