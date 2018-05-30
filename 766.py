class Solution:
    def isToeplitzMatrix(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: bool
        """
        for i in range(0,len(matrix)-1):
            if matrix[i][:-1]!=matrix[i+1][1:]:
                return False
        else:
            return True
