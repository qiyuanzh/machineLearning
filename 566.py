class Solution:
    def matrixReshape(self, nums, r, c):
        """
        :type nums: List[List[int]]
        :type r: int
        :type c: int
        :rtype: List[List[int]]
        """
       
        temp=[]
        res=[]
        if (r*c) != (len(nums)*len(nums[0])):
            return nums
        
        else:
            for i in range(len(nums)):
                temp=temp+nums[i] # this will give you a list with one dimension
                
            for j in range(r):
                res.append(temp[j*c:(j+1)*c]) #this will return list of list, that's the difference between '+' and append for list
                
            return res
