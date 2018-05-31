class Solution:
    def findMaxConsecutiveOnes(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        t=0
        m=0
        for i in range(len(nums)):
          if nums[i]==0:
            m=max(t,m)
            t=0
          else:
            t+=1
        m=max(m,t)
        return m
