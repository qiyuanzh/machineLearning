class Solution:
    def maximumProduct(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        temp=sorted(nums)[::-1]
        return max(temp[0]*temp[1]*temp[2],temp[0]*temp[-1]*temp[-2])
