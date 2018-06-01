class Solution:
    def minMoves(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        min_num=min(nums)
        return sum(map(lambda x: x-min_num,nums))
        
        
Assume input = [1,2,3]
Let's say we want to elevate the first element to match the second element
The steps are 2-1 = 1
And now we have: [2, 2, 4]
If you check carefully, the difference between the third element the first element is still the same. This is because we also elevated the third element altogether w/ the first element.
Now we want to match the first and the second element w/ the third element, the steps are (4-2) = 2
Total steps = 1 + 2 = 3

Which equals to summing the difference of every elements in the list w/ the minimum element.
