class Solution:
    def addDigits(self, num):
        """
        :type num: int
        :rtype: int
        """
        
        def summ(ss):
            temp=0
            for digit in str(ss):
                temp+=int(digit)
            return temp

        while num/10>=1:
            num=summ(num)
        return num 
            
            
