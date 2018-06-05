class Solution:
    def addStrings(self, num1, num2):
        """
        :type num1: str
        :type num2: str
        :rtype: str
        """
        temp1=0
        temp2=0
        for i in num1:
            temp1*=10
            temp1+=(ord(i)-ord('0')) #get ascill code
            
        for i in num2:
            temp2*=10
            temp2+=(ord(i)-ord('0'))
            
        return str(temp1+temp2)
            
