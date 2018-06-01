class Solution:
    def romanToInt(self, s):
        """
        :type s: str
        :rtype: int
        """
        roman={'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}
        
        num=[roman.get(letter) for letter in s] #first create a list of numbers
        total=0
        for i, x in enumerate(num):#then use enumerate
            if max(num[i:])<=x:#this is very smart, if the max number follow the number is less than the number then add
                total+=x
            else:   #otherwise subtract
                total-=x
        return total
