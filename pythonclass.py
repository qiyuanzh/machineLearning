# -*- coding: utf-8 -*-
"""
Created on Thu May 10 10:18:45 2018

@author: zhaogra
"""

#python class tutorial 
import datetime
class Employee:
    numberofem=0 #this is the class variable
    pay_factor=1.04
    def __init__(self,first, last, pay):
        self.first=first
        self.last=last
        self.payamount=pay
        Employee.numberofem+=1
    def pay(self):
        return self.payamount*self.pay_factor #you can use class variable by Employee.pay_factor or using instance variable by self.pay_factor
    
    @classmethod
    def tranform(cls,st):
        first, last, pay=st.split('-')
        return cls(first, last, pay)
    @staticmethod
    def is_weekday(day):
        if day.weekday()==5 or day.weekday()==6:
            return False
        return True
    
print(Employee.numberofem)
grant=Employee('grant','zhao',1000000)
yaoming=Employee('yaoming','he',1000)        
    

d=datetime.datetime(2016,1,1)

Employee.is_weekday(d)

st1='Michael-Jordan-10000000'
michael=Employee.tranform(st1)
michael.payamount
grant.pay_factor=2


print(grant.pay())
print(yaoming.pay())

print(Employee.numberofem)

