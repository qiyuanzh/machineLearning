class solution:
    def selfdividingnumber(self, left,right):
        result=[]
        
        for i in range(left,right+1):
            if '0' in str(i):
                continue
            if sum([i%int(digit) for digit in str(i)])==0:
                result.append(i)
                
        return result
    
test=solution()
test.selfdividingnumber(1,22)
