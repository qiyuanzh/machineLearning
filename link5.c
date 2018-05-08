# -*- coding: utf-8 -*-
"""
Created on Tue May  8 15:19:57 2018

@author: zhaogra
"""



from scipy import optimize
import pandas as pd
import numpy as np
import queue
import datetime
from dateutil.relativedelta import relativedelta









if __name__ == '__main__':
    ##
    ## Check that the port is the same as on the Gateway
    ## ipaddress is 127.0.0.1 if one same machine, clientid is arbitrary
    
  
        
        
   
##calculate first momentum basket
    price_equity=pd.read_csv("price_equity.csv")
    price_equity=price_equity.set_index('date')
    price_equity.index=pd.to_datetime(price_equity.index,format='%Y%m%d')
    
    return_equity=np.log(price_equity).diff()
    
 #define a new way to calculate cove
    def variance(x):
        
        temp=pd.DataFrame(data=x.copy())
        ind=range(len(x.copy()))[::-1]
        temp['exp']=ind
        temp.columns=["return","exp"]
        ii=temp['exp'].values
        temp['factor']=pd.Series(data=(1-0.95)*pow(0.95,ii),index=temp.index)
        temp['return']=pow(temp['return'],2)
        temp['weight']=temp['factor']/(temp['factor'].sum())
        temp['weighted return2']=temp['weight']*temp['return']
        tt=np.sqrt(temp['weighted return2'].sum()*252)
        return tt
    
    def adjust_variance(x):
        
        
        temp=pd.DataFrame(data=x.copy())
        ind=range(len(x.copy()))[::-1]
        temp['exp']=ind
        temp.columns=["return","exp"]
        ii=temp['exp'].values
        temp['factor']=pd.Series(data=(1-0.95)*pow(0.95,ii),index=temp.index)
        temp['weight']=np.sqrt(len(x.copy())*temp['factor']/(temp['factor'].sum()))
        temp['weighted return']=temp['weight']*temp['return']
        tt=temp['weighted return'].std()*(252**0.5)
        return tt
    
    return_equity_sigma=return_equity.rolling(63).apply(adjust_variance)
    #adjust_return_equity is based on sharpe ratio
    adjust_return_equity=return_equity.rolling(63).sum()/return_equity_sigma
    ##temp=return_equity.rolling(63).std()
   
  
    adjust_return_equity.index=adjust_return_equity.index.to_datetime()
    ##get the condition of momentum    
    
    condition=adjust_return_equity[adjust_return_equity.index.day==5].shift(1)
    condition_rank=condition.rank(1)
    #assume we only start trading from 2017-12-05 due to calendar date
    rank_mapping={1:0.3,2:0.25,3:0.2,4:0.15,5:0.1,6:0,7:0,8:0,9:0}    
    equity_weight=condition_rank.replace(rank_mapping)
    equity_weight=equity_weight.reindex(adjust_return_equity.index,fill_value=np.nan)
    #equity_weight=equity_weight.shift(-1)
    equity_weight=equity_weight.fillna(method='ffill')
    #the above dataframe saved all the weightings for etf momentum
    #equity_weight[equity_weight['XLP'].notnull()]
    #assume 1mm dollars investment for each momentum basket
    equity_weight_cal=equity_weight[datetime.datetime(2017,12,5):]
    equity_return_cal=return_equity
    equity_return_cal.index=equity_return_cal.index.to_datetime()
    equity_return_cal=equity_return_cal[datetime.datetime(2017,12,5):]
    equity_port=pd.DataFrame(np.diag(equity_weight_cal.dot(equity_return_cal.T)),index=equity_return_cal.index)
    
    equity_port.columns=['ret_d']
    equity_port['modify_ret_d']=equity_port['ret_d']
    equity_port.loc[equity_port.index.day==5,'modify_ret_d']=0
    equity_port['1plusR']=equity_port['modify_ret_d'].cumsum()+1
    equity_port['value']=equity_port['1plusR']*1000000
    equity_port['value'].plot()
    
    ##calculate the total equity basket price
    total_equity_weight=equity_weight.fillna(method='bfill')
    ##this is all return for equity
    total_equity_port=pd.DataFrame(np.diag(total_equity_weight.dot(return_equity.T)),index=return_equity.index) 
    total_equity_port.columns=['ret_d']
    total_equity_port['modify_ret_d']=total_equity_port['ret_d']
    total_equity_port.loc[equity_port.index.day==5,'modify_ret_d']=0
    total_equity_port['1plusR']=total_equity_port['modify_ret_d'].cumsum()+1
    total_equity_port['value']=total_equity_port['1plusR']*1000000
    total_equity_port['value'].plot()
    ##calculate equity basket    
    total_equity_port['equity_bask']=252*total_equity_port['modify_ret_d'].rolling(63).sum()/63
    #this equity_bask is annulized return
    total_equity_port['equity_bask'].plot()
    
    
    
    #dealing with other assets
    price_exequity=pd.read_excel('otherasset.xlsx')
    price_exequity=price_exequity[datetime.datetime(2017,4,21):]
    #price_exequity=price_etf.iloc[:,9:]
    price_exequity=price_exequity.dropna(how='any')
    price_exequity.reindex(total_equity_port.index).plot()
    return_exequity=np.log(price_exequity).diff()
    adjust_return_exequity=252*return_exequity.rolling(63).sum()/63
    adjust_return_exequity.index=adjust_return_exequity.index.to_datetime()    
    adjust_return_exequity.plot()
    
    adjust_return_all=adjust_return_exequity
    final_return=pd.concat([adjust_return_all,total_equity_port['equity_bask']],axis=1)
    
#    
#    final_return.cov()
#    final_return.describe()
    
    
    final=final_return.dropna(how='any')
    
#    final_cov=(final-final.mean()).T.dot((final-final.mean()))/(189)
#    final_cov.plot()
#    
#    a=np.matrix(1/np.diag(final_cov))
#    dd=np.sqrt(pd.DataFrame(np.diag(a.A1)))
#    dd.columns=final_cov.columns
#    dd.index=final_cov.index
#    (dd.dot(final_cov)).dot(dd)
#    final.corr().plot()
    
    #covariance matrix of heatmap
    import seaborn as sns
    sns.heatmap(final.rolling(63,63).cov().loc['2018-04-05'])
    final.rolling(63,63).corr().loc['2018-04-05']
    #create the rebalancing framework
    rebalance=final[final.index.day==5]
    rebalance
    
    final_return.loc[rebalance.index[1]]
    #define a new way to calculate cove
    def variance(x):
        
        temp=pd.DataFrame(data=x.copy())
        ind=range(len(x.copy()))[::-1]
        temp['exp']=ind
        temp.columns=["return","exp"]
        ii=temp['exp'].values
        temp['factor']=pd.Series(data=(1-0.95)*pow(0.95,ii),index=temp.index)
        temp['return']=pow(temp['return'],2)
        temp['weight']=temp['factor']/(temp['factor'].sum())
        temp['weighted return2']=temp['weight']*temp['return']
        tt=np.sqrt(temp['weighted return2'].sum()*252)
        return tt
    
    #finally fouond out how to do that you need to data=x.copy() use the copy
  
    
    a1=return_equity.rolling(63).apply(variance)
    
    def adjust_variance(x):
        
        
        temp=pd.DataFrame(data=x.copy())
        ind=range(len(x.copy()))[::-1]
        temp['exp']=ind
        temp.columns=["return","exp"]
        ii=temp['exp'].values
        temp['factor']=pd.Series(data=(1-0.95)*pow(0.95,ii),index=temp.index)
        temp['weight']=np.sqrt(len(x.copy())*temp['factor']/(temp['factor'].sum()))
        temp['weighted return']=temp['weight']*temp['return']
        tt=temp['weighted return'].std()*(252**0.5)
        return tt
    
    a2=return_equity.rolling(63).apply(adjust_variance)
    
#    def adjust_return(x):
#        temp=pd.DataFrame(data=x.copy())
#        ind=range(len(x.copy()))[::-1]
#        temp['exp']=ind
#        ii=temp['exp'].values
#        temp['factor']=pd.Series(data=(1-0.95)*pow(0.95,ii),index=temp.index)
#        temp['weight']=np.sqrt(len(x.copy())*temp['factor']/(temp['factor'].sum()))
#        index=temp['weight']
#        #return x.copy().mul(index,axis=0).corr()
#        return pd.DataFrame((x.copy().T*index.values).T).values
    
    
    
    def adjust_cov(x):
            temp=pd.DataFrame(data=x.copy())
            ind=range(len(x.copy()))[::-1]
            temp['exp']=ind
            ii=temp['exp'].values
            temp['factor']=pd.Series(data=(1-0.95)*pow(0.95,ii),index=temp.index)
            temp['weight']=np.sqrt(len(x.copy())*temp['factor']/(temp['factor'].sum()))
            index=temp['weight']
            return x.copy().mul(index,axis=0).cov()
    
    
    
    def dynamic_cov(y):
        total=pd.DataFrame(index=y.copy().index)
        tt={}
        for i in range(len(y))[::-1]:
            if i>(63):
                j=i+1-63
                tt[total.iloc[i].name]=adjust_cov(y.iloc[j:i+1])        
        ff=pd.Series(tt)
        ff=ff.reindex(y.index,fill_value=np.nan)
        return ff
        
               
#
#
#            
#        return pd.DataFrame((x.copy().T*index.values).T).values
#
#    adjust_return(return_equity)
    
    #get rebalance date
    bench=datetime.datetime(2017, 10, 5)
    bb=[]
    for i in range(7):
        bb.append(bench+relativedelta(months=i))
    bb_right=[]
    for i in bb:
        x=i
        while x not in final.index:
            x=x+relativedelta(days=1)
        bb_right.append(x)
        
    rebalance=final.loc[pd.to_datetime(bb_right)]
            
        
        
        
    rebalance=rebalance['2017-12-05':]
    def get_optweight(reba):
        #try to optimize the return
        optweight=pd.DataFrame(index=reba.index,columns=reba.columns)
        for i in reba.index:
            
            return_for_weight=final_return.loc[i]
            #asset_cov=final.rolling(63,63).cov().loc[i]
            asset_cov=dynamic_cov(final).loc[i]
            
            def obj_func(weight):
                return -1*(weight*return_for_weight).sum()
        
            def obj_func_gd(weight):
                return -1*return_for_weight.values.flatten()
            
            def constr_0 (weight):
                
                return 0.05-np.sqrt(np.dot(weight, np.dot(asset_cov,weight)))
            
            def constr_0_gd(weight):
                return -1*np.dot(asset_cov,weight).flatten()
            
            #all bigger than or equal to 0
            def constr_1(weight):
                return weight.sum()-1
            def constr_1_gd(weight):
                return np.ones(weight.shape[0]).flatten()
            
            def constr_2(weight):
                return 0.75-(weight*pd.Series([0,0,0,1,1,0,0,0,0,0,1])).sum()
            def constr_2_gd(weight):
                return -1*pd.Series([0,0,0,1,1,0,0,0,0,0,1]).values.flatten()
            
            def constr_3(weight):
                return 0.5-(weight*pd.Series([0,0,1,0,0,1,0,0,0,0,0])).sum()
            def constr_3_gd(weight):
                return -1*pd.Series([0,0,1,0,0,1,0,0,0,0,0]).values.flatten()
                
            def constr_4(weight):
                return 0.2-(weight*pd.Series([1,1,0,0,0,0,0,0,0,0,0])).sum()
            def constr_4_gd(weight):
                return -1*pd.Series([1,1,0,0,0,0,0,0,0,0,0]).values.flatten()
                
            def constr_5(weight):
                return 0.3-(weight*pd.Series([0,0,0,0,0,0,1,1,1,0,0])).sum()
            def constr_5_gd(weight):
                return -1*pd.Series([0,0,0,0,0,0,1,1,1,0,0]).values.flatten()
  
            bound_low= 0
            bound_high= 0.225
           
            opt= optimize.minimize( fun= obj_func, jac= obj_func_gd, 
                               x0 = np.ones( asset_cov.shape[0]).flatten()* 1/asset_cov.shape[0] ,
                               constraints= ({'type': 'eq', 'fun': constr_0, 'jac': constr_0_gd}, 
                                            {'type': 'eq', 'fun': constr_1, 'jac': constr_1_gd},
                                            {'type': 'ineq', 'fun': constr_2, 'jac': constr_2_gd},
                                            {'type': 'ineq', 'fun': constr_3, 'jac': constr_3_gd},
                                            {'type': 'ineq', 'fun': constr_4, 'jac': constr_4_gd},
                                            {'type': 'ineq', 'fun': constr_5, 'jac': constr_5_gd}),
                
                               method='SLSQP',
                               bounds= [[bound_low, bound_high]]* asset_cov.shape[0],
                               options= {'disp':True}
                              )
            optimized_weight=pd.Series(opt.x)
            optweight.loc[i]=optimized_weight.values
            
        optweight=optweight.reindex(adjust_return_equity.index,fill_value=np.nan)
        optweight=optweight.fillna(method='ffill')
        return optweight
    
    opt_weight=get_optweight(rebalance)
    opt_weight=opt_weight[1:]
    
    regular_return=return_exequity.dropna(how='any')
    regular_return['equity_bask']=total_equity_port['modify_ret_d']
    regular_return=regular_return[:-1]
    np.dot(opt_weight,regular_return.T)
    
    
    opt_portreturn=pd.DataFrame(np.diag(np.dot(opt_weight,regular_return.T)),index=opt_weight.index)
    opt_portreturn["return plus"]=opt_portreturn[0]+1
    opt_portreturn["return plus"].fillna(1,inplace=True)
    opt_portreturn['return plus'].plot()
    opt_portreturn['equity line']=opt_portreturn["return plus"].cumprod()
    opt_portreturn['equity line'].loc['2017-12-05':].plot()
    opt_portreturn.head()
    opt_portreturn[0].plot()
    opt_weight.tail()
    
    regular_return.tail()
    (return_exequity.dropna(how='any').T).shape
    regular_return=return_exequity.dropna(how='any')
    return_exequity.dropna(how='any')
    opt_weight.loc['2017-12-05':].plot()
    
    equity_port['1plusR'].plot()
    tt=opt_portreturn.loc['2017-12-05':]
    tt['equity']=equity_port['1plusR']   
    tt.columns=[[0,1,"port equity line","US ETF momentum"]]
    tt[["port equity line","US ETF momentum"]].plot()
    
  
    opt_weight.loc[rebalance.index]
    
    #create shares matrix
    shares=pd.DataFrame(0,index=opt_weight.index,columns=opt_weight.columns)
    price_exequity['momentum']=total_equity_port['value']    
    
    #adjust your shares matrix according to your optimal weight
    shares['code']=range(len(shares))
    rebalance_period=[]
    #set the initial shares of assets
    shares.loc['2017-12-04',:'equity_bask']=np.array([100,100,100,100,100,100,100,100,100,100,100])
    for reba_index in rebalance.index:
        selection_value=shares.loc[reba_index]['code']-1
        selection_index=shares[shares['code']==selection_value].index
        
        #print(shares[shares['code']==selection_value].index)
        #show each rebalancing date
        for i in range(7):
            each_reba_index=shares[shares['code']==(selection_value+i+1)].index
            each_prev_index=shares[shares['code']==(selection_value+i)].index
            shares.loc[each_reba_index,:'equity_bask']=shares.loc[each_prev_index,:'equity_bask'].values-(shares.loc[selection_index,:'equity_bask'].values/7)+((shares.loc[selection_index,:'equity_bask'].values/7)*price_exequity.loc[each_reba_index].values).sum()*opt_weight.loc[each_reba_index].values/price_exequity.loc[each_reba_index].values
            rebalance_period.append(each_reba_index)
        i=i+1
        while (shares[shares['code']==(selection_value+i+1)].index.values in shares.index.values) and (shares[shares['code']==(selection_value+i+1)].index.values not in rebalance.index.values):
            temp_index=shares.loc[shares['code']==(selection_value+i+1)].index
            temp_prev_index=shares.loc[shares['code']==(selection_value+i)].index
            shares.loc[temp_index,:'equity_bask']=shares.loc[temp_prev_index,:'equity_bask'].values
#            print(temp_index)
            i=i+1
        #print(shares.loc[reba_index]['code']-1)
    price_exequity=price_exequity.loc[:'2018-04-20']
    new_port_value=np.diag(np.dot(shares.loc['2017-12-04':,:'equity_bask'],price_exequity.loc['2017-12-04':,:].T))
    new_port=pd.DataFrame(new_port_value,index=price_exequity.loc['2017-12-04':,:].index)
    new_port.columns=['name'] #sounds good to know if you want to change the column if you see that's the case
                 
    
    #now we need to calculate the portfolio index values
    #first we need to calculate the volatility control
    def dynamic_cov_st(y):
        total=pd.DataFrame(index=y.copy().index)
        tt={}
        for i in range(len(y))[::-1]:
            if i>(40):
                j=i+1-40
                tt[total.iloc[i].name]=adjust_cov(y.iloc[j:i+1])        
        ff=pd.Series(tt)
        ff=ff.reindex(y.index,fill_value=np.nan)
        return ff
    
    
    def dynamic_cov_lt(y):
        total=pd.DataFrame(index=y.copy().index)
        tt={}
        for i in range(len(y))[::-1]:
            if i>(80):
                j=i+1-80
                tt[total.iloc[i].name]=adjust_cov(y.iloc[j:i+1])        
        ff=pd.Series(tt)
        ff=ff.reindex(y.index,fill_value=np.nan)
        return ff
    
    df_lt=0.95
    df_st=0.9
    #calculate ST vol
    port_variance_st=pd.DataFrame(index=opt_weight.loc['2017-12-05':].index,columns=['variance'])
    
    for i in port_variance_st.index:
        if i in pd.Series(rebalance_period).values:
            port_variance_st.loc[i]=np.dot(np.dot(opt_weight.loc[i].T,dynamic_cov_st(final).loc[i]),opt_weight.loc[i])
        else:
            port_variance_st.loc[i]=df_st*port_variance_st.loc[port_variance_st.index[port_variance_st.index.get_loc(i)-1]].values+(1-df_st)*252*((np.log(new_port.loc[i].values/new_port.loc[new_port.index[new_port.index.get_loc(i)-1]].values))**2)
        
        
   
    port_variance_lt=pd.DataFrame(index=opt_weight.loc['2017-12-05':].index,columns=['variance'])
    
    for i in port_variance_lt.index:
        if i in pd.Series(rebalance_period).values:
            port_variance_lt.loc[i]=np.dot(np.dot(opt_weight.loc[i].T,dynamic_cov_lt(final).loc[i]),opt_weight.loc[i])
        else:
            port_variance_lt.loc[i]=df_lt*port_variance_lt.loc[port_variance_lt.index[port_variance_lt.index.get_loc(i)-1]].values+(1-df_lt)*252*((np.log(new_port.loc[i].values/new_port.loc[new_port.index[new_port.index.get_loc(i)-1]].values))**2)
             
        #it's a nice catch to get the location of the index
    port_variance=pd.DataFrame(index=opt_weight.loc['2017-12-05':].index)
    port_variance['ST']=port_variance_st
    port_variance['LT']=port_variance_lt
    port_variance['final']=port_variance.max(axis=1)
    port_variance=port_variance**0.5
    
    
    
    
    
    np.dot(opt_weight.loc['2018-01-05'], np.dot(dynamic_cov(final).loc['2018-01-05'], opt_weight.loc['2018-01-05']))**(0.5)
    #this is very right hahahaahahahahahahaha:) 5% annualized return
        
    #using stackbar to show component
#    n=opt_portreturn[opt_weight.index.day==5].loc['2017-12-05':].index.values
#    a=plt.plot(n)
#        n.plot()
    
#
#
#datetime.datetime.today()
#now=datetime.datetime.now()
#
#dd=loca_tz=pytz.timezone('US/Pacific')
#dd.astimezone(pytz.timezone('US/Mountain'))
#
#
#dt_str='January 01, 1985'
#datetime.datetime.strptime(dt_str,'%B %d, %Y')
#pd.to_datetime(dt_str,format='%B %d, %Y')
#
#now=loca_tz.localize(now)
#now.isoformat()
#bday.strftime('%b-%d-%Y')
#
#bday=datetime.datetime(1985,1,1,14,30,00,100000,tzinfo=pytz.timezone('Asia/Shanghai'))
#
#bday_eastern = bday.astimezone(pytz.timezone('US/Eastern'))
#
#
#
#
#bday_pacific=bday.astimezone(pytz.timezone('US/Pacific'))
#
#now
#dt=pd.to_datetime('2017-01-01',format='%Y-%m-%d')
    #try to optimize the return
    
    
    
    tt_inv=pd.DataFrame(np.linalg.pinv(tt.values),tt.columns,tt.index)
    tt.dot(tt_inv).describe()
    
    
    
    
