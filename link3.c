from ibapi.wrapper import EWrapper
from ibapi.client import EClient
from ibapi.contract import Contract
from threading import Thread

from scipy import optimize
import pandas as pd
import numpy as np
import queue
import datetime
from dateutil.relativedelta import relativedelta

DEFAULT_HISTORIC_DATA_ID=50
DEFAULT_GET_CONTRACT_ID=43

FINISHED = object()
STARTED = object()
TIME_OUT = object()





class finishableQueue(object):

    def __init__(self, queue_to_finish):

        self._queue = queue_to_finish
        self.status = STARTED

    def get(self, timeout):
        """
        Returns a list of queue elements once timeout is finished, or a FINISHED flag is received in the queue
        :param timeout: how long to wait before giving up
        :return: list of queue elements
        """
        contents_of_queue=[]
        finished=False

        while not finished:
            try:
                current_element = self._queue.get(timeout=timeout)
                if current_element is FINISHED:
                    finished = True
                    self.status = FINISHED
                else:
                    contents_of_queue.append(current_element)
                    ## keep going and try and get more data

            except queue.Empty:
                ## If we hit a time out it's most probable we're not getting a finished element any time soon
                ## give up and return what we have
                finished = True
                self.status = TIME_OUT


        return contents_of_queue

    def timed_out(self):
        return self.status is TIME_OUT






class TestWrapper(EWrapper):
    """
    The wrapper deals with the action coming back from the IB gateway or TWS instance
    We override methods in EWrapper that will get called when this action happens, like currentTime
    """
    def __init__(self):
        self._my_contract_details = {}
        self._my_historic_data_dict = {}

    ## error handling code
    def init_error(self):
        error_queue=queue.Queue()
        self._my_errors = error_queue

    def get_error(self, timeout=5):
        if self.is_error():
            try:
                return self._my_errors.get(timeout=timeout)
            except queue.Empty:
                return None

        return None


    def is_error(self):
        an_error_if=not self._my_errors.empty()
        return an_error_if

    def error(self, id, errorCode, errorString):
        ## Overriden method
        errormsg = "IB error id %d errorcode %d string %s" % (id, errorCode, errorString)
        self._my_errors.put(errormsg)

    ## Time telling code
    def init_time(self):
        time_queue=queue.Queue()
        self._time_queue = time_queue

        return time_queue

    def currentTime(self, time_from_server):
        ## Overriden method
        self._time_queue.put(time_from_server)



 ## get contract details code
    def init_contractdetails(self, reqId):
        contract_details_queue = self._my_contract_details[reqId] = queue.Queue()

        return contract_details_queue

    def contractDetails(self, reqId, contractDetails):
        ## overridden method

        if reqId not in self._my_contract_details.keys():
            self.init_contractdetails(reqId)

        self._my_contract_details[reqId].put(contractDetails)

    def contractDetailsEnd(self, reqId):
        ## overriden method
        if reqId not in self._my_contract_details.keys():
            self.init_contractdetails(reqId)

        self._my_contract_details[reqId].put(FINISHED)

 ## Historic data code
    def init_historicprices(self, tickerid):
        historic_data_queue = self._my_historic_data_dict[tickerid] = queue.Queue()

        return historic_data_queue


    def historicalData(self, tickerid , bar):

        ## Overriden method
        ## Note I'm choosing to ignore barCount, WAP and hasGaps but you could use them if you like
        bardata=(bar.date, bar.open, bar.high, bar.low, bar.close, bar.volume)

        historic_data_dict=self._my_historic_data_dict

        ## Add on to the current data
        if tickerid not in historic_data_dict.keys():
            self.init_historicprices(tickerid)

        historic_data_dict[tickerid].put(bardata)

    def historicalDataEnd(self, tickerid, start:str, end:str):
        ## overriden method

        if tickerid not in self._my_historic_data_dict.keys():
            self.init_historicprices(tickerid)

        self._my_historic_data_dict[tickerid].put(FINISHED)

class TestClient(EClient):
    """
    The client method
    We don't override native methods, but instead call them from our own wrappers
    """
    def __init__(self, wrapper):
        ## Set up with a wrapper inside
        EClient.__init__(self, wrapper)

    def speaking_clock(self):
        """
        Basic example to tell the time
        :return: unix time, as an int
        """

        print("Getting the time from the server... ")

        ## Make a place to store the time we're going to return
        ## This is a queue
        time_storage=self.wrapper.init_time()

        ## This is the native method in EClient, asks the server to send us the time please
        self.reqCurrentTime()

        ## Try and get a valid time
        MAX_WAIT_SECONDS = 10

        try:
            current_time = time_storage.get(timeout=MAX_WAIT_SECONDS)
        except queue.Empty:
            print("Exceeded maximum wait for wrapper to respond")
            current_time = None

        while self.wrapper.is_error():
            print(self.get_error())

        return current_time
        
    def resolve_ib_contract(self, ibcontract, reqId=DEFAULT_GET_CONTRACT_ID):

        """
        From a partially formed contract, returns a fully fledged version
        :returns fully resolved IB contract
        """

        ## Make a place to store the data we're going to return
        contract_details_queue = finishableQueue(self.init_contractdetails(reqId))

        print("Getting full contract details from the server... ")

        self.reqContractDetails(reqId, ibcontract)

        ## Run until we get a valid contract(s) or get bored waiting
        MAX_WAIT_SECONDS = 10
        new_contract_details = contract_details_queue.get(timeout = MAX_WAIT_SECONDS)

        while self.wrapper.is_error():
            print(self.get_error())

        if contract_details_queue.timed_out():
            print("Exceeded maximum wait for wrapper to confirm finished - seems to be normal behaviour")

        if len(new_contract_details)==0:
            print("Failed to get additional contract details: returning unresolved contract")
            return ibcontract

        if len(new_contract_details)>1:
            print("got multiple contracts using first one")

        new_contract_details=new_contract_details[0]

        resolved_ibcontract=new_contract_details.summary

        return resolved_ibcontract


    def get_IB_historical_data(self, ibcontract, durationStr="1 Y", barSizeSetting="1 day",
                               tickerid=DEFAULT_HISTORIC_DATA_ID):

        """
        Returns historical prices for a contract, up to today
        ibcontract is a Contract
        :returns list of prices in 4 tuples: Open high low close volume
        """


        ## Make a place to store the data we're going to return
        historic_data_queue = finishableQueue(self.init_historicprices(tickerid))

        # Request some historical data. Native method in EClient
        self.reqHistoricalData(
            tickerid,  # tickerId,
            ibcontract,  # contract,
            datetime.datetime.today().strftime("%Y%m%d %H:%M:%S %Z"),  # endDateTime,
            durationStr,  # durationStr,
            barSizeSetting,  # barSizeSetting,
            "TRADES",  # whatToShow,
            1,  # useRTH,
            1,  # formatDate
            False,  # KeepUpToDate <<==== added for api 9.73.2
            [] ## chartoptions not used
        )



        ## Wait until we get a completed data, an error, or get bored waiting
        MAX_WAIT_SECONDS = 10
        print("Getting historical data from the server... could take %d seconds to complete " % MAX_WAIT_SECONDS)

        historic_data = historic_data_queue.get(timeout = MAX_WAIT_SECONDS)

        while self.wrapper.is_error():
            print(self.get_error())

        if historic_data_queue.timed_out():
            print("Exceeded maximum wait for wrapper to confirm finished - seems to be normal behaviour")

        self.cancelHistoricalData(tickerid)


        return historic_data


class TestApp(TestWrapper, TestClient):
    def __init__(self, ipaddress, portid, clientid):
        TestWrapper.__init__(self)
        TestClient.__init__(self, wrapper=self)

        self.connect(ipaddress, portid, clientid)

        thread = Thread(target = self.run)
        thread.start()

        setattr(self, "_thread", thread)

        self.init_error()




if __name__ == '__main__':
    ##
    ## Check that the port is the same as on the Gateway
    ## ipaddress is 127.0.0.1 if one same machine, clientid is arbitrary
    
  
        
        
    app = TestApp('127.0.0.1',4001,10)

    current_time = app.speaking_clock()


    ##define the function to get price etf
    def hist_etf(lis):
        price_etf=pd.DataFrame()
        for i in lis:
            contract = Contract()
            contract.symbol = i
            contract.secType = "STK"
            contract.currency = "USD"
            if i=="EMB":
                contract.exchange="AMEX"
            else:                
                contract.exchange = "SMART"   
            resolved_ibcontract=app.resolve_ib_contract(contract)
            historic_data = app.get_IB_historical_data(resolved_ibcontract)   
            temp=pd.DataFrame(historic_data)
            temp.set_index([0],inplace=True)
            temp.index.name='date'
            temp=temp[4]
            price_etf[i]=temp
            print(i)
        return price_etf

    def hist_bond(lis):
        price_bond=pd.DataFrame()
        for i in lis:
            contract = Contract()
            contract.symbol = i
            contract.secType = "BOND"
            contract.currency = "USD"
            if i=="EMB":
                contract.exchange="AMEX"
            else:                
                contract.exchange = "SMART"   
            resolved_ibcontract=app.resolve_ib_contract(contract)
            historic_data = app.get_IB_historical_data(resolved_ibcontract)   
            temp=pd.DataFrame(historic_data)
            temp.set_index([0],inplace=True)
            temp.index.name='date'
            temp=temp[4]
            price_bond[i]=temp
            print(i)
        return price_bond
    ticker_bond=['IBCID19238708','BCID306382844','IBCID48921642']
    price_bond=hist_bond(ticker_bond)

    ticker_etf=['XLF','XLV','XLU','XLI','XLE','XLY','XLP','XLB','XLK','EEM','EFA','HYG','EMB','USO','IAU','SOYB','VNQ']
    price_etf=hist_etf(ticker_etf)


##calculate first momentum basket
    #price_equity=pd.read_csv("price_equity.csv")
    #price_equity=price_equity.set_index('date')
    #price_equity.index=pd.to_datetime(price_equity.index,format='%Y%m%d')
    price_equity=price_etf.iloc[:,:9]
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
    
    
    shares=pd.DataFrame(0,index=opt_weight.index,columns=opt_weight.columns)
    price_exequity['momentum']=total_equity_port['value']
    
    
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
    return_for_weight=final_return.loc['2018-04-05']
    
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
        
    
    asset_cov=final.rolling(63,63).cov().loc['2018-04-05']
    
    bound_low= 0
    bound_high= 0.225
    from scipy import optimize
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
    optimized_std=np.sqrt(np.dot(np.dot(optimized_weight,asset_cov),optimized_weight))
 
    #calculate the daily return of optimized portfolio
    return_weight=return_exequity
    return_weight['equity_basket']=total_equity_port['modify_ret_d']
    pd.Series(np.dot(return_weight.loc['2018-04-05':],optimized_weight),index=final_return.loc['2018-04-05':].index).sum()
    return_weight['equity_basket'].loc['2018-04-05':].sum()
    
    weight=pd.Series(opt.x, index=final_return.columns)
    weight.T.plot()
    
    import matplotlib.pyplot as plt

    
    
    tt_inv=pd.DataFrame(np.linalg.pinv(tt.values),tt.columns,tt.index)
    tt.dot(tt_inv).describe()
    
    
    
    
    
    contract = Contract()
    contract.symbol = "US-T"
    contract.secType = "BOND"
    contract.currency = "USD"
    contract.localSymbol='IBCID192387082'
    contract.exchange = "SMART"   
    resolved_ibcontract=app.resolve_ib_contract(contract)

    historic_data = app.get_IB_historical_data(resolved_ibcontract)   
    
    print(historic_data)
    
    tencent=pd.DataFrame(historic_data)
    tencent.set_index([0],inplace=True)
    tencent.index.name='date'
    tencent=tencent[4]
    tencent.head()
    tencent.columns
    
    '''
     queryTime=(datetime.today()-timedelta(hours=3)-timedelta(days=1)) 
     querytime=queryTime.strftime("%Y%m%d %H:%M:%S")
    aa=app.reqHistoricalData(1000,contract,querytime,"1 M","1 Day","MIDPOINT", 1, 1, False,[])
    
    
    print(current_time)
'''
    #app.disconnect()
