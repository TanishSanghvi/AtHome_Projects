#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 11:18:58 2020

@author: apple
"""


import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
from bs4 import BeautifulSoup
from dateutil import parser
from datetime import datetime, timedelta
from scipy import optimize
import math


browser=webdriver.Chrome("/Users/apple/Downloads/chromedriver 3")
browser.get('https://www.moneycontrol.com/')
time.sleep(2)

try:
    browser.find_element_by_xpath('//*[@id="mc_mainWrapper"]/header/div[1]/div/div[2]/div[2]/div[4]/a').click()
except: 
    time.sleep(10)
    browser.find_element_by_xpath('//*[@id="mc_mainWrapper"]/header/div[1]/div/div[2]/div[2]/div[4]/a').click()
######################
browser.find_element_by_xpath('//*[@id="mc_mainWrapper"]/header/div[4]/div/div[2]/ul/li[11]/a').click()
time.sleep(3)

browser.find_element_by_xpath('//*[@id="clickid1"]').click()
time.sleep(3)
browser.find_element_by_xpath('//*[@id="clickid2"]').click()
time.sleep(3)


#funds=['Canara Robeco Blue Chip Equity Fund - Regular Plan', 'PGIM India Midcap Opportunities Fund']

def xnpv(rate, cashflows):
    return sum([cf/(1+rate)**((t-cashflows[0][0]).days/365.0) for (t,cf) in cashflows])
 
def xirr(cashflows, guess=0.1):
    try:
        return optimize.newton(lambda r: xnpv(r,cashflows),guess)
    except:
        print('Calc Wrong')


def  meta(title, i, meta_data):            
            
            meta_data['title']=title
            meta_data['latest_value']=browser.find_element_by_xpath('//*[@id="portSumm2"]/div[2]/table/tbody/tr/td[2]/table/tbody/tr/td/table/tbody/tr['+str(i)+']/td[9]').text
            print("Meta Data Extracted")
            
            browser.find_elements_by_xpath('//div[starts-with(@id,"opn_2")]')[i-x].click()
            time.sleep(2)
            print("Getting Dates and Investments")
            dates=[]
            investments=[]
            
            try:
                count=(int(title.split(' ')[-1][1]))*2
            except:
                count=2
            
            for j in range(2,count+2,2):  
                    dates.append(browser.find_element_by_xpath('//*[@id="portSumm2"]/div[2]/table/tbody/tr/td[2]/table/tbody/tr/td/table/tbody/tr['+str(i+1)+']/td/table/tbody/tr[3]/td/table[1]/tbody/tr/td/form/table/tbody/tr['+str(j)+']/td[4]').text)
                    investments.append(browser.find_element_by_xpath('//*[@id="portSumm2"]/div[2]/table/tbody/tr/td[2]/table/tbody/tr/td/table/tbody/tr['+str(i+1)+']/td/table/tbody/tr[3]/td/table[1]/tbody/tr/td/form/table/tbody/tr['+str(j)+']/td[5]').text)
                    
            print("Dates Extracted")
            today=datetime.date(datetime.today())
            dates=[datetime.date(datetime.strptime(x, '%d/%m/%y')) for x in dates]
            
            diff=[]
            for d in range(0,len(dates)):
                diff.append(((today-dates[d]).days)-2)
                investments[d]=int(investments[d].replace(',' , ''))
                
            number_of_days=math.floor((sum(diff)/len(diff))+0.4)
            meta_data['number_of_days']=number_of_days
            meta_data['investment'] = sum(investments)
            
            dates.append(today)
            
            investments2=[-y for y in investments]
            latest_value=int(meta_data['latest_value'].replace(',' , ''))
            investments2.append(latest_value)
            
            merged=list(tuple(zip(dates,investments2)))
            merged=sorted(merged,key = lambda x: x[0])
            
            meta_data['xirr']=xirr(merged)*100
            
            avg_days=meta_data['number_of_days']/365
            meta_data['cagr']=((latest_value/meta_data['investment'])**(1/avg_days)-1)*100

            return meta_data
  
def process(i, x, meta_data, err, totals):
    
    funds=['Canara Robeco Blue Chip Equity Fund - Regular Plan', 'PGIM India Midcap Opportunities Fund', 'Quant Active Fund']
    title=browser.find_element_by_xpath('//*[@id="portSumm2"]/div[2]/table/tbody/tr/td[2]/table/tbody/tr/td/table/tbody/tr['+str(i)+']/td[1]').text
    ed_title=title.split('(')[0].strip()
    if ed_title in funds:
    #if 'PGIM' in title:
        print("Found: {0}".format(title))
        metrics=meta(title, i, meta_data)
        totals.append(metrics)
        print("Data Pushed \n")
        browser.find_elements_by_xpath('//div[starts-with(@id,"opn_2")]')[i-x].click()
        
        return totals

html = browser.page_source
soup = BeautifulSoup(html, features="lxml")
totals=[]
error=[]
x=1        
    
for i in range(2,100,2):
    meta_data={}
    err={}
    x+=1
    
    try:
        process(i, x, meta_data, err, totals)
        
    except:
        if title!='':
            try:
                totals=process(i, x, meta_data, err, totals)
                
            except Exception as e:
                print(e)
                print("Error: {0}\n".format(title))
                err['title']=title
                err['index']=i
                err['x']=x
                error.append(err)
            
        else:
            print("Done")
            break;
            
 
    
final_df=pd.DataFrame(totals)
final_df=final_df[['title','investment','latest_value', 'number_of_days', 'xirr', 'cagr']]

#browser.close()



