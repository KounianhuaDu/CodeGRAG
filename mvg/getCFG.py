import os
import tqdm
import threading
from threading import Thread
import argparse
import time

parser = argparse.ArgumentParser(description = 'designate the rootpath, cfgpath and astpath')
parser.add_argument('--rootpath', type=str, help = 'root path of .cpp source file')
parser.add_argument('--astpath', type=str, help = 'path store the parse cfg')
parser.add_argument('--cfgpath', type=str, help = 'path store the parse ast')
args = parser.parse_args()
command="./cfg "
# rootpath="testGetJson/"
rootpath = args.rootpath
astpath = args.astpath
cfgpath = args.cfgpath
# command="ls "
# rootpath="testGetJson/"
typeListTemp=[]
lock=threading.Lock()
detectSignal=0
def add1(ls):
    lock.acquire()
    global typeListTemp
    global detectSignal
    detectSignal=detectSignal+1
    typeListTemp = typeListTemp + ls
    lock.release()
def getResult(doc):
    # print("123444")
    global typeListTemp
    global detectSignal
    print(command + rootpath + doc + " -- " + astpath + ' ' + cfgpath)
    os.system(command + rootpath + doc + " -- " + astpath + ' ' + cfgpath)


docs = os.listdir(rootpath)

docsLen=len(docs)
# threadNum = 800
threadNum = 1200
if threadNum > docsLen:
    threadNum = docsLen
threadTime=int(docsLen/threadNum+1)
for j in range(0,threadTime):
    threadNumThisTurn=threadNum
    nextTarget=threadNum*j+threadNum
    lowerBound=threadNum*(j-1)+threadNum
    if nextTarget>docsLen:
        threadNumThisTurn=docsLen-lowerBound
    # print("aaaaaaaaaaaaaaa")
    for i in range(0,threadNumThisTurn):
        # print("processing ",file+"/"+docs[lowerBound+i])
        t=Thread(target=getResult, args=(docs[lowerBound+i],),name=docs[lowerBound+i])
        t.start()
    time.sleep(10)
    #time.sleep(120)
        




