import re
import os.path
import numpy as np


def load(path):
    mails = os.listdir(path)
    spam=[]
    mss=[]
    for mail in mails:
        if re.search(r"^spmsg", mail):
            spam.append(mail)
        else:
            mss.append(mail)
            #print(mail)
    return spam, mss

def train(spam,mss,path):
    ns=0
    nm=0
    Vs={}
    Vm={}
    for mail in spam:
        with open(path+"/"+mail) as f:
            words = re.findall(r"[A-Za-z]+", f.read())
            ns+=len(words)
            for word in words:
                if word not in Vs:
                    Vs[word]=1
                    Vm[word]=0
                else:
                    Vs[word]+=1
    V=Vs
    for mail in mss:
        with open(path+"/"+mail) as f:
            words = re.findall(r"[A-Za-z]+", f.read())
            nm+=len(words)
            for word in words:
                if word not in Vm:
                    Vm[word]=1
                    Vs[word]=0
                else:
                    Vm[word]+=1

    Ps=len(spam)/(len(spam)+len(mss))
    Pm=len(mss)/(len(spam)+len(mss))
    Ns=dict(zip(Vs.keys(),np.log((np.array(list(Vs.values()))+1)/(ns+len(Vs)))))
    Nm=dict(zip(Vm.keys(),np.log((np.array(list(Vm.values()))+1)/(nm+len(Vm)))))

    return V,Ns,Nm,Ps,Pm

def pred(V,Ns,Nm,Ps,Pm,mail,path):
    result_spam=np.log(Ps)
    result_mss=np.log(Pm)
    with open(path + "/" + mail) as f:
        words = re.findall(r"[A-Za-z]+", f.read())
        for word in words:
            if word in V:
                result_spam+=Ns[word]
                result_mss+=Nm[word]
    return 1 if result_spam>result_mss else 0

def test(spam_test,mss_test,V,Ns,Nm,Ps,Pm,path):
    result_spam=[]
    result_mss=[]

    for mail in spam_test:
        result_spam.append(pred(V,Ns,Nm,Ps,Pm,mail,path))

    for mail in mss_test:
        result_mss.append(pred(V,Ns,Nm,Ps,Pm,mail,path))

    tp=sum(result_spam)
    fp=len(spam_test)-sum(result_spam)
    tn=len(mss_test)-sum(result_mss)
    fn=sum(result_mss)
    p=tp/(tp+fp)
    r=tp/(tp+fn)
    fpr = fp/(fp+tn)
    fnr = fn/(tp+fn)
    ACY = (tp+tn)/(tp+tn+fp+fn)
    print("TP=", tp, "\nFN=", fn, "\nFP=", fp, "\nTN=", tn)
    print("ACY=",ACY)
    print("\nP=",p,"\nTPR=",r)
    print("\nFPR=", fpr, "\nFNR=",fnr)

def main():
    path_train="hw3_nb/train_mails"
    spam_train,mss_train=load(path_train)
    V,Ns,Nm,Ps,Pm=train(spam_train,mss_train,path_train)
    path_test="hw3_nb/test_mails"
    spam_test,mss_test=load(path_test)
    test(spam_test,mss_test,V,Ns,Nm,Ps,Pm,path_test)
main()