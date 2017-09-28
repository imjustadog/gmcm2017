#-*-coding:utf-8-*-
import matplotlib.pyplot as pl
from matplotlib.ticker import MultipleLocator, FuncFormatter
import numpy as np
from six import byte2int
from matplotlib.lines import lineStyles
from pylab import mpl
from numpy import average
th = 600

def draw_line(z):
    Q = 1e-5 # process variance
    n_iter = len(z)
    sz = (n_iter,) # size of array

    # allocate space for arrays
    xhat=np.zeros(sz)      # a posteri estimate of x
    P=np.zeros(sz)         # a posteri error estimate
    xhatminus=np.zeros(sz) # a priori estimate of x
    Pminus=np.zeros(sz)    # a priori error estimate
    K=np.zeros(sz)         # gain or blending factor

    R = 0.1**2 # estimate of measurement variance, change to see effect
    # intial guesses
    xhat[0] = 0.0
    P[0] = 1.0
    
    for k in range(1,n_iter):
        # time update
        xhatminus[k] = xhat[k-1]  #X(k|k-1) = AX(k-1|k-1) + BU(k) + W(k),A=1,BU(k) = 0
        Pminus[k] = P[k-1]+Q      #P(k|k-1) = AP(k-1|k-1)A' + Q(k) ,A=1
    
        # measurement update
        K[k] = Pminus[k]/( Pminus[k]+R ) #Kg(k)=P(k|k-1)H'/[HP(k|k-1)H' + R],H=1
        xhat[k] = xhatminus[k]+K[k]*(z[k]-xhatminus[k]) #X(k|k) = X(k|k-1) + Kg(k)[Z(k) - HX(k|k-1)], H=1
        P[k] = (1-K[k])*Pminus[k] #P(k|k) = (1 - Kg(k)H)P(k|k-1), H=1

    mpl.rcParams['font.sans-serif'] = ['STSong'] # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False
    pl.style.use('seaborn-bright')
    #fig = pl.figure()
    fig = pl.figure(figsize=(4,3))
    ax = pl.gca()
    
    l1 = ax.plot(z, 'k-')
    
    length = len(z)
    
    #l1 = ax.plot(z, 'y-',label=u'滤波前')
    #l2 = ax.plot(xhat, 'k-', label=u'滤波后 ')
    l3 = ax.plot([0,length],[th,th],'r-',linewidth = 1.5)
    
    pl.xticks(fontsize = 13)
    pl.yticks(fontsize = 13)
    pl.xlim(0, length)
    #leg = pl.legend(fontsize = 13) #标注，显示内容位label
    leg = pl.legend()
    #pl.show() #展示 
    
    #pl.savefig("wcg.pdf") #保存
    
    return pl,xhat

def read_frame(xhat):
    state = 'no'
    out = ''
    for index,i in enumerate(xhat):
        if state == 'no':
            if i < th:
                state = 'down'
                continue
            else:
                state = 'up'
                out = out + str(index + 1) + '-'
                continue
        elif state == 'up':
            if i < th:
                state = 'down'
                out = out +  str(index + 1) + ','
                continue
        elif state == 'down':
            if i >= th:
                state = 'up'
                out = out +  str(index + 1) + '-'
                continue
    
    if state == 'up':
        out = out +  str(index + 1)
        
    return out    
    


if __name__ == '__main__': #入口在这里！
    file = open("data_overpass.txt","r")
    z = file.read()
    z = z.split(',')
    z = map(int,z)
    pl,xhat = draw_line(z)
    out = read_frame(xhat)
    print out
    pl.show()