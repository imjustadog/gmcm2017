#-*-coding:utf-8-*-
import matplotlib.pyplot as pl
from matplotlib.ticker import MultipleLocator, FuncFormatter
import numpy as np
from six import byte2int
from matplotlib.lines import lineStyles
from pylab import mpl


def draw_line(z):
    mpl.rcParams['font.sans-serif'] = ['STSong'] # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False
    pl.style.use('seaborn-bright')
    fig = pl.figure()
    ax = pl.gca()
    l1 = ax.plot(z, 'y-')
    
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


    l2 = ax.plot(xhat, 'k-')
    
    pl.xlim(0, len(z))
    pl.show() #展示 
    
    #pl.savefig("wcg.pdf") #保存
    
    return xhat


if __name__ == '__main__': #入口在这里！
    file = open("data.txt","r")
    z = file.read()
    z = z.split(',')
    z = map(int,z)
    draw_line(z)