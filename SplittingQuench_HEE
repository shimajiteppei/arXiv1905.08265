'''
Created on 2019/04/15

@author: tomoesaturn

This python numerical computation is based on our paper arXiv:1905.08265, sec5.
If you have an environment to excute python programs,
you can calculate HEE(holographic entanglement entropy) for DOUBLE SPLITTING QUENCH (and SINGLE SPLITTING QUENCH) by this program.


In order to calculate elliptic theta function, I use mpmath library.

http://mpmath.org/

"mpmath is a free (BSD licensed) Python library for real and complex floating-point arithmetic with arbitrary precision."

'''
import mpmath as mp
from mpmath import j,pi,exp,jtheta,re,im,sin,sinh,log,atan,cos,diff
#runtime measurement
import time



"""
DOUBLE SPLITTING QUENCH
"""
def w(v,s,b):
    """
    conformal map for double splitting quench (Equation 5.1). In this code v means \nu.

    We can interpret the convention for elliptic theta function,
    \theta_1(\nu,\tau) in Equation 5.3 = jtheta(1,pi*nu,exp(pi*j*tau)).

    s : moduli parameter. The Hawking Page phase transition point is s=1.
    b : quench point. We consider double splitting at x=b,-b.

    s and b must be positive number to get right answer.
    """
    return -j*b*( jtheta(1,pi*v,exp(-pi*s),1)/jtheta(1,pi*v,exp(-pi*s),0) + jtheta(1,pi*(v+j*s/2),exp(-pi*s),1)/jtheta(1,pi*(v+j*s/2),exp(-pi*s),0) + j )

def Dw(v,s,b):
    """
    derivative of w. We use this function to transform UV cutoff.
    """
    return diff(lambda x: w(x,s,b), v, 1)
# def Dw(v,s,d):
#     t1 = jtheta(1, pi*v,exp(-pi*s), 0)
#     s1 = jtheta(1, pi*(v+j*s/2),exp(-pi*s), 0)
#     return (-j*d*pi)*( (jtheta(1,pi*v,exp(-pi*s),2)*t1-jtheta(1,pi*v,exp(-pi*s),1)**2)/(t1**2) + (jtheta(1,pi*(v+j*s/2),exp(-pi*s),2)*s1-jtheta(1,pi*(v+j*s/2),exp(-pi*s),1)**2)/(s1**2) )

def v(x,t,s,b):
    """
    inverse map of w(v) = x+i\tau --> x-t.
    """
    # c is atrificial parameter for findroot.
    c = 10**(-9)
    if x>b:
        return j*mp.findroot(lambda y: re(w(j*y,s,b)-(x-t)), [-s/2+c,0-c],"bisect")
    elif x<-b:
        return j*mp.findroot(lambda y: re(w(j*y,s,b)-(x-t)), [0+c,s/2-c],"bisect")
    else:
        return 1/2+j*mp.findroot(lambda y: re(w(1/2+j*y,s,b)-(x-t)), 0)

def vb(x,t,s,b):
    """
    vb is for complex conjugate of v.
    """
    if abs(x)>b:
        return -v(x,-t,s,b)
    else:
        return 1-v(x,-t,s,b)

def FDoubleConn(x1,x2,t,s,b):
    """
    inside of log() for connected HEE. We take single interval subsystem A=[x1,x2].
    """
    v1 = v(x1,t,s,b)
    v2 = v(x2,t,s,b)
    vb1 = vb(x1,t,s,b)
    vb2 = vb(x2,t,s,b)
    F = (pi**(-4))*Dw(v1,s,b)*(-Dw(-vb1,s,b))*Dw(v2,s,b)*(-Dw(-vb2,s,b))
    if s<1:
        return F * (s**4)*((sinh(pi*(v1-v2)/s)*sinh(pi*(vb1-vb2)/s))**2)
    else:
        return F * (sin(pi*(v1-v2))*sin(pi*(vb1-vb2)))**2

def SDoubleConn(x1,x2,t,s,b):
    """
    HEE for connected geodesic (Equation 5.18, 5.23). We take single interval subsystem A=[x1,x2].
    """
    return re(log( FDoubleConn(x1,x2,t,s,b) ))/12-log((x2-x1)**2)/6

def FDoubleDisc(x1,x2,t,s,b):
    """
    inside of log() for disconnected HEE.
    We take single interval subsystem A=[x1,x2].
    This has very complicated minimizasion because the configuration of boundary surface Q
    varys by Hawking-Page transition.
    """
    v1 = v(x1,t,s,b)
    v2 = v(x2,t,s,b)
    vb1 = vb(x1,t,s,b)
    vb2 = vb(x2,t,s,b)
    h1 = v1-vb1
    hp1 = h1+j*s/2
    hm1 = h1-j*s/2
    h2 = v2-vb2
    hp2 = h2+j*s/2
    hm2 = h2-j*s/2
    F = (pi**(-4))*Dw(v1,s,b)*(-Dw(-vb1,s,b))*Dw(v2,s,b)*(-Dw(-vb2,s,b))
    if s<1:
        g1 = min([re(sinh(pi*hp1/s)**2),re(sinh(pi*hm1/s)**2)])
        g2 = min([re(sinh(pi*hp2/s)**2),re(sinh(pi*hm2/s)**2)])
        return F * (s**4)*g1*g2
    else:
        m1 = (sin(pi*hp1)*sin(pi*hp2))**2
        m2 = ((sin(pi*hp1)*sin(pi*hm2))**2)*exp(2*pi*s)
        m3 = ((sin(pi*hm1)*sin(pi*hp2))**2)*exp(2*pi*s)
        m4 = (sin(pi*hm1)*sin(pi*hm2))**2
        return F * min([re(m1),re(m2),re(m3),re(m4)])

def SDoubleDisc(x1,x2,t,s,b):
    """
    HEE for disconnected geodesic (Equation 5.19, 5.24). We take single interval subsystem A=[x1,x2].
    """
    return re(log( FDoubleDisc(x1,x2,t,s,b) ))/12-log((x2-x1)**2)/6

def SDouble(x1,x2,t,s,b):
    """
    HEE for double splitting quench. The vacuum contribution is subtracted.
    """
    return min([SDoubleConn(x1, x2, t, s, b),SDoubleDisc(x1, x2, t, s, b)])


"""
SINGLE SPLITING QUENCH
"""
def T(x,t,a):
    """
    conformal map for single splitting quench. a is qunech cutoff.
    """
    if x>0:
        return atan(-(x-t)/a)
    else:
        return -pi+atan(-(x-t)/a)

def TB(x,t,a):
    """
    complex conjugate of T
    """
    if x>0:
        return atan(-(x+t)/a)
    else:
        return -pi+atan(-(x+t)/a)

def FSingleConn(x1,x2,t,a):
    """
    inside of log() for connected HEE. We take single interval subsystem A=[x1,x2].
    """
    t1 = T(x1,t,a)
    t2 = T(x2,t,a)
    tb1 = TB(x1,t,a)
    tb2 = TB(x2,t,a)
    num = (a**4)*( (exp(j*t1)-exp(j*t2))*(exp(-j*tb1)-exp(-j*tb2)) )**2
    den = exp(j*(t1-tb1+t2-tb2))*((cos(t1)*cos(tb1)*cos(t2)*cos(tb2))**2)
    return num/den

def SSingleConn(x1,x2,t,a):
    """
    HEE for connected geodesic. We take single interval subsystem A=[x1,x2].
    """
    return re(log( FSingleConn(x1,x2,t,a) ))/12-log((x1-x2)**2)/6

def FSingleDisc(x1,x2,t,a):
    """
    inside of log() for connected HEE. We take single interval subsystem A=[x1,x2].
    """
    t1 = T(x1,t,a)
    t2 = T(x2,t,a)
    tb1 = TB(x1,t,a)
    tb2 = TB(x2,t,a)
    num = (a**4)*( (exp(j*t1)-exp(-j*tb1))*(exp(j*t2)-exp(-j*tb2)) )**2
    den = exp(j*(t1-tb1+t2-tb2))*((cos(t1)*cos(tb1)*cos(t2)*cos(tb2))**2)
    return num/den

def SSingleDisc(x1,x2,t,a):
    """
    HEE for disconnected geodesic. We take single interval subsystem A=[x1,x2].
    """
    return re(log( FSingleDisc(x1,x2,t,a) ))/12-log((x2-x1)**2)/6

def SSingle(x1,x2,t,a):
    """
    HEE for single splitting quench. The vacuum contribution is subtracted.
    """
    return min([SSingleConn(x1, x2, t, a),SSingleDisc(x1, x2, t, a)])


"""
difference between SINGLE and DOUBLE
"""
def cutoffD(s,b):
    """
    compute quench cutoff a for double splitting quench from s and b.
    """
    return im(w(mp.findroot(lambda y: diff(lambda x: im(w(x-j*s/4,s,b)) ,y,1),1/2)-j*s/4,s,b))



"""
PLOTTING
"""
if __name__ == '__main__':

    mp.dps = 500  #dps for precision
    mp.pretty = True

    """
    if you'd like to check the precision (compared to Mathematica or something), this sample code may be helpful.
    """
#     print(re(v(-100,0,5,50)))
#     print(re(vb(-100,0,5,50)))

    start = time.time()
    """
    plot of DOUBLE SPLITTING QUENCH
    """
#     b = 50
#     maxtime = 300
#     plotpoints = 100
#     s = 0.945
#     x1, x2 = 100, 200
#     mp.plot([lambda t: SDoubleConn(x1, x2, t, s, b),lambda t: SDoubleDisc(x1, x2, t, s, b)],[0,maxtime],points=plotpoints)
#     x1, x2 = 10, 30
#     mp.plot([lambda t: SDoubleConn(x1, x2, t, s, b),lambda t: SDoubleDisc(x1, x2, t, s, b)],[0,maxtime],points=plotpoints)
#     x1, x2 = 30, 100
#     mp.plot([lambda t: SDoubleConn(x1, x2, t, s, b),lambda t: SDoubleDisc(x1, x2, t, s, b)],[0,maxtime],points=plotpoints)
#     x1, x2 = -150, 100
#     mp.plot([lambda t: SDoubleConn(x1, x2, t, s, b),lambda t: SDoubleDisc(x1, x2, t, s, b)],[0,maxtime],points=plotpoints)
#     s = 5.28
#     x1, x2 = 100, 200
#     mp.plot([lambda t: SDoubleConn(x1, x2, t, s, b),lambda t: SDoubleDisc(x1, x2, t, s, b)],[0,maxtime],points=plotpoints)
#     x1, x2 = 10, 30
#     mp.plot([lambda t: SDoubleConn(x1, x2, t, s, b),lambda t: SDoubleDisc(x1, x2, t, s, b)],[0,maxtime],points=plotpoints)
#     x1, x2 = 30, 100
#     mp.plot([lambda t: SDoubleConn(x1, x2, t, s, b),lambda t: SDoubleDisc(x1, x2, t, s, b)],[0,maxtime],points=plotpoints)
#     x1, x2 = -150, 100
#     mp.plot([lambda t: SDoubleConn(x1, x2, t, s, b),lambda t: SDoubleDisc(x1, x2, t, s, b)],[0,maxtime],points=plotpoints)
    """
    plot of SINGLE SPLITTING QUENCH
    """
#     x1, x2 = 100, 150
#     a = cutoffD(0.945,50)
#     print(a)
#     maxtime = 300
#     plotpoints = 1000
#     mp.plot([lambda t: SSingleConn(x1, x2, t, a),lambda t: SSingleDisc(x1, x2, t, a)],[0,maxtime],points=plotpoints)
#     x1, x2 = -20, 50
#     mp.plot([lambda t: SSingleConn(x1, x2, t, a),lambda t: SSingleDisc(x1, x2, t, a)],[0,maxtime],points=plotpoints)

    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
