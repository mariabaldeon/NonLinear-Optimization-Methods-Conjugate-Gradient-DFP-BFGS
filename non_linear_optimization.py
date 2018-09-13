import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from numpy import linalg as LA


#Computes the function 
def f(x1,x2): 
    return -12*x2+4*(x1)**2+4*(x2)**2+4*x1*x2


#The gradient of the function
def gradf(x1,x2): 
    return np.matrix([[8*x1+4*x2],[-12+8*x2+4*x1]])

#Graph the function to check the results
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

fig = plt.figure()
ax = fig.gca(projection='3d')

X = np.arange(-15, 15, 0.25)
Y = np.arange(-15, 15, 0.25)
X, Y = np.meshgrid(X, Y)
Z = f(X,Y)

surf = ax.plot_surface(X, Y, Z,cmap=cm.coolwarm, linewidth=0, antialiased=False)

plt.show()

#Golden search method customized to minimize the lambda in the above methods
#ak, bk are the interval. length of uncertainty the uncertainty admitted for the solution
#Y is the initial point and d the search direction. 
def golden_search(ak, bk,length_uncertainty, Y, d):
    
    k=1
    termination=True
    
    #Set the X1 and X2 direction and initial point value
    x1_in=float(Y[0])
    x2_in=float(Y[1])
    d1=float(d[0])
    d2=float(d[1])
    
    #Initialization Step. Compute lambda, u and the functions value 
    alpha=((-1+math.sqrt(5))/2)
    lambdk=alpha*ak+(1-alpha)*bk
    uk=(1-alpha)*ak+alpha*bk
    
    #Compute the functions value using the point and direction given and 
    #the lambda and u calculated.
    flambdk=f(x1_in+lambdk*d1, x2_in+lambdk*d2)
    fuk=f(x1_in+uk*d1,x2_in+uk*d2 )
    
    while termination: 
        #Step 1.Set the new values of the interval 
        if flambdk<fuk: 
            ak=ak
            bk=uk
            uk=lambdk
            lambdk=alpha*ak+(1-alpha)*bk
            fuk=flambdk
            flambdk=f(x1_in+lambdk*d1, x2_in+lambdk*d2)
    
        else: 
            ak=lambdk
            bk=bk
            lambdk=uk
            uk=(1-alpha)*ak+alpha*bk
            flambdk=fuk
            fuk=f(x1_in+uk*d1,x2_in+uk*d2 )
    
    #Step 3. Check termination criteria
        if (bk-ak)<length_uncertainty: 
            #The optimal solution is within the interval [ak,bk] 
            #Choosing the point the middle as the optimal solution
            search_magnitude=(bk+ak)/2
            termination=False
            
        else: 
            k+=1
    return search_magnitude

# a) Fletcher-Reeves (Conjugate Gradient)
#Define a dataframe to save the solutions and directions for each interation
points_checked=pd.DataFrame(columns=["xk","alphak","dk","lambda","x(k+1)","gradient","Objective Function"])
    
#Step 0. Choose the initial point,length of uncertainty and calculate the first descent direction
len_uncert=0.5
Xk=np.matrix([[11],[11]])
n=int(Xk.shape[0])
dk=-gradf(Xk[0,0],Xk[1,0])
termination=True

while termination: 
            
    #Compute the magnitude of the direction (lambda) using the Golden search method
    lambd=golden_search(0,100,0.00002,Xk,dk)
        
    #Compute the new point
    Xk1=Xk+lambd*dk
        
    #Compute alphak to obtain a conjugate direction
    alphak=(LA.norm(gradf(Xk1[0,0],Xk1[1,0]))**2)/(LA.norm(gradf(Xk[0,0],Xk[1,0]))**2)
    
    #Save the information
    points_checked=points_checked.append({"xk":(np.round(Xk[0,0],2),np.round(Xk[1,0],2)),"alphak":np.round(alphak,3),"dk":(np.round(dk[0,0],2),np.round(dk[1,0],2)),"lambda":np.round(lambd,3),"x(k+1)":np.round(Xk1,2),"gradient":np.round(gradf(Xk1[0,0],Xk1[1,0]),2),"Objective Function":np.round(f(Xk1[0,0],Xk1[1,0]),2)}, ignore_index=True)
    
    #Update the descent direction
    dk=-gradf(Xk1[0,0],Xk1[1,0])+alphak*dk
    
    #Check termination criteria
    if LA.norm(gradf(Xk1[0,0],Xk1[1,0]))<len_uncert: 
        termination=False
        print("The optimal solution is X*=",Xk1[0],Xk1[1], "and Objective Function= ",np.round(f(Xk1[0,0],Xk1[1,0]),2)  )
        
    #Update Xk
    Xk=Xk1
        
print("The table with the iterations: ")
points_checked


# b) Davidon-Fletcher-Powell (DFP)
#Define a dataframe to save the solutions and directions for each interation
points_checked=pd.DataFrame(columns=["xk","Dk","dk","lambda","x(k+1)","gradient","Objective Function"])
    
#Step 0. Choose the initial point,length of uncertainty and set matrix D as the identity
len_uncert=0.5
Xk=np.matrix([[11],[11]])
n=int(Xk.shape[0])
Dk=np.eye(n)
termination=True

while termination: 
            
    #Compute the directions using the matrix Dk
    dk=-Dk*gradf(Xk[0,0],Xk[1,0])
        
    #Compute the magnitude of the direction (lambda) using the Golden search method
    lambd=golden_search(0,100,0.002,Xk,dk)
        
    #Compute the new point
    Xk1=Xk+lambd*dk
        
    #Compute Ck using the Davidon-Fletcher-Powell Method
    pk=Xk1-Xk
    qk=gradf(Xk1[0,0],Xk1[1,0])-gradf(Xk[0,0],Xk[1,0])
    Ck=((pk*pk.T)/(pk.T*qk))-((Dk*qk*qk.T*Dk)/(qk.T*Dk*qk))
    
    #Save the information
    points_checked=points_checked.append({"xk":(np.round(Xk[0,0],2),np.round(Xk[1,0],2)),"Dk":np.round(Dk,3),"dk":(np.round(dk[0,0],2),np.round(dk[1,0],2)),"lambda":np.round(lambd,3),"x(k+1)":np.round(Xk1,2),"gradient":np.round(gradf(Xk1[0,0],Xk1[1,0]),2),"Objective Function":np.round(f(Xk1[0,0],Xk1[1,0]),2)}, ignore_index=True)
        
    #Update Dk
    Dk=Dk+Ck
        
    #Check termination criteria
    if LA.norm(gradf(Xk1[0,0],Xk1[1,0]))<len_uncert: 
        termination=False
        print("The optimal solution is X*=",Xk1[0],Xk1[1], "and Objective Function= ",np.round(f(Xk1[0,0],Xk1[1,0]),2)  )
        
    #Update Xk
    Xk=Xk1
        
print("The table with the iterations: ")
points_checked


# c) Broyden-Fletcher-Goldfarb-Shanno (BFGS)

#Define a dataframe to save the solutions and directions for each interation
points_checked=pd.DataFrame(columns=["xk","Dk","dk","lambda","x(k+1)","gradient","Objective Function"])
    
#Step 0. Choose the initial point,length of uncertainty and set matrix D as the identity
len_uncert=0.5
Xk=np.matrix([[11],[11]])
n=int(Xk.shape[0])
Dk=np.eye(n)
termination=True
    
while termination: 
            
    #Compute the directions using the matrix Dk
    dk=-Dk*gradf(Xk[0,0],Xk[1,0])
        
    #Compute the magnitude of the direction (lambda) using the Golden search method
    lambd=golden_search(0,100,0.0025,Xk,dk)
        
    #Compute the new point
    Xk1=Xk+lambd*dk
        
    #Compute Ck using the Broyden-Fletcher-Goldfarb-Shanno Method
    pk=Xk1-Xk
    qk=gradf(Xk1[0,0],Xk1[1,0])-gradf(Xk[0,0],Xk[1,0])
    Ck=(((pk*pk.T)/(pk.T*qk))*float((1+((qk.T*Dk*qk)/(pk.T*qk)))))-((Dk*qk*pk.T+pk*qk.T*Dk)/(pk.T*qk))
    
    points_checked=points_checked.append({"xk":(np.round(Xk[0,0],2),np.round(Xk[1,0],2)),"Dk":np.round(Dk,3),"dk":(np.round(dk[0,0],2),np.round(dk[1,0],2)),"lambda":np.round(lambd,3),"x(k+1)":np.round(Xk1,2),"gradient":np.round(gradf(Xk1[0,0],Xk1[1,0]),2),"Objective Function":np.round(f(Xk1[0,0],Xk1[1,0]),2)}, ignore_index=True)
        
    #Update Dk
    Dk=Dk+Ck
              
    #Check termination criteria
    if LA.norm(gradf(Xk1[0,0],Xk1[1,0]))<len_uncert:    
        termination=False
        print("The optimal solution is X*=",Xk1[0],Xk1[1], "and Objective Function= ",np.round(f(Xk1[0,0],Xk1[1,0]),2))
            
    #Update Xk
    Xk=Xk1
        
print("The table with the iterations: ")
points_checked

