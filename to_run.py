import New as n
import numpy as np
import matplotlib.pyplot as plt
import time

#Please find below all the lines required to produce any results used in report and required in Project Outline
#Parts can be switched on and off via Boolean switches for ease - some parts take a long time to run, so don't 
#want to have to wait for these to complete every time you want to see a certain graph.

#PLEASE NOTE: All calculated values in below examples may slightly disagree with those in the report - I have 
#coded this part to use the INTERPOLATED NLL function, to allow methods to be seen quickly for visualisation purposes
#If you want to see the reported results (takes longer to run in pretty much all cases): set arguments = "NLL" 
#wherever the argument f_interp appears.

#Part 3.1 Access the Data
first_test = n.decay_analysis('lifetime.txt',"all",0) #object of class to access data from file
visualise_raw_data = False #set to True to see histograms of data 

#Part 3.2 Fit Function
fit_test = n.fitting(first_test.get_split_data(),1e-7) #object of class to perform fitting and minimisation routines
visualise_fit_functions = False #set to True to see fit functions, and ROUGH overlay of fit function on raw data for first guess of tau

#Part 3.3 Likelihood Function
visualise_NLL = False #set to True to see the NLL graph with a "calculated by eye" function minimum
#Part 3.4
minimise_NLL = False #set to True to use parabolic minimisation to find minimum value of 1-dimensional NLL
                    #Also returns validaion of parabolic minimisation routine on cosh(x)
                    #This will also generate Figure in report explaining how "most efficient" precision was chosen for this method
#Part 3.5
accuracy = False #set to True to evalute 1d error in value calculated above
accuracy_test = False #set to True to see analysis on Number of data points required for accuracy of 10^-15s
#Part 4
two_dim_min = False #set to True to see all workings for 2-dimensional minimisation routines
two_d_error = False #set to True to see 2-d error ellipse
pseudo_test = False #set to True to test pseudo-data generation procedure
data_generation = False #set to True to generate pseudo data to test accuracy relationship with number of data points

#Obtain required results above using functions explained in depth in script New.py
if visualise_raw_data == True:
    first_test.visualise_raw_data()    
if visualise_fit_functions == True:
    fit_test.visualise_f_t()
    fit_test.visualise_fit_function()
    fit_test.rough_overlay()
def cosh(x):
    return np.cosh(x)
if visualise_NLL == True:
    fit_test.visualise_NLL()
if minimise_NLL == True:
    #fit_test.test_precision()
    print "Parabolic Minimisation validation: ", fit_test.parabolic_minimise(cosh,[0.1,1.0,2.0])[0]
    tau_optimal, three_points = fit_test.parabolic_minimise("NLL",[0.4,0.5,0.6])
    print "Minimum Value of 1-dimensional NLL: ", tau_optimal  
if accuracy_test == True:
    acc = n.accuracy_test(np.linspace(1000,10000,5),first_test.get_data_set(),1e-15)
    print "Number of Data points required for accuracy of 10^-15: ", acc.find_number()
if accuracy == True:
    print "Upper and Lower tau errors using +0.5 method: ", fit_test.standard_deviation_NLL("NLL",False,0)
    print "Upper and Lower tau errors using curvature method: ", fit_test.standard_deviation_quadratic("NLL")
def test_function(x,y):
    return (x**2) + 2*(y**2) + x*y + 3*x 
if two_dim_min == True:
    fit_test.visualise_contour()
    results_one,p1,num = fit_test.two_dimensional_minimise_grad_method([1.0,1.0],test_function,0.01,0.02)
    results_two,p2,num2 = fit_test.quasi_newton(test_function,[1.0,1.0],0.01,0.02)
    print "Result of Validation on Simplistic Parabolic Funciton (Gradient Method): ", results_one
    print "Result of Validation on Simplistic Parabolic Funciton (Quasi-Newton): ", results_two
    T,A,Z,f_interpolated = fit_test.interpolate()
    def f_interp(t,a):
        r1 = f_interpolated(t,a)
        return float(r1)
    start_point = [0.4,0.8]
    t = time.time()
    results, list_points,num = fit_test.two_dimensional_minimise_grad_method(start_point,f_interp,0.0001,0.00001)
    print "Minimum Value found by Gradient Method: ", results
    print "Number Steps: ", num
    print "Average step time: ",(time.time()-t)/num
    t1 = time.time()
    a,d,num1 = fit_test.annealing(f_interp,start_point)
    print "Minimum Value found by simulating annealing", a
    print "Number Steps: ", num1
    print "Average step time: ",(time.time()-t1)/num1
    t2 = time.time()
    g,h,num2 =  fit_test.quasi_newton(f_interp,start_point,0.000016,0.0001)
    print "Minimum Value found by Quasi Newton", g
    print "Number Steps: ", num2
    print "Average step time: ",(time.time()-t2)/num2
    x = [i[0] for i in d]
    y = [i[1] for i in d]
    x_1 = [i[0] for i in list_points]
    y_1 = [i[1] for i in list_points]
    x_2 = [i[0] for i in h]
    y_2 = [i[1] for i in h]
    fit_test.compare_contours(x,y,x_1,y_1,x_2,y_2)   
if two_d_error == True:    
    fit_test.close_contours()
if data_generation == True:
    acc = n.accuracy_test(np.linspace(1000,10000,5),first_test.get_data_set(),1e-15)
    acc.loop() 
if pseudo_test == True: 
    acc = n.accuracy_test(np.linspace(1000,10000,5),first_test.get_data_set(),1e-15)
    acc.test_pseudo()