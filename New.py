import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy import stats, special, interpolate
import matplotlib.cm as cm
import random
import part5 as p1 #includes rejection method from Project A
from mpl_toolkits.mplot3d import Axes3D
import time

#Class for cleaning and observing raw datas
#Contains methods for obtaining, splicing, and visualising data
#Also contains method for generating `pseudo-data' according to raw data distributions 
#Inputs:
#       file_location = .txt file location of data points in (time,sigma) pairs
#       Num = number of data points that are required for analysis
#               if "all" - all data points are used
class decay_analysis:
    
    #Initialise variables to be used throughout
    def __init__(self,file_location,Num,seed):
        
        self.data_set = [] #contains all (time,sigma) pairs in data set
        self.time_data = [] #list of time data points
        self.sigma_data = [] #list of sigma data points
        self.file_location = file_location #file path to .txt file containing method
        self.seed = seed
        random.seed(self.seed)
        #Access .txt file from file location
        with open(self.file_location) as f:
            for line in f:
                t_i,sig_i = line.split()
                #Update self.data_set with tuples of (time,sigma) data point pairs
                self.data_set.append((float(t_i),float(sig_i))) #convert from strings to floats
      
        #Allow for user to not know the number of data points - just input "all"
        if Num == "all":
            self.Num = len(self.data_set)
        else:
            self.Num = Num
            
        #Once everything is set up, splice the data into separate lists of time and sigma values.
        self.splice_data()
               
    #Function: Returns data set: list of tuples of (time, sigma) pairs
    def get_data_set(self):
        return self.data_set
    
    #Function to choose N random data points from data set
    #Output: data_set variable is updated with N (time,sigma) pairs (N<= Total Number of Data Points)
    def splice_data(self):
        random.shuffle(self.data_set) #Randomly shuffle data set
        self.data_set = self.data_set[:self.Num] #Choose first N data pairs
    
    #Function to split up data into time set, and sigma set
    #Output: Updates variables self.time_data and self.sigma_data as list containing 
    #        the specific data points (e.g time or sigma)
    def split_data(self):
        self.time_data = [i[0] for i in self.data_set]
        self.sigma_data = [i[1] for i in self.data_set]
       
    #Function to return array of split data, [time data, sigma data]
    def get_split_data(self):
        self.split_data() #Perform the splitting        
        return [self.time_data,self.sigma_data]
    
    #Function to plot histograms of raw data for visualisation purposes
    #Output is two histograms (time and sigma) - parameters for these have been chosen to give most appropriate histograms and fits.
    def visualise_raw_data(self):
        self.split_data() #split the data into time and sigma sets
        
        #generate appropriate bins for both histograms (parameters decided upon through various simulations)
        t_bins = np.linspace(min(self.time_data),max(self.time_data),500)
        sigma_bins = np.linspace(min(self.sigma_data),max(self.sigma_data),150)
    
        #Plot the histograms
        f, (ax1,ax2) = plt.subplots(1,2)
        ax1.hist(self.time_data,t_bins)
        ax2.hist(self.sigma_data,sigma_bins)
        ax1.set_title("Histogram of Time Data")
        ax1.set_xlabel("Bins (ps)")
        ax2.set_xlabel("Bins (ps)")
        ax1.set_ylabel("Number in each bin")
        ax2.set_title("Histogram of Sigma Data")
       
    #Function to empty contents of class, useful for iterating over MC Simulations
    def empty(self):
        self.data_set = []
        self.time_data = []
        self.sigma_data = []
        
    #Function to generate 'pseudo data' for more in-depth accuracy analysis
    #Function returns list of generated times, and list of generated sigmas
    #These new variables will follow SAME distribution as the original raw data
    #Input: Number_input - how many new data points do you want to create
    #       tau_optimal - the optimal tau value around which you want to centre the data creation
    #       want_to_plot = Boolean value if you want to check generated histograms
    def pseudo_data(self, Number_input,tau_optimal,want_to_plot):
        Num = Number_input
        t_raw,sigma_raw = self.get_split_data() #get raw time and sigma values
        
        #Define contant comparison function for rejection method (value chosen by looking at time histograms)
        def const_comparison_func(x):
            if x>=-2.0 and x<=6.0: #range defined by raw data histogram
                return 175.0 #value defined by max value of raw data histogram
            else:
                return 0.0 #Function 'padded' with zeros to avoid errors later on with choosing random values outside range
        
        #Redefine normalised comparison function, also for use in rejection method
        def normalised_comparison_func(x):
            return (const_comparison_func(x)/(175.0*8.0))
        
        tau_sorted = sorted(t_raw) #sort t values in ascending order
        hist_t, bins_t = np.histogram(tau_sorted,np.linspace(min(tau_sorted),max(tau_sorted),500)) #generate raw data histogram    
        new_bins_t = [] #list of bin CENTRES
        for i in range(len(bins_t)-1):
            new_bins_t.append((bins_t[i]+bins_t[i+1])/2) #Update list with bin centres
        new_bins_t = sorted(new_bins_t) #sort list of bin centres in ascending order  for interpolation
        #Interpolate over this histogram to generate function for rejection method 
        #This is the function which we want our new data points to follow
        f_one = interpolate.interp1d(new_bins_t,hist_t,fill_value="extrapolate")
        
        #Redefine this interpolating function with zero-padding - again to avoid any problems beyond interpolating range
        def f_one_padded(x):
            if x < min(new_bins_t):
                return 0.0
            else:
                return f_one(x)
        #Perform rejection method using function from Project A
        taus,t1 = p1.get_result_rej(Number_input,normalised_comparison_func,const_comparison_func,np.linspace(min(new_bins_t),max(new_bins_t),Num),f_one_padded,0,True,0,True,self.seed)
        
        #Esentially do the same thing again, this time for sigma values
        sigma_sorted = sorted(sigma_raw) #sorted raw data in ascending order 
        hist, bins = np.histogram(sigma_sorted,np.linspace(min(sigma_sorted),max(sigma_sorted),150)) #generate raw data histograms
        new_bins = [] #list of bin centres
        for i in range(len(bins)-1):
            new_bins.append((bins[i]+bins[i+1])/2) #update list#
        new_bins = sorted(new_bins) #sort list in ascending order for using in interpolation
        #Perform 1-d interpolation over raw data histogram to be used in rejection algorithm
        f = interpolate.interp1d(new_bins,hist,fill_value="extrapolate")
        
        #Redefine the interpolating function with zero-padding to avoid issues outside interpolating range
        def f_padded(x):
            if x < min(new_bins):
                return 0.0
            else:
                return f(x)
        #Constant comparison function for use in rejetion method for sigma data generation
        def comp_func(x):
            if x>=0.0 and x<=0.6: #values degined by raw data histogram
                return 120.0 #value defined by peak of raw data histogram
            else:
                return 0.0
        #Define normalised constant comparison for use in rejection method algorithm
        def norm_comp_func(x):
            return comp_func(x)/(120.0*0.6)
        
        #Perform rejection method for sigma values using function generated for Project A
        r1,t2 = p1.get_result_rej(Num,norm_comp_func,comp_func,np.linspace(min(sigma_sorted),max(sigma_sorted),Num),f_padded,0,True,0,True,self.seed)
        
        if want_to_plot == True: #PLOT generated data if the user has passed in the argument True
            
            fig, (ax1, ax2) = plt.subplots(1,2)
            
            ax1.hist(taus,np.linspace(min(tau_sorted),max(tau_sorted),50))
            #plt.hist(taus)
            ax1.set_xlabel("tau (ps)")
            ax1.set_title("Generated times")
            ax1.set_ylabel("Number in each bin")
            ax2.set_title("Sigmas generated")
            ax2.set_xlabel("Associated Error")
            ax2.set_ylabel("Number in each bin")
            #plt.hist(r1)
            ax2.hist(r1,np.linspace(min(sigma_sorted),max(sigma_sorted),50))
        
        return taus, r1 #return generated data
    
    #Function to generate a uniform deviate of pseudo random numbers using random library
    #This function is required for the pseudo-data generation process - these are the random numbers
    #that will be transformed into the required data which follows the required distributions
    #Input: Num: number of random numbers you want to generate
    #       value: the upper limit of the generator (ie generate pseudo-random numbers between 0.0 and value)
    def generate_uni_deviate(self,Num,value):
        uniform_deviate = []
        for i in range(int(Num)):
            r = random.uniform(0,value)
            uniform_deviate.append(r)
        return uniform_deviate
    
    #Function: Appends any new/ generated data on self.data_set variable
    #Most useful for when 'pseudo-data' is generated, and you want to incude it in your fitting analysis below
    #Inputs: new_t: list of generated t values
    #        new_sigma: list of generated sigma values
    def append_data(self,new_t,new_sigma):
        for i in range(len(new_t)): #loop through data to append
            self.data_set.append((new_t[i],new_sigma[i])) #append data

        
#Class for performing analysis and error calculations on data 
#Inputs:
    #   split_data = array of two arrays [time data array, sigma data array]
    #               generated using decay_analysis class
    #   precision = required precision in calculation of standard deviation using Negative Log Likelihood
class fitting:
    
    #set up variables to be used throughout multiple methods
    def __init__(self,split_data,precision):
        
        self.time_data, self.sigma_data = split_data #assign individual lists for each set of data points
        self.precision = precision #required precision

        #Initialise parameters for later calculations with trivial entries (0.0)
        self.brackets = [0,0] #coordinate 'brackets' to be updated throughout the parabolic minimisation method
        self.tau_optimal = 0 #tau value calculated which minimises NLL
        self.coordinates = [0.0] #previous coordinate generated in a minimisation step
        self.h = 0.0 #parameter required for central difference scheme for approximate gradient of function
        self.alpha = 0.0 #parameter in Gradient Method and Quasi-Newton Method 
        self.function = 0.0 #function to be minimised
    
    #Function returns a list of all time data points    
    def get_t_data(self):
        return self.time_data
    #Function returns a list of all sigma data points
    def get_s_data(self):
        return self.sigma_data
    
    #Function of idealised exponential decay function (with no smearing or uncertainties)
    #Input: time parameter t
    #       theoretical decay lifetime tau
    #Returns value of function at all t
    def f_t(self,t,tau):
        if t <0.0:
            return 0.0
        else:
            return (1.0/tau)*(np.exp(-t/tau))
        
    #Function to visualise theoretical decay function f_t
    #Plots graph for varying tau to examine effect of this parameters on overall shape
    def visualise_f_t(self):
    
        #Create dummy variable input to examine effects
        tau_input = np.linspace(0.5,5,6)
        t_range = np.linspace(-5,10,500)
    
        fig, ax = plt.subplots(1,1)
        for tau in tau_input:
            #Plot graph for various tau
            ax.plot(t_range, [self.f_t(t,tau) for t in t_range],label = "tau = {}".format(tau))
        ax.set_title("Idealised Decay Function")
        ax.set_xlabel('time')
        ax.set_ylabel("f(t)")
        ax.legend()

    #Fit function (convolution of theoretical decay function with Gaussian function centred at 0)
    #Inputs: time variable t
    #        tau - theoretical decay lifetime
    #        sigma - width of blurring Gaussian function (uncertainty)
    #Returns: value of function at value t
    def fit_func(self,t,tau,sigma):
        exp = np.exp(((sigma**2)/(2.0*(tau**2)))-(t/tau))
        erfc = special.erfc((1.0/np.sqrt(2.0))*((sigma/tau)-(t/sigma)))
        prefactor = (1.0/(2.0*tau))
        return prefactor*exp*erfc
    
    #Function to visualise the above fit function
    #Output: generates plot of fit function for varying tau at constant sigma, and varying sigma at constant tau
    #in order to examine the effects of these parameters on the overall shape
    #Function also set-up to evaluate the integral of these fit functions over all space to check it is a normalised PDF (ie. area = 1)
    #Evaluating this integral between [-inf,inf] is very intensive and slow, therefore approximations are made to the interval
    #The integral is evaluated between increasing, finite, boundaries to examine the trend of the results (hoping it tends towards one)
    def visualise_fit_function(self):
        
        #Input values for plotting of the parameters and variables
        t_input = np.linspace(-50,50,400)
        tau_range = np.linspace(4,32,4)
        sigma_range = tau_range
        sigma_const, tau_const = [2.0,2.0]
    
        
        f, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
        #Perform plotting and integral analysis for varying tau at constant sigma
        for tau in tau_range:
            ax1.plot(t_input, [self.fit_func(t,tau,sigma_const) for t in t_input], label = 'tau = %d ps' %(tau))    
            #Perform integral of fit function between different boundaries, to check it is a normalised PDF
            for i in np.linspace(10,2000,10):
                ax3.scatter(i,quad(self.fit_func,-1*(i/2.0),i/2.0,args=(tau,sigma_const))[0],color = 'red',marker = 'x')
        
        
        #Perform plotting and intergral analysis for varying sigma at constant tau
        for sigma in sigma_range:
            ax2.plot(t_input,[self.fit_func(t,tau_const,sigma) for t in t_input], label = 'Sigma = %d ps' %(sigma))
            #Perform integral of fit function between different boundaries, to check it is a normalised PDF
            for i in np.linspace(10,2000,10):
                ax4.scatter(i,quad(self.fit_func,-1*(i/2.0),i/2.0,args=(tau_const,sigma))[0],color = 'red',marker='x')        
        #Plot all results
        ax1.set_title("Fit Function for Increasing tau")
        ax1.set_xlabel("Time (ps)")
        ax1.set_ylabel("f(t)")
        ax2.set_title("Fit Function for Increasing Sigma")   
        ax2.set_xlabel("Time (ps)")
        ax2.set_ylabel("f(t)")
        ax3.set_title("Integral over increasing range")  
        ax3.set_xlabel("Size of Integral Range About Zero")
        ax3.set_ylabel("Value of Integral")
        ax3.plot(np.linspace(10,2000,10),[1]*10,'--')
        ax4.set_title("Integral over increasing range")
        ax4.set_xlabel("Size of Integral Range About Zero")
        ax4.plot(np.linspace(10,2000,10),[1]*10,'--')
        ax4.set_ylabel('Value of Integral')
        ax2.legend()  
        ax1.legend()
        
    #Function to roughly overlay a fit function on top of the raw data histograms.
    #This is in an attempt to establish a guess for the value of tau
    #Output is this overlay function on top of the data histogram
    def rough_overlay(self):
        tau_guess = 0.64/2.5 #Tau guess fiddled with manually, and by eye to get a close estimate
        #Sigma guess found by naively taking the average of the sigma values in the data set
        sigma_guess = np.mean(self.sigma_data) 
        
        #Generate appropraite bin widths and t range for smooth plotting
        t_bins = np.linspace(min(self.time_data),max(self.time_data),500)
        t_range = np.linspace(-10,10,200)
    
        f,ax = plt.subplots(1,1)        
        #Plot histogram of data obtained by experiment
        ax.hist(self.time_data,t_bins,normed=True,label = "Time data histogram")
        #Overlay fit function with manually tuned parameters
        ax.plot(t_range, [self.fit_func(t,tau_guess,sigma_guess) for t in t_range],label = "Fit Function Overlay with tau = 0.256 ps, sigma = 0.28 ps ")        
        ax.set_title("Rough estimate of tau by overlaying fit function")
        ax.set_xlabel("t (ps)")
        ax.set_ylabel("f(t), rough overlay")
        ax.legend()
        
    #Function to produce the Negative Likelihood function of the data
    #Input: tau - decay lifetime
    #Output: value of NLL at that tau
    def NLL(self,tau):
        
        N = len(self.time_data) #Number of data points
        result = 0.0 #set result = 0.0, to be updated by summing over all data points
        #Sum over all data points, taking combined logarithm of probablility of measurement of tau
        for i in range(N):
            value = self.fit_func(self.time_data[i],tau,self.sigma_data[i])
            if value != 0.0: #avoid taking logarithm of negative values (shouldn't really be an issue)
                result += (-1.0*(np.log(value)))
        return result
    
    #Function to visualise NLL function to see general shape and make rough guess at position of minimum
    #minimum of NLL will occur at the optimal value of parameter tau - ie a 'best estimate' for decay lifetime
    def visualise_NLL(self):
    
        #Dummy range for plotting purposes, chosen to cover the majority of the behaviour of the function
        tau_range = np.linspace(0.1,0.99,100)
        #Plot NLL function over this dummy tau range
        f,ax = plt.subplots(1,1)
        ax.plot(tau_range,[self.NLL(g) for g in tau_range])
        ax.scatter(0.403973509934,6220.45295362307,color = 'red', marker='x')
        ax.set_title("Negative Log Likelihood Function")  
        ax.set_xlabel("tau (ps)")
        ax.set_ylabel("NLL(tau)")
    
    #Function to perform intermediate step in the parabolic minimisation algorithm
    # Takes three points (x0,x1,x2) and a function, and finds the 'next point'
    #       Fits a parabola between the three points, then finds the minimum of this parabola
    # Returns the minimum 
    def get_x3(self,x0,x1,x2,func):
        
        #evaluate the function at the three points
        f0 = func(x0)
        f1 = func(x1)
        f2 = func(x2)
    
        #Fit parabola and find minimum analytically 
        numerator = (((x2**2)-(x1**2))*f0) + (((x0**2)-(x2**2))*f1) + (((x1**2)-(x0**2))*f2)
        denominator = ((x2-x1)*f0) + ((x0-x2)*f1) + ((x1-x0)*f2)
        x3 = 0.5*(numerator/denominator) #this is the minimum value of the fitted parabola
        return x3 #return minimum value of 2nd order Lagrange Polynomial/ fitted parabola 
    
    #Function to perform intermediate step in the parabolic minimisation routine
    # Given function and 4 points (three points + minimum of parabola fitted through these three points)
    # Returns array containing only 3 points - the points which all give the LOWEST values of the function to be minimised
    def remove_max(self,function,array,point):
        array.append(point) #create array of all 4 points
        new_array = [function(h) for h in array] #evaluate function at all four points
        del array[new_array.index(max(new_array))] #find whihc f(xi) is the largest, and remove it
        return array #return array of the three points which produce the smallest f(xi)
    
    #Function to minimise a given function - through PARABOLIC MINIMISATION ROUTINE
    #Input: Function - function to be minimised
    #       three_points - three guess points (must all lie in a region of the function where the curvature is positive)
    #Return Calculated minimum of the function, and the final three points generated in the method
    #User can input function as "NLL", and analysis will be performed specifically for the Negative Log Likelihood
    #generated in this class from the given data set
    def parabolic_minimise(self,function,three_points):

        if function == "NLL": #check user input, map "NLL" input to self.NLL function 
            function = self.NLL
            
        #Set required precision of final value (relative precision)
        delta_x3_req = self.precision
        
        #Parabolic Minimisation routine invovles using three guess points at every stage
        #A fourth point is found using these points, and this routine is continued until convergence (to within required precision)
        #To start off the algorithm, a dummy 'previous point' needs to be created: in this case the maximum input value is used
        #This guarantees the routine doesn't fail before it's even started
        
        #Perform first step of routine independently of below while loop, as a first check step
        values = [function(j) for j in three_points] #evaluate function at three guess points
        first_x3 = three_points[values.index(max(values))] #generate first 'Fourth' point in the routine to initialise the below while loop
        new_x3 = self.get_x3(three_points[0],three_points[1],three_points[2],function) #find minimum of parabola fitted through three guess points
        delta_x3 = first_x3-new_x3 #Find change between previous of guess, and new calculated value
        three_points = self.remove_max(function,three_points,new_x3) #remove point which yeilds largest f(x) value
        prev_x3 = new_x3 #old point= point from step before
        
        #delta_x3 is the change of new guess for minimum value, after each step of process
        #The algorithm is run conintually until this change is within the required precision
        while abs(delta_x3) >= delta_x3_req:
            
            #Perform parabolic minimisation routine
            #Fit parabola between previous three points, and find minimum of this parabola
            x3_next = self.get_x3(three_points[0],three_points[1],three_points[2],function)
            #Find three points which give the lowest function value, these are the ones kept for the next iteration
            three_points = self.remove_max(function,three_points,x3_next)
            delta_x3 = x3_next - prev_x3 #change in x value
            prev_x3 = x3_next #update previous x_minimum value
          
        #prev_x3 is the last value obtained for the function minimum
        #Updated self.tau_optimal with the final guess for the minimum of the NLL function
        self.tau_optimal = prev_x3
        
        #Return this minimum tau, and the final three points used in the calculation (to be used later for estimation of error)
        return prev_x3, three_points       
    
    #Function to test the effects of relative precision on the calculated minimum of NLL
    #This function was used before running the full minimisation, to find the most efficient value of the relative precision
    #Returns a plot of Minimum Value obtained against Log(precision value)
    def test_precision(self):
        oldprecisions = np.linspace(-2,-13,20) #set a range of precisions to test over (in log space)
        precisions = 10**(oldprecisions) #map outside of log space
        value = [] 
        for i in precisions:
            self.precision = i #update the precision used in the scheme with the test value
            #Parabolic minimise with the given relative precision range as part of the convergence criteria
            value.append(self.parabolic_minimise("NLL",[0.4,0.5,0.6])[0])
        #Plot Results
        plt.figure()
        plt.plot(-1*oldprecisions,value,color='red')
        plt.title("Effect of precision on Parabolic Minimisation Routine")
        plt.xlabel("-Log(precision)")
        plt.ylabel("Minimum Value Found")
        plt.show()
            
    #Function: Perform intermediate step in calculation of standard deviation of NLL function
    #Determines mid point of two points
    #Input: Array containing two points
    #Output: Teturns middle of these two points
    def mid_point(self, brackets_input):
        return (brackets_input[0] + brackets_input[1])/2.0
    
    #Function: Perform intermediate step in calculation of standard deviation of NLL function
    #Update 'brackets' with new calculated brackets, unless value within precision: then only final value is returned
    #Input: mid_input - middle value of new 'brackets'
    #       function - function that the bracketing technique is being applied to 
    #       min_val - "target value" of bracketing mechanism
    def update(self,mid_input,function,min_val):
        array = [self.brackets[0],self.brackets[1],mid_input] #brackets
        val_mid = function(array[2]) #Evaluate function at middle value of brackets
        
        if abs(val_mid - min_val - 0.5) <= self.precision: #Check to see if we have reached our "target value"
            return [array[2]] #If so, return the "target value" calculated
        elif val_mid - min_val > 0.5: #If not, see which bracket lies closet to new guess of "target value"
            return [array[0],array[2]]
        else:
            return [array[2],array[1]]    
        
    #Function to perform Bisection to find where NLL changes by +0.5
    #Inputs: initial_val - first "start" value in algorithm
    #        final_guess - Upper bracket input
    #        function - function to perform bracketing on
    def bracketing(self,initial_val,final_guess,function):
        self.brackets = [initial_val, final_guess] #set up brackets
        min_val = function(initial_val) #evaluate function at first value
        mid = self.mid_point(self.brackets) #Find new point for bracketing regime
                
        self.brackets = self.update(mid,function,min_val) #Perform first step of bracketing process
        while len(self.brackets) != 1: #perform loop until we reach within our target precision
            self.brackets = self.update(self.mid_point(self.brackets),function,min_val) #Continue performing loop until "convergence"
           
        return self.brackets[0], abs(function(self.brackets[0])- min_val) #return final obtained value, and accuracy to which it was calculated
    
    #Function to automatically find the standard deviation associated with the calculated minimum of the NLL function (using Bracketing Routine)
    #Input: function - function whose minimum is being evaluated for its uncertainty
    #       user can input "NLL" and function will use NLL function established in this class
    #       sliced - Boolean value - if we are using a smaller subset of the function
    #       input_min_val - the calculated minimum value of which we want to find the uncertainty.
    def standard_deviation_NLL(self,function,sliced,input_min_val):
        
        if function == "NLL": #check user input for function to be analysed
            function = self.NLL
        if sliced == True: #check user input, is data set sliced
            min_val = function(input_min_val)
            optimal = input_min_val 
        else:
            if self.tau_optimal == 0.0: #if there is no minimum value that's been calculated already
                self.parabolic_minimise("NLL",[0.4,0.5,0.6])
            min_val = function(self.tau_optimal) #function evaluated at best-fit tau value (ie the minimum value of the function)
            optimal = self.tau_optimal

        tau_plus_guess = 1.01*optimal #initial guess for upper standard deviation (doesn't matter as long as >tau_best_fit)
        tau_minus_guess = 0.99*optimal #initial guess for lower standard deviation (deoesn't matter as long as <tau_best_fit)
        
        #Update these initial guesses until we straddle a change of +0.5 in NLL function
        while (function(tau_minus_guess)-0.5) <= min_val: #if we're below the +0.5
            tau_minus_guess -= (0.01*optimal) #reduce value by 1%
        while (function(tau_plus_guess)-0.5) <= min_val: #if we're above the +0.5
            tau_plus_guess += (0.01*optimal) #increase value by 1%
        
        result_one = self.bracketing(optimal,tau_plus_guess,function)[0] #perform bracketing to calculate upper value
        result_two = self.bracketing(optimal,tau_minus_guess,function)[0] #perform bracketing to calculate lower value
        return result_one, result_two #return upper and lower errors
    
    #Function: function to return calculated minimum
    def get_tau_optimal(self):
        return self.tau_optimal
    
    #Function: Calculate standard deviation in minimum tau value using parabolic curvature method
    #input: function to find error of minimum of
    def standard_deviation_quadratic(self,function):
        if function == "NLL": 
            function = self.NLL
        
        x0,x1,x2 = self.parabolic_minimise(function,[0.4,0.5,0.6])[1] #perform parabolic minimisation and take minimum value
 
        #Analytic calculation of curvature of 2nd order Lagrange Polynomial 
        term1 = 2.0*(function(x0)/((x0-x1)*(x0-x2)))
        term2 = 2.0*(function(x1)/((x1-x0)*(x1-x2)))
        term3 = 2.0*(function(x2)/((x2-x0)*(x2-x1)))
        r = term1+term2+term3 #Add up the terms
        return self.tau_optimal + np.sqrt(1.0/r), self.tau_optimal - np.sqrt(1.0/r) #Return error calculated by 1/(curvature**2)


    #Function: new background error function with background fraction included
    #Input: tt - time variable
    #       tau_b - theoretical lifetime
    #       sigma_b - theoretical uncertainty in lifetime
    def f_background(self,tt,tau_b,sigma_b):
        prefactor = 1.0/(sigma_b*(np.sqrt(2*np.pi)))
        return prefactor*(np.exp(-0.5*(((tt**2)/(sigma_b**2)))))
                
    #Function: New Fit Function with background included
    #Input: t - time variable
    #       tau - theoretical lifetime
    #       sigma - theoretical uncertainty in lifetime 
    #       a - fraction of signal in background
    def new_fit_func(self,t,tau,sigma,a):
        term_1 = a*self.fit_func(t,tau,sigma)
        term_2 = (1.0-a)*self.f_background(t,tau,sigma)            
        return term_1 + term_2
         
    #Function: evaluate new 2-dimensional NLL function
    #Input: tau - theoretical lifetime
    #       a - fraction of signal in background
    def NLL_new(self,tau,a):
        N = len(self.time_data) #Number of data points
        result = 0.0 #Initialise value with 0.0, to update with addition
        for i in range(N):
            value = self.new_fit_func(self.time_data[i],tau,self.sigma_data[i],a)
            if type(value) == float:
                if value != 0.0: #ensure that there are no (randomly occuring) unphysical values that can't be logged
                    result += (-1.0*(np.log(value)))
            else:
                if value.any() != 0.0: #ensure that there are no (randomly occuring) unphysical values that can't be logged
                    result += (-1.0*(np.log(value)))
        return result  
         
    #Function: To visualise 2D contour plot of 2-dimensional NLL function   
    #Return a 2d contour plot and 3d plot
    def visualise_contour(self):
        tau_set = np.linspace(0.1,2.0,90) #set up range of tau values for plotting
        a_set = np.linspace(0.1,0.99,90) #set up range of fraction values for plotting
        
        T,A = np.meshgrid(tau_set,a_set) #build grid of data points
        Z = self.NLL_new(T,A) #Calculate NLL at all points on grid
    
        plt.figure() #2d contour plot
        levels = 12 #number of contours
        plt.title("Contour Plot of new NLL function")
        contour = plt.contour(T,A,Z,levels,colors='k')
        plt.clabel(contour,colors='k',fmt='%2.1f',fontsize=12)
        contour_filled = plt.contourf(T,A,Z,levels)
        plt.colorbar(contour_filled)
        plt.xlabel("tau (ps)")
        plt.ylabel("Fraction a")
        plt.show()
                
        fig = plt.figure() #3d plot
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(T, A, Z, cmap=cm.plasma,
                       linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)  
        plt.xlabel("tau (ps)")
        plt.ylabel("Fraction a")
        ax.set_zlabel("NLL(tau,a)")
        plt.show()
        
   #Function: Calculate gradient vector for intermediate step in minimisation algorithms
   # Using a CENTRAL DIFFERENCE scheme
    def gradient_vector(self):
        x_pos = self.coordinates[0] #x position to be calculated at 
        y_pos = self.coordinates[1] #y position to be calculated at 
        
        #Central difference scheme in both directions
        a = (self.function(x_pos + self.h,y_pos)-self.function(x_pos-self.h,y_pos))/(2.0*self.h)
        b = (self.function(x_pos,y_pos+self.h)-self.function(x_pos,y_pos-self.h))/(2.0*self.h)
        return [float(a),float(b)] #return result
   
    #Function: Perform gradient method to minimise function
    #Input: first_guess - starting point for algorithm
    #       function - function to be minimised
    #       h - parameter in central difference scheme
    #       alpha - parameter for gradient method step
    #Return minimum of function, list of points evaluated during the minimisation process, number of iterations required
    def two_dimensional_minimise_grad_method(self,first_guess,function,h,alpha):
        self.h = h
        self.alpha = alpha
        self.coordinates = first_guess
        
        if function == "NLL": #check is user has input "NLL" as value
            self.function = self.NLL_new
        else:
            self.function = function
            
        list_points = [first_guess] #list points will contain all points found during algorithm
        grad_initial = self.gradient_vector() #calculate first gradient
        new_point = [0,0]
        new_point[0] = self.coordinates[0]-(self.alpha*grad_initial[0]) #perform step of process in first dimension
        new_point[1] = self.coordinates[1]-(self.alpha*grad_initial[1]) #perform step of process in second dimension        
        list_points.append(new_point) #append calculated point to list of points
                
        #Take small steps prescribed by Gradient Method
        num_steps = 1 #counter for number of steps
        while self.function(new_point[0],new_point[1]) < self.function(self.coordinates[0],self.coordinates[1]): #check if convergence criteria are fulfilled
            self.coordinates = new_point #latest point
            grad = self.gradient_vector() #calculate gradient
            x_change = self.alpha*grad[0] #step in x
            y_change = self.alpha*grad[1] #step in y
            new_point = [self.coordinates[0] -x_change, self.coordinates[1] - y_change] #move to new point
            list_points.append(new_point)
            num_steps +=1 #count this iteration
            
        self.tau_optimal = self.coordinates[0] #final value found
        return self.coordinates,list_points, num_steps
    
    #Functions: Interpolate over the NLL function
    #Return grid points, and interpolated function
    #This is useful for checking minimisation methods quickly (evaluating the NLL function takes longer due to large summation)
    def interpolate(self):
        tau_set = np.linspace(0.3,1.0,400) #tau range to interpolate over
        a_set = np.linspace(0.4,0.99,20) #a range to interpolate over
        T,A = np.meshgrid(tau_set,a_set) #set up grid
        Z = self.NLL_new(T,A) #evaluate over grid
        f_interpolated = interpolate.interp2d(tau_set,a_set,Z,kind='cubic') #Interpolate over this grid
        return T,A,Z,f_interpolated
    
    #Function: Plot contours for comparison of minimisation routes
    #Inputs: x_points, y_points - list of coordinates from first minimisation method
    #       x1_points, y1_points - list of coordinates from second minimisation method
    #       x2_points, y2_points - list of coordinates from third minimisation method
    def compare_contours(self,x_points,y_points,x1_points,y1_points,x2_points,y2_points):        
        T,A,Z,f_interpolated = self.interpolate() #interpolate
        tau_set_more = np.linspace(0.3,1.0,1000) #dummy tau set for plotting
        a_set_more = np.linspace(0.4,0.99,1000) #dummy a set for plotting
        levels = 50 #number of contours
        T_new,A_new = np.meshgrid(tau_set_more,a_set_more) #set up grid
        Z_two = f_interpolated(tau_set_more,a_set_more) #evaluate interpolating function over grid
        plt.figure()
        plt.title("Contour Plot with Minimisation Routes")
        plt.plot(x_points,y_points,label='Simulated Annealing')
        plt.plot(x1_points,y1_points,label ='Gradient Method')
        plt.plot(x2_points,y2_points,label = 'Quasi Newton')
        contour = plt.contour(T_new,A_new,Z_two,levels,colors='k')
        plt.clabel(contour,colors='k',fmt='%2.1f',fontsize=12)
        contour_filled= plt.contourf(T_new,A_new,Z_two,levels)
        plt.colorbar(contour_filled)
        plt.xlabel("tau (ps)")
        plt.ylabel("Fraction, a")
        plt.legend()
        plt.show()
        
    #Function: Plot contours near the minimum to get 2D "error ellipse"
    #Graphical representation of 2D uncertainty in tau and a 
    def close_contours(self):
        
        T,A,Z, f_interpolated = self.interpolate() #perform interpolation
        tau_set = np.linspace(0.40,0.42,1000) #dummy tau range for plotting
        a_set = np.linspace(0.97,0.99,1000) #dummy a range for plotting
        levels = [6218.39558633 + 0.5,6218.39558633 + 1.5,6218.39558633 + 2.5,6218.39558633 + 10] #contours (at minimum value +0.5)

        T_new,A_new = np.meshgrid(tau_set,a_set) #set up grid
        Z_two = f_interpolated(tau_set,a_set) #evaluate on grid
        plt.figure()
        plt.title("2D NLL, Error Calculation: Minimum at value 6218.396")
        contour = plt.contour(T_new,A_new,Z_two,levels,colors='k')
        plt.clabel(contour,colors='k',fmt='%2.1f',fontsize=12)
        contour_filled= plt.contourf(T_new,A_new,Z_two,levels)
        plt.colorbar(contour_filled)
        plt.scatter(0.404458, 0.988522,color='red',marker='x') #plot cross at lowest tau value within ellipse
        plt.scatter(0.414718, 0.979958,color='red',marker='x') #plot cross at largest tau value within ellipse
        plt.scatter(0.4095,0.9841,color='red',marker='x') #plot cross at minimum value within ellipse
        plt.xlabel("tau (ps")
        plt.ylabel("Fraction, a")
        plt.show()
        print "Minimum Log Likelihood at: ", self.NLL_new(0.40954811,0.98409594)
 
    #Function: Intermediate step in simulated annealing algorithm
    #Input: solution, to which we want to find a "neighbour"
    #Neighbour found by randomly varying ONE parameter by a random amount
    def neighbour(self,solution_input):
        parameter = 1e-4
        #choose whether to modify tau or a
        value = random.random()
        #Create random parameter between 2 and -2
        alpha = random.uniform(-1.0,1.0)
        if value > 0.5:
            tau = solution_input[0]*(1+(alpha*parameter)) #slightly modify tau 
            other = solution_input[1]
            neighbour = (tau,other)
        else:
            other = solution_input[0]
            a = solution_input[1]*(1+(alpha*parameter)) #slightly modify a
            neighbour = [other,a]
        return neighbour
        
    #Boltzmann acceptance probability for use in simulated annealing process
    #old_value: previous "system energy" - value of function at previous step
    #new_value: new "system energy" - value of function at new step
    #T - 'temperature'
    def acceptance_probability(self,old_value,new_value,T):
        delta_E = new_value-old_value
        if delta_E <= 0.0:
            return 1.0
        else:
            return np.exp((-delta_E)/(T))
        
    #Function: carry out simulated annealing algorithm
    #Inputs: function - function to be minimised
    #        initial - starting point
    #Return: solution - minimum value calculated
    #        list_points - all points during minimisation routine
    #        count - number of iterations
    def annealing(self,function,initial):
        solution = initial
        if function == "NLL": #check if user wants to minimise NLL
            self.function = self.NLL_new
        else:
            self.function = function
        overall_count = 0 #count total number of iterations required
        list_points = [solution] #list of points calculated during minimisation process
        T = 1.0 #start off with temperature of 1
        T_min = 1e-6 #minimum value to which the temperature will be lowered
        alpha = 0.99 #parameter which will minimise temperature slowly 
        num_accepted = 0 #count number of accepted values 
        while T > T_min: #continue algorithm until temperature reaches minimum
            count = 1 #count number of iterations at certain temperature
            old = self.function(solution[0],solution[1]) #previous value of algorithm
            while count <= 500: #perform multiple "steps" at each temperature
                new_solution =  self.neighbour(solution) #Find close neighbour to solution
                new = self.function(new_solution[0],new_solution[1]) #evaluate at this neighbour
                ap = self.acceptance_probability(old,new,T) #calculate acceptance probability of this step
                if ap > random.random(): #if this value greater than a random probability, accept step
                    solution = new_solution
                    old = new
                    num_accepted +=1  #count number of steps
                    list_points.append(solution) # add point to list of points#
                count +=1
                overall_count += 1
            T = T*alpha #reduce temperature#
        return solution,list_points,overall_count
    
    #Function: Function to perform Quasi-Newton minimisation routine
    #Inputs: function - function to be minimised
    #        start - starting point for algorithm
    #        alpha - parameter for taking steps in QN algorithm
    #        h - parameter in central difference scheme to calculate gradient 
    #Returns: minimum value calculated
    #         list of points evaluated throuhgout the minimisation routine
    #         count number of iterations required
    def quasi_newton(self,function,start,alpha,h):
        list_points = [start] #list of points started with initial point
        self.alpha = alpha
        self.h = h
        if function == "NLL": #check if NLL is function to be minimised
            self.function = self.NLL_new
        else:
            self.function = function
          
        #Inverse Hession approximation - set to identity matrix
        G = np.identity(2)
        old = start #previous point = starting point
        self.coordinates = old
        
        #Perform one step outside while loop to start off the process
        grad = self.gradient_vector() #calculate gradient 
        new_x = self.coordinates[0] - (self.alpha*grad[0]) #update in x direction
        new_y = self.coordinates[1] - (self.alpha*grad[1]) #update in y direction
        new = [new_x,new_y]
        list_points.append(new)
        self.coordinates = new
        grad_new = self.gradient_vector()
        self.coordinates = old

        count = 1 #count number of iterations 
        while (((new[0]-old[0])**2)+((new[1]-old[1])**2))**0.5 >= 1e-6: #check convergence criteria
            delta = [0,0]
            gamma = [0,0]            
            delta[0] = new[0] - old[0] #change in x position
            delta[1] = new[1]- old[1] #change in y position
            gamma[0] = grad_new[0]-grad[0] #change in x component of gradient
            gamma[1] = grad_new[1]-grad[1] #change in y component of gradient
            #Terms to update G matrix
            term_1 = (np.outer(delta,delta))/(np.dot(gamma,delta))
            term_2 = (np.dot(G,np.dot(np.outer(delta,delta),G)))/(np.dot(gamma,np.dot(G,gamma)))                    
            old = new
            new_new = [0,0]
            for i in range(2):
                for j in range(2):
                    G[i][j] = G[i][j] + term_1[i][j] - term_2[i][j] #update entries in G matrix
            for i in range(2):
                new_new[i] = new[i] - self.alpha*(np.dot(G,grad)[i]) #take step in x and y
            list_points.append(new_new)
            new = new_new    
            grad = grad_new
            self.coordinates = new
            grad_new_new = self.gradient_vector()
            grad_new = grad_new_new    
            count +=1
        return new,list_points,count
                
        
#Class to perform test of how many data points you would need for a target accuracy
#Inputs: array_lengths - list of number of data points to be tested for what accuracy they give
#        data_set - list of time and sigma values over which analysis should be performed
#        Target accuracy - accuracy that you are testing for 
class accuracy_test:
    def __init__(self,array_lengths,data_set,target_accuracy):
        self.lengths = array_lengths
        self.data_set = data_set
        self.target_accuracy = target_accuracy/1e-12
        
    #Function to find number of data points needed for target accuracy 
    #Return: Number of points for required target accuracy
    def find_number(self):
        accuracies = []
        for number in self.lengths:
            new_test = decay_analysis('lifetime.txt',int(number),0) #create object of data analysis class
            fitting_test = fitting(new_test.get_split_data(),1e-8) #create object of fitting class
            fitting_test.parabolic_minimise("NLL",[0.4,0.5,0.6]) #perform parabolic 1D minimisation routine
            accuracies.append((number,fitting_test.standard_deviation_NLL("NLL",False,1)[0]-fitting_test.get_tau_optimal())) #Find accuracy

        lengths = [np.log(p[0]) for p in accuracies] #Update list of log(data_set lengths)
        uncertainties = [np.log(p[1]) for p in accuracies] #Update list of log(accuracy)
    
        #Perform linear regression  over these values
        slope, intercept, r_value, p_value, std_err = stats.linregress(lengths,uncertainties)
        plt.figure()
        plt.scatter(lengths,uncertainties)
        t_range = np.linspace(min(lengths),max(lengths),100)
        plt.plot(t_range,[((slope*j) + intercept) for j in t_range]) 
        plt.xlabel("log(Number of data points)")
        plt.ylabel("log(Accuracy of result)")
        plt.title("Behaviour of result accuracy with varying data set size")
        num_points = np.exp(((np.log(self.target_accuracy))-intercept)/slope) #trace linear regression line forward to target accuracy

        return num_points

    #Function to loop over data generation technique (Monte Carlo Simulations) for different seeds
    #Plot histogram of obtained tau values to look at distribution and check that error agrees
    def loop(self):
        some = []
        seed = 0 #set start seed
        for i in range(10):
            updated = decay_analysis('lifetime.txt',"all",seed) 
            taus, sigmas = updated.pseudo_data(10000,0.4095,False) #generate 10,000 pseudo data points
            new_test = decay_analysis('lifetime.txt',0,seed)
            new_test.append_data(taus,sigmas) #add pseudo-data to empty data_set
            fit_test_new = fitting(new_test.get_split_data(),1e-8)        
            tau_optimal, three_points = fit_test_new.parabolic_minimise("NLL",[0.4,0.5,0.6]) #minimise to calculate tau
            some.append(tau_optimal)
            seed +=1        #change the seed for the next run
        plt.figure() #plot histogram
        plt.hist(some,np.linspace(0.4,0.45,20))
        plt.xlim([0.4,0.45])
        plt.xlabel("Tau Optimal (ps)")
        plt.ylabel("Number in each bin")
        plt.title("Monte Carlo Simulation of Pseudo-Data Generation")

    #Function to test performance of pseudo-data generation by plotting histograms for comparison with raw data
    def test_pseudo(self):
        first_test = decay_analysis('lifetime.txt',"all",0) #object of class to access data from file
        taus, sigmas = first_test.pseudo_data(10000,0.4095,True)  #generate pseudo-data
        new_test = decay_analysis('lifetime.txt',0,0)
        new_test.append_data(taus,sigmas) #append pseudo data
        fit_test_new = fitting(new_test.get_split_data(),1e-8)#
        t = time.time()
        one = fit_test_new.standard_deviation_NLL("NLL",False,1)[0]
        two = fit_test_new.get_tau_optimal()  
        print "Upper Error ", one
        print "Minimum Value Calculated for tau in 1-d using pseudo data ", two 
        print "Accuracy: ", one-two
        print "Time Taken: ", time.time()-t
    
        T,A,Z,f_interpolated = fit_test_new.interpolate() #interpolate over new data
        def f_interp(t,a):
            r1 = f_interpolated(t,a)
            return float(r1)
    
        results, list_points,number = fit_test_new.two_dimensional_minimise_grad_method([0.41,0.8],f_interp,0.0001,0.00001)
        print "Minimum value found using pseudo-data for tau optimal and a", results 