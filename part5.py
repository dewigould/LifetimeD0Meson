import random 
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
import time


see_results = False
#True if user wants to perform analysis numerically, False if want to use provided CDF inverse calculated analytically
numerical_solution = False

validation_routine = False


if see_results == True:
    #----------------------------------
    #PART A
    
    #Number of random numbers to be generated
    N = 1e5    
    #Number of bins of histogram required
    num_bins = 100
    
    #Generate bins over required ranges
    bins = np.linspace(0,1,num_bins)
    bins_pdf = np.linspace(0,np.pi,num_bins)

    #Set Seed
    seed = 3
    random.seed(seed)

    #----------------------------------
    #PART B

    #Create x range of target deviate distribution over a given number of increments
    increments = 1000
    t_range = np.linspace(0,np.pi,increments)

    #Define required pdf
    def pdf_1(x):
        if 0.0 <= x <= np.pi:
            result = (1.0/2.0)*(np.sin(x))
        else:
            result = 0
        return result

    #Define second required pdf
    def pdf_2(x):
        if 0.0 <= x <= np.pi:
            result = (2.0/np.pi)*((np.sin(x))**2)
        else:
            result = 0
        return result    
    
    #----------------------------------
    #PART C 
    
    #Define constant comparison function
    def const_comparison(y):
        C = 0.0
        if 0.0 <= y <= np.pi:
            C = 0.65
        return C
    
    #Define normalised comparison function
    def norm_const_comparison(y):
        return (const_comparison(y)/0.65)*(1.0/np.pi)

    #Define new, better-conforming comparison function
    def new_comparison(y):
        return (4.0/np.pi)*pdf_1(y)

    #Define the normalised version of the above better-conforming comparison function
    def normalised_comparison(y):
        return pdf_1(y)

    #Define inverse of CDF functions of better-conforming comparison function - calculated analytically
    def y_inverse_1(x):
        if 0.0 <= x <=1.0:
            return np.arccos(1-(2*x))

    
    #Required number of samples for rejection method
    N_trials = 1e5
    #Number of bins of resulting histogram for rejection method
    num_bins_rej = 100
    #Create bins for plot
    bins_y_i = np.linspace(0,np.pi,num_bins_rej)

#----------------------
    # PART A
#Function: Generate N random numbers evenly distributed between 0 and value
#Input: Num - number of required random numbers (integer), upper range of random number
#Output: Array of generateted random numbers
def generate_uni_deviate(Num,value):
    uniform_deviate = []
    for i in range(int(Num)):
        r = random.uniform(0,value)
        uniform_deviate.append(r)    
    return uniform_deviate

def generate_uni_deviate_variable(Num,smallest,largest):
    uniform_deviate = []
    for i in range(int(Num)):
        r = random.uniform(smallest,largest)
        uniform_deviate.append(r)
    return uniform_deviate

#----------------------
    # PART B

#Function: Perform cumulative distribution integration step of transformation method
    #Note - this function is only required when the user hasnt provided an analytic CDF inverse
#Input: y - value to be integrated up to
#       prob_function - probability density function to be integrated
#Output: Integral from [-inf,y] of input probability density function
def cumulative_distribution_integrate(y,prob_function):
    #return quad(prob_function,-1*np.inf,y)[0]
    return quad(prob_function,-3.0,y)[0]


#Function: Perform Transformation method to obtain new distribution of random numbers
#Input: pdf_required - desired shape of new random number distribution (in form of probability density function)
#       initial_deviate - array of uniform deviate to be transformed
#       new_range - x range over which new distribution should be determined 
#       analytic - on/off switch takes two values (True, False)..defines whether function performs intergration numerically or given result analytically
#       y_input - analytically calculated inverse of CDF - if analytic = False, this is ignored so can be given any value whatsoever
#Output: array of random numbers distributed according to new, target distribution
def transformation_method(pdf_required,initial_deviate,new_range,analytic,y_input):
    
    #if given an analytically calculated function, this is called to map the random numbers to their new value
    if analytic == True:
        to_return = [y_input(x) for x in initial_deviate]
    
    #If no function given for inverse of CDF: result performed numerically  
    if analytic == False:  
    
    #Perform integration to get cumulative distribution function evaluated at all points in target x range
        F_y = []
        y_values = []
        for i in new_range:
            a = i
            b = cumulative_distribution_integrate(i,pdf_required)
            F_y.append((a,b))
    
            #Invert Result - calculated inverse of function by "swapping" x,y values
            #y_values is array of tuples of the form ((x1,y1),(x2,y2).....(xN,yN)) - points of function calculated by inverting CDF
            y_values.append((b,a))
        
        

        

        #Function: map value x (from uniform deviate) to y found using inverted function above
        #Input: x - random number between 0 and 1
        #       y_points_input - list of tuples of x,y values of inverted function 
        #Output: new random number, mapped into new distribution
        def y(x,y_points_input):
        
            #sort in ascending order of x component
            y_points = sorted(y_points_input, key = lambda a: a[0])
            #Find which x value is closest to our random number - and then map with this number

            for i in range(len(y_points)):
                if x == y_points[i][0]:
                    return y_points[i][1]
                elif x < y_points[i][0]:
                    if i == 0:
                        return y_points[i][1]
                    elif (y_points[i][0]-x) == (x-y_points[i-1][0]):
                        return ((y_points[i][1] + y_points[i-1][1]))/2.0
                    elif (y_points[i][0]-x) < (x-y_points[i-1][0]):
                        return y_points[i][1]
                    else:
                        return y_points[i-1][1]
                else:
                    if i == len(y_points)-1:
                        return y_points[i][1]
    
        #Gather results - perform mapping operation for all random numbers in uniform deviate
        to_return = [y(x,y_values) for x in initial_deviate]
        
    return to_return


#----------------------------------
#PART C

#Implement Rejection Method

#Function: Implement the Rejection Method
#Input:
#       Num - number of required samples
#       comparison - comparison function
#       input_deviate - array of random numbers to be transformed 
#       target_pdf - the distribution to be transformed into
#Output:
#       not_rejected - array of new random numbers distributed according to target_pdf
def rej_method(Num,comparison,input_deviate,target_pdf):
    
    #Empty arrays to be filled by algorithm
    y_i = []
    p_i = []
    for i in range(len(input_deviate)):
        #get random number distributed according to input_deviate
        to_append = input_deviate[i]
        y_i.append(to_append)
        #get second random number distributed between 0 and max value of comparison function evaluated at first random number
        p_i.append(random.uniform(0,comparison(to_append)))
        
    
    not_rejected = []
    #Final algorithm - values accepted or rejected with probablility determined by comparison two pdf's
    for i in range(int(Num)):
        if target_pdf(y_i[i]) >= p_i[i]:
            not_rejected.append(y_i[i])
            
    return not_rejected
     
#Function: Iterate over rejection method multiple times to get required number of samples
#Input: 
#       Number_required - required number of samples at end of process
#       normalised_comparison - normalised comparison function
#       comparison_fun - un-normlaised comparison function
#       required_range - range over which new random numbers are to be distributed
#       start_seed - input seed to start process
#       analytic - on/off switch takes two values (True, False)..defines whether function performs intergration numerically or given result analytically
#       y_input - analytically calculated inverse of CDF - if analytic = False, this is ignored so can be given any value whatsoever
#       const_comparison - True if your comparison function is a constant one, False otherwise
#Output:
#       not_rejected - array of REQUIRED NUMBER of new random numbers distributed according to target_pdf
    
#Note - I appreciate there are a silly amount of arguments to this function - this was just to allow easy plotting at the end
    #and to allow easy switching between analytic and numerical regimes... if this was to be properly used it could be easily trimmed down.
def get_result_rej(Number_required,
                   normalised_comparison,
                   comparison_func,
                   required_range,
                   target_pdf,
                   start_seed,
                   analytic,
                   y_required,
                   const_comparison,
                   seed_input):
    start = time.time()
    not_rejected = []
    Num = Number_required
    
    random.seed(start_seed)
    random.seed(seed_input)


    #Number of samples taken in this iteration
    Num_samples = Num
    count = 0
    #Keep performing process until Num of samples generated = that required
    while len(not_rejected) < Num:
        count += Num_samples

        #Generate new uniform deviate for each iteration - new random numbers from same seed
        new_uniform_deviate = generate_uni_deviate(Num_samples,1.0)
        
        #Generate new comparison function deviate for each iteration
        #using transformation method for non-constant comparison functions
        if const_comparison == False:
            new_comparison_deviate = transformation_method(normalised_comparison,new_uniform_deviate,required_range,analytic,y_required)
            result = rej_method(Num_samples,comparison_func,new_comparison_deviate,target_pdf)
        #Using uniform deviate generator above for constant comparison functions
        if const_comparison == True:
            result = rej_method(Num_samples,comparison_func,generate_uni_deviate_variable(Num_samples,min(required_range),max(required_range)),target_pdf)
                     
        for i in result:
            not_rejected.append(i)
        
            #Updated number of samples with number of samples left to finish process
            Num_samples = (Num-len(not_rejected))   
            
        if len(not_rejected) != Num:
            for i in range(int(len(not_rejected)-Num)):
                del not_rejected[-1]
    
        
    end = time.time()
    print "Rejection method Time Taken ", end-start  
    print "Number of samples generated: ", len(not_rejected)
    print "Number of iterations required to get number of samples: ", count
    return not_rejected,end-start
  



#Get Results

if see_results == True:
    
    uniform_deviate = generate_uni_deviate(N,1.0)

    start = time.time()
    pdf_deviate_1 = transformation_method(pdf_1,uniform_deviate,np.linspace(0,np.pi,100),True,y_inverse_1)
    end = time.time()
    print "Time for Transformation Method (0.5sinx): ", end-start
    time_1 = end-start
    print "Number of Samples: ", len(pdf_deviate_1)
    start = time.time()
    pdf_deviate_2 = transformation_method(pdf_2,uniform_deviate,np.linspace(0,np.pi,100),False,0)  
    end = time.time() 
    print " "
    print "Time for Transformation Method (sin^2x): ", end-start
    print "Number of Samples: ", len(pdf_deviate_1)   
    
    print " "


    if numerical_solution == False:
        #Do rejection method with constant comparison function
        print "Rejection Method for sin^2(x) with constant comparison function"
        not_rejected,time_taken_one = get_result_rej(N_trials,norm_const_comparison,const_comparison,np.linspace(0,np.pi,100),pdf_2,0,True,0,True)
    
        print " "
    
        #Do rejection method with better conforming comparison function
        print "Rejection Method for sin^2(x) with ~sin(x) comparison function"
    
        not_rejected_2,time_taken_two = get_result_rej(N_trials,normalised_comparison,new_comparison,np.linspace(0,np.pi,100),pdf_2,1,True,y_inverse_1,False) 
    
    if numerical_solution == True:    
    
        #Do rejection method with constant comparison function
        print "Rejection Method for sin^2(x) with constant comparison function"
        not_rejected,time_taken_one = get_result_rej(N_trials,norm_const_comparison,const_comparison,np.linspace(0,np.pi,100),pdf_2,0,False,0,False)
    
        print " "
    
        #Do rejection method with better conforming comparison function
        print "Rejection Method for sin^2(x) with ~sin(x) comparison function"
    
        not_rejected_2,time_taken_two = get_result_rej(N_trials,normalised_comparison,new_comparison,np.linspace(0,np.pi,100),pdf_2,1,False,0,False) 
    

    
    
    
    
    #Plotting Results
    f, ax = plt.subplots(3, 2)
    
    ax[0,0].hist(uniform_deviate,bins,color="blue",label="Number of bins: %d" %(num_bins))
    ax[0,0].plot(bins,[N/num_bins]*num_bins,color="red",label="target uniform deviate")
    ax[0,0].set_title("Uniform Deviate",fontsize = 'small')
    ax[0,0].legend(fontsize='x-small')
    
    ax[0,1].plot(t_range,[pdf_1(x) for x in t_range],color="blue",label = "0.5*sin(x)")
    ax[0,1].plot(t_range,[pdf_2(x) for x in t_range],color = "red",label = "(2/pi)sin^2(x)")
    ax[0,1].plot(t_range,[new_comparison(x) for x in t_range],color = "green", label = "Comparison Function")
    ax[0,1].plot(t_range,[normalised_comparison(x) for x in t_range],color = "orange", label = " Normalised Comparison Function")
    ax[0,1].plot(t_range,[const_comparison(x) for x in t_range],color="purple",label = "Constant Comparison Function (=0.65)")
    ax[0,1].legend(fontsize='x-small')
    ax[0,1].set_title("PDF's and Comparison Functions",fontsize = 'small')
    
    ax[1,0].hist(pdf_deviate_1,bins_pdf,normed=True,color="blue",label = "Transformation of 0.5sin(x)")
    ax[1,0].plot(t_range,[pdf_1(i) for i in t_range],color = "red",label = "Sin^2x fit")
    ax[1,0].set_title("Transformation Method (0.5sin(x))",fontsize = 'small')
    ax[1,0].set_xlabel("x")
    
    ax[1,1].hist(pdf_deviate_2,bins_pdf,normed=True,color="blue",label="(2/pi)sin^2(x)")
    ax[1,1].plot(t_range,[pdf_2(i) for i in t_range],color = "red",label = "Sin^2x fit")
    ax[1,1].set_title("Transformation Method of (2/pi)sin^2(x)",fontsize = 'small')
    ax[1,1].set_xlabel("x")
    
    ax[2,0].hist(not_rejected,bins_y_i,normed=True,color="blue",label = "Constant Comparison Function")
    ax[2,0].plot(t_range,[pdf_2(i) for i in t_range],color = "red",label = "Sin^2x fit")
    ax[2,0].set_title("Rejection Method - (2/pi)sin^2(x) ",fontsize = 'small')
    ax[2,0].legend(fontsize='x-small')
    
    ax[2,1].hist(not_rejected_2,bins_y_i,normed=True, color="blue",label = "Better Fitting Comparison Function (~sinx)")
    ax[2,1].plot(t_range,[pdf_2(i) for i in t_range],color = "red",label = "Sin^2x fit")
    ax[2,1].set_title("Rejection Method - (2/pi)sin^2(x)",fontsize = 'small')
    ax[2,1].legend(fontsize='x-small')
    

#Perform transformation and Rejection methods to simple constant distribution
if validation_routine == True:
    
    def f(x):
        if 0.0 <= x <= np.pi:
            return (1.0/np.pi)
        else:
            return 0.0
        
    not_rejected, t = get_result_rej(1e5,f,f,np.linspace(0,np.pi,100),f,1,True,0,True)
    num_bins = 100
    bins = np.linspace(0,1,num_bins)
    bins_pdf = np.linspace(0,np.pi,num_bins)
    plt.hist(not_rejected,bins)
            