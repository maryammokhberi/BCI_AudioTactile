# -*- coding: utf-8 -*-
"""
Created on Tue Jul 04 11:16:54 2017

@author: mokhberim
"""
## training procedure for Bayesian LDA
## INPUT:
##    b       - object of type bayeslda
##    x       - m*n matrix containing n feature vectors of size m*1
##    y       - 1*n matrix containing class labels (-1,1)
##
## OUTPUT:
##    b       - updated object of type bayeslda
##
## Author: Ulrich Hoffmann - EPFL, 2006
## Copyright: Ulrich Hoffmann - EPFL
##
## The algorithm implemented here was originally described by
## MacKay, D. J. C., 1992. Bayesian interpolation.
## Neural Computation 4 (3), pp. 415-447.

## revised by erwei in order to include in our code
#Maryam Mokhberi converted the code to a python script


def BLDA_train(x, y):

    from numpy import linalg as LA

    
    
#    x = x'
#    y = y' #Maryam: I commented these lines since my input data is already in 
            #in the form of n*1 and n*m
    
    for i in range(y.shape[0]):
        if y[i]==0:
            y[i]=-1
    
    ### compute regression targets from class labels (to do lda via regression)
    n_posexamples = sum(y==1)
    n_negexamples = sum(y==-1)
    n_examples    = n_posexamples + n_negexamples
    for i in range(y.shape[0]):
        if y[i]==1:
            y[i]= n_examples/n_posexamples
        if y[i]==-1:
            y[i]= n_examples/n_negexamples
            
    
    
    ## add feature that is constantly one (bias term)
    ## x = [x; ones(1,size(x,2))];  
    
    
    ## initialize variables for fast iterative estimation of alpha and beta
    n_features = x.shape[1]            # dimension of feature vectors 
    d_beta = float('inf')                      # (initial) diff. between new and old beta  
    d_alpha = float('inf')                    # (initial) diff. between new and old alpha 
    alpha    = 25                     # (initial) inverse variance of prior distribution
    biasalpha = 0.00000001            # (initial) inverse variance of prior for bias term
    beta     = 1                      # (initial) inverse variance around targets
    stopeps  = 0.0001                 # desired precision for alpha and beta
    i        = 1                      # keeps track of number of iterations
    maxit    = 500                    # maximal number of iterations 
    w,v = LA.eig(np.matmul(np.transpose(x),x)  # needed for fast estimation of alpha and beta  #v: vector #w:eigen values
    vxy    = np.matmul(np.matmul(np.transpose(v),x), np.transpose(y))                  # dito
    w = np.diag(w)                       # dito
    e = np.ones([n_features-1,1],dtype=float)          # dito
    
    
    ## estimate alpha and beta iteratively
    while ((d_alpha > stopeps) or (d_beta > stopeps)) and (i < maxit): #Maryam:changed syntax
        alphaold = alpha
        betaold  = beta
        m = beta*v*(LA.inv(beta*w+[alpha*e, biasalpha])*vxy)
        err = sum((y-m'*x).^2)
        gamma = sum(beta*w./(beta*w+[alpha*e; biasalpha]))
        alpha = gamma/(m'*m);
        beta  = (n_examples - gamma)/err
    ##     if b.verbose
    ##         fprintf('Iteration %i: alpha = %f, beta = %f\n',i,alpha,beta);
    ##     end
        d_alpha = abs(alpha-alphaold)
        d_beta  = abs(beta-betaold)
        i = i + 1
    
    
    #
    ### process results of estimation 
    if (i < maxit)
        
        # compute the log evidence
        #this can be used for simple model selection tasks
        #(see MacKays paper)
        blda.evidence = (n_features/2)*log(alpha) + (n_examples/2)*log(beta) - ...
        (beta/2)*err - (alpha/2)*m'*m - ...
        0.5*sum(log((beta*w+[alpha*e; biasalpha]))) - (n_examples/2)*log(2*pi);
    
        # store alpha, beta, the posterior mean and the posterrior precision-
        # matrix in class attributes
        blda.alpha = alpha;
        blda.beta  = beta;
        blda.w     = m;
        blda.p     = v*diag((beta*w+[alpha*e; biasalpha]).^-1)*v';
     
    #     if b.verbose
    #         fprintf('Optimization of alpha and beta successfull.\n');
    #         fprintf('The logevidence is %f.\n',b.evidence);
    #     end
    
    else
    
        fprintf('Optimization of alpha and beta did not converge after %i iterations.\n',maxIt);
        fprintf('Giving up.');
    
    end
    

    return