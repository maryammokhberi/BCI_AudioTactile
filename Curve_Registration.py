# -*- coding: utf-8 -*-
"""
Created on Wed Nov 08 22:20:04 2017

@author: Amarantine
"""
import numpy as np
import matplotlib.pyplot as plt
#% Register a set of curves
#% Synopsis
#% regdata=registrethis(data)
#%
def registerthis(data):
    plt.close('all')
    NumberBasis=20;
    NPts=data.shape[1]
    
    gaittime=np.transpose(np.linspace(0,1,NPts))
    
#    [NRows NumCurves]=size(data);
    NRows=data.shape[0]
    NumCurves=data.shape[1]
    gaitarray = np.zeros((NRows, NumCurves, 1))
    gaitarray[:,:,1] = data
    
#    %  ---------------  set up the fourier basis  ------------------------
    
    gaitbasis = create_fourier_basis(np.array([0,1]), NumberBasis); 
#    %[1,100] is the range for evaluating functions
    
#    %  -----------  create the fd object (no smoothing)  -----------------
    
    gaitfd = data2fd(gaitarray, gaittime,  gaitbasis) #%functional data
    
    gaitfd_fdnames{1} = 'time';
    gaitfd_fdnames{2} = 'analysis';
    gaitfd_fdnames{3} = 'amplitude';
    gaitfd = putnames(gaitfd, gaitfd_fdnames);
    
    %  set up basis for warping function
    nbasis = 7;
    wbasis = create_fourier_basis([0,1],nbasis);
    Lfd    = 3;
    lambda = 1e-3;
    
    %  set parameters for registerfd
    
    periodic = 1;  %  data are periodic
    
    index  = 1:NumCurves;  %  curves to be registered
    
    %  set initial value for warping functions
    
    Wfd0   = fd(zeros(nbasis,length(index)),wbasis); % create functional data object
    % 1st argument is coefficient matrix: nbasis x number of signals
    % 2nd argument is basis object
    
    WfdPar = fdPar(Wfd0, Lfd, lambda);
    
    %  set up target for registration of first derivatives
    Dgaitfd = deriv(gaitfd(index),1);
    y0fd    = meanfdata(Dgaitfd);
    yfd     = Dgaitfd(index);
    
    %Register raw curves themselves
    %y0fd = meanfdata(gaitfd);
    %yfd=gaitfd(index);
    
    xfine   = linspace(0,1,NPts)'; %generate 100 equally spaced points between 0 and 1
    ofine   = ones(NPts,1); %vector of ones
    y0vec   = squeeze(eval_fd(y0fd, xfine)); %remove singleton dimensions; 
    %evaluate functional datda observation at xfine
    yvec    = eval_fd(yfd, xfine); %evaluate derivative curves at xfine
    
    
    %  carry out the registration
    [regfd, Wfd, shift] = registerfd(y0fd, yfd, WfdPar, periodic);
    
    %  compute registered function and warping function values
    
    yregmat = eval_fd(xfine, regfd); %registered curves with correct time axis
    Wfd     = Wfd;
    % shift   = shift;
    warpmat = monfn(xfine, Wfd);
    warpmat = ofine*shift' + warpmat./(ofine*warpmat(NPts,:));
    
    %  plot the registered derivatives
    %plot(yregmat); % registred curves wrt to gait cycle
    %plot(regfd); %registered curves wrt to [0,1]
    
    % plot registered curves themselves
    % Create functional data object that spans new range of time (after
    % warping)
    MinTime=min(min(warpmat)); %new range of times
    MaxTime=max(max(warpmat));
    gaittime=linspace(MinTime,MaxTime,NPts)';
    gaitbasis = create_fourier_basis([MinTime, MaxTime], NumberBasis);  %create basis
    newgaitfd = data2fd(gaitarray, gaittime,  gaitbasis); %create functional data object
    
    regdata=zeros(NPts,NumCurves);
    for i=1:NumCurves,
        a=eval_fd(warpmat(:,i),newgaitfd); %evaluate the gait functions at warped times
        regdata(:,i)=a(:,i);
    end
    
    % plot(data);
    % figure;
    % plot(regdata);
    
    % compare variance before and after registration
    % figure;
    % plot(var(data'));hold;
    % plot(var(regdata'),'r');
    
    % Create new functional data object for registered curves
    gaitarray=regdata;
    reggaitfd=data2fd(gaitarray,gaittime,gaitbasis);
    
    return regdata , shift
