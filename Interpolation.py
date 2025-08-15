import numpy as np
import subprocess
import random
import concurrent.futures as cf
import multiprocessing
import time
import matplotlib.pyplot as plt
from numba import njit
import matplotlib
matplotlib.use('TkAgg')

########## VERSION 3 (2D matrix) ##########:
def Interpolator(data, category, thinning, saving_id):

    def normalize(D):
        min = np.nanmin(D)
        max = np.nanmax(D)
        return min, max, (D-min)/(max-min)


    min, max, Data = normalize(data)
    Cats = category

    ## Defining the parameters of the problem:

    # # SETUP:
    J = 1
    delta = 1/100
    T_start = 1/np.log(2)
    dimensions = np.shape(Data)
    r = delta

    @njit
    def spins_binner(spins,r):
        binned_spins = np.zeros_like(spins)
        [m,n] = np.shape(binned_spins)
        for i in range(m):
            for j in range(n):
                s = spins[i,j]
                left_boundary = r
                right_boundary = left_boundary + r
                while right_boundary <= 1:
                    if left_boundary < s <= right_boundary:
                        binned_spins[i,j] = (right_boundary + left_boundary)/2
                        break
                    else:
                        left_boundary += r
                        right_boundary += r
                        pass
                if s <=r:
                    binned_spins[i,j] = r/2
                elif s >= 1 - r:
                    binned_spins[i,j] = 1 - r/2
        return binned_spins

    @njit
    def spin_initialization2(data):

      [N,M] = np.shape(data)

      spin_lattice = np.zeros_like(data)

      for i in range(N):
        for j in range(M):
          spin = Data[i,j]
          if np.isnan(spin):
              spin_value_choices = np.arange(r / 2, 1 + r / 2, r / 2)
              spin_guess = np.random.choice(spin_value_choices)
              spin_lattice[i,j] = spin_guess
          else:
              spin_lattice[i,j] = data[i,j]

      return spin_lattice

    @njit
    def padder(matrix):

        [m,n] = np.shape(matrix)

        padded_matrix = np.zeros((m+2,n+2))*np.nan

        padded_matrix[1:m+1,1:n+1] = matrix

        return padded_matrix









    ss = spin_initialization2(Data)

    print("Initializations done!")

    @njit
    def OneSpinHamiltonian(spins, spin_loc, J):
        """
        Calculates the Hamiltonian for the entire grid.
        """
        i, j = spin_loc[0] + 1, spin_loc[1] + 1

        neighbours = np.array([spins[i-1,j],spins[i+1,j],spins[i,j-1],spins[i,j+1]])
        MSE = (1/4)*np.nansum((neighbours - spins[i,j])**2)

        return MSE

    @njit
    def RMS_to_binsize_ratio(config1,config2,delta,thinning):

        [m,n] = np.shape(config1)
        RMS = np.sqrt(np.nansum((config1-config2)**2)/(m*n*(1-thinning)))

        return (RMS/delta)




    #---------------------------------------------------------------------------------------------------------------------

    def MCMC_0(spin_lattice_categories, total_steps):

        inference_spin_locs = np.argwhere(spin_lattice_categories == 1)
        chosen_indices = np.random.choice(len(inference_spin_locs), total_steps, replace=True)
        spin_picks = inference_spin_locs[chosen_indices]
        spin_value_choices = np.arange(r / 2, 1 + r / 2, r / 2)
        spin_guesses = np.random.choice(spin_value_choices, total_steps, replace=True)

        return [spin_picks, spin_guesses]


    def MCMC(spin_lattice, spin_lattice_categories, T, steps):

        [m, n] = np.shape(spin_lattice)
        spins_matrix = padder(spin_lattice)
        path = MCMC_0(spin_lattice_categories, steps)
        for step in range(steps):
            # referencing the old spins:
            spins = spins_matrix.copy()

            # picking a random spin from the matrix:
            spin_pick = path[0][step]
            E_old = OneSpinHamiltonian(spins, np.asarray(spin_pick), J)

            # updating that chosen spin:
            new_spin = path[1][step]
            spins[spin_pick[0] + 1, spin_pick[1] + 1] = new_spin
            E_new = OneSpinHamiltonian(np.array(spins), np.asarray(spin_pick), J)

            # Change in energy calculated here:
            energy_change = E_new - E_old

            if energy_change < 0:
                spins_matrix = spins
            else:
                if np.exp(-energy_change / T) > random.random():
                    spins_matrix = spins
                else:
                    pass

        return np.array([spins_matrix[1:m + 1, 1:n + 1]])

    @njit
    def linear_correction(spins_matrix,spin_categories,rounds, mode):  #computationally inefficient!
        spins = spins_matrix.copy()
        cats = spin_categories
        [m,n] = np.shape(spins)
        b = 1
        for _ in range(m*n*rounds):
          i = np.random.randint(1,m-1)
          j = np.random.randint(1,n-1)
          if cats[i,j] == mode:
              N_1 = spins[i-1, j]
              N_2 = spins[i+1, j]
              N_3 = spins[i, j-1]
              N_4 = spins[i, j+1]
              C_1 = cats[i - 1, j]
              C_2 = cats[i + 1, j]
              C_3 = cats[i, j - 1]
              C_4 = cats[i, j + 1]
              importance = np.ones(4)
              if C_1 == -1:
                  importance[0] = b
              elif C_2 == -1:
                  importance[1] = b
              elif C_3 == -1:
                  importance[2] = b
              elif C_4 == -1:
                  importance[3] = b

              neighbours = np.array([N_1,N_2,N_3,N_4])*importance
              spins[i,j] = np.nansum(neighbours)/np.nansum(importance)


        return spins



    n = int(1/(delta))
    R_lst = []
    spin_lattice, spin_lattice_categories = ss, Cats
    steps=int(dimensions[0]*dimensions[1]*n*(1-thinning))
    #first iteration to get R0:
    S = MCMC(spin_lattice,spin_lattice_categories,T_start,steps)[0]
    R0 = RMS_to_binsize_ratio(spin_lattice,S,delta,thinning); R_lst.append(R0)
    check_points = []
    check_points.append(S)
    T_lst = [T_start]
    cooling = 1.15


    # continuing the cycle + annealing:
    while R_lst[-1] > 0.5:
        T = T_lst[-1]/cooling
        S_next = MCMC(check_points[-1], spin_lattice_categories, T, steps)[0]
        R = RMS_to_binsize_ratio(check_points[-1], S_next, delta, thinning)
        R_lst.append(R)
        print(R)
        check_points.append(S_next)
        T_lst.append(T)

        if R < 0.5:
            final_corrected = linear_correction(S_next,spin_lattice_categories, 250, 1)
            final_corrected = linear_correction(final_corrected, spin_lattice_categories, 25, -1)
            final_corrected = final_corrected*(max-min) + min
            np.save("./crops/"+"V_"+str(saving_id)+".npy",final_corrected)
            np.save("./crops/checkpoints.npy",np.array(check_points))
            print("Interpolation if Complete.")
        else:
            pass







D = np.load("K_train.npy")
C = np.load("categories_train.npy")

plt.imshow(D)
plt.show()

plt.imshow(C)
plt.show()

Interpolator(D,C,0.1983,"unbiased")
#######################################################################################################################
