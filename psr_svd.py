import sys
import os
import os.path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import math
from scipy import fftpack, optimize, interpolate, linalg, integrate
import copy

NHARMONIC = 640 
NMODES = 4

NPHASEBIN = 512
NCENTRALBINS = 512
NCENTRALBINSMAIN = 512

#plot_name = 'U_fit_'

def main():    

    if False: #reform the format of raw data
        '''raw_data is in the shape of (pulse rotation, phase, pol)'''
        raw_data = np.load('/scratch2/p/pen/hsiuhsil/psr_B1957+20/data_file/B1957pol3_512g_2014-06-15T07:06:23.00000+536s.npy')
        '''Separate raw__data into L, R pols, which in the shape of ((pulse rotation, phase, L/R pol))'''
        data = np.sum(raw_data.reshape(raw_data.shape[0], raw_data.shape[1], 3,2), axis=2)
        L_data = np.zeros((data[:,:,0].shape))
        R_data = np.zeros((data[:,:,1].shape))
        for ii in xrange(len(L_data)):
            L_data[ii] = data[ii,:,0] - np.mean(data[ii,:,0])
            R_data[ii] = data[ii,:,1] - np.mean(data[ii,:,1])
        B_data = np.concatenate((L_data, R_data), axis=1) # B means both of L and R
        np.save('B_data.npy', B_data)
        print 'save B_data'

    B_data = np.load('/scratch2/p/pen/hsiuhsil/psr_B1957+20/data_file/B_data.npy')
    if True: # Stack the pulse profiles
        profile_stack = 50
        B_data_stack = stack(B_data, profile_stack)
        print 'B_data_stack.shape: ', B_data_stack.shape
        
    if False:
        rebin_pulse = 1
        filename = 'B_data_rebin_' + str(rebin_pulse)
        B_data_rebin = B_data_stack#rebin_spec(B_data, rebin_pulse, 1)
        np.save(filename + '.npy', B_data_rebin)
        print 'B_data_rebin.shape', B_data_rebin.shape
#        svd(B_data_rebin, rebin_pulse)
        plot_svd(B_data_rebin, rebin_pulse, filename)

    if True: 
        '''Reconstruct V modes'''
        profile = B_data_stack

        # remove signal modes, and reconstruct noise profiles.
        U, s, V = svd(profile)
        V0_raw = copy.deepcopy(V[0])
        V1_raw = copy.deepcopy(V[1])
        V[0] = 0
        V[1] = 0
        noise_profiles = reconstruct_profile(U,s,V)
        print 'noise_profiles.shape', noise_profiles.shape

        # get an estimate of the noise level, for each the R and L polarizations.
        noise_var_L = np.mean(np.var(noise_profiles[:, 0:noise_profiles.shape[1]/2],axis=1))
        noise_var_R = np.mean(np.var(noise_profiles[:, noise_profiles.shape[1]/2:noise_profiles.shape[1]],axis=1))

        # transform origin profiles with unit variance.
        profile_norm_var_L = np.zeros((profile.shape[0], profile.shape[1]/2))
        profile_norm_var_R = np.zeros((profile.shape[0], profile.shape[1]/2))

        for ii in xrange(profile.shape[0]):
            profile_norm_var_L[ii] = norm_variace_profile(profile[ii, 0:profile.shape[1]/2], noise_var_L)
            profile_norm_var_R[ii] = norm_variace_profile(profile[ii, profile.shape[1]/2:profile.shape[1]], noise_var_R)
        profile_norm_var = np.concatenate((profile_norm_var_L, profile_norm_var_R), axis=1)
        print 'finish profile_norm_var'
        # SVD on the normalized variance profile.
        U, s, V = svd(profile_norm_var)
        plot_svd(profile_norm_var, 'profile_norm_var')
        print 'finish SVD'
        check_noise(profile_norm_var)      

        '''reshape profiles of L and R into periodic signals (pulse number, L/R, phases)'''
        profile = profile.reshape(profile.shape[0], 2, profile.shape[1]/2)        
        V_recon = V.reshape(V.shape[0], 2, V.shape[1]/2)

        phase_fitting(profile, V_recon)


def reconstruct_V(V, V0_raw, V1_raw):
    '''reconstruct V modes by adding raw V0 and V1 to the V modes created without raw V0 and V1'''
    V[0] += V0_raw
    V[1] += V1_raw

    # check V_L and V_R
    V_L = V[:, 0: V.shape[1]/2]
    V_R = V[:, V.shape[1]/2:V.shape[1]]
    plot_V(V_L, 'recon_V_L.png')
    plot_V(V_R, 'recon_V_R.png')

    # reshape V into (mode numbers, L/R, phases)
    V_recon = V.reshape(V.shape[0], 2, V.shape[1]/2)

    return V_recon

def plot_V(V, plot_name_V): 
    '''plot V modes'''
    fontsize = 16
    plt.close('all')
    plt.figure()
    n_step = -0.3
    x_range = np.arange(0 , len(V[0]))
    color = ['r', 'g', 'b', 'y', 'c', '0.0', '0.2', '0.4', '0.6', '0.8']
    for ii in xrange(len(color)):
        plt.plot(x_range, np.roll(V[ii] + ii *n_step, 0), color[ii], linewidth=1.0)
    plt.xlim((0, len(V[0])))
    plt.ylabel('V values', fontsize=fontsize)
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
#    plot_name_V = 'recon_V.png'
    plt.savefig(plot_name_V, bbox_inches='tight')

def reconstruct_profile(U,s,V):
    '''reconstruct a profile using U, s, and V.'''
    profile_recon = np.dot(U, np.dot(np.diag(s), V))
    return profile_recon 

def norm_variace_profile(profile, variance):
    '''transform an array into the array with zero mean and unit variance.'''
    profile_norm_var = (profile - np.mean(profile)) / np.sqrt(variance)
    return profile_norm_var

def check_noise(profile):

    var_L = np.zeros((profile.shape[0]))
    var_R = np.zeros((profile.shape[0]))
    for ii in xrange(profile.shape[0]):
        var_L[ii] = np.var(profile[ii, 0:profile.shape[1]/2])
        var_R[ii] = np.var(profile[ii, profile.shape[1]/2:profile.shape[1]])

    rebin_factor = 100
    rebin_var_L = rebin_array(var_L, rebin_factor)
    rebin_var_R = rebin_array(var_R, rebin_factor)

    fontsize = 16

    plt.close('all')
    plt.figure()
    x_range = np.arange(0 , len(rebin_var_L))
    plt.plot(x_range, rebin_var_L, 'ro', linewidth=2.5, label='rebin_var_L')
    plt.plot(x_range, rebin_var_R, 'bs', linewidth=2.5, label='rebin_var_R')
    plt.xlim((0, len(rebin_var_L)))
    plt.xlabel('profile numbers', fontsize=fontsize)
    plt.ylabel('Variance', fontsize=fontsize)
    plt.legend(loc='upper right', fontsize=fontsize)
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    plt.savefig('variance_rl.png', bbox_inches='tight')


def phase_fitting(profiles, V):
    profile_numbers = []
    profile_numbers_lik = []
    phase_model = []
    phases = []
    phase_errors = []
    phases_lik = []
    phase_errors_lik = []

#    profiles = B_data_rebin
    nprof = len(profiles)
    profiles = profiles[:nprof]
#    profiles = np.mean(profiles, 1)
    V_fft = fftpack.fft(V, axis=1)
    V_fft_L = fftpack.fft(V[:,0,:], axis=1)
    V_fft_R = fftpack.fft(V[:,1,:], axis=1)

    for ii, profile in list(enumerate(profiles))[1:2]:
        print "Profile: ", ii
        profile_numbers.append(ii)
        profile_L = profile[0]
        profile_R = profile[1]

        profile_fft = fftpack.fft(profile)
        profile_fft_L = fftpack.fft(profile_L)
        profile_fft_R = fftpack.fft(profile_R)

        phase_init = 0.
        phase_model.append(phase_init)
        amp_init = np.sqrt(np.sum(profile**2))
#        pars_init = [phase_init, amp_init] + [0.] * (NMODES - 1)
        pars_init = [phase_init, amp_init, amp_init/50, amp_init/1000, amp_init/100000]
        pars_fit, cov, infodict, mesg, ier = optimize.leastsq(
                residuals,
                pars_init,
                (profile_fft, V_fft),
                full_output=1,
                )
        fit_res = residuals(pars_fit, profile_fft, V_fft)

        chi2_fit = chi2(pars_fit, profile_fft, V_fft)
        dof = len(fit_res) - len(pars_init)
        red_chi2 = chi2_fit / dof
        print "chi1, dof, chi2/dof:", chi2_fit, dof, red_chi2

        cov_norm = cov * red_chi2

        errs = np.sqrt(cov_norm.flat[::len(pars_init) + 1])
        corr = cov_norm / errs[None,:] / errs[:,None]

        print "phase, amplidudes (errors):"
        print pars_fit
        print errs
#        print "correlations:"
#        print corr

        phases.append(pars_fit[0])
        phase_errors.append(errs[0])

        model_fft_L = model(pars_fit, V_fft_L) 
        model_fft_R = model(pars_fit, V_fft_R)
        print 'model_fft_L.shape', model_fft_L.shape
        print 'profile_fft_L.shape', profile_fft_L.shape
        plot_phase_fft(pick_harmonics(profile_fft_L), model_fft_L, ii, 'phase_fit_L_')
        plot_phase_ifft(pars_fit, profile_L, V_fft_L, ii, 'phase_fit_L_')
        plot_phase_fft(pick_harmonics(profile_fft_R), model_fft_R, ii, 'phase_fit_R_')
        plot_phase_ifft(pars_fit, profile_R, V_fft_R, ii, 'phase_fit_R_')

        if True:
            # Fix phase at set values, then fit for amplitudes. Then integrate
            # the likelihood over the parameters space to get mean and std of
            # phase.
            phase_diff_samples = np.arange(-50, 50, 0.3) * errs[0]
            chi2_samples = []
            for p in phase_diff_samples:
                this_phase = p + pars_init[0]
                if True:
                    # Linear fit.
#                    P = shift_trunc_modes(this_phase, V_fft)
#                    d = pick_harmonics(profile_fft)
                    P_L = shift_trunc_modes(this_phase, V_fft_L)
                    P_R = shift_trunc_modes(this_phase, V_fft_R)
                    P = np.concatenate((P_L, P_R), axis=1)

                    d_L = pick_harmonics(profile_fft_L)
                    d_R = pick_harmonics(profile_fft_R)
                    d = np.concatenate((d_L, d_R))

                    N = linalg.inv(np.dot(P, P.T))
                    this_pars_fit = np.dot(N, np.sum(P * d, 1))
                else:
                    # Nonlinear fit.
                    residuals_fix_phase = lambda pars: residuals(
                            [this_phase] + list(pars),
                            profile_fft,
                            V_fft,
                            ) / red_chi2
                    #pars_sample = [p + pars_fit[0]] + list(pars_fit[1:])
                    this_pars_fit, cov, infodict, mesg, ier = optimize.leastsq(
                        residuals_fix_phase,
                        list(pars_init[1:]),
                        #(profile_fft, V_fft),
                        full_output=1,
                        )
                chi2_sample = chi2(
                        [this_phase] + list(this_pars_fit),
                        profile_fft,
                        V_fft,
                        1. / red_chi2,
                        )
                chi2_samples.append(chi2_sample)
            phase_diff_samples = np.array(phase_diff_samples)
            chi2_samples = np.array(chi2_samples, dtype=np.float64)
#            print 'chi2_samples', chi2_samples

            if False:
                #Plot of chi-squared (ln likelihood) function.
                plt.figure()
                plt.plot(phase_diff_samples, chi2_samples - chi2_fit / red_chi2)
                plt.ylabel(r"$\Delta\chi^2$")
                plt.xlabel(r"$\Delta_{\rm phase}$")

            # Integrate the full liklihood, taking first and second moments to
            # get mean phase and variance.
            likelihood = np.exp(-chi2_samples / 2)
            norm = integrate.simps(likelihood)
            print 'norm', norm
            mean = integrate.simps(phase_diff_samples * likelihood) / norm
            print 'mean', mean
            var = integrate.simps(phase_diff_samples**2 * likelihood) / norm - mean**2
            std = np.sqrt(var)
            print 'std', std
            print "Integrated Liklihood:", pars_init[0] + mean, std
            phases_lik.append(pars_init[0] + mean)
            phase_errors_lik.append(std)
            profile_numbers_lik.append(ii)
            plot_phase_diff_chi2(phase_diff_samples, likelihood, norm, ii)

def stack(profile, profile_stack):
    nprof = len(profile)
    nprof -= nprof % profile_stack
    profile = profile[:nprof].reshape(nprof // profile_stack, profile_stack, profile.shape[-1])
    profile = np.mean(profile, 1)
    return profile


def residuals(parameters, profile_fft, V_fft):
    res_L = pick_harmonics(profile_fft[0]) - model(parameters, V_fft[:,0,:]) 
    res_R = pick_harmonics(profile_fft[1]) - model(parameters, V_fft[:,1,:])
    res_all = np.concatenate((res_L, res_R))
    return res_all


def model(parameters, V_fft):
    phase_bins = parameters[0]
    amplitudes = np.array(parameters[1:])
    shifted_modes = shift_trunc_modes(phase_bins, V_fft)
    return np.sum(amplitudes[:,None] * shifted_modes, 0)


def shift_trunc_modes(phase_bins, V_fft):
    V_fft_shift = apply_phase_shift(V_fft, phase_bins)
    V_harmonics = pick_harmonics(V_fft_shift)
    return V_harmonics[:NMODES]

def chi2(parameters, profile_fft, V_fft, norm=1):
    return np.sum(residuals(parameters, profile_fft, V_fft)**2) * norm


def pick_harmonics(profile_fft):
    harmonics = profile_fft[..., 1:NHARMONIC]
    harmonics = np.concatenate((harmonics.real, harmonics.imag), -1)
    return harmonics


def apply_phase_shift(profile_fft, phase_bins_shift):
    "Parameter *phase_shift* takes values [0 to 1)."

    phase_shift = phase_bins_shift / NPHASEBIN
    n = profile_fft.shape[-1]
    freq = fftpack.fftfreq(n, 1./n)
    phase = np.exp(-2j * np.pi * phase_shift * freq)
    return profile_fft * phase

def rebin_array(input_data, rebin_factor):
    xlen = input_data.shape[0] / rebin_factor
    output_data = np.zeros((xlen,))
    for ii in range(xlen):
        output_data[ii]=input_data[ii*rebin_factor:(ii+1)*rebin_factor].mean()
    return output_data
    
def rebin_spec(input_data, rebin_factor_0, rebin_factor_1):
    xlen = input_data.shape[0] / rebin_factor_0
    ylen = input_data.shape[1] / rebin_factor_1
    output_data = np.zeros((xlen, ylen))
    for ii in range(xlen):
        for jj in range(ylen):
            output_data[ii,jj]=input_data[ii*rebin_factor_0:(ii+1)*rebin_factor_0,jj*rebin_factor_1:(jj+1)*rebin_factor_1].mean()
    return output_data

def fft(file):
    profile_fft = np.fft.fft(file)
    profile_fft[0] = 0
    return profile_fft

def ifft(file):
    profile_ifft = np.fft.ifft(file)
    return profile_ifft

def fft_phase_curve_inverse(parameters, profile_fft):
    '''inverse phase for chaning 1.0j to -1 1.0j'''
    freq = np.fft.fftfreq(len(profile_fft))
    n= len(profile_fft)
    fft_model = parameters[1] * np.exp(-1.0j * 2 * np.pi * freq * ( n - parameters[0])) * profile_fft
    return fft_model

def svd(file):

    time_matrix = np.zeros(file.shape)
    for ii in xrange(len(time_matrix)):
        time_matrix[ii] = ifft(fft_phase_curve_inverse([0, 1], fft(file[ii]))).real

    U, s, V = np.linalg.svd(time_matrix, full_matrices=False)

    if np.abs(np.amax(V[0])) < np.abs(np.amin(V[0])):
        V[0] = -V[0]

    if True:
        np.save('B_U_.npy', U)
        np.save('B_s_.npy', s)
        np.save('B_V_.npy', V)

    return U, s, V

def plot_spec():

    time_length = B_data.shape[1]
    fontsize = 16

    plt.close('all')
    plt.figure()
    n_step = -5
    x_range = np.arange(0 , len(B_data[35197]))
    plt.plot(x_range, B_data[31597])
    plt.xlim((0, len(B_data[0])))
    plt.xlabel('Phase', fontsize=fontsize)
    plt.ylabel('B data values', fontsize=fontsize)
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
#    plot_name_B_data = plot_name + '_V.png'
    plt.savefig('B_data_31597.png', bbox_inches='tight')


def plot_svd(file, plot_name):

    U, s, V= svd(file)

    V_name = plot_name + '_V.npy'
    np.save(V_name, V)
 
    print 'len(V[0])', len(V[0])
    print 's.shape', s.shape

    fontsize = 16

    plt.close('all')
    plt.figure()
    x_range = np.arange(0, len(s))
    plt.semilogy(x_range, s, 'ro-')
#    plt.plot(x_range, s, 'ro-')
    plt.xlabel('Time')
    plt.ylabel('Log(s)')
    plt.ylim((700, np.max(s)))
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    plot_name_s = plot_name + '_s.png'
    plt.savefig(plot_name_s, bbox_inches=None)
#    plt.show()

#    print 'np.max(V[0])',np.max(V[0])
#    print 'np.max(V[1])',np.max(V[1])

    plt.close('all')
    plt.figure()
    n_step = -0.3
    x_range = np.arange(0 , len(V[0]))
    color = ['r', 'g', 'b', 'y', 'c', '0.0', '0.2', '0.4', '0.6', '0.8']
#    color = ['r', 'g', 'b']
    for ii in xrange(len(color)):
        plt.plot(x_range, np.roll(V[ii] + ii *n_step, 0), color[ii], linewidth=1.0)
    plt.xlim((0, len(V[0])))
#    plt.xlabel('Phase', fontsize=fontsize)
    plt.ylabel('V values', fontsize=fontsize)
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    plot_name_V = plot_name + '_V.png'
    plt.savefig(plot_name_V, bbox_inches='tight')

def plot_rebin_ut(xmin, xmax):
    plot_name = 'B_rebin_ut'
    fontsize = 16

    rebin_u = np.load('/scratch2/p/pen/hsiuhsil/psr_B1957+20/data_file/B_rebin_U_t3334.npy')
    rebin_ut = rebin_u.T
    print 'rebin_ut.shape', rebin_ut.shape
#    np.save('rebin_ut.npy', rebin_ut)
    
    plt.close('all')
    plt.figure()
    n_step = -0.03
    x_range = np.arange(xmin, xmax)
    color = ['b']
#    color = ['r', 'g', 'b', 'y', 'c', '0.0', '0.2', '0.4', '0.6', '0.8']
#    color = ['r', 'g', 'b']
    for ii in xrange(len(color)):
        plt.plot(x_range, np.roll(rebin_ut[2,xmin:xmax] + ii *n_step, 0), color[ii], linewidth=1.0)
#    plt.xlim((0, len(rebin_ut[0])))
    plt.xlabel('Pulse numbers', fontsize=fontsize)
    plt.ylabel('U values', fontsize=fontsize)
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    plot_name_U = plot_name + '_' + str(xmin) + '_' + str(xmax) + '_rebin_ut.png'
    plt.savefig(plot_name_U, bbox_inches='tight')
    
def plot_phase_fft(data_fft, model_fft, ii, plot_name):

    freq = np.fft.fftfreq(len(data_fft))   
    '''Real part'''
    model_fft_real = np.concatenate((model_fft[:(len(model_fft)/2)], model_fft[:(len(model_fft)/2)][::-1]))
    data_fft_real = np.concatenate((data_fft[:(len(data_fft)/2)], data_fft[:(len(data_fft)/2)][::-1]))
    res_fft_real = data_fft_real - model_fft_real

    '''Imag part'''
    model_fft_imag = np.concatenate((model_fft[(len(model_fft)/2):], -model_fft[(len(model_fft)/2):][::-1]))
    data_fft_imag = np.concatenate((data_fft[(len(data_fft)/2):], -data_fft[(len(data_fft)/2):][::-1]))
    res_fft_imag = data_fft_imag - model_fft_imag

    freq_range = np.linspace(np.amin(np.fft.fftfreq(len(data_fft))), np.amax(np.fft.fftfreq(len(data_fft))), num=len(data_fft), endpoint=True)
    freq_min = np.amin(freq_range)
    freq_max = np.amax(freq_range)

#    plot_name = 'phase_fit_'
    plot_name += str(ii) + '_'
    fontsize = 16
    markersize = 4

    '''Plot for real and imag parts in the Fourier space.'''
    plt.close('all')
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(16,9))
    f.subplots_adjust(wspace=0.09, hspace=0.07)
    mode_range = np.linspace(-len(freq)/2, len(freq)/2, num=len(freq), endpoint=True)
    print 'len(freq)', len(freq)
    xmax = np.amax(mode_range)
    xmin = np.amin(mode_range)   
    ax1.plot(mode_range, np.roll(data_fft_real, -int(len(freq)/2)),'bo', markersize=markersize)
    ax1.plot(mode_range, np.roll(model_fft_real, -int(len(freq)/2)),'r-', markersize=markersize)
    ax1.set_title('Real', size=fontsize)
    ax1.set_xlim([xmin,xmax])
    ax1.tick_params(axis='both', which='major', labelsize=fontsize)

    ax2.plot(mode_range, np.roll(data_fft_imag, -int(len(freq)/2)),'bo', markersize=markersize)
    ax2.plot(mode_range, np.roll(model_fft_imag, -int(len(freq)/2)),'r-', markersize=markersize)
    ax2.set_title('Imag', size=fontsize)
    ax2.set_xlim([xmin,xmax])
    ax2.tick_params(axis='both', which='major', labelsize=fontsize)

    ax3.plot(mode_range, np.roll(res_fft_real, -int(len(freq)/2)),'gs', markersize=markersize)
    ax3.set_xlabel('Harmonic modes', fontsize=fontsize)
    ax3.set_ylabel('Residuals (T/Tsys)', fontsize=fontsize)
    ax3.set_xlim([xmin,xmax])
    ax3.tick_params(axis='both', which='major', labelsize=fontsize)

    ax4.plot(mode_range, np.roll(res_fft_imag, -int(len(freq)/2)),'gs', markersize=markersize)
    ax4.set_xlabel('Harmonic modes', fontsize=fontsize)
    ax4.set_xlim([xmin,xmax])
    ax4.tick_params(axis='both', which='major', labelsize=fontsize)
      
    plt.savefig(plot_name + 'fft.png', bbox_inches='tight')

def plot_phase_ifft(pars_fit, data, V_fft, ii, plot_name):

    '''Plot for real part in real space'''
    fit_model = 0
    V_fft_shift = apply_phase_shift(V_fft, pars_fit[0])
    for jj in range(NMODES):
        fit_model += pars_fit[jj+1] * fftpack.ifft(V_fft_shift[jj]).real
 
    model_ifft = fit_model
    data_ifft = data
    res_ifft = data_ifft - model_ifft

#    plot_name = 'phase_fit_'
    plot_name += str(ii) + '_'
    fontsize = 16
    markersize = 4

    plt.close('all')
    f, ((ax1, ax2)) = plt.subplots(2, 1, sharex='col', figsize=(8,9))
    f.subplots_adjust(hspace=0.07)
    phase_bins_range = np.linspace(0, len(data), num=len(data), endpoint=True)
    xmax = np.amax(phase_bins_range)
    xmin = np.amin(phase_bins_range)
    ax1.plot(phase_bins_range, data_ifft,'bo', markersize=markersize)
    ax1.plot(phase_bins_range, model_ifft,'r-', markersize=markersize)
    ax1.set_xlim([xmin,xmax])
    ax1.set_ylabel('T/Tsys', fontsize=fontsize)
    ax1.tick_params(axis='both', which='major', labelsize=fontsize)
    ax2.plot(phase_bins_range, res_ifft,'gs', markersize=markersize)
    ax2.set_xlim([xmin,xmax])
    ax2.set_xlabel('Phase Bins', fontsize=fontsize)
    ax2.set_ylabel('Residuals (T/Tsys)', fontsize=fontsize)
    ax2.tick_params(axis='both', which='major', labelsize=fontsize)

    plt.savefig(plot_name + 'ifft.png', bbox_inches='tight')    

def plot_phase_diff_chi2(phase_diff_samples, likelihood, norm, ii):
    plot_name = 'phase_fit_'
    fontsize = 16
    plot_name += str(ii) + '_'
    plt.close('all')
    phase_diff_range = np.linspace(np.amin(phase_diff_samples), np.amax(phase_diff_samples), num=len(phase_diff_samples), endpoint=True)
    plt.semilogy(phase_diff_range, likelihood / norm / (0.02/NPHASEBIN))
    plt.xlabel('Phase Bins', fontsize=fontsize)
    plt.ylabel('log(Likelihood)', fontsize=fontsize)
    plt.xlim((phase_diff_range[np.where((likelihood / norm / (0.02/NPHASEBIN))>np.amax(likelihood / norm / (0.02/NPHASEBIN)) * 10**-4)[0][0]],phase_diff_range[np.where((likelihood / norm / (0.02/NPHASEBIN) )>np.amax(likelihood / norm / (0.02/NPHASEBIN)) * 10**-4)[0][-1]]))
    plt.ylim((np.amax(likelihood / norm / (0.02/NPHASEBIN)) * 10**-4, np.amax(likelihood / norm / (0.02/NPHASEBIN)) * 4.5))
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    plt.savefig(plot_name+'phase_chi2.png', bbox_inches='tight')


if __name__ == '__main__':
    main()


