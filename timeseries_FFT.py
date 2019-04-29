#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 5 09:00:12 2018

Python 3!!

To perform FFT on any time series data and show the response in the frequency domain

time_Series: must be a pandas series object
error: must be a pandas series object of same length as above
start: should be datetime.datetime object
end: should be datetime.datetime object
input_time_format: if the time_series index is timestamp rather than datetime then specify here


@author: exr007
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from matplotlib import gridspec
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


class timeseries_FFT_plot():

    def __init__(self, time_series, error, start, end, data_label, power_label, FFT_option = 'astropy', maj_tick = 10, min_tick=1, input_time_format = 'datetime'):
        """-----------------------------------------------------------------------"""
        """ time_series, error MUST be pd.DataFrame format with time index column """
        """-----------------------------------------------------------------------"""
        self.df = time_series
        self.error = error

        self.start_date = start
        self.end_date = end

        self.FFT_option = FFT_option #FFT options: 'gatspy' or 'astropy'
        self.input_time_format = input_time_format #options: 'datetime' / 'timestamp'

        # sets FFT plot ticks and timeseries plot y-axis label
        self.ts_y_axis_label = data_label
        self.ps_y_axis_label = power_label
        self.FFT_min_tick = min_tick
        self.FFT_maj_tick = maj_tick


    def setDate(self, start, end):
        """----------------------------------------"""
        """sets the start and end dates on interest"""
        """----------------------------------------"""
        self.start_date = start
        self.end_date = end

    def setTime(self):
        """---------------------------------------------------------------"""
        """converts from datetime.datetime object to a timstamp in seconds"""
        """---------------------------------------------------------------"""
        # self.times = pd.Series([pd.to_datetime(t).to_pydatetime().timestamp() - pd.to_datetime(self.df.index.values[0]).to_pydatetime().timestamp() for t in self.df.index.values],
                                # index=self.df.index)# set time axis in seconds (starting at 0s)
        self.times = pd.Series([pd.to_datetime(t).to_pydatetime().timestamp() - pd.to_datetime(self.start_date).to_pydatetime().timestamp() for t in self.df.index.values],
                                index=self.df.index)# set time axis in seconds (starting at 0s)
        #self.times = self.df['times']

    def mask_data(self):
        """---------------------------------------------------"""
        """mask out any data that is outside of the date range"""
        """---------------------------------------------------"""

        # print('Times are: ')
        # print(self.times)
        # print('Data: ')
        # print(self.df)
        # print('Time: ')
        # print(self.start_date)
        # print(self.end_date)

        #date mask
        mask_date = [(pd.to_datetime(t).to_pydatetime() >= self.start_date) & (pd.to_datetime(t).to_pydatetime() < self.end_date) for t in self.df.index.values]
        self.times = self.times[mask_date]
        self.df = self.df[mask_date]
        self.error = self.error[mask_date]

        # print('mask: ')
        # print(mask_date)

        #convert from pandas DF to np array
        l = len(self.df)
        self.times = np.reshape(self.times.values,l) #turn into np array
        self.df = np.reshape(self.df.values,l) #turn into np array
        self.error = np.reshape(self.error.values,l) #turn into np array

        """
        #remove data gaps
        self.times = self.times[(self.df != 0) & (np.isfinite(self.df))]
        self.error = self.error[(self.df != 0) & (np.isfinite(self.df))]
        self.df = self.df[(self.df != 0) & (np.isfinite(self.df))]
        """
        #remove nans
        self.times = self.times[np.isfinite(self.df)]
        self.error = self.error[np.isfinite(self.df)]
        self.df = self.df[np.isfinite(self.df)]

    def FFT_data(self, oversample = False):
        """------------------------------------------------------------------"""
        """perform FFT on timeseries, two options available: gatspy / astropy"""
        """------------------------------------------------------------------"""
        self.setTime()
        print("Pre-mask times length: %s"%len(self.times))
        self.mask_data() #mask out useless data
        print("Post-mask times length: %s"%len(self.times))
        N = len(self.times)

        if self.FFT_option == 'gatspy':
            print('... using gatspy')
            #gatspy LS
            from gatspy.periodic import LombScargleFast
            #ls = LombScargleFast().fit(self.times, self.df)#, self.error)
            #periods, self.power = ls.periodogram_auto(nyquist_factor=2)
            #self.frequencies = (1/periods)*(1e6) #frequencies in microHz
            
            fmin = 0
            dtmed = np.median(np.diff(self.times))
            df = 1./(dtmed*N)
            ls = LombScargleFast().fit(self.times, self.df, np.ones(N))
            power = ls.score_frequency_grid(fmin, df, N/2)
            freqs = fmin + df * np.arange(N/2)
            var = np.std(self.df)**2
            power /= np.sum(power) # convert so sums to unity
            power *= var # Parseval's: time-series units [G], variance units [G^2]
            power /= df * 1e6 # convert to G^2/muHz
            self.power = power
            self.frequencies = freqs*1e6
            
            ts_energy = np.sum(np.abs(np.power(self.df,2)))/N
            fd_energy = np.sum(np.abs(np.power(self.power,2)))
            print('TS: %s'%ts_energy)
            print('FD: %s'%fd_energy)
            
            #df = pd.DataFrame({'periods':periods, 'power':power})
            #df.to_csv('/mnt/storage/003. Data/004. HiSPARC Data/FFT.csv')

        elif self.FFT_option == 'astropy':
            print('... using astropy')
            #astropy LS
            from astropy.stats import LombScargle
            from scipy.integrate import simps
            """ts_energy = simps(np.power(self.df,2), self.times)
            print(ts_energy)"""
            dtmed = np.median(np.diff(self.times))
            df = 1./(dtmed*N)
            fmin = 0
            if oversample != False:
                frequencies = fmin + (df/oversample) * np.arange(oversample*N/2 + 1)
            else:
                frequencies = fmin + df * np.arange(N/2 + 1)
            power = LombScargle(self.times, self.df).power(frequencies)
            power[0] = 0.0
            #frequencies, power = LombScargle(self.times, self.df).autopower(nyquist_factor=1)
            var = np.std(self.df)**2
            
            ts_energy = np.sum(np.abs(np.power(self.df,2)))/N
            
            power /= np.sum(power) # convert so sums to unity
            power *= var # Parseval's: time-series units [G], variance units [G^2]
            
            fd_energy = np.sum(np.abs(np.power(power,1)))
            
            power /= df * 1e6 # convert to G^2/muHz
            self.power = power
            self.frequencies = frequencies*1e6 #frequencies in microHz
            
            #ts_energy = np.sum(np.abs(np.power(self.df,2)))/N
            #fd_energy = np.sum(np.abs(np.power(self.power,2)))
            print('TS: %s'%ts_energy)
            print('TS_var: %s'%var)
            print('FD: %s'%fd_energy)
    
        return self.frequencies, self.power, self.times


    def FT_data(self, type = 'zeros'):
        """ Compute fourier transform on data """
        """ Options to handle data when gaps filled with nans or with zeros """
        from astropy.stats import LombScargle
        from scipy.integrate import simps
            
        # Ensure time array is in seconds
        self.setTime()
        print("Pre-mask times length: %s"%len(self.times))
        N = len(self.times)
            
        if type == 'nans':
            self.mask_data() #mask out NaN data
            print("Post-mask times length: %s"%len(self.times))
            var = np.std(self.df)**2 # compute variance in timeseries            
        elif type == 'zeros':
            #f.times = f.times[f.df != 0]
            #f.error = f.error[f.df != 0]
            #f.df = f.df[f.df != 0]
            var = np.std(self.df)**2 # compute variance in timeseries
            fill = len(self.df[self.df != 0])/N # (No. == 0 / N)
            print('Fill is: %s'%fill)
            var = var / fill
            print("Post-mask times length: %s"%len(self.times))
            
        dtmed = np.median(np.diff(self.times)) # median cadence
        df = 1./(dtmed*N) # freq resolution
        fmin = 0 # zero freq bin
        if N%2 == 0:
            frequencies = fmin + df * np.arange(N/2 + 1) # set freq range
        elif N%2 != 0:
            frequencies = fmin + df * np.arange(N//2 + 1) # set freq range
        power = LombScargle(self.times, self.df).power(frequencies)
        power[0] = 0.0
        power /= np.sum(power) # convert so sums to unity
        power *= var # Parseval's: time-series units [], variance units []^2
        fd_energy = np.sum(np.abs(np.power(power,1)))
        power /= df * 1e6 # convert to G^2/muHz
        self.power = power
        self.frequencies = frequencies*1e6 #frequencies in microHz
        print('TS_var: %s'%var)
        print('FD: %s'%fd_energy)
            
        return self.frequencies, self.power, self.times
        

    def plot_time_series_and_frequency_spectra(self, save=True):
        majorLocator = MultipleLocator(self.FFT_maj_tick)
        majorFormatter = FormatStrFormatter('%.1f')
        minorLocator = MultipleLocator(self.FFT_min_tick)
        minorFormatter = FormatStrFormatter('%.6f')

        fig = plt.figure(figsize = (10,10))
        fig.subplots_adjust(hspace = 0.3, wspace=0.1)
        #fig.suptitle('Muon count and FFT')
        gs = gridspec.GridSpec(2,1)

        ax1 = plt.subplot(gs[0])
        #ax1.set_xlim(plot_start_datetime, plot_end_datetime)
        ax1.errorbar(self.times/3600,
                     self.df,
                     yerr=self.error,
                     c='k',
                     alpha=0.8,
                     markersize = 2.0,
                     linewidth = 0.5)
        ax1.set_xlabel('Time [hours since %s]'%(datetime.datetime.strftime(self.start_date, "%d/%m/%Y")), fontsize=14)
        ax1.set_ylabel(self.ts_y_axis_label, fontsize=14)
        ax1.tick_params('x', rotation=20, labelsize=12)

        ax2 = plt.subplot(gs[1])
        #ax2.set_xlim(plot_start_datetime, plot_end_datetime)
        ax2.plot(self.frequencies,
                 self.power,
                 c='k',
                 alpha=0.8,
                 lw=0.5)
        ticks = np.arange(0, self.frequencies.max(), 0.5)
        ax2.set_xticks(ticks, minor=True)
        # Make the y-axis label, ticks and tick labels match the line color.
        ax2.set_ylabel(self.ps_y_axis_label, fontsize=14)
        #ax2.set_xlim([0, 5000])
        ax2.set_xlabel('Frequency [$\mu$Hz]', fontsize=14)
        ax2.tick_params('x', rotation=20, labelsize=12)
        #        ax2.xaxis.set_major_locator(majorLocator)
        #        ax2.xaxis.set_major_formatter(majorFormatter)
        #        ax2.xaxis.set_minor_locator(minorLocator)
        #        ax2.xaxis.set_minor_formatter(minorFormatter)

        if save == True:
            plt.savefig('FFT.eps', bbox_inches='tight')
        plt.show()

    def make_plot(self):
        if self.FFT_option == 'gatspy':
            self.plot_time_series_and_frequency_spectra()

        elif self.FFT_option == 'astropy':
            self.plot_time_series_and_frequency_spectra()
