"""
===============
Signal Analyser
===============

Calculates, plots and exports a single-sided FFT spectrum.

"""

import os
import sys
if not (os.path.abspath('../../thesdk') in sys.path):
    sys.path.append(os.path.abspath('../../thesdk'))

import numpy as np
import tempfile
import scipy.fftpack as ffp
import scipy.signal as ss
import matplotlib.pyplot as plt

import copy

import pdb
import traceback

from thesdk import *
from vhdl import *

class signal_analyser(thesdk):
    """

    Attributes
    ----------

    IOS.Members['in'].Data: ndarray (or list of ndarrays)
        Time-domain input signal to use for FFT calculation. If the number of
        columns is 1, the data is assumed to be sampled. If the number of
        columns is 2, the first column is assumed to be the time-vector, and
        the second column is assumed to be the signal. If input data is a list
        of signals arrays, signal shall be extracted from each element and RMS
        average of their spectra will be analysed instead.
    IOS.Members['out'].Data: ndarray
        Calculated FFT of the input signal. Shape is (nsamp,2), where the first
        column is the frequency vector, and second column is the power vector.
    fs: float, default 2e9
        Sampling frequency used for FFT calculation.
    nsamp: int, default 1024
        Number of samples used for the FFT calculation.
    nadc: int, default 1
        Number of time-interleaved ADC channels (used for mismatch annotation
        only).
    sig_osr: int, default 1
        Signal oversampling ratio (related to signal_generator).
    window: bool, default False
        Should windowing be applied or not? True -> Hanning window, False ->
        rectangular.
    plot: bool, default True
        Should the figure be drawn or not? True -> figure is drawn, False ->
        figure not drawn.
    figformat: str, default 'pdf'
        Format of the file in which the figure is saved. Possible options (check
        from Matplotlib documentation): 'pdf', 'eps', 'png', 'svg'
    export: (bool,str), default (False,'')
        Should the figure(s), as well as the DNL/INL datapoints, be exported to
        image and csv files or not? The filetypes .csv and .pdf are automatically
        appended to the 'filepath/filename' -string given.

        For example::

            export = (True,'./figures/result')

        would create 'result.pdf' and 'result.csv' in the directory called
        'figures', if self.figformat is 'pdf'.
    export_csv: bool, default True
        Flag to disable exporting datapoints as .csv files. Exports datapoints to csv by default.
    gridaxis: str, {'both' (default), 'x', 'y', or ''}
        Which axis should the grid be shown for.
    freqscale: str, {'Hz', 'kHz', 'MHz' (default), 'GHz', 'THz'}
        The scale of the frequency axis. The input strings are case in-sensitive.
        Note: The sampling rate should always be given in Herz. 
    linlog: str, {'lin' (default), 'log'}
        X-axis type, linear or logarithmic.
    ylim: str, {'noisecross' (default), 'noisemean', 'auto', 'noisemin'}
        Location of y-axis limit. 'noisecross' sets the limit to the noise
        floor detected based on zero crossings, 'noisemean' tries to place the
        limit at the mean value of the visual noise floor, 'auto' fits the full
        data to view and 'noisemin' sets the limit to minimum component of the
        noise floor.
    xlim: int, default automatically determined
        Location of x-axis maximum limit in the freqscale units.
    title: str, default ''
        Title for the produced figure.
    annotations: list(str), {'SNDR','SNR','ENOB','SFDR','THD','Range'}, default ['']
        Configures the metrics to annotate to the figure as a list of strings.
        The order of the metrics defines the order in the figure, and the words
        are case in-sensitive.
    textbgcolor: str, default '#ffffffa0'
        Background color for the annotation text. Can be any valid Python color
        string. The default is transparent white. Purpose is to make text more
        readable over a grid.
    rangeunit: str, default 'bits'
        Unit to be used for range display. If not defined, the input data is
        assumed to be quantized (integer values), and the range is expressed as
        bits according to log2(maxcode-mincode). The result would look like
        'Range = 8.21 bits'. If, on the other hand, the rangeunit is defined as
        'V' for voltage, the range is printed as 'Range = 1 V', and it's
        calculated as V_max-V_min.
    sfdr_pointer: bool, default False
        Should the limiting spurious tone be annotated with an arrow in the
        figure? This does not work currently.
    annotate_mismatch: list(str), {'skew','gain','offset'}, default None
        Configures the time-interleaving mismatch effects to be annotated in
        the spectrum. Set to None (default) for no annotations. Only applicable
        to time-interleaved ADC outputs.
    annotate_harmonics: bool, default False
        Annotates the most significant harmonic components on the spectrum.
    snr_order: int, default 8 (up to 10th harmonic)
        The order of harmonics to be considered in SNR and THD calculations.
    adj_bins: int, default 1
        The number of adjacent bins to include in a signal/harmonic power
        calculation.
    snr: float
        Read the calculated SNR from this attribute, if needed. Accessible
        after calling run().
    sndr: float
        Read the calculated SNDR from this attribute, if needed. Accessible
        after calling run().
    sfdr: float
        Read the calculated SFDR from this attribute, if needed. Accessible
        after calling run().
    enob: float
        Read the calculated ENOB from this attribute, if needed. Accessible
        after calling run().
    thd: float
        Read the calculated THD from this attribute, if needed. Accessible
        after calling run().
    fullscale: float
        Read the calculated fullscale range from this attribute, if needed.
        Accessible after calling run().
    spur_hd: float
        Read the calculated highest spur level of harmonic distortion. Only
        applicable to time-interleaved ADC outputs. Accessible after calling
        run().
    spur_skew: float
        Read the calculated highest spur level of timing skew mismatch / gain
        mismatch. Only calculated when any of the mismatch annotations is
        enabled, else None. Only applicable to time-interleaved ADC outputs.
        Accessible after calling run().
    spur_offs: float
        Read the calculated highest spur level of offset mismatch. Only
        calculated when any of the mismatch annotations is enabled, else None.
        Only applicable to time-interleaved ADC outputs. Accessible after
        calling run().
    skew_mismatch_data: ndarray of shape (2,Ng). Ng = number of gain spurs
        Calculated spur levels (1st row) and corresponding frequency bin values
        (0th row) of timing skew mismatch / gain mismatch. Only calculated when
        any of the mismatch annotations is enabled, else None. Only applicable
        to time-interleaved ADC outputs. Accessible after calling run().
    offs_mismatch_data: ndarray of shape (2,No). No = number of offset spurs
        Calculated spur levels (1st row) and corresponding frequency bin values
        (0th row) of offset mismatch. Only calculated when any of the mismatch
        annotations is enabled, else None. Only applicable to time-interleaved
        ADC outputs. Accessible after calling run().
    harmpowers: list
        List containing the power levels of the harmonics. Index 0 corresponds
        to 2nd harmonic, index 1 to 3rd, and so on. Accessible after calling
        run().
    """
    @property
    def _classfile(self):
        return os.path.dirname(os.path.realpath(__file__)) + "/"+__name__

    def __init__(self,*arg): 
        self.print_log(type='I', msg='Initializing %s' %(__name__)) 
        self.proplist = [ 'sig_osr', 'fs','nsamp', 'window', 'nadc', 'plot'];

        self.fs = 2e9

        # Options for bandwidth limited noise calculation
        self.fstart = 0 #Start of the bandwidth
        self.fstop = 0 #End of the bandwidth. Keep fstop at 0 to calculate for whole nyquist band

        # Save powers at these frequencies
        self.retfreqs = []

        self.nsamp = 2**10
        self.sig_osr = 1
        self.nadc = 1

        self.window = False

        # Plotting options
        self.plot = True
        self.export = (False,'')
        self.export_csv = True
        self.figformat='pdf'
        self.gridaxis = 'both'
        self.freqscale = 'MHz'
        self.linlog = 'lin'
        self.ylim = 'noisecross'
        self.xlim = None
        self.title = ''

        # Things to annotate in this order (case insesitive)
        self.annotations = ['']
        self.textbgcolor = '#ffffffa0'
        self.annotate_mismatch = None
        self.rangeunit = 'bits'
        self.sfdr_pointer = False
        self.annotate_harmonics = False
        self.snr_order = 8
        self.adj_bins = 1

        # Readout attributes
        self.sfdr = 0
        self.sndr = 0
        self.enob = 0
        self.snr = 0
        self.thd = 0
        self.sigfreq = -1
        self.spur_hd = None
        self.spur_skew = None
        self.spur_offs = None
        self.skew_mismatch_data = None
        self.offs_mismatch_data = None
        self.maxampdBV = 0
        self.retampsdBV = [0]
        self.harmpowers = []

        self.IOS=Bundle()
        self.IOS.Members['in']=IO()
        self.IOS.Members['out']=IO()
        self.model='py'

        if len(arg)>=1:
            parent=arg[0]
            self.copy_propval(parent,self.proplist)
            self.parent =parent

        self.init()

    def init(self):
        """This function has no purpose currently.
        """
        ### Lets fix this later on
        if self.model=='vhdl':
            self.print_log(type='F', msg='VHDL simulation is not supported with v1.2\n Use v1.1')

    def calculate_sndr(self,psd,freq_axis):
        """Internally called function to calculate the SNDR and ENOB from the spectrum.
        """
        try:
            if self.fstop == 0:
                fstart_ind = 0
                fstop_ind = psd.size
            else:
                fstart_ind = np.where(freq_axis >= self.xscale*self.fstart)[0][0]
                fstop_ind = np.where(freq_axis >= self.xscale*self.fstop)[0][0]
            # These should be refactored as attributes or something
            tmppsd = np.copy(psd)
            tmppsd = tmppsd[fstart_ind:fstop_ind]
            sigbin = np.where(tmppsd == np.max(tmppsd))[0][0]
            distpow = 0
            sigpow = 0
            for i in range(tmppsd.size):
                binpow = (10**(tmppsd[i]/20))**2
                if i >= sigbin-self.adj_bins and i <= sigbin+self.adj_bins:
                    sigpow += binpow
                else:
                    distpow += binpow
            self.sndr = 10*np.log10(sigpow/distpow)
            self.enob = (self.sndr-1.76)/6.02
        except:
            self.print_log(type='W',msg='Something went wrong while calculating SNDR. Too few samples?')
        finally:
            return

    def calculate_snr(self,psd,freq_axis,fstart,fstop):
        """Internally called function to calculate the SNR and THD from the spectrum.
        """
        try:
            if fstop == 0:
                fstart_ind = 0
                fstop_ind = psd.size
                sigbin = np.where(psd == np.max(psd))[0][0]
            else:
                fstart_ind = np.where(freq_axis >= fstart)[0][0]
                fstop_ind = np.where(freq_axis >= fstop)[0][0]
                sigbin = np.where(psd == np.max(psd[fstart_ind:fstop_ind]))[0][0]

            self.harmidcs = np.zeros(self.snr_order, dtype=int)
            self.max_harm_freq = None
            self.max_harm_pow = None
            for i in range(self.snr_order):
                fharm = (i+2)*self.sigfreq
                rem = fharm % (self.fs_scaled/2)
                alias = np.floor(fharm/(self.fs_scaled/2))
                if alias % 2 == 0:
                    self.harmidcs[i] = np.where(freq_axis >= rem)[0][0]
                else:
                    self.harmidcs[i] = np.where(freq_axis >= self.fs_scaled/2-rem)[0][0]
                hd_pow = psd[self.harmidcs[i]]
                hd_freq = freq_axis[self.harmidcs[i]]
                if self.max_harm_pow is None:
                    self.max_harm_pow = hd_pow
                    self.max_harm_freq = hd_freq
                elif hd_pow > self.max_harm_pow:
                    self.max_harm_pow = hd_pow
                    self.max_harm_freq = hd_freq

            noisepow = 0
            sigpow = 0
            # First calculating SNDR
            for i in range(fstart_ind, fstop_ind):
                binpow = (10**(psd[i]/20))**2
                if i >= sigbin-self.adj_bins and i <= sigbin+self.adj_bins:
                    sigpow += binpow
                else:
                    noisepow += binpow

            # Then subtracting harmonics from SNDR to get SNR
            harmpow = 0
            for i in range(self.harmidcs.size):
                harmidx = self.harmidcs[i]
                if (harmidx >= fstart_ind) and (harmidx <= fstop_ind):
                    harmstart = int(np.max([harmidx-self.adj_bins,0]))
                    harmstop = int(np.min([harmidx+self.adj_bins+1,len(psd)-1]))
                    binpow = np.sum((10**(psd[harmstart:harmstop]/20))**2)
                    harmpow += binpow
                    noisepow -= binpow

            self.snr = 10*np.log10(sigpow/noisepow)
            self.thd = 10*np.log10(harmpow/sigpow)

            # Saving highest harmonic tone
            #self.spur_hd = np.max(psd[self.harmidcs.astype(int)])
            self.spur_hd = self.max_harm_pow
            #self.print_log(msg='Max HD = %.03f dBc (@ %.03f %s).' % (self.spur_hd,self.max_harm_freq,self.freqscale))
        except:
            self.print_log(type='W',msg=traceback.format_exc())
            self.print_log(type='W',msg='Something went wrong while calculating SNR. Too few samples?')
        finally:
            return

    def calculate_sfdr(self,psd,freq_axis):
        """Internally called function to calculate the SFDR from the spectrum.
        """
        try:
            if self.fstop == 0:
                fstart_ind = 0
                fstop_ind = psd.size
            else:
                fstart_ind = np.where(freq_axis >= self.xscale*self.fstart)[0][0]
                fstop_ind = np.where(freq_axis >= self.xscale*self.fstop)[0][0]
            tmppsd = np.copy(psd)
            tmppsd = tmppsd[fstart_ind:fstop_ind]
            tmpfreq = np.copy(freq_axis)
            tmpfreq = tmpfreq[fstart_ind:fstop_ind]
            peak_idcs = ss.find_peaks(tmppsd,distance=2)[0]
            peaks_sorted = np.flipud(np.sort(tmppsd[peak_idcs]))
            self.sigidx = np.where(tmppsd == peaks_sorted[0])[0][0]
            self.sigfreq = tmpfreq[np.where(tmppsd == peaks_sorted[0])[0][0]]
            self.sfdr = peaks_sorted[0]-peaks_sorted[1]
            self.spuridx = np.where(tmppsd == peaks_sorted[1])[0][0]
            self.spurfreq = tmpfreq[self.spuridx]
        except:
            self.sigidx = 0
            self.print_log(type='W',msg='Something went wrong while calculating SFDR. Too few samples?')
        finally:
            return 

    def calculate_maxamp(self,nyq_mag,nyq_index):
        """Internally called function to calculate the amplitude of maximum tone from the spectrum.
        """
        try:
            temp_mag = np.copy(nyq_mag)
            #fidx = np.argmax(temp_mag)
            #idxstart = int(np.max([fidx-self.adj_bins,0]))
            #idxstop = int(np.min([fidx+self.adj_bins+1,len(temp_mag)-1]))
            if self.window:
                #scaling
                temp_mag *= 2*nyq_index/sum(np.hanning(2*nyq_index))
                #self.maxampdBV=20*np.log10(sum(temp_mag[idxstart:idxstop])/nyq_index)
                self.maxampdBV=20*np.log10(max(temp_mag)/nyq_index)
            else:
                self.maxampdBV=20*np.log10(max(temp_mag)/nyq_index)
        except:
            self.print_log(type='W',msg='Something went wrong while calculating maxamplitude.')
        finally:
            return 

    def calculate_retamps(self,nyq_mag,nyq_index,freq_axis):
        """Internally called function to calculate the powers at defined frequencies from the spectrum.
        """
        try:
            temp_mag2 = np.copy(nyq_mag)
            for i in range(len(self.retfreqs)):
                fidx = np.where(freq_axis >= self.retfreqs[i]*self.xscale)[0][0]
                if temp_mag2[fidx-1] > temp_mag2[fidx]:
                    fidx -= 1 # peak was propably at previous bin
                idxstart = int(np.max([fidx-self.adj_bins,0]))
                idxstop = int(np.min([fidx+self.adj_bins+1,len(temp_mag2)-1]))

                if self.window:
                    #scaling
                    temp_mag2 *= 2*nyq_index/sum(np.hanning(2*nyq_index))
                    self.retampsdBV[i]=20*np.log10(sum(temp_mag2[idxstart:idxstop])/nyq_index)
                else:
                    self.retampsdBV[i]=20*np.log10(sum(temp_mag2[idxstart:idxstop])/nyq_index)
        except:
            self.print_log(type='W',msg='Something went wrong while calculating return powers.')
        finally:
            return 

    def annotate_plot(self):
        """Internally called function to add the text annotation to the figure.
        """
        ax = plt.gca()
        if self.linlog == 'lin':
            sigpos = self.sigfreq/(self.fs_scaled/4)
        else:
            sigpos = self.sigfreq/np.sqrt((self.fs_scaled/2)*(self.fs_scaled/self.nsamp))
        if sigpos >= 1:
            halign = 'left'
            xpos = 0.025
        else:
            halign = 'right'
            xpos = 0.975

        textstr = ""
        for a in self.annotations:
            if a.upper() == "SFDR":
                textstr += "SFDR = %.02f dB\n" % self.sfdr
            if a.upper() == "SNDR":
                textstr += "SNDR = %.02f dB\n" % self.sndr
            if a.upper() == "RANGE":
                textstr += "Range = %.02f %s\n" % (self.fullscale,self.rangeunit)
            if a.upper() == "ENOB":
                textstr += "ENOB = %.02f bits\n" % self.enob
            if a.upper() == "SNR":
                textstr += "SNR = %.02f dB\n" % self.snr
            if a.upper() == "THD":
                textstr += "THD = %.02f dBc\n" % self.thd

        if textstr != '':
            textstr = textstr.rstrip()
            self.meastxt = plt.text(xpos,0.95,textstr,usetex=plt.rcParams['text.usetex'], \
                    horizontalalignment=halign,verticalalignment='top', \
                    multialignment='left',fontsize=plt.rcParams['legend.fontsize'], \
                    fontweight='normal',transform=ax.transAxes,\
                    bbox=dict(boxstyle='square,pad=0',fc=self.textbgcolor,ec='none'))
        return

    def add_sfdr_pointer(self):
        """Internally called function to add the limiting spurious tone pointer to the figure.
        """
        #if abs(self.spurfreq-self.sigfreq) > self.fs/2-self.spurfreq:
        #    xoffs = -1
        #    halign = 'right'
        #else:
        #    xoffs = 1
        #    halign = 'left'
        #plt.annotate("-%.02f dBc" % self.sfdr, \
        #        xy=(self.spurfreq,-self.sfdr),xytext=(xoffs*40,30),\
        #        xycoords='data',textcoords='offset points', \
        #        horizontalalignment=halign,verticalalignment='bottom', \
        #        arrowprops=dict(arrowstyle='-|>',connectionstyle='arc3'))
        #return
        pass

    def add_harmonic_annotation(self,psd,freq_axis):
        """Internally called function to annotate strongest harmonics in the figure.
        """
        ax = plt.gca()
        peak_idcs = ss.find_peaks(psd,distance=1)[0]
        peaks_sorted = np.flipud(np.sort(psd[peak_idcs]))
        if len(peaks_sorted) >= len(self.harmidcs):
            #baseline = psd[np.where(psd == peaks_sorted[self.snr_order])[0][0]]
            #baseline = self.maxnoise
            baseline = self.psdmean+10
            #harmonics = psd[self.harmidcs]
            for i in range(len(self.harmidcs)):
                idx = int(self.harmidcs[i])
                self.harmpowers.append(psd[idx])
                if self.annotate_harmonics and psd[idx] > baseline:
                    #harmtxt = "-%.02f dBc" % -psd[harmonics[i]]
                    harmtxt = "H%d" % (i+2)
                    fontsize = plt.rcParams['xtick.labelsize']
                    if type(fontsize) == str:
                        fontsize='x-small'
                    else:
                        fontsize -= 2

                    t=plt.text(freq_axis[idx],psd[idx]+1,harmtxt,\
                            horizontalalignment='center',\
                            fontsize=fontsize,\
                            verticalalignment='bottom')

    def add_mismatch_annotation(self,norm_nyq_mag_db,freq_axis):
        """Internally called function to annotate time-interleaving mismatch
        spurs in the figure.
        """
        self.spur_skew = None
        self.spur_offs = None
        legendentries = []
        # this should be automatic for plots annotated with multiple legends
        if 0 < self.sigfreq <= self.fs_scaled/8:
            legend_x = 0.375
        elif self.fs_scaled/8 < self.sigfreq <= 2*self.fs_scaled/8:
            legend_x = 0.135
        elif 2*self.fs_scaled/8 < self.sigfreq <= 3*self.fs_scaled/8:
            legend_x = 0.865
        else:
            legend_x = 0.625
        for mm in self.annotate_mismatch:
            if mm.lower() == 'skew' or mm.lower() == 'gain':
                legendentries.append(plt.Line2D([0],[0],color='k',marker='o',linestyle='',fillstyle='none',label='Gain/Skew'))
                for k in range(1,self.nadc):
                    spurfreq = (-self.sigfreq+k/self.nadc*self.fs_scaled) % (self.fs_scaled/2)
                    if spurfreq > 0 and spurfreq < self.fs_scaled:
                        spuridx = np.where(np.abs(freq_axis-spurfreq) == np.min(np.abs(freq_axis-spurfreq)))[0]
                        if spuridx[0] != self.sigidx:
                            plt.plot(freq_axis[spuridx[0]],norm_nyq_mag_db[spuridx[0]],'ko',fillstyle='none')
                            if self.spur_skew is None or norm_nyq_mag_db[spuridx[0]] > self.spur_skew:
                                self.spur_skew = norm_nyq_mag_db[spuridx[0]]
                            if self.skew_mismatch_data is None:
                                self.skew_mismatch_data = np.array([freq_axis[spuridx[0]],norm_nyq_mag_db[spuridx[0]]])
                            else:
                                self.skew_mismatch_data = np.column_stack((self.skew_mismatch_data,np.array([freq_axis[spuridx[0]],norm_nyq_mag_db[spuridx[0]]])))
                    spurfreq = (self.sigfreq+k/self.nadc*self.fs_scaled) % (self.fs_scaled/2)
                    if spurfreq > 0 and spurfreq < self.fs_scaled:
                        spuridx = np.where(freq_axis >= spurfreq)[0]
                        if spuridx[0] != self.sigidx:
                            plt.plot(freq_axis[spuridx[0]],norm_nyq_mag_db[spuridx[0]],'ko',fillstyle='none')
                            if self.spur_skew is None or norm_nyq_mag_db[spuridx[0]] > self.spur_skew:
                                self.spur_skew = norm_nyq_mag_db[spuridx[0]]
                            if self.skew_mismatch_data is None:
                                self.skew_mismatch_data = np.array([freq_axis[spuridx[0]],norm_nyq_mag_db[spuridx[0]]])
                            else:
                                self.skew_mismatch_data = np.column_stack((self.skew_mismatch_data,np.array([freq_axis[spuridx[0]],norm_nyq_mag_db[spuridx[0]]])))
            if mm.lower() == 'offset':
                legendentries.append(plt.Line2D([0],[0],color='k',marker='s',linestyle='',fillstyle='none',label='Offset'))
                for k in range(1,self.nadc):
                    spurfreq = (k/self.nadc*self.fs_scaled) % (self.fs_scaled/2)
                    if spurfreq > 0 and spurfreq < self.fs_scaled:
                        spuridx = np.where(np.abs(freq_axis-spurfreq) == np.min(np.abs(freq_axis-spurfreq)))[0]
                        #self.offs_mismatch_data.append((freq_axis[spuridx[0]],norm_nyq_mag_db[spuridx[0]]))
                        if spuridx[0] != self.sigidx:
                            plt.plot(freq_axis[spuridx[0]],norm_nyq_mag_db[spuridx[0]],'ks',fillstyle='none')
                            if self.spur_offs is None or norm_nyq_mag_db[spuridx[0]] > self.spur_offs:
                                self.spur_offs = norm_nyq_mag_db[spuridx[0]]
                            if self.offs_mismatch_data is None:
                                self.offs_mismatch_data = np.array([freq_axis[spuridx[0]],norm_nyq_mag_db[spuridx[0]]])
                            else:
                                self.offs_mismatch_data = np.column_stack((self.offs_mismatch_data,np.array([freq_axis[spuridx[0]],norm_nyq_mag_db[spuridx[0]]])))
            plt.legend(handles=legendentries,frameon=False,loc='upper center',\
                    bbox_to_anchor=(legend_x,1.0),fontsize=plt.rcParams['legend.fontsize'],\
                    markerscale=1,ncol=1,handlelength=0.3,columnspacing=1.)
        self.skew_mismatch_data = np.unique(self.skew_mismatch_data,axis=1)
        self.offs_mismatch_data = np.unique(self.offs_mismatch_data,axis=1)

    def main(self):
        """Main functionality.
        """
        in_data = copy.deepcopy(self.IOS.Members['in'].Data)

        if not isinstance(in_data,list):
            signal_list = [in_data]
            n_batch = 1
        else:
            signal_list = in_data
            n_batch = len(signal_list)

        self.harmpowers = []
        self.retampsdBV = np.zeros(len(self.retfreqs))
        self.sigfreq = -1
        tempfs=self.fs
        temposr=1.0
        fullscale = 0
        magnitude = np.zeros(self.nsamp)
        for signal in signal_list:
            if isinstance(signal.shape,tuple) and len(signal.shape) > 1 and signal.shape[1] == 1:
                signal = signal[:,0]
            if isinstance(signal.shape,tuple) and len(signal.shape) > 1 and signal.shape[1] > 1:
                # When analog input has extra points defined (osr > 1)
                # -> Take every self.sig_osr point and call it good.
                signal = signal[::self.sig_osr,1]
            if len(signal) > self.nsamp:
                signal = signal[-self.nsamp:]
            signal = signal-np.mean(signal)
            if self.window:
                signal *= np.hanning(len(signal))
            # runtime averaging of fullscale and fft magnitude
            fullscale += (max(signal)-min(signal))/n_batch
            magnitude = np.add(magnitude,(np.square(np.absolute(ffp.fft(signal,n=self.nsamp)))/n_batch))

        if self.rangeunit == 'bits':
            self.fullscale = np.log2(fullscale)
        else:
            self.fullscale = fullscale

        magnitude = np.sqrt(magnitude)+np.finfo(float).eps # square root of squared and averaged fft bins
        nyq_index = int(len(magnitude)/2.)
        nyq_mag = magnitude[1:nyq_index+1]
        nyq_mag_db = 20.*np.log10(nyq_mag)
        norm_nyq_mag_db = nyq_mag_db-max(nyq_mag_db)
        #freq_axis = np.linspace(0,tempfs/2.,num=nyq_index,endpoint=False)
        freq_axis = np.linspace(0,tempfs/2.,num=nyq_index+1)
        #norm_nyq_mag_db[0] = np.nan
        freq_axis = np.delete(freq_axis,0)

        if self.export[0] and self.export_csv:
            exportarray = np.stack((freq_axis,norm_nyq_mag_db),axis=-1)
            np.savetxt("%s.csv"%self.export[1],exportarray,delimiter=",")

        # Scaling the frequency axis based on user input
        if self.freqscale.lower() == 'hz':
            self.xscale = 1
            self.freqscale = 'Hz'
        elif self.freqscale.lower() == 'khz':
            self.xscale = 1e-3
            self.freqscale = 'kHz'
        elif self.freqscale.lower() == 'mhz':
            self.xscale = 1e-6
            self.freqscale = 'MHz'
        elif self.freqscale.lower() == 'ghz':
            self.xscale = 1e-9
            self.freqscale = 'GHz'
        elif self.freqscale.lower() == 'thz':
            self.xscale = 1e-12
            self.freqscale = 'THz'

        freq_axis *= self.xscale
        self.fs_scaled = self.fs*self.xscale

        # Calculating stuff
        self.calculate_sfdr(norm_nyq_mag_db,freq_axis)
        self.calculate_sndr(norm_nyq_mag_db,freq_axis)
        self.calculate_snr(norm_nyq_mag_db,freq_axis,self.fstart*self.xscale,self.fstop*self.xscale)
        self.calculate_maxamp(nyq_mag,nyq_index)
        self.calculate_retamps(nyq_mag,nyq_index,freq_axis)

        infostr = 'SNDR = %.03f dB (@ f_in = %.03f %s, f_s = %.03f %s.)' % (self.sndr,\
                self.sigfreq,self.freqscale,self.fs_scaled,self.freqscale)
        self.print_log(type='I',msg=infostr)

        try:
            figure = plt.figure(constrained_layout=True)
            plt.plot(freq_axis[0:int(len(freq_axis)/temposr)],norm_nyq_mag_db[0:int(len(freq_axis)/temposr)])
            if self.title != '':
                plt.title(self.title)
            plt.xlabel('Frequency (%s)' % self.freqscale)
            plt.ylabel('Magnitude (dBFS)')
            if self.linlog.lower() == 'log':
                ax = plt.gca()
                ax.set_xscale('log')
            plt.autoscale(True,'x',tight=True)
            if self.gridaxis == '':
                plt.grid(False)
            else:
                plt.grid(True,axis=self.gridaxis)

            # Perceived noise floor
            harmmask = self.harmidcs
            for i in range(1,self.adj_bins+1):
                harmmask = np.concatenate((harmmask,self.harmidcs-i))
                harmmask = np.concatenate((harmmask,self.harmidcs+i))
            harmmask = np.append(harmmask,range(self.sigidx-self.adj_bins,self.sigidx+self.adj_bins+1))
            tmpsig = np.delete(norm_nyq_mag_db,harmmask,0)
            self.maxnoise = np.max(tmpsig)
            self.minnoise = np.min(tmpsig[np.where(tmpsig>-120)[0]])
            self.psdmean = np.mean(tmpsig[np.where(tmpsig>-120)[0]])

            bottom_lift = np.abs(np.min(norm_nyq_mag_db)).astype(int)
            cross_count = np.empty(bottom_lift)
            for lvl in range(bottom_lift):
                zero_crossings = np.where(np.diff(np.signbit(norm_nyq_mag_db+lvl)))[0]
                cross_count[lvl] = len(zero_crossings)
            self.noisecross = -np.argmax(cross_count)

            if self.fstart != 0:
                plt.axvline(self.fstart*self.xscale,ls=':',c='0.5')
            if self.fstop != 0:
                plt.axvline(self.fstop*self.xscale,ls=':',c='0.5')

            if self.ylim == 'noisecross':
                plt.ylim(self.noisecross, 5)
            elif self.ylim == 'noisemean':
                plt.ylim(self.psdmean-5, 5)
            elif self.ylim == 'noisemin':
                plt.ylim(self.minnoise, 5)
            if self.linlog != 'log':
                plt.xlim(left=0)
            if self.xlim != None:
                plt.xlim(right=self.xlim)
            #plt.ylim((self.maxnoise+self.psdmean)/2, 5)
            if self.sfdr != 0:
                if self.sfdr_pointer:
                    self.add_sfdr_pointer()
                self.add_harmonic_annotation(norm_nyq_mag_db,freq_axis)
                if self.annotate_mismatch is not None:
                    self.add_mismatch_annotation(norm_nyq_mag_db,freq_axis)
                self.annotate_plot()
            if self.export[0]:
                fname = "%s.%s"%(self.export[1], self.figformat)
                self.print_log(type='I',msg='Saving figure to %s.' % fname)
                figure.savefig(fname,format=self.figformat)
            if self.plot:
                plt.show(block=False)
                plt.pause(0.5)
            else:
                plt.close(figure)

            self.IOS.Members['out'].Data = np.hstack((freq_axis.reshape(-1,1),norm_nyq_mag_db.reshape(-1,1)))
        except Exception as e:
            self.print_log(type='I',msg=traceback.format_exc())
            self.print_log(type='E',msg='Failed to plot spectrum.')
            if not self.plot:
                plt.close(figure)

    def run(self,*arg):
        """Called externally to execute the signal analyser.
        """
        if len(arg)>0:
            self.par=True      #flag for parallel processing
            self.queue=arg[0]  #multiprocessing.queue as the first argument
        if self.model=='py':
            self.main()
        else: 
          if self.model=='sv':
              self.vlogparameters=dict([ ('g_Rs',self.fs),]) #Defines the sample rate
              self.run_verilog()
                            
              #This is for parallel processing
              if self.par:
                  self.queue.put(self.IOS.Members[Z].Data)
              del self.iofile_bundle #Large files should be deleted

          elif self.model=='vhdl':
              self.print_log(type='F', msg='VHDL simulation is not supported with v1.2\n Use v1.1')

if __name__=="__main__":
    from  signal_analyser import *
    import pdb

    try:
        import plot_format
        plot_format.set_style('ieeetran')
    except:
        self.print_log(type='W',msg='Module \'plot_format\' not in path. Plot formatting might look incorrect.')

    fs=2e9
    #f = 75.1953125e6
    f = 700.1953125e6
    nsamp = 2**12
    t = np.linspace(0,nsamp/fs,num=nsamp,endpoint=False)
    indata = np.sin(2*np.pi*f*t)
    for i in range(7):
        indata += np.sin((i+2)*2*np.pi*f*t)/(np.random.randint(7)+4)**4
    indata += np.random.normal(0,1e-5,len(indata))
    indata = np.floor(2**10*(0.5*indata+0.5))

    duts=[signal_analyser() for i in range(1) ]
    duts[0].model='py'
    for d in duts: 
        d.fs = fs
        d.nsamp = nsamp
        d.window = False
        d.annotate_harmonics = True
        d.sfdr_pointer= True
        d.annotations = ['SFDR','SNR','SNDR','THD']
        #d.annotate_mismatch = ['skew','offset']
        d.freqscale = 'mhz'
        d.linlog = 'lin'
        #d.gridaxis = 'y'
        d.IOS.Members['in'].Data=indata
        d.init()
        d.run()
    input()

