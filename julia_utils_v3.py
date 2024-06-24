# %load julia_utils.py
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import os
#from datetime import datetime
from datetime import timezone
from dateutil.tz import *
import matplotlib.dates as mdates
from datetime import timedelta
#from iri2016 import timeprofile, timeprofile2
from model_utils import *
from scipy import stats


def Diff(li1, li2):
    return (list(list(set(li1)-set(li2)) + list(set(li2)-set(li1))))


def GetBinaryMatrix(d_prima):
    d_prima = d_prima.copy()
#d_prima[d_prima==np.nan]=0
#d_prima[d_prima!=np.nan]=1
    fils, cols = d_prima.shape
    matCount = np.zeros(d_prima.shape)
    for i in range(fils):
        for j in range(cols):
            if np.isnan(d_prima[i,j]):
                d_prima[i,j]=0
            #matCount[i,j]+=1
            else:
                d_prima[i,j]=1
                matCount[i,j]+=1
    return d_prima, matCount

def GetTimeRangeArrays(year, month, day, hour_i, hour_f, h_min, h_max, delta_ran):
    range_fixed = np.arange(h_min,h_max+delta_ran,delta_ran)
    string_t0 = '%d-%02d-%02d %02d:00:00' % (year, month, day, hour_i)
    string_tf = '%d-%02d-%02d %02d:00:00' % (year, month, day, hour_f)
    dt0 = datetime.datetime.strptime(string_t0, '%Y-%m-%d %H:%M:%S')
    dtf = datetime.datetime.strptime(string_tf, '%Y-%m-%d %H:%M:%S') + timedelta(days=1)
#ax.set_xlim(mdates.date2num(dt0),mdates.date2num(dtf))
    time_range = np.arange(dt0,dtf,timedelta(seconds=600))
    #print(time_range[1]-time_range[0])
    #print("time_range.shape: ", time_range.shape)
    #print("Type: ", type(time_range[0]))
    time_range = np.array(time_range).astype(datetime.datetime)#,dtype='datetime64[s]')
    return time_range, range_fixed

def GetArrays(directory, filename, PlotFlag):
    hour_i= 18.0
    hour_f = 7.0
    h_min = 90.0
    h_max = 1008.75
    delta_ran = 3.75

    file_hf5 = directory + filename
    hf = h5py.File(file_hf5, 'r')
    rango = hf['Data/Table Layout/']['gdalt']
    #rango2D = hf['Data/Array Layout/2D Parameters/gdalt']
    snl =  hf['Data/Table Layout/']['snl']
    snl2 = hf['Data/Array Layout/2D Parameters/snl']
    #vipe1 = hf['Data/Array Layout/2D Parameters/vipe']
    vipn1 = hf['Data/Array Layout/2D Parameters/vipn']
    timestamps = hf['Data/Array Layout/']['timestamps']
    vvert = hf['Data/Table Layout/']['vipn']
    time_vector = []
    #v_zonal = np.array(vipe1).T
    v_vertical = np.array(vipn1).T
    snl2 = np.array(snl2).T
    #print("v_zonal.shape",v_zonal.shape)
    print("v_vertical.shape",v_vertical.shape)
    print("snl2.shape",snl2.shape)
    #snl = np.array(snl2).flatten()
    rango = getattr(rango, "tolist", lambda: rango)()
    ###########################################################
    ran_max = max(rango)
    ran_min = min(rango)
    #rang_list = list(rango)
    max_index = rango.index(ran_max)
    min_index = rango.index(ran_min)
    rango = np.array(rango)
    range_diff = np.diff(rango)
    #delta_range = range_diff[0]
    #array = np.array([1,2,3,4,4,5])
    mode = stats.mode(range_diff)
    print('delta_range', mode[0])
    delta_range = mode[0]
    #print('delta range: ', delta_range)#valor constante para todo el arreglo
    MinRange, MaxRange = np.min(rango),np.max(rango)
    DataMatrixRows = int((MaxRange-MinRange)/delta_range)
#range_array = np.linspace(MinRange, MaxRange, DataMatrixRows+1)
    DataMatrix = np.ones((DataMatrixRows, snl2.shape[0]))*np.nan
    DataMatrixVert = np.ones((DataMatrixRows, snl2.shape[0]))*np.nan
    DriftMatrix = np.ones((DataMatrixRows+1, snl2.shape[0]))*np.nan
    RowInMatrix = np.array((rango-MinRange)/delta_range, dtype=int)
    range_array = np.linspace(MinRange, MaxRange, DataMatrixRows+1)
    RangeMatrix = np.ones((DataMatrixRows+1, snl2.shape[0]))*np.nan
    prev_stamps = []
       #################################
    datetime_objects = []
    for ts in timestamps:
        if ts in prev_stamps:
            print('Same timestap')
        else:
            date_time_obj = datetime.datetime.fromtimestamp(ts)
            datetime_objects.append(date_time_obj)
    new_index = pd.DatetimeIndex(datetime_objects) # timedelta(hours=5)
    index = pd.DatetimeIndex(datetime_objects) # timedelta(hours=5)
    datetime_objects = np.array(datetime_objects)#,dtype='datetime64[s]')
    day = index[0].day
    anio = index[0].year
    year = index[0].year
    month = index[0].month
    mes = GetMonth(month)
    time_range, range_fixed = GetTimeRangeArrays(year, month, day, hour_i, hour_f,h_min,h_max,delta_range)

    dir_plots = 'Plots-%s-%d' % (mes, anio)
    dir_plots_EEJ = 'Plots-%s-%d/EEJ' % (mes, anio)
    col = 0 #counter for current columns
    PastRow = 0 #saving past row index
    #print("range(rango.size) ",range(rango.size))
    for k in range(len(rango)):
        row = RowInMatrix[k]
        # Putting snr in corresponding matrix element
        DataMatrix[row,col] = snl[k]
        DataMatrixVert[row,col] = vvert[k]
        if row<PastRow:
            col += 1
        PastRow = row
    data = DataMatrix.T#[::-1]
    dataVert = DataMatrixVert.T
    #line.split()[0]
    string_t0 = '%d-%02d-%02d %02d:00:00' % (year, month, day, 18)
    print("Time string: ", string_t0)
    dt0 = datetime.datetime.strptime(string_t0, '%Y-%m-%d %H:%M:%S')
    #dt0 = datetime_objects[0]
    #print(string_t0)
    dt_indices = np.array((datetime_objects-dt0)/timedelta(seconds=600),dtype=int)
    #print("Datetime_objs: ",datetime_objects )
    #print("dt0: ", dt0)
    #print(dt_indices)
    #print(rango2D.shape, snl2.shape, data.shape, datetime_objects.shape)
    m = np.r_[True,dt_indices[:-1]!=dt_indices[1:],True]
    counts = np.diff(np.flatnonzero(m))
    unq = dt_indices[m[:-1]]
    times_repeated=np.c_[unq,counts]
    #time_Range_index = np.linspace(0,78,77)
    time_range_index = np.arange(0,78)
    print(time_range_index)

    ###############################################################################
    #d_prom = np.zeros((times_repeated.shape[0], data.shape[1]))
    d_prom = np.zeros((times_repeated.shape[0], range_fixed.shape[0]))
    for j in range(data.shape[1]):

        for i in range(times_repeated.shape[0]):
            if i==0:
                d_prom[i,j] = np.nanmean(data[0:times_repeated[0,1],j])
            else:
                aux = np.sum(times_repeated[0:i,1])
                aux2 = times_repeated[i,1]
                d_prom[i,j] = np.nanmean(data[aux:aux+aux2,j])
    #'''
    index_aux = times_repeated[:,0]
    diff = Diff(time_range_index, index_aux)
    diff=np.array(diff)
    diff=np.sort(diff)
    diff=diff.astype(np.dtype('int64'))
    #d_prom2 = np.copy(d_prom)
    #print("diff")
    #print(diff)
    b3=np.ones((time_range_index.shape[0],range_fixed.shape[0]))*np.nan

    for j in range(data.shape[1]):
        b_aux=d_prom[:,j]
        for i in list(diff):
            aux=b_aux[:i]
            aux2=b_aux[i:]
            aux=np.append(aux,np.nan)
            aux=np.append(aux,b_aux[i:])
            b_aux=aux
        b3[:,j]=b_aux

    #'''

    #timedelta(seconds=600)
    #print(time_range.shape, range_fixed.shape, d_prom.shape)
    #print(time_range[0],time_range[-1])
    if PlotFlag:
        fig, ax = plt.subplots(figsize=(12, 6))
#        clrs= ax.pcolormesh(mdates.date2num(time_range), range_fixed, b3.T,cmap='jet')
        clrs= ax.pcolormesh(mdates.date2num(time_range), range_fixed, b3.T,cmap='jet')

        ax.xaxis_date()
    #ax.set_xlim(mdates.date2num(dt0),mdates.date2num(dtf))
        date_format = mdates.DateFormatter('%H:%M')
        ax.xaxis.set_major_formatter(date_format)
        ax.set_xlabel("Hora Local (h)", fontsize=16)
        ax.set_ylabel("Rango (km)", fontsize=17)
        ax.set_ylim([170,900])
#dia = dia - 1
        fig_title = r'F-Dispersa - Promedio(%d-%02d-%02d)' % (year, month, day)
        plt.title(fig_title, fontsize=15)
        str_date = '%d-%02d-%02d' % (year, month, day)
    # This simply sets the x-axis data to diagonal so it fits better.
        fig.autofmt_xdate()
        box=ax.get_position()
        cbarax=fig.add_axes([box.x0+box.width+0.01, box.y0, 0.025, box.height])
        cb=plt.colorbar(clrs,cax=cbarax)
    #cb2 = fig.colorbar(im2)
        cb.set_label(r'$log_{10}SNR$', fontsize=17)
        #print(str_date)
        #plt.show()
        #plt.savefig('%s/promedio-%s.png' % (dir_plots,str_date))

        #plt.close(fig)
        fig, ax = plt.subplots(figsize=(12, 6))
#        clrs= ax.pcolormesh(mdates.date2num(time_range), range_fixed, b3.T,cmap='jet')
        clrs= ax.pcolormesh(mdates.date2num(index), range_array, data.T,cmap='jet')

        ax.xaxis_date()
    #ax.set_xlim(mdates.date2num(dt0),mdates.date2num(dtf))
        date_format = mdates.DateFormatter('%H:%M')
        ax.xaxis.set_major_formatter(date_format)
        ax.set_xlabel("Hora Local (h)", fontsize=16)
        ax.set_ylabel("Rango (km)", fontsize=17)
        ax.set_ylim([90,900])
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=60))
        date_format = mdates.DateFormatter('%H:%M')
        ax.xaxis.set_major_formatter(date_format)
#dia = dia - 1
        fig_title = r'F-Dispersa (%d-%02d-%02d)' % (year, month, day)
        plt.title(fig_title, fontsize=15)
        str_date = '(%d-%02d-%02d)' % (year, month, day)
    # This simply sets the x-axis data to diagonal so it fits better.
        fig.autofmt_xdate()
        box=ax.get_position()
        cbarax=fig.add_axes([box.x0+box.width+0.01, box.y0, 0.025, box.height])
        cb=plt.colorbar(clrs,cax=cbarax)
    #cb2 = fig.colorbar(im2)
        cb.set_label(r'$log_{10}SNR$', fontsize=17)
        #print(str_date)
        #plt.show()
        plt.setp( ax.xaxis.get_majorticklabels(), rotation=0 ,ha='center')
        #plt.savefig('%s/ESF-not-avg-%s.png' % (dir_plots,str_date))

        fig, ax = plt.subplots(figsize=(12, 6))
#        clrs= ax.pcolormesh(mdates.date2num(time_range), range_fixed, b3.T,cmap='jet')
        clrs= ax.pcolormesh(mdates.date2num(index), range_array, data.T,cmap='jet')

        ax.xaxis_date()
    #ax.set_xlim(mdates.date2num(dt0),mdates.date2num(dtf))
        date_format = mdates.DateFormatter('%H:%M')
        ax.xaxis.set_major_formatter(date_format)
        ax.set_xlabel("Hora Local (h)", fontsize=16)
        ax.set_ylabel("Rango (km)", fontsize=17)
        ax.set_ylim([80,160])
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=60))
        date_format = mdates.DateFormatter('%H:%M')
        ax.xaxis.set_major_formatter(date_format)
#dia = dia - 1
        fig_title = r'Electrochorro Ecuatorial (%d-%02d-%02d)' % (year, month, day)
        plt.title(fig_title, fontsize=15)
        str_date = '%d-%02d-%02d' % (year, month, day)
    # This simply sets the x-axis data to diagonal so it fits better.
        fig.autofmt_xdate()
        box=ax.get_position()
        cbarax=fig.add_axes([box.x0+box.width+0.01, box.y0, 0.025, box.height])
        cb=plt.colorbar(clrs,cax=cbarax)
    #cb2 = fig.colorbar(im2)
        cb.set_label(r'$log_{10}SNR$', fontsize=17)
        #print(str_date)
        #plt.show()
        plt.setp( ax.xaxis.get_majorticklabels(), rotation=0 ,ha='center')
        #plt.savefig('%s/EEJ-not-avg-%s.png' % (dir_plots_EEJ,str_date))
        #plt.close(fig)
        fig, ax = plt.subplots(figsize=(12, 6))
        clrs= ax.pcolormesh(mdates.date2num(datetime_objects), range_array, dataVert.T, cmap='jet')
        ax.xaxis_date()
        date_format = mdates.DateFormatter('%H:%M')
        ax.xaxis.set_major_formatter(date_format)
        ax.set_xlabel("Hora Local (h)", fontsize=16)
        ax.set_ylabel("Rango (km)", fontsize=17)
        ax.set_ylim([80,160])
        #fig_title = r'F-Dispersa (%d-%02d-%02d)' % (year, month, day)
        #plt.title(fig_title, fontsize=15)
        #        str_date = '(%d-%02d-%02d)' % (year, month, day)
        # This simply sets the x-axis data to diagonal so it fits better.
        #fig.autofmt_xdate()
        box=ax.get_position()
        cbarax=fig.add_axes([box.x0+box.width+0.01, box.y0, 0.025, box.height])
        cb=plt.colorbar(clrs,cax=cbarax)
    #cb2 = fig.colorbar(im2)
        cb.set_label(r'E$\times$B (m/s)', fontsize=17)
        plt.setp( ax.xaxis.get_majorticklabels(), rotation=0 ,ha='center')
        #plt.savefig('%s/Velocidad-vertical/vertical-velocity-%s.png' % (dir_plots,str_date))

    return time_range, range_fixed, b3,data,snl, snl2, index, range_array, rango, dir_plots, np.array(datetime_objects), dataVert
    ################################################################################

def GetMatrix(directory, filename, PlotFlag):

    file_hf5 = directory + filename
    hf = h5py.File(file_hf5, 'r')
    #with h5py.File(file_hf5, 'r') as f:
    #    g = f.visit(print)

    rango = hf['Data/Table Layout/']['gdalt']
    snl =  hf['Data/Table Layout/']['snl']
    snl2 = hf['Data/Array Layout/2D Parameters/snl']
    vipe1 = hf['Data/Array Layout/2D Parameters/vipe']
    vipn1 = hf['Data/Array Layout/2D Parameters/vipn']
    timestamps = hf['Data/Array Layout/']['timestamps']
    #snl = np.array(snl2).flatten()

    rango = getattr(rango, "tolist", lambda: rango)()
    ###########################################################
    ran_max = max(rango)
    ran_min = min(rango)
    #rang_list = list(rango)
    max_index = rango.index(ran_max)
    min_index = rango.index(ran_min)
    range_diff = np.diff(rango)
    #delta_range = range_diff[-1] #valor constante para todo el arreglo
    mode = stats.mode(range_diff)
    print('delta_range', mode[0])
    delta_range = mode[0]
    MinRange, MaxRange = np.min(rango), ran_max#np.max(rango)
    DataMatrixRows = int((MaxRange-MinRange)/delta_range)
    rango = np.array(rango)
    range_array = np.linspace(MinRange, MaxRange, DataMatrixRows+1)
    DataMatrix = np.ones((DataMatrixRows+1, snl2.shape[0]))*np.nan
    RowInMatrix = np.array((rango-MinRange)/delta_range, dtype=int)
    range_array = np.linspace(MinRange, MaxRange, DataMatrixRows+1)
    RangeMatrix = np.ones((DataMatrixRows+1, snl2.shape[0]))*np.nan
    prev_stamps = []
#################################
    datetime_objects = []
    for ts in timestamps:
        if ts in prev_stamps:
            print('Same timestap')
        else:
            date_time_obj = datetime.datetime.fromtimestamp(ts)
            datetime_objects.append(date_time_obj)
    index = pd.DatetimeIndex(datetime_objects) # timedelta(hours=5)
#################################
    string_date = index[0].strftime('%B %d, %Y, %r')
    #line.split()[0]
    mes = string_date.split()[0]
    month_prime = index[0].month
    dia = index[0].day
    anio = index[0].year
    mes = GetMonth(month_prime)
    dir_plots = 'Plots-%s-%d' % (mes, anio)
    col = 0 #counter for current columns
    PastRow = 0 #saving past row index
    #print("range(rango.size) ",range(rango.size))
    for k in range(len(rango)):
        row = RowInMatrix[k]
    # Putting snr in corresponding matrix element
        DataMatrix[row,col] = snl[k]
        if row<PastRow:
            col += 1
        PastRow = row
    data = DataMatrix#[::-1]

    print("Shapes involving in the plotting: ", data.T.shape, range_array.shape, len(datetime_objects))
    print("snl2.shape: ", snl2.shape)
    #######################################################################################################
    if (PlotFlag):
        fig, ax = plt.subplots(figsize=(12, 6))
        x_min = mdates.date2num(np.min(index))
        x_max = mdates.date2num(np.max(index))
        extent=[x_min, x_max,ran_min,ran_max]
        #im2 = plt.imshow(data, cmap='jet',aspect='auto',interpolation='nearest',origin="lower", extent=extent)
        clrs= ax.pcolormesh(mdates.date2num(datetime_objects), range_array, data, cmap='jet')
        ax.xaxis_date()
        date_format = mdates.DateFormatter('%H:%M')
        ax.xaxis.set_major_formatter(date_format)
        ax.set_xlabel("Hora Local (h)", fontsize=16)
        ax.set_ylabel("Rango (km)", fontsize=17)
        fig_title = r'F-Dispersa (%d-%02d-%02d)' % (anio, month_prime, dia)
        plt.title(fig_title, fontsize=15)
        # This simply sets the x-axis data to diagonal so it fits better.
        fig.autofmt_xdate()
        box=ax.get_position()
        cbarax=fig.add_axes([box.x0+box.width+0.01, box.y0, 0.025, box.height])
        cb=plt.colorbar(clrs,cax=cbarax)
        #cb2 = fig.colorbar(im2)
        cb.set_label(r'$log_{10}SNR$', fontsize=17)
        #plt.savefig(r'ESF-no-labels-%d-%02d-%02d.png' % (year[0], month[0], days[0]))
        plt.savefig(r'%s/ESF-prueba-escalado-%d-%02d-%02d.png' % (dir_plots,anio, month_prime, dia))
        plt.show()
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 6))
        x_min = mdates.date2num(np.min(index))
        x_max = mdates.date2num(np.max(index))
        #extent=[x_min, x_max,ran_min,ran_max]
        #im2 = plt.imshow(data, cmap='jet',aspect='auto',interpolation='nearest',origin="lower", extent=extent)
        clrs= ax.pcolormesh(mdates.date2num(datetime_objects), range_array, data, cmap='jet')
        ax.xaxis_date()
        date_format = mdates.DateFormatter('%H:%M')
        ax.xaxis.set_major_formatter(date_format)
        ax.set_xlabel("Hora Local (h)", fontsize=16)
        ax.set_ylabel("Rango (km)", fontsize=17)
        ax.set_ylimit(80,160)
        fig_title = r'Electrochorro Ecuatorial (%d-%02d-%02d)' % (anio, month_prime, dia)
        plt.title(fig_title, fontsize=15)
        # This simply sets the x-axis data to diagonal so it fits better.
        fig.autofmt_xdate()
        box=ax.get_position()
        cbarax=fig.add_axes([box.x0+box.width+0.01, box.y0, 0.025, box.height])
        cb=plt.colorbar(clrs,cax=cbarax)
        #cb2 = fig.colorbar(im2)
        cb.set_label(r'$log_{10}SNR$', fontsize=17)
        #plt.savefig(r'ESF-no-labels-%d-%02d-%02d.png' % (year[0], month[0], days[0]))
        plt.savefig(r'%s/EEJ-prueba-escalado-%d-%02d-%02d.png' % (dir_plots_EEJ,anio, month_prime, dia))
        plt.show()
        plt.close(fig)

    return  data ,snl, snl2, index, range_array, rango, dir_plots, np.array(datetime_objects)
