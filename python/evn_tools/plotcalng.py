import os
import re
from matplotlib import pylab as pl
from matplotlib import dates
from datetime import datetime, timedelta

from casatools import table as tbtool
tb = tbtool()
    
params = {'legend.fontsize': 'x-small',
          'axes.labelsize': 'small',
          'axes.titlesize': 'small',
          'xtick.labelsize': 'x-small',
          'ytick.labelsize': 'x-small',
          'figure.dpi': 160}
pl.rcParams.update(params)

def __casa2datetime(t):
    days = int(t // 86400)
    temp = int(round((t - days * 86400) * 1000))
    seconds = temp // 1000
    useconds = temp * 1000 - seconds * 1000000
    return datetime(1858, 11, 17) + timedelta(days, seconds, useconds)

def __get_antennas(caltable):
    tb.open(os.path.join(caltable, 'ANTENNA'))
    antennas = tb.getcol('NAME').tolist()
    tb.close()
    return antennas

def __get_table(caltable, table):
    tb.open(os.path.join(caltable, table))
    rows = []
    for i in range(tb.nrows()):
        row = {}
        for name in tb.colnames():
            try:
                row[name] = tb.getcell(name, i)
            except RuntimeError:
                pass
        rows.append(row)
    tb.close()
    return rows

def __get_data(caltable, xaxis, yaxis, iteration, poln, field, antenna, spw, timerange):
    antenna_list = __get_antennas(caltable)
    fields_list = __get_table(caltable, 'FIELD')
    spw_list = __get_table(caltable, 'SPECTRAL_WINDOW')
    caltype = __get_caltype(caltable)
    
    tb.open(caltable)
    timecol = tb.getcol('TIME')
    antcol = tb.getcol('ANTENNA1')
    spwcol = tb.getcol('SPECTRAL_WINDOW_ID')
    fieldcol = tb.getcol('FIELD_ID')
    column = 'CPARAM' if "CPARAM" in tb.colnames() else "FPARAM"
    paramcol = tb.getcol(column)
    tb.close()

    # Select data; would for loop + numba be preferred for large tables? 
    rows = pl.where(pl.isin(fieldcol,field))[0]
    rows = rows[pl.where(pl.isin(spwcol[rows], spw))[0]]
    rows = rows[pl.where(pl.isin(antcol[rows], antenna))[0]]
    rows = rows[pl.where(pl.logical_and(timecol[rows] >= timerange[0], timecol[rows] <= timerange[1]))[0]]
    
    # Key and Label templates for each subfigure, iteration determines the number of subfigures
    # iteratation = field, antenna, or spw
    figtempl = ""
    fighdrs = []
    iterset = set(it.upper() for it in iteration.split(','))
    #print('iteration="{}", len(iterset)="{}", iterset="{}"'.format(iteration, len(iterset), iterset))
    for it in iterset:
        if it == "":
            continue
        elif it == "ANTENNA":
            figtempl += "A{1}"
            fighdrs.append("Antenna = {1}")
        elif it == "SPW":
            figtempl += "S{2}"
            fighdrs.append("Spw = {2}")
        elif it == "FIELD":
            figtempl += "F{3}"
            fighdrs.append("Field = {3}")
        else:
            print("Error, invalid iteration parameter:", it)
            raise ValueError()
    fighdrtempl = ", ".join(fighdrs)
    
    # Label and key template for each plot
    #plotcols = {"TIME", "ANTENNA", "SPW", "FIELD"} - iterset
    plotcols = {"ANTENNA", "SPW", "FIELD"} - iterset
    if "TIME" in (xaxis, yaxis):
        plotcols -= {"TIME"}
    if "CHANNEL" in (xaxis, yaxis):
        plotcols -= {"SPW"}
    if "ANTENNA" in (xaxis, yaxis):
        plotcols -= {"ANTENNA"}
    plottempl = ""
    plothdrs = []
    for pc in plotcols:
        if pc == "TIME":
            plottempl += "T{0}"
            plothdrs.append("{0}")
        elif pc == "ANTENNA":
            plottempl += "A{1}"
            plothdrs.append("{1}")
        elif pc == "SPW":
            plottempl += "S{2}"
            plothdrs.append("{2}")
        elif pc == "FIELD":
            plottempl += "F{3}"
            plothdrs.append("{3}")
    plothdrtempl = ", ".join(plothdrs)
    
    #field,spw,antenna,pol,time / val
    plotdata = {}
    for row in rows:
        r = (timecol[row], antcol[row], spwcol[row], fieldcol[row])
        rlabel = (__casa2datetime(r[0]), antenna_list[r[1]], r[2], fields_list[r[3]]['NAME'])
        figkey = figtempl.format(*r)
        try:
            f = plotdata[figkey]
        except KeyError:
            label = fighdrtempl.format(*rlabel)
            f = {"label": label}
            plotdata[figkey] = f
        plotkey = plottempl.format(*r)
        try:
            p = f[plotkey]
        except KeyError:
            label = plothdrtempl.format(*rlabel)
            p = ([], [], label) # X, Y, Label
            f[plotkey] = p
        #print('figkey = \"{}\" ; plotkey = {}; label = {}'.format(figkey, plotkey, label))
        for pol in poln:
            for i, ax in enumerate((xaxis, yaxis)):
                # Should be possible to do more efficient
                if ax == "TIME":
                    p[i].append(rlabel[0])
                elif ax == "ANTENNA":
                    p[i].append(r[1])
                elif ax == "CHAN":
                    nchan = spw_list[r[2]]['NUM_CHAN']
                    p[i].extend([ch for ch in range(nchan)])
                elif ax == "FREQ":
                    p[i].extend(spw_list[r[2]]['CHAN_FREQ'])
                elif ax == "DELAY":
                    if caltype == "FRINGE":
                        p[i].append(paramcol[pol*4 + 1, 0, row])
                    else:
                        p[i].append(paramcol[pol*4, 0, row])
                elif ax == "RATE":
                    if caltype == "FRINGE":
                        p[i].append(paramcol[pol*4+2, 0, row]) 
                    else:
                        print("Error: Rate is not available in calibration table")
                        raise ValueError()
                elif ax == "DISP":
                    if caltype == "FRINGE":
                        p[i].append(paramcol[pol*4+3, 0, row]) 
                    else:
                        print("Error: Dispersive delay is not available in calibration table")
                        raise ValueError()
                elif ax == "AMP":
                    p[i].extend(pl.absolute(paramcol[pol, :, row]))
                elif ax == "PHASE":
                    if caltype == "FRINGE":
                        v = paramcol[pol*4, :, row]
                    else:
                        v = paramcol[pol, :, row]
                    p[i].extend(pl.arctan2(pl.imag(v), pl.real(v)) * 180 / pl.pi)
                elif ax == "TSYS":
                    if caltype == "EVLASWPOW":
                        p[i].append(paramcol[pol*4 + 1], 0, row)
                    else:
                        p[i].append(paramcol[pol, 0, row])
                else:
                    #print(len(spw_list))
                    #print(spw_list)
                    #print(paramcol.shape)
                    print("Error: unknown axis ", ax)
                    raise ValueError()

    # Convert data to sorted numpy arrays
    for figkey, fig in plotdata.items():
        for plotkey, data in fig.items():
            if (plotkey == "label"):
                continue
            x = pl.array(data[0])
            y = pl.array(data[1])
            srt = x.argsort()
            fig[plotkey] = [x[srt], y[srt], data[2]]
    return plotdata

def __ensure_iterable(x):
    if isinstance(x, str):
        return [x]
    elif isinstance(x, bytes):
        return [x.decode('UTF8')]
    try:
        for i, v in enumerate(x):
            if isinstance(v, bytes):
                x[i] = v.decode('UTF8')
    except TypeError:
        # x is not iterable
        return([x])
    return x

def __get_selection(selection, table_key, casa_table):
    if (selection == "") or (selection == []):
        return [i for i in range(len(casa_table))]

    results = []
    error_msg = "Invalid value passed to parameter " + table_key
    for entry in __ensure_iterable(selection):
        if isinstance(entry, str):
            for item in entry.split(','):
                m = re.match(r'(\d+)~(\d+)', item)
                if m != None:
                    smin, smax = [int(x) for x in m.groups()]
                    if (smin > smax) or (smax >= len(casa_table)):
                        print(error_msg)
                        raise ValueError(error_msg)
                    results += [i for i in range(smin, smax+1)]
                else:
                    m = re.match(r'\d+$', item)
                    if m != None:
                        val = int(item)
                        if (val >= len(casa_table)):
                            print(error_msg)
                            raise ValueError(error_msg)
                        results.append(val)
                    else:
                        idx = -1
                        for i, table_entry in enumerate(casa_table):
                            if table_entry["NAME"] == item:
                                idx = i
                                break
                        if idx == -1:
                            print(error_msg + " : \'{}\'".format(item))
                            raise ValueError(error_msg)
                        results.append(idx)
        elif isinstance(entry, int):
            if (entry < 0) or (entry >= len(casa_table)):
                print(error_msg + " : {}".format(entry))
                raise ValueError(error_msg)
            results.append(entry)
        else:
            raise TypeError("Invalid data type for parameter \"{}\", should be string or int".format(table_key))
    if len(set(results)) != len(results):
        dups = {"{}(spw #{})".format(casa_table[b]["NAME"], b) for b in results if results.count(b) > 1}
        raise ValueError("Error: Duplicate entries in parameter {}: {}".format(table_key, ",".join(dups)))
    return results

def __parse_time(t, obs_start):
    # First try matching on YYYY/MM/DD/hh:mm:ss
    m = re.match(r'(\d{4})/(\d+)/(\d+)/(\d+):(\d+):(\d+)', t)
    if m == None:
        # Try matching on hh:mm:ss
        m = re.match(r'(\d+):(\d+):(\d+)', t)
        if m == None:
            print('Invalid time', t)
            raise ValueError()

    g = m.groups()
    # first extract time
    hour, minute, seconds = g[-3:]
    if (minute > 60) or (second > 60):
        # NB hour > 24 is valid
        print('Invalid time', t)
        raise ValueError()
    
    # Date part is optional
    if len(g) == 3:
        return obs_start + hour * 3600 + minute * 60 + second

    year, month, day = g[:3]
    d = datetime(year, month, day, hour, minute, second) 
    return (d - datetime(1858, 11, 17)).total_seconds()

def __parse_timerange(timerange, obs_list):
    obs_range = obs_list[0]['TIME_RANGE']
    if timerange == '' or timerange == []:
        return (obs_range[0], obs_range[1])

    times = []
    for entry in __ensure_iterable(timerange):
        if isinstance(entry, str):
            # Format YYYY/MM/DD/hh:mm:ss[~YYYY/MM/DD/hh:mm:ss]
            #   or              hh:mm:ss[~YYYY/MM/DD/hh:mm:ss]
            d = entry.partition("~")
            times.append(__parse_time(d[0], obs_range[0]))
            if d[2] != "":
                times.append(__parse_time(d[2], obs_range[0]))
        elif isinstance(entry, datetime):
            times.append((entry - datetime(1858, 11, 17)).total_seconds())
        else:
            print("Invalid data time for parameter \"timerange\", should be string or datetime")
            raise TypeError()
    if (len(times) != 2) or (times[0] > times[1]):
        print('Invalid time in timerange:', times)
        raise ValueError()
    return tuple(times)

def __get_caltype(caltable):
    tb.open(caltable)
    subType = [x.upper() for x in tb.info()['subType'].split()]
    tb.close()
    if (subType[-1] == "TSYS") or (subType[-1] == "EVLASWPOW"):
        caltype = subType[-1]
    else:
        caltype = subType[0]
    print("subType = {}, caltype = {}".format(subType, caltype))
    return caltype

def __parse_axis(caltable, xaxis_, yaxis_):
    xaxis = xaxis_.upper()
    yaxis = yaxis_.upper()
    
    # The axis are determined by which calibration table type we are plotting
    caltype = __get_caltype(caltable)    
    if xaxis == "" :
        if caltype in ("G", "T", "M", "A", "GSPLINE", "EVLAWSPOW", "F", "FRINGE"):
            xaxis = "TIME"
        elif caltype == "D":
            xaxis = "ANTENNA"
        else:
            xaxis = "CHAN"
    if yaxis == "":
        if caltype == "EVLASWPOW":
            yaxis = "SPGAIN"
        elif caltype == "TSYS":
            yaxis = "TSYS"
        elif caltype == "F":
            yaxis = "TEC"
        elif caltype == "FRINGE":
            yaxis = "DELAY"
        else:
            yaxis = "AMP"

    # AMP has a different meaning for TSYS and EVLASWPOW
    if caltype == "TSYS":
        if xaxis == "AMP":
            xaxis = "TSYS"
        if yaxis == "AMP":
            yaxis = "TSYS"
    elif caltype == "EVLASWPOW":
        if xaxis == "AMP":
            xaxis = "SPGAIN"
        if yaxis == "AMP":
            yaxis = "SPGAIN"

    return xaxis, yaxis

def __parse_input(caltable, xaxis, yaxis, field, spw, antenna, poln, timerange):
    antenna_list = __get_table(caltable, 'ANTENNA')
    fields_list = __get_table(caltable, 'FIELD')
    spw_list = __get_table(caltable, 'SPECTRAL_WINDOW')
    obs_list = __get_table(caltable, 'OBSERVATION')
    selection = {}
    
    # If axis aren't specified determine defaults from calibration table type
    selection["xaxis"], selection["yaxis"] = __parse_axis(caltable, xaxis, yaxis)

    # Timerange to be plotted, also accepts python datetime objects
    selection['timerange'] = __parse_timerange(timerange, obs_list)

    # Get list of fields to be plotted
    selection['field'] = __get_selection(field, 'field', fields_list)

    # Spectral windows, like CASA polplot the channel syntax is ignored
    # Use plotrange to determine which channels should be plotted
    selection['spw'] = __get_selection(spw, 'spw', spw_list)

    # Create Antenna list
    selection['antenna'] = __get_selection(antenna, 'antenna', antenna_list)

    # Polarizations
    if (poln == '') or (poln == []):
        selection['poln'] = [0, 1]
    else:
        for entry in __ensure_iterable(poln):
            polarizations = []
            if isinstance(entry, str):
                    if (entry == 'XY') or (entry == 'RL'):
                        polarizations += [0, 1]
                    elif (entry == 'X') or (entry == 'R'):
                        polarizations.append(0)
                    elif (entry == 'Y') or (entry == 'L'):
                        polarizations.append(1)
                    else:
                        raise ValueError("Invalid polarization parameter {}".format(entry))
            elif isinstance(entry, int):
                if (entry < 0) or (entry > 1):
                    raise ValueError("Invalid polarization index {}".format(entry)) 
                polarizations.append(entry)
            else:
                raise TypeError("Invalid data type for parameter \"poln\", should be string or int")
        if len(set(polarizations)) != len(polarizations):
            raise ValueError("Error: Duplicate polarizations entries")
        selection['poln'] = polarizations

    return selection

def plotcalng(caltable='', xaxis='', yaxis='', poln='', field='', antenna='', spw='', 
              timerange='', plotncolumns=2, subplot='', overwrite=False, clearpanel='Auto', iteration='', 
              plotrange=[], showflags=False, plotsymbol='.', plotcolor='blue', markersize=5.0, 
              fontsize=10.0, showgui=False, figfile='', showlegend=True):
    if subplot != "":
        print("Warning, the subplot parameter is ignored")
    
    selection = __parse_input(caltable, xaxis, yaxis, field, spw, antenna, poln, timerange)
    data = __get_data(caltable, 
                      selection['xaxis'], 
                      selection['yaxis'], 
                      iteration, 
                      selection['poln'], 
                      selection['field'], 
                      selection['antenna'], 
                      selection['spw'],
                      selection['timerange'])
    
    # Create grid of subplots
    nplots = max(1, len(data))
    ncolumns = min(nplots, plotncolumns)
    nrows = int(pl.ceil(nplots / ncolumns))
    # Unfortunately, matplot doesn't properly scale fig size with nrows / ncolumns
    if nplots == 1:
        figsize = (4,3)
    else:
        figsize = (8, 3 * nrows)
    
    fig, axs = pl.subplots(nrows, ncolumns, figsize = figsize, constrained_layout=True)
    if not isinstance(axs, pl.ndarray):
        axs = pl.array([[axs]])
    elif len(axs.shape) == 1:
        axs = axs.reshape([1, axs.shape[0]])
    nremove = nrows * ncolumns - nplots
    for i in range(0, nremove):
        axs[-1][ncolumns - i - 1].set_visible(False)

    ticklocator = dates.AutoDateLocator()
    tickformatter = dates.AutoDateFormatter(ticklocator)
    #print(data.keys())
    #print('nplots = {}, ncolumns = {}, nrows = {}, nremove = {}'.format(nplots, ncolumns, nrows, nremove))
    #print(len(axs), len(axs[-1]))
    for i, key in enumerate(data):
        row = i // ncolumns
        column = i % ncolumns
        ax = axs[row][column]
        f = data[key]
        for plotkey in f:
            if (plotkey == "label"):
                ax.set_title(f[plotkey])
                continue
            xy = f[plotkey]
            axislabels = []
            for j, axistype in enumerate((selection['xaxis'], selection['yaxis'])):
                if axistype == "TIME":
                    ax.xaxis.set_major_locator(ticklocator)
                    ax.xaxis.set_major_formatter(tickformatter)
                    xy[j] = dates.date2num(xy[j])
                    axislabels.append('Time')
                elif axistype == "CHAN":
                    axislabels.append("Channel")
                elif axistype == "ANTENNA":
                    axislabels.append("Antenna")
                elif axistype == "FREQ":
                    maxval = max(xy[j])
                    if maxval >= 1e10:
                        xy[j] /= 1e9
                        unit = "GHz"
                    elif maxval > 1e7:
                        xy[j] /= 1e6
                        unit = "MHz"
                    elif maxval > 1e4:
                        xy[j] /= 1000
                        unit = "KHz"
                    else:
                        unit = "Hz"
                    axislabels.append("Frequency [{}]".format(unit))
                elif axistype == "DELAY":
                    axislabels.append("Delay [ns]")
                elif axistype == "RATE":
                    xy[j] *= 1e12
                    axislabels.append("Delay rate [ps / s]")
                elif axistype == "DISP":
                    # TEC conversion parameter from Mevius LOFAR school notes
                    xy /= 1.334537
                    axislabels.append("Dispersive delay [miliTEC]")
                elif axistype == "AMP":
                    axislabels.append("Gain Amplitude")
                elif axistype == "PHASE":
                    axislabels.append("Gain Phase [degrees]")
                elif axistype == "TSYS":
                    axislabels.append("TSYS [K]")
                else:
                    print("Unknown axis type:", axistype)
                    raise ValueError()
           
            ax.plot(xy[0], xy[1], plotsymbol, label=xy[2])
            ax.set_xlabel(axislabels[0])
            ax.set_ylabel(axislabels[1])
            if showlegend:
                ax.legend()

            if plotrange != []:
                # CASA convention: if min==max in plotrange for axis then autoscale that axis
                if plotrange[0] != plotrange[1]:
                    ax.set_xlim(plotrange[0], plotrange[1])
                if plotrange[2] != plotrange[3]:
                    ax.set_ylim(plotrange[2], plotrange[3])
                
    if xaxis.upper() == "TIME":
        fig.autofmt_xdate()
    if figfile != "":
        fig.savefig(figfile)
    #return data
