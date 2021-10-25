import numpy as np
import adios as bp
import os, sys, yaml, json
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
plt.rcParams.update({'font.size': 16})

from create_manage_analyze.analyze_file import energies_from_hist
from create_manage_analyze.plot_file import ax_and_cbar


def compress_bp(folder, suffix=None):
    bp_folder = f"{folder}/simOutput/bp" if suffix is None else f"{folder}/rerun_{suffix}/bp" 

    names_list = [fn.split('.')[0] for fn in os.listdir(bp_folder) if fn[-3:]=='.bp']
    iters = set([int(name.split('_')[-1]) for name in names_list])
    lastiter = max(iters)
    if False: # protect last iteration, may be not written out fully
        names_list = [name for name in names_list if not int(name.split('_')[-1])==lastiter]
    
    fieldsinfiles = {}
    for name in names_list:
        kind, nr = name.split('_')
        if kind=='em':
            fieldsinfiles[name] = {
                fname: dict(
                    outname=f"{fname.replace('/', '')}_{nr}.dat.gz",
                    compress = (10, 32)
                )
                for fname in "E/x, E/y, E/z, B/x, B/y, B/z".split(', ')
            }
        elif kind=='dens':
            fieldsinfiles[name] = {
                fname: dict(
                    outname=f"{'e' if fname[6]=='e' else ''}dens_{fname[0]}_{nr}.dat.gz",
                    compress = (10, 32),
                )
                for fname in 'e_all_density, H_all_density, e_all_energyDensity, H_all_energyDensity'.split(', ')
            }
        else:
            continue

    # only for console output:
    fromts = {}
    for name, fieldnames in fieldsinfiles.items():
        kind, nr = name.split('_')
        types = fromts.setdefault(int(nr), [])
        for fname in fieldnames:
            types.append(fname)
    print(f"In folder {folder} I will compress {len(fromts)} timesteps with following outputs:")
    for nr, types in sorted(fromts.items()):
        print(f"for ts {nr} the fields: {types}")


    for name, fieldnames in sorted(fieldsinfiles.items()):
        kind, nr = name.split('_')
        print(f"{kind} for ts {nr}", end=", ")
        filename = f"{bp_folder}/{name}.bp"
        with bp.File(filename) as fh:
            ds = fh[f"/data/{nr}/fields"]
            for k, attrs in fieldnames.items():
                Ny, Nx = ds[k].shape
                cy, cx = attrs['compress'] # care, the order is not x-y
                ny, nx = Ny//cy, Nx//cx

                data = np.reshape(ds[k][:], (ny, cy, nx, cx))
                datanew = data.mean(axis=(1,3)) * ds[f"{k}/unitSI"].value
                os.system(f'echo reducing {k} for ts {nr} by {cx}x{cy} in x/y dimension >> {bp_folder}/info.txt')

                outname = attrs['outname']
                np.savetxt(f"{bp_folder}/{outname}", datanew)

        os.remove(filename)
        os.system(f"rm -r {filename}.dir")

    print(f"\nFinished compressing bp for folder {folder}\n")


def plot_dens(nr, basefolder="/bigdata/hplsim/production/ilja/runs/pulseshape1d"):
    folder = f"{basefolder}/{nr}"
    bp_folder = f"{folder}/simOutput/bp"
    paramfile = f"/home/goethe93/simulations/210425_pulse_and_ramp_1d/custom_params/{nr}"
    dx, pos = 0.8 / 40, 80 # hardcoded length of cell of arrays in microns

    with open(paramfile, 'r') as fh:
        p = yaml.safe_load(fh)

    ts_main, dt = p['cfgparams']['ts_main'], p['cfgparams']['dt']
    step_to_t = lambda ts: (ts-ts_main)*1e15*dt

    names_list = [fn.split('.')[0] for fn in os.listdir(bp_folder) if fn[-3:]=='.gz']
    iters_to_kind = {}
    for name in names_list:
        *rest, ts = name.split('_')
        lis = iters_to_kind.setdefault(int(ts), [])
        lis.append('_'.join(rest))
    
    # create axes for plotting and outfolder
    fig, ax1 = plt.subplots(1, 1, figsize=(9,7))
    ax2 = ax1.twinx()
    outfolder = f"{folder}/analysis"
    os.system(f'mkdir -p {outfolder}')
    
    n_c = (1.11485e21 * 1.e6 / 0.8 / 0.8)
    
    # do the plotting for all ts that have e and H density
    for ts, kinds in iters_to_kind.items():
        if 'dens_e' in kinds and 'dens_H' in kinds:
            print(f"plotting dens for run {nr} for ts {ts}")
            dens_e = np.loadtxt(f"{bp_folder}/dens_e_{ts}.dat.gz") / n_c
            dens_H = np.loadtxt(f"{bp_folder}/dens_H_{ts}.dat.gz") / n_c
            if not len(dens_e) == len(dens_H):
                raise ValueError("I expect the data for densities to be of same length")
            
            coords = np.arange(len(dens_e)) * dx - pos
            coords2 = coords[5::10] # for smoothed delta_n
            ax1.cla()
            ax2.cla()
            ax2.plot(coords2, (dens_H-dens_e).reshape((-1, 10)).mean(axis=-1), 'g--', alpha=0.8)
            ax1.plot(coords, dens_e, 'b-', lw=2, label = "$n_e$")
            ax1.plot(coords, dens_H, 'r-', lw=2, label = '$n_H$')
            ax1.plot([], [], 'g--', label = '$n_H$ - $n_e$')
            ax1.set_yscale('log')
            ax1.set_xlabel('pos in $\mu\,$m')
            ax1.set_ylabel('$n_{e/H}$ in $n_c$')
            ax2.set_ylabel('$\Delta n$ in $n_c$')
            # cut smaller limits:
            wheredens = np.where(dens_e > 5e-5)[0]
            indstart = wheredens[0] - 160 if wheredens[0] > 160 else 0
            indend = wheredens[-1] + 160 if wheredens[-1] < len(dens_e)-162 else -1
            ax1.set_xlim([coords[indstart], coords[indend]])
            ax1.set_ylim([1e-3, max(dens_e) * 1.5])
            ax1.legend()
            ax1.set_title(f"dens of $H^+$ and $e^-$ $~~|~~$ {nr} $~~|~~$ step {ts} / {round(step_to_t(ts))} fs ")
            fig.savefig(f"{outfolder}/dens_{ts}.png")
    
    plt.close()

def rerun_create(folder, suffix=None, force=False):
    filenames = os.listdir(f"{folder}/tbg")
    suffix = "a" if suffix is None else suffix
    if f"submit_{suffix}.start" in filenames and not force:
        raise AssertionError(f"requested file submit_{suffix}.start already exists")
    
    if "submit.start" in filenames:
        with open(f"{folder}/tbg/submit.start", "r") as fh:
            text = fh.read()
    
    text = text.replace("simOutput", f"rerun_{suffix}").replace("stdout", f"stdout_{suffix}").replace("stderr", f"stderr_{suffix}")
    with open(f"{folder}/tbg/submit_{suffix}.start", "w") as fh:
        fh.write(text)
    
    return text



def plot_heatmap(scandata, **metadata):
    if not "axfig" in metadata:
        axes, cax, fig = ax_and_cbar(retfig=True)
    else:
        axes, cax, fig = metadata["axfig"] 
    title = metadata.get("title", lambda key: str(key))
    suptitle = metadata.get("suptitle", lambda key: "")
    datafilter = metadata.get("datafilter", lambda x: x)
    xlabel = metadata.get("xlabel", "scan quantity 1")
    ylabel = metadata.get("ylabel", "scan quantity 2")
    iflogx = metadata.get("iflogx", True)
    iflogy = metadata.get("iflogy", True)
    iftext = metadata.get("iftext", False)
    nrn = metadata.get("nrn", {})
    
    valsx, valsy = [sorted(set(item)) for item in zip(*scandata.keys())]    
    # construct bins for x and y axis
    bins = {}
    for name, vals, iflog in [('x', valsx, iflogx), ('y', valsy, iflogy)]:
        sqrtratios = np.exp(np.diff(np.log(vals))) ** 0.5
        halfdiffs = np.diff(vals) / 2
        bins[name] = b = []

        if iflog:
            b.append(vals[0] / sqrtratios[0])
            for val, rat in zip(vals, sqrtratios):
                b.append(val * rat)
            b.append(vals[-1] * sqrtratios[-1])
        else:
            b.append(vals[0] - halfdiffs[0])
            for val, diff in zip(vals, halfdiffs):
                b.append(val + diff)
            b.append(vals[-1] + halfdiffs[-1])

    # construct data array
    data = np.zeros((len(valsx), len(valsy)))
    textcoords = {}
    for i, x in enumerate(valsx):
        for j, y in enumerate(valsy):
            try:
                data[i, j] = datafilter(scandata[(x, y)])
                textcoords[(x, y)] = (bins['x'][i], (bins['y'][j] + bins['y'][j+1])/2)
            except KeyError:
                print(f"missing {x}, {y}")

    im = axes.pcolormesh(
        bins['x'], bins['y'],
        data.T, norm=LogNorm(vmin=np.min(data[data>0]), vmax=np.max(data))
    )
    axes.set_xscale('log' if iflogx else 'linear')
    axes.set_yscale('log' if iflogy else 'linear')
    axes.set_ylabel(ylabel)
    axes.set_xlabel(xlabel)
    fig.suptitle(suptitle)
    axes.set_title(title)
    plt.colorbar(im, cax=cax)
    
    if iftext:
        textargs = dict(verticalalignment='center', fontsize=8, color='red')
        for (x, y), val in scandata.items():
            value = datafilter(val)
            if nrn:
                text = f"{value}\n{nrn[(x, y)]}"
            else: 
                text = value
            axes.text(*textcoords[(x, y)], text, **textargs)
            
    
    return (axes, cax, fig)

def plot_heatmaps(scansdata, **metadata):
    axes, cax, fig = metadata.get("axfig", ax_and_cbar(retfig=True))
    key_to_title = metadata.get("key_to_title", lambda key: str(key))
    key_to_suptitle = metadata.get("key_to_suptitle", lambda key: "")
    key_to_filename = metadata.get("key_to_filename", lambda key: f"scan_{key}.png")
    datafilter = metadata.get("datafilter", lambda x: x)
    xlabel = metadata.get("xlabel", "scan quantity 1")
    ylabel = metadata.get("ylabel", "scan quantity 2")
    iflogx = metadata.get("iflogx", True)
    iflogy = metadata.get("iflogy", True)
    iftext = metadata.get("iftext", False)
    scans_to_nrn = metadata.get("scans_to_nrn", {})

    for key, scan in scansdata.items():
        title = key_to_title(key)
        suptitle = key_to_suptitle(key)
        axes.cla()
        cax.cla()
        print(f"plotting heatmap for {key}")
        axes, cax, fig = plot_heatmap(
            scan,
            axfig=(axes, cax, fig),
            title=title,
            suptitle=suptitle,
            datafilter=datafilter,
            xlabel=xlabel,
            ylabel=ylabel,
            iflogy=iflogy,
            iflogx=iflogx,
            iftext=iftext,
            nrn=scans_to_nrn.get(key, {})
        )
        fig.savefig(key_to_filename(key))

def get_params_from_nrn(nrn, basefolder="/bigdata/hplsim/production/ilja/runs/pulseshape1d"):
    with open(f"{basefolder}/params.yaml", 'r') as fh:
        all_params = yaml.safe_load(fh)

    params = {}

    for nr in nrn:
        base, suff = (nr, None) if not "/" in nr else nr.split("/")
        if base in all_params:
            params[nr] = all_params[base]
            continue

        print(f"reading params for {nr}")
        paramfile = f"/home/goethe93/simulations/210425_pulse_and_ramp_1d/custom_params/{base}" # hardcoded
        try:
            with open(paramfile, 'r') as fh:
                p = yaml.safe_load(fh)
        except FileNotFoundError:
            print(f"No paramfile for {nr}")
            continue

        params[nr] = p
        all_params[base] = p

    with open(f"{basefolder}/params.yaml", 'w') as fh:
        yaml.dump(all_params, fh)
    
    return params


def check_which_nrn_completed(nrn, basefolder="/bigdata/hplsim/production/ilja/runs/pulseshape1d"):
    completed = []
    for nr in nrn:
        if "/" in nr:
            base, suff = nr.split("/")
        else:
            base, suff = nr, None

        if not base in os.listdir(basefolder):
            print(f"{nr} not created")
            continue
                  
        folder = f"{basefolder}/{base}/"
        name = "stdout" if suff is None else f"stdout_{suff}"
        if not name in os.listdir(folder):
            print(f"{nr} not started")
            continue
                  
        with open(f"{folder}/{name}", 'r') as fh:
            zeilen = fh.readlines()
        if not 'full sim' in zeilen[-1]:
            print(f"{nr} not completed")
        else:
            completed.append(nr)
    
    return completed


def compute_emax_from_nrn(nrn, basefolder="/bigdata/hplsim/production/ilja/runs/pulseshape1d"):
    with open(f"{basefolder}/energies.json", 'r') as fh:
        efromname = json.load(fh)
        
    for nr in nrn:
        if not nr in efromname:
            print(f"Getting cutoff energy for {nr}")
            if "/" in nr:
                base, suff = nr.split("/")
                filename = f"{basefolder}/{base}/rerun_{suff}/H_energyHistogram_all.dat"
            else:
                filename = f"{basefolder}/{nr}/simOutput/H_energyHistogram_all.dat"
                
            energies, steps, arr, last_bin, first_gap = energies_from_hist(filename=filename)
            e1, e2 = [], []
            for ts in steps:
                e1.append(first_gap(ts, threshold=30))
                #e2.append(last_bin(ts, threshold=2e2))

            cutoff = max(e1)
            efromname[nr] = cutoff
            
    with open(f"{basefolder}/energies.json", 'w') as fh:
        json.dump(efromname, fh)
    
    return efromname


def plot_absorb_from_nrn(nrn, params={}, basefolder="/bigdata/hplsim/production/ilja/runs/pulseshape1d"):
    figa, axesa = plt.subplots(1, 1, figsize=(9, 7))

    with open("/bigdata/hplsim/production/ilja/runs/pulseshape1d/absorbtions.json", 'r') as fh:
        absorbtions_all = json.load(fh)
    
    absorbtions = {}
    for nr in nrn:
        if nr in absorbtions_all:
            absorbtions[nr] = absorbtions_all[nr]
            continue
            
        folder = f"{basefolder}/{nr}"
        outfolder = f"{folder}/analysis"
        os.system(f'mkdir -p {outfolder}')
        if not nr in params:
            paramfile = f"/home/goethe93/simulations/210425_pulse_and_ramp_1d/custom_params/{nr}" # hardcoded
            try:
                with open(paramfile, 'r') as fh:
                    p = yaml.safe_load(fh)
            except FileNotFoundError:
                continue
        else:
            p = params[nr]

        ts_main, dt = p['cfgparams']['ts_main'], p['cfgparams']['dt']
        step_to_t = lambda ts: (ts-ts_main)*1e15*dt

        dat = np.loadtxt(f"{folder}/simOutput/fields_energy.dat", skiprows=2)
        times = step_to_t(dat[:, 0])
        datf = dat[:, 1]
        date = np.loadtxt(f"{folder}/simOutput/e_energy_all.dat", skiprows=2)[:, 1]
        datH = np.loadtxt(f"{folder}/simOutput/H_energy_all.dat", skiprows=2)[:, 1]

        axesa.cla()
        axesa.plot(times, datH, 'r-', label="sum $E_H$")
        axesa.plot(times, date, 'b-', label="sum $E_e$")
        axesa.plot(times, datf, 'g-', label="sum fields $E$")
        axesa.set_title(f"timelines of energies $~~|~~$ {nr}")
        axesa.set_xlabel('time in fs')
        axesa.set_ylabel('Energies in J')
        axesa.legend()
        figa.savefig(f"{outfolder}/timeline_sume.png")

        absorbtions[nr] = absorbtions_all[nr] = (max(date+datH) / max(datf))
        print(f"computed absorbtion for {nr}: {absorbtions[nr]}")
              
    with open("/bigdata/hplsim/production/ilja/runs/pulseshape1d/absorbtions.json", 'w') as fh:
        json.dump(absorbtions_all, fh)
    
    return absorbtions

# for electron temperature
def bin_momentum(
        filename, ts=None,
        species="e_all",
        mom_direction="y",
        bins_mom_SI=None,
    ):
    """ returns bincounts of histogram of momenta along given axis """
    if ts is None: # then try to guess
        ts = filename.replace('.bp', '').split('_')[-1]

    if bins_mom_SI is None:
        bins_mom_SI = np.linspace(0, 20, 101)
        
    with bp.File(filename) as fh:
        ds = fh[f"/data/{ts}/particles/{species}/"]
        Np = ds["weighting"].shape[0]
        unitmom = ds[f"momentum/{mom_direction}"].unitSI.value
        bins_mom_pic = bins_mom_SI / unitmom

        # divide in chunks of less than 1e8 particles:
        nrchunks = (Np + 1) // 100000000 + 1
        limits = list(np.linspace(0, Np, nrchunks, endpoint=False, dtype=int))[1:]
        indices = [slice(i, j, None) for (i, j) in zip([None] + limits, limits + [None])]
        hists = []
        for index in indices:
            # read the data
            weights = ds["weighting"][index]
            N = len(weights)
            pos = np.empty(N)
            mom = np.empty(N)
            mom[:] = ds[f"momentum/{mom_direction}"][index] / weights

            # do binning
            hist, bin_edges = np.histogram(
                mom,
                bins=bins_mom_pic,
                weights=weights
            )
            hists.append(hist)

    hist = sum(hists)
    return hist

def electron_temp_thomas_from_particles(folder, steps=[0]):
    mass = 9.10938356e-31 # in SI
    c = 3e8             # in SI
    J2MeV = 6.242e+12
    E_0 = mass * c**2    # in SI
    Emin = 3**0.5 * 0.51122 # in Thomas' script minimal momentum is sqrt(2), i.e. gamma=sqrt(3)
    Emax = 15.51122 # maximal kinetic energy is 15 MeV
    NE = 100 # I take less bins, fluctuations are still bigger
    bins_E_MeV = np.linspace(Emin, Emax, NE+1)
    bins_E_SI = bins_E_MeV / J2MeV
    bins_p_SI = (bins_E_SI**2 - E_0**2)**0.5 / c
    hists = {}
    for ts in steps:
        filename = f"{folder}/simOutput/bp/species_{ts}.bp"
        hist = bin_momentum(filename, species='e_all', bins_mom_SI=bins_p_SI)
        hists[ts] = hist
    
    return bins_E_MeV, hists

def get_or_plot_electron_temp_thomas_from_particles(nrn, params={}, basefolder="/bigdata/hplsim/production/ilja/runs/pulseshape1d"):
    with open(f"{basefolder}/electron_temp_thomas_particles.json", 'r') as fh:
        electron_temp_thomas_particles = json.load(fh)

    fig, (axesh, axest) = plt.subplots(2, 1, figsize=(8, 10))

    for nr in nrn:
        axesh.cla()
        axest.cla()
        if nr in electron_temp_thomas_particles:
            continue

        if not nr in params:
            paramfile = f"/home/goethe93/simulations/210425_pulse_and_ramp_1d/custom_params/{nr}" # hardcoded
            try:
                with open(paramfile, 'r') as fh:
                    p = yaml.safe_load(fh)
            except FileNotFoundError:
                continue
        else:
            p = params[nr]

        ts_main, dt = p['cfgparams']['ts_main'], p['cfgparams']['dt']
        step_to_t = lambda ts: (ts-ts_main)*1e15*dt

        folder = f"{basefolder}/{nr}"
        steps = [int(name.split('.')[0].split('_')[-1]) for name in os.listdir(f"{folder}/simOutput/bp") if 'species' in name and name[-3:]=='.bp']
        print(f"{len(steps)} steps for sim {nr}")
        bins_E_MeV, hists = electron_temp_thomas_from_particles(folder, steps=steps)
        values_E = 0.5*(bins_E_MeV[:-1] + bins_E_MeV[1:])
        temperatures = {}
        plt.axes(axesh)
        for ts, hist in sorted(hists.items()):
            if not np.where(hist)[0].size > 0:
                continue
            time = step_to_t(ts)
            temperatures[time] = np.average(values_E, weights=hist)
            plt.plot(values_E, hist)
        plt.xlabel("$E_{e^-}$ in MeV")
        plt.ylabel(r"bin count $e^-$ in fw. direction")
        plt.yscale('log')
        plt.title("electron forward spectra for all times")
        
        plt.axes(axest)
        plt.plot(*zip(*sorted(temperatures.items())), 'b-x')
        plt.xlabel("time in fs")
        plt.ylabel(r"$T_{e^-}$ in MeV")
        plt.title('electron "temperature" over time')
        plt.xlim([None, 400])
        
        t_to_ts = {v: k for k, v in temperatures.items()}
        temp, tsmax = max(t_to_ts.items())
        print(f"in {nr}: temp {temp} at {tsmax}")
        electron_temp_thomas_particles[nr] = temp
        
        plt.savefig

    with open(f"{basefolder}/electron_temp_thomas_particles.json", 'w') as fh:
        json.dump(electron_temp_thomas_particles, fh)
    
    return {nr: temp for nr, temp in electron_temp_thomas_particles.items() if nr in nrn}









if sys.argv[1] == 'compress_bp':
    if len(sys.argv) > 4:
        raise ValueError("I expect at most two arguments after compress_bp subcommand: folder [suffix]")
    elif len(sys.argv)==3:
        compress_bp(folder=sys.argv[2])
    elif len(sys.argv)==4:
        compress_bp(folder=sys.argv[2], suffix=sys.argv[3])
    else:
        raise ValueError("I expect a path to a folder after compress_bp subcommand")
elif sys.argv[1] == 'plot_dens':
    if not len(sys.argv) == 3:
        raise ValueError("I expect exactly one run number after plot_dens subcommand")
    else:
        plot_dens(nr=sys.argv[2])
elif sys.argv[1] == 'rerun_create':
    if not len(sys.argv) <= 4:
        raise ValueError("I expect at most two arguments after rerun_create subcommand: nr [suffix]")
    else:
        suffix = sys.argv[3] if len(sys.argv) == 4 else None
        folder = f"/bigdata/hplsim/production/ilja/runs/pulseshape1d/{sys.argv[2]}/"
        rerun_create(folder=folder, suffix=suffix)

