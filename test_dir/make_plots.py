import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
from itertools import product
import os

component = 'all'
colors = 'krb'
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 18
rcParams['legend.loc'] = 'right'
rcParams['mathtext.fontset'] = 'dejavuserif'

def main():

    rocket_grids = ['smooth', 'bumpy']
    rocket_fields = ['smooth', 'bumpy']
    rocket_orientation = ['xy']
    rocket_bins = ['67']
    rocket_var = ['cp', 'cfx', 'all']

    flat_grids = ['smooth', 'cubic', 'step']
    flat_fields= ['step', 'wavy']
    flat_orientation = ['xy', 'xz', 'yz']
    flat_bins = ['33', '67']

    print('ROCKET')
    for grid, field, orientation, nll, var in product(rocket_grids, 
                                                      rocket_fields, 
                                                      rocket_orientation,
                                                      rocket_bins, 
                                                      rocket_var):
        print((grid,field,orientation))
        for axis in orientation:
            ll_exact = None
            ll_fn = '_'.join(('rocket', grid, field, orientation, nll, var, 'LL.dat')) 
            print(ll_fn)
            try:
                ll_old = np.loadtxt(os.path.join('lineloads','b0',axis+'_axis', ll_fn),
                                    skiprows=2)
            except:
                ll_old = False
            
            try:
                ll_new = np.loadtxt(os.path.join('lineloads','b1',axis+'_axis',ll_fn),
                                    skiprows=2)
            except:
                ll_new = False

            if ll_new is not False:
                plot_lls(ll_new, ll_old, ll_exact, axis, grid, orientation, field,  nll, var, 'Rocket')

    print('FLAT')
    var = 'all'
    for grid, field, orientation, nll in product(flat_grids, flat_fields, flat_orientation, flat_bins):
        print((grid,field,orientation))
        for axis in orientation:
            ll_fn = '_'.join(('flat', grid, field, orientation, nll, 'all', 'LL.dat')) 
            print(ll_fn)
            ll_exact = np.loadtxt(os.path.join('lineloads',
                                  '_'.join(('flat', field, orientation, 'exact_cp_'+axis+'_LL.dat'))),
                                  skiprows=2)
            try:
                ll_old = np.loadtxt(os.path.join('lineloads','b0',axis+'_axis', ll_fn),
                                    skiprows=2)
            except:
                ll_old = False
            
            try:
                ll_new = np.loadtxt(os.path.join('lineloads','b1',axis+'_axis', ll_fn),
                                    skiprows=2)
            except:
                ll_new = False

            if ll_new is not False:
                plot_lls(ll_new, ll_old, ll_exact, axis, grid, orientation, field,  nll, var, 'Flat')

    os.system('gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite -dAutoRotatePages=/None -sOutputFile=plots/'+component+'.pdf plots/*'+component+'*_LL.pdf')

def plot_lls(ll_new, ll_old, ll_exact, axis, grid, orientation, field,  nll, var, case):
    plt.figure(figsize=(18,6))
    ax1 = plt.subplot(1,2,1)
    ax2 = plt.subplot(1,2,2)
    dirs = 'xyz'
    dots = 'os^'
    for i in range(1,4):
        if ll_exact is not None:
            ax1.plot(ll_exact[:,0], ll_exact[:,i],
                    colors[i-1]+':',
                    label=r"$\hat{ll}_{" + dirs[i-1] + "}$ or $\hat{lm}_{" + dirs[i-1] + "}$ Analytical")
                    #label='Analytical '+dirs[i-1]+' LL')
        ax1.plot(ll_new[:,0], ll_new[:,i],
                colors[i-1]+dots[i-1],
                markersize=5,
                label=r"$\hat{ll}_{" + dirs[i-1] + "}$ or $\hat{lm}_{" + dirs[i-1] + "}$ with $b_1$ bins")
                #label=r'$b_1$ '+dirs[i-1]+' LL')
        if ll_old is not False:
            ax1.plot(ll_old[:,0], ll_old[:,i],
                    colors[i-1]+dots[i-1],
                    markersize=10, markerfacecolor='none',
                    label=r"$\hat{ll}_{" + dirs[i-1] + "}$ or $\hat{lm}_{" + dirs[i-1] + "}$ with $b_0$ bins")
                    #label=r'$b_0$ '+dirs[i-1]+ ' LL')
        if ll_exact is not None:
            ax2.plot(ll_exact[:,0], ll_exact[:,i+3],
                    colors[i-1]+':', 
                    label='Analytical '+dirs[i-1]+' LM')
        ax2.plot(ll_new[:,0], ll_new[:,i+3], 
                colors[i-1]+dots[i-1], markersize=5,
                label=r'$b_1$ '+dirs[i-1]+ ' LM')
        if ll_old is not False:
            ax2.plot(ll_old[:,0],ll_old[:,i+3],
                    colors[i-1]+dots[i-1],
                    markersize=10,markerfacecolor='none',
                    label=r'$b_0$ '+dirs[i-1]+ ' LM')

    handles, legend_labels = ax1.get_legend_handles_labels()

    # makes legend as seperate figure
    fig_legend = plt.figure(figsize=(12,2))
    fig_legend.legend(handles, legend_labels, ncol=3, loc='center')
    legend_file = os.path.join('plots', case.lower() + '_ll_legend.pdf')
    fig_legend.savefig(legend_file, transparent=True)
    plt.close(fig_legend)

    ax1.set_ylabel('$\hat{ll}_d$')
    ax1.set_xlabel(axis)
    ax1.grid(which='major',axis='x')

    ax2.set_xlabel(axis)
    ax2.set_ylabel('$\hat{lm}_d$')
    ax2.grid(which='major',axis='x')

    plt.suptitle(case +', '+grid+' grid, '+field+' forcing, '+orientation+' orientation, '+component+' data')
    out_fn = '_'.join((case.lower(), grid, field, orientation, axis,  var, str(nll), 'LL.pdf'))

    plt.savefig(os.path.join('plots', out_fn), bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main()
