p_cent = 0.5* (period[1:] + period[:-1])
r_cent = 0.5* (rp[1:] + rp[:-1])

a = np.linspace(np.log10(10), np.log10(500), num=5)
b = np.diff(np.linspace(np.log10(10), np.log10(500), num=5))
print(a)
c = 10**np.append(a, a[-1]+b[0])
print(a, b, c)

d = np.diff(c)
print(d)

plt.plot(d, [1, 1, 1, 1, 1], 'o')
plt.xscale('log')

a = np.linspace(np.log10(2.5), np.log10(5), num=5)
b = np.diff(np.linspace(np.log10(2.5), np.log10(5), num=5))
print(a)
c = 10**np.append(a, a[-1]+b[0])
print(a, b, c)

d = np.diff(c)
print(d)

plt.plot(d, [1, 1, 1, 1, 1], 'o')
plt.xscale('log')

def dfm(pg, rg, prng, rrng, pbins, rbins, samples, model, comp, cplt=False):
    samples = np.atleast_2d(samples)
    pop = np.empty((len(samples), pg.shape[0], pg.shape[1]))
    for i, p in enumerate(samples):
        pop[i] = rateModel(pg, rg, prng, rrng, p, model)
    #pop = rateModel(pg, rg, prng, rrng, sample, model)
    if cplt:
        pop = pop * comp[None, :, :]
        alt_pop = np.swapaxes(pop, 1, 2)
    else:
        #pop = pop[None, :, :]
        alt_pop = np.swapaxes(pop, 1, 2)
    
    pop = 0.5*(pop[:, 1:] + pop[:, :-1])
    pop = np.sum(pop * np.diff(pbins)[None, :, None], axis=1)
    
    alt_pop = 0.5*(alt_pop[:, 1:] + alt_pop[:, :-1])
    alt_pop = np.sum(alt_pop * np.diff(rbins)[None, :, None], axis=1)
    
    x = np.append(np.diff(rbins), 0.94603558)
    y = np.append(np.diff(pbins), 829.57397424)
    
    r = np.percentile(pop * x, [2.5, 16, 50, 84, 97.5], axis=0)
    p = np.percentile(alt_pop * y, [2.5, 16, 50, 84, 97.5], axis=0)
    
    return(r, p)

def recreate(pg, rg, prng, rrng, pbins, rbins, samples, model, comp, cplt=False):
    samples = np.atleast_2d(samples)
    pop = np.empty((len(samples), pg.shape[0], pg.shape[1]))
    for i, p in enumerate(samples):
        pop[i] = rateModel(pg, rg, prng, rrng, p, model)
    #pop = rateModel(pg, rg, prng, rrng, sample, model)
    if cplt:
        pop = pop * comp[None, :, :]
        alt_pop = np.swapaxes(pop, 1, 2)
    else:
        #pop = pop[None, :, :]
        alt_pop = np.swapaxes(pop, 1, 2)
    
    pop = trapz(pop, x=pbins, axis=1)
    alt_pop = trapz(alt_pop, x=rbins, axis=1)
    
    x = np.append(np.diff(rbins), 0.94603558)
    y = np.append(np.diff(pbins), 829.57397424)
    
    r = np.percentile(pop * x, [2.5, 16, 50, 84, 97.5], axis=0)
    p = np.percentile(alt_pop * y, [2.5, 16, 50, 84, 97.5], axis=0)
    return(r, p)

from scipy.interpolate import griddata

points = np.array(list(zip(period_grid.flatten(), rp_grid.flatten())))
values = summedCompleteness.flatten()

pc = np.array(list(zip(kois.koi_period.values, getRadii(kois).values)))

grid = griddata(points, values, pc, method='linear')

kois['completeness'] = grid/ kiclist.size
kois['koi_rp'] = getRadii(kois)

occ = np.zeros((len(period)-1, len(rp)-1))

for i in range(len(period)-1):
    for j in range(len(rp)-1):
        #print(i, j)
        #print([period[i], period[i+1]], [rp[j], rp[j+1]])
        bin_ = kois[((kois.koi_period < period[i+1]) & (kois.koi_period > period[i]) & 
                     (kois.koi_rp < rp[j+1]) & (kois.koi_rp > rp[j]))]
        occ[i, j] = np.sum(bin_.totalReliability / bin_.completeness) / kiclist.size
        
print(occ)

plt.figure()
plt.plot(r_cent, np.sum(occ, axis=0), '.')
plt.xscale('log')

plt.figure()
plt.plot(p_cent, np.sum(occ, axis=1), '.')
plt.xscale('log')

plt.show()

dfm_rad, dfm_per = dfm(period_grid, rp_grid, period_rng, rp_rng, period, rp, 
                        allSamples, model, summedCompleteness)
rad, per = recreate(period_grid, rp_grid, period_rng, rp_rng, period, rp, 
                     allSamples, model, summedCompleteness)

dfm_rad_num, dfm_per_num = dfm(period_grid, rp_grid, period_rng, rp_rng, period, rp, 
                        allSamples, model, summedCompleteness, cplt=True)
rad_num, per_num = recreate(period_grid, rp_grid, period_rng, rp_rng, period, rp, 
                     allSamples, model, summedCompleteness, cplt=True)

plt.figure()
#plt.fill_between(rp, dfm_rad[0], dfm_rad[4], color="m", alpha=0.1, edgecolor="none")
#plt.fill_between(rp, dfm_rad[1], dfm_rad[3], color="m", alpha=0.3, edgecolor="none")
#plt.plot(rp, dfm_rad[2], label='dfm, radius', linestyle='--', color='m')
plt.fill_between(rp, rad[0], rad[4], color="g", alpha=0.1, edgecolor="none")
plt.fill_between(rp, rad[1], rad[3], color="g", alpha=0.3, edgecolor="none")
plt.plot(rp, rad[2], color='g', label='mine, radius')
plt.legend()
plt.ylabel("dN/dX", fontsize=14)
plt.xlabel("radius", fontsize=14)
plt.xscale('log')

plt.figure()
#plt.fill_between(period, dfm_per[0], dfm_per[4], color="m", alpha=0.1, edgecolor="none")
#plt.fill_between(period, dfm_per[1], dfm_per[3], color="m", alpha=0.3, edgecolor="none")
#plt.plot(period, dfm_per[2], label='dfm, period', linestyle='--', color='m')
plt.fill_between(period, per[0], per[4], color="g", alpha=0.1, edgecolor="none")
plt.fill_between(period, per[1], per[3], color="g", alpha=0.3, edgecolor="none")
plt.plot(period, per[2], color='g', label='mine, period')
plt.legend()
plt.ylabel("dN/dX", fontsize=14)
plt.xlabel("period", fontsize=14)
plt.xscale('log')

plt.show()

plt.figure()
#plt.fill_between(rp, dfm_rad_num[0], dfm_rad_num[4], color="m", alpha=0.1, edgecolor="none")
#plt.fill_between(rp, dfm_rad_num[1], dfm_rad_num[3], color="m", alpha=0.3, edgecolor="none")
#plt.plot(rp, dfm_rad_num[2], label='dfm, radius', linestyle='--', color='m')
plt.fill_between(rp, rad_num[0], rad_num[4], color="g", alpha=0.1, edgecolor="none")
plt.fill_between(rp, rad_num[1], rad_num[3], color="g", alpha=0.3, edgecolor="none")
plt.plot(rp, rad_num[2], color='g', label='mine, radius')

plt.errorbar(r_cent, np.sum(H, axis=0), yerr=np.sqrt(np.sum(H, axis=0)), fmt=".k", label='observed')

plt.legend()
plt.ylabel("num", fontsize=14)
plt.xlabel("radius", fontsize=14)
plt.xscale('log')

plt.figure()
#plt.fill_between(period, dfm_per_num[0], dfm_per_num[4], color="m", alpha=0.1, edgecolor="none")
#plt.fill_between(period, dfm_per_num[1], dfm_per_num[3], color="m", alpha=0.3, edgecolor="none")
#plt.plot(period, dfm_per_num[2], label='dfm, period', linestyle='--', color='m')
plt.fill_between(period, per_num[0], per_num[4], color="g", alpha=0.1, edgecolor="none")
plt.fill_between(period, per_num[1], per_num[3], color="g", alpha=0.3, edgecolor="none")
plt.plot(period, per_num[2], color='g', label='mine, period')

plt.errorbar(p_cent, np.sum(H, axis=1), yerr=np.sqrt(np.sum(H, axis=1)), fmt=".k", label='observed')

plt.legend()
plt.ylabel("num", fontsize=14)
plt.xlabel("period", fontsize=14)
plt.xscale('log')

plt.show()

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

#axs[0, 0].fill_between(rp, dfm_rad_num[0], dfm_rad_num[4], color="m", alpha=0.1, edgecolor="none")
#axs[0, 0].fill_between(rp, dfm_rad_num[1], dfm_rad_num[3], color="m", alpha=0.3, edgecolor="none")
#axs[0, 0].plot(rp, dfm_rad_num[2], label='dfm, radius', linestyle='--', color='m')
axs[0, 0].fill_between(rp, rad_num[0], rad_num[4], color="g", alpha=0.1, edgecolor="none")
axs[0, 0].fill_between(rp, rad_num[1], rad_num[3], color="g", alpha=0.3, edgecolor="none")
axs[0, 0].plot(rp, rad_num[2], color='g', label='mine, radius')
axs[0, 0].errorbar(r_cent, np.sum(H, axis=0), yerr=np.sqrt(np.sum(H, axis=0)), fmt=".k", label='observed')
axs[0, 0].set_xlabel('Radius', fontsize=14)
axs[0, 0].set_ylabel('# detections', fontsize=14)
#axs[0, 0].set_xscale('log')
#axs[0, 0].legend()

#axs[0, 1].fill_between(rp, dfm_rad[0], dfm_rad[4], color="m", alpha=0.1, edgecolor="none")
#axs[0, 1].fill_between(rp, dfm_rad[1], dfm_rad[3], color="m", alpha=0.3, edgecolor="none")
#axs[0, 1].plot(rp, dfm_rad[2], label='dfm, radius', linestyle='--', color='m')
axs[0, 1].fill_between(rp, rad[0], rad[4], color="g", alpha=0.1, edgecolor="none")
axs[0, 1].fill_between(rp, rad[1], rad[3], color="g", alpha=0.3, edgecolor="none")
axs[0, 1].plot(rp, rad[2], color='g', label='mine, radius')
axs[0, 1].scatter(r_cent, np.sum(occ, axis=0), c='k', marker='.')
axs[0, 1].set_xlabel('Radius', fontsize=14)
axs[0, 1].set_ylabel('dN/dlogR', fontsize=14)
#axs[0, 1].set_xscale('log')
#axs[0, 1].legend()

#axs[1, 1].fill_between(period, dfm_per[0], dfm_per[4], color="m", alpha=0.1, edgecolor="none")
#axs[1, 1].fill_between(period, dfm_per[1], dfm_per[3], color="m", alpha=0.3, edgecolor="none")
#axs[1, 1].plot(period, dfm_per[2], label='dfm, period', linestyle='--', color='m')
axs[1, 1].fill_between(period, per[0], per[4], color="g", alpha=0.1, edgecolor="none")
axs[1, 1].fill_between(period, per[1], per[3], color="g", alpha=0.3, edgecolor="none")
axs[1, 1].plot(period, per[2], color='g', label='mine, period')
axs[1, 1].scatter(p_cent, np.sum(occ, axis=1), c='k', marker='.')
axs[1, 1].set_xlabel('Period', fontsize=14)
axs[1, 1].set_ylabel('dN/dlogP', fontsize=14)
#axs[1, 1].set_xscale('log')
#axs[1, 1].legend()

#axs[1,0].fill_between(period, dfm_per_num[0], dfm_per_num[4], color="m", alpha=0.1, edgecolor="none")
#axs[1,0].fill_between(period, dfm_per_num[1], dfm_per_num[3], color="m", alpha=0.3, edgecolor="none")
#axs[1,0].plot(period, dfm_per_num[2], label='dfm, period', linestyle='--', color='m')
axs[1,0].fill_between(period, per_num[0], per_num[4], color="g", alpha=0.1, edgecolor="none")
axs[1,0].fill_between(period, per_num[1], per_num[3], color="g", alpha=0.3, edgecolor="none")
axs[1,0].plot(period, per_num[2], color='g', label='mine, period')
axs[1,0].errorbar(p_cent, np.sum(H, axis=1), yerr=np.sqrt(np.sum(H, axis=1)), fmt=".k", label='observed')
axs[1,0].set_xlabel('Period', fontsize=14)
axs[1,0].set_ylabel('# detected', fontsize=14)
#axs[1,0].set_xscale('log')
#axs[1,0].legend()

plt.show()

