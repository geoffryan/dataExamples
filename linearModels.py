import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import scipy.optimize as opt
import emcee
import ptemcee
import corner
import h5py as h5
import autocorr


def evaluatePolyModel(t, x):
    return np.polyval(x[::-1], t)


def genPriorSample(mup, sigp):

    x = mup + sigp * np.random.normal(0.0, 1.0, mup.shape)
    return x


def genData(Ndata, Tmax, x, sigLevel):

    dt = Tmax/Ndata
    t = dt * (0.5 + np.arange(Ndata))

    yexact = evaluatePolyModel(t, x)

    yerr = sigLevel*np.abs(yexact)
    yerr[yerr==0] = sigLevel
    
    dy = np.random.normal(0.0, yerr, t.shape)

    y = yexact + dy

    return t, y, yerr


def logprior(x, mup, sigp):
    N = len(x)
    res = (x-mup)/sigp
    lp = -0.5*N*math.log(2*np.pi) - math.log(sigp.prod()) - 0.5*(res*res).sum()

    return lp


def loglike(x, t, y, sigy):

    N = t.shape[0]

    res = (evaluatePolyModel(t, x) - y)/sigy
    chi2 = (res*res).sum()

    lp = -0.5*N*math.log(2*np.pi) - np.log(sigy).sum() - 0.5*chi2

    return lp


def logpost(x, t, y, sigy, mup, sigp):

    lp = logprior(x, mup, sigp)
    lp += loglike(x, t, y, sigy)

    return lp


def logprior_opt(x, *args, **kwargs):
    return -logprior(x, *args, **kwargs)


def loglike_opt(x, *args, **kwargs):
    return -loglike(x, *args, **kwargs)


def logpost_opt(x, *args, **kwargs):
    return -logpost(x, *args, **kwargs)


def getDataPars(t, y, sigy):
   
    sigy2 = sigy*sigy
    isigy2 = 1.0/(sigy2)
    wD = 0.5*(y*isigy2*y).sum()

    detcovD = np.exp(np.log(sigy2).sum())

    print((sigy2 == 0.0).any())

    print(np.log(sigy2).sum())
    print(detcovD)

    return detcovD, wD


def getLikePars(ndim, t, y, sigy):

    tp = np.ones(t.shape)
    f = np.empty((ndim, t.shape[0]))
    f[0, :] = tp
    for i in range(1, ndim):
        tp *= t
        f[i, :] = tp

    isigy2 = 1.0/(sigy*sigy)

    f_icovd_f = (f[:, None, :] * isigy2[None, None, :] * f[None, :, :]
                 ).sum(axis=2)
    y_icovd_f = (y[None, :] * isigy2[None, :] * f[:, :]).sum(axis=1)

    icovL = 0.5*(f_icovd_f + f_icovd_f.T)  # Just to ensure symmetry
    covL = np.linalg.inv(icovL)
    muL = (covL[:, :] * y_icovd_f[None, :]).sum(axis=1)

    detcovL = np.linalg.det(covL)
    wL = 0.5 * (muL[:, None] * icovL[:, :] * muL[None, :]).sum()

    return muL, covL, icovL, detcovL, wL


def getPriorPars(mup, sigp):

    sigp2 = sigp*sigp
    isigp2 = 1.0/(sigp2)

    covp = np.diag(sigp2)
    icovp = np.diag(isigp2)

    wp = 0.5 * (mup * isigp2 * mup).sum()
    detcovp = sigp2.prod()

    return mup, covp, icovp, detcovp, wp


def getBetaPars(beta, mup, icovp, muL, icovL):

    scalar = False
    if np.isscalar(beta):
        scalar = True
    beta = np.atleast_1d(beta)

    icovB = icovp[None, ...] + beta[:, None, None]*icovL[None, ...]
    covB = np.linalg.inv(icovB)
    detcovB = np.linalg.det(covB)

    muB = (covB[:, :, :] * (beta[:, None]
                               * np.matmul(icovL, muL)[None, :]
                               + np.matmul(icovp, mup)[None, :]
                               )[:, None, :]).sum(axis=2)

    wB = (muB[:, :, None] * icovB[:, :, :] * muB[:, None, :]).sum(axis=(1, 2))
    wB *= 0.5

    if scalar:
        return muB[0], covB[0], icovB[0], detcovB[0], wB[0]

    return muB, covB, icovB, detcovB, wB


def getZBeta(beta, Ndata, detcovD, detcovp, detcovB, wD, wp, wB):

    ZB = np.power(2*np.pi, -0.5*beta*Ndata)
    ZB *= np.sqrt(detcovB/(np.power(detcovD, beta) * detcovp))
    ZB *= np.exp(wB - beta*wD - wp)

    return ZB


def getDlogZDBeta(beta, Ndata, detcovD, muL, muB, icovL, covB, wD, wL):

    scalar = False
    if np.isscalar(beta):
        scalar = True
    beta = np.atleast_1d(beta)

    if scalar:
        dlzdb = -0.5*Ndata*math.log(2*math.pi)
    else:
        dlzdb = np.empty(beta.shape)
        dlzdb[:] = -0.5*Ndata*math.log(2*math.pi)

    dmuBmuL = muB - muL[None, :]

    dlzdb -= 0.5*math.log(detcovD)
    dlzdb -= 0.5*np.trace(np.matmul(icovL[None, ...], covB), axis1=1, axis2=2)
    dlzdb -= 0.5*(dmuBmuL[:, :, None] * icovL[None, :, :]
                  * dmuBmuL[:, None, :]).sum(axis=(1, 2))
    dlzdb -= wD
    dlzdb += wL

    return dlzdb


def sampleEmcee(t, y, ye, mup, sigp, Tmax, nwalkers=20, nsteps=1000,
                nburn=None, sampleFile=None):

    ndim = len(mup)
    ndata = len(t)

    if nburn is None:
        nburn = nsteps//4

    doTheSampling = True

    if sampleFile is not None:
        with h5.File(sampleFile, "a") as f:
            if 'emcee/chain' in f and 'emcee/lnprobability' in f:
                chain = f['emcee/chain'][...]
                lnprobability = f['emcee/lnprobability'][...]
                try:
                    assert chain.shape == (nwalkers, nsteps, ndim)
                    assert lnprobability.shape == (nwalkers, nsteps)
                    samps = chain.reshape((-1, ndim))
                    lnprobs = lnprobability.reshape((-1, ))
                    doTheSampling = False
                except AssertionError:
                    pass
        

    if doTheSampling:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, logpost,
                                        args=(t, y, ye, mup, sigp))

        p0 = mup[None, :] + np.random.normal(0.0, 1.0e-4, (nwalkers, ndim))

        if nburn > 0:
            for i, result in enumerate(sampler.sample(p0, iterations=nburn,
                                       storechain=False)):
                print("Burn in {0:d} steps: {1:.1f}%"
                      .format(nburn, 100*(i+1)/nburn), end='\r')
            print('')
            sampler.reset()
        else:
            result = (p0, )

        for i, result in enumerate(sampler.sample(*result, iterations=nsteps,
                                   storechain=True)):
            print("Sampling {0:d} steps: {1:.1f}%".format(
                  nsteps, 100*(i+1)/nsteps),
                  end='\r')
        print('')

        chain = sampler.chain
        samps = sampler.flatchain
        lnprobs = sampler.flatlnprobability
        lnprobability = sampler.lnprobability

        if sampleFile is not None:
            f = h5.File(sampleFile, 'a')
            if 'emcee/chain' in f:
                f['emcee/chain'].resize(chain.shape)
                f['emcee/chain'][...] = chain[...]
            else:
                f.create_dataset('emcee/chain', data=chain,
                                 maxshape=(None, None, None))
            
            if 'emcee/lnprobability' in f:
                f['emcee/lnprobability'].resize(lnprobability.shape)
                f['emcee/lnprobability'][...] = lnprobability[...]
            else:
                f.create_dataset('emcee/lnprobability', data=lnprobability,
                                 maxshape=(None, None))
            f.close()

    labels = ['C{0:01d}'.format(i) for i in range(ndim)] 

    fig = corner.corner(samps, labels=labels)
    figname = "emcee_corner.png"
    print("Saving", figname)
    fig.savefig(figname)
    plt.close(fig)

    for i in range(ndim):
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        for j in range(nwalkers):
            ax.plot(chain[j, :, i], alpha=2.0/nwalkers, color='k')
        ax.set_xlabel('# Iterations')
        ax.set_ylabel(labels[i])
        
        figname = "emcee_trace_" + labels[i] + ".png"
        print("Saving", figname)
        fig.savefig(figname)
        plt.close(fig)
    imap = lnprobs.argmax()

    xmap = samps[imap]
    means = samps.mean(axis=0)
    diffs = samps-means
    cov = (diffs[:, :, None]*diffs[:, None, :]).mean(axis=0)

    tau = autocorr.integrated_time(chain, timeAxis=1, walkerAxis=0)

    print("Emcee AutoCorrTau:", tau)

    return xmap, means, cov, samps, lnprobs


def sampleEmceePT(t, y, ye, mup, sigp, Tmax, nwalkers=100, nsteps=1000,
                  nburn=None, ntemps=20, sampleFile=None, maxTemp=1e4):

    ndim = len(mup)
    ndata = len(t)

    if nburn is None:
        nburn = nsteps//4

    doTheSampling = True

    if sampleFile is not None:
        with h5.File(sampleFile, "a") as f:
            if ('emcee-pt/chain' in f
                    and 'emcee-pt/lnprobability' in f
                    and 'emcee-pt/lnlikelihood' in f
                    and 'emcee-pt/betas' in f):
                chain = f['emcee-pt/chain'][...]
                lnprobability = f['emcee-pt/lnprobability'][...]
                lnlikelihood = f['emcee-pt/lnlikelihood'][...]
                betas = f['emcee-pt/betas'][...]
                try:
                    assert chain.shape == (ntemps, nwalkers, nsteps, ndim)
                    assert lnprobability.shape == (ntemps, nwalkers, nsteps)
                    assert lnlikelihood.shape == (ntemps, nwalkers, nsteps)
                    assert betas.shape == (ntemps,)
                    chain = chain
                    samps = chain[0].reshape((-1, ndim))
                    lnprobs = lnprobability[0].reshape((-1, ))
                    lnlikes = lnlikelihood
                    doTheSampling = False
                except AssertionError:
                    pass

    if doTheSampling:

        if ntemps % 2 == 0:
            betas = np.geomspace(1.0/maxTemp, 1.0, ntemps)[::-1]
            betas = np.concatenate((betas, [0.0]))
        else:
            betas = np.geomspace(1.0/maxTemp, 1.0, ntemps-1)[::-1]
            betas = np.concatenate((betas, [0.0]))

        sampler = emcee.PTSampler(ntemps, nwalkers, ndim, loglike, logprior,
                                  loglargs=(t, y, ye), logpargs=(mup, sigp),
                                  betas=betas)

        p0 = mup[None, None, :] + np.random.normal(0.0, 1.0e-4,
                                                   (ntemps, nwalkers, ndim))

        if nburn > 0:
            for i, result in enumerate(sampler.sample(p0, iterations=nburn,
                                       storechain=False)):
                print("Burn in {0:d} steps: {1:.1f}%"
                      .format(nburn, 100*(i+1)/nburn), end='\r')
            print('')
            sampler.reset()
        else:
            result = (p0, )

        for i, result in enumerate(sampler.sample(*result, iterations=nsteps,
                                   storechain=True)):
            print("Sampling {0:d} steps: {1:.1f}%".format(nsteps,
                  100*(i+1)/nsteps),
                  end='\r')
        print('')

        chain = sampler.chain
        samps = sampler.flatchain[0]
        lnprobs = sampler.lnprobability[0].reshape((-1, ))
        lnlikes = sampler.lnlikelihood
        betas = sampler.betas

        if sampleFile is not None:
            f = h5.File(sampleFile, 'a')
            if 'emcee-pt/chain' in f:
                f['emcee-pt/chain'].resize(sampler.chain.shape)
                f['emcee-pt/chain'][...] = sampler.chain[...]
            else:
                f.create_dataset('emcee-pt/chain', data=sampler.chain,
                                 maxshape=(None, None, None, None))
            
            if 'emcee-pt/lnprobability' in f:
                f['emcee-pt/lnprobability'].resize(sampler.lnprobability.shape)
                f['emcee-pt/lnprobability'][...] = sampler.lnprobability[...]
            else:
                f.create_dataset('emcee-pt/lnprobability',
                                 data=sampler.lnprobability,
                                 maxshape=(None, None, None))
            
            if 'emcee-pt/lnlikelihood' in f:
                f['emcee-pt/lnlikelihood'].resize(sampler.lnlikelihood.shape)
                f['emcee-pt/lnlikelihood'][...] = sampler.lnlikelihood[...]
            else:
                f.create_dataset('emcee-pt/lnlikelihood',
                                 data=sampler.lnlikelihood,
                                 maxshape=(None, None, None))
            
            if 'emcee-pt/betas' in f:
                f['emcee-pt/betas'].resize(betas.shape)
                f['emcee-pt/betas'][...] = betas[...]
            else:
                f.create_dataset('emcee-pt/betas', data=betas,
                                 maxshape=(None,))
            f.close()

    labels = ['C{0:01d}'.format(i) for i in range(ndim)] 

    fig = corner.corner(samps, labels=labels)
    figname = "emceePT_corner.png"
    print("Saving", figname)
    fig.savefig(figname)
    plt.close(fig)

    # for k in range(ntemps):
    for k in [0, ntemps-1]:
        for i in range(ndim):
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            for j in range(nwalkers):
                ax.plot(chain[k, j, :, i], alpha=2.0/nwalkers, color='k')
            ax.set_xlabel('# Iterations')
            ax.set_ylabel(labels[i])
            
            figname = "emceePT_trace_T{0:01d}_{1:s}.png".format(k, labels[i])
            print("Saving", figname)
            fig.savefig(figname)
            plt.close(fig)

    imap = lnprobs.argmax()

    taus = autocorr.integrated_time(chain, timeAxis=2, walkerAxis=1)
    lnlike_taus = autocorr.integrated_time(lnlikes, timeAxis=2, walkerAxis=1)
    print("emceePT AutoCorrTau:", taus)
    print("emceePT AutoCorrTau logLike:", lnlike_taus)

    lnlike_adj = lnlikes - lnlikes.mean(axis=(1, 2), keepdims=True)
    lnlike_var = (lnlike_adj*lnlike_adj).mean(axis=(1, 2))

    xmap = samps[imap]
    means = samps.mean(axis=0)
    diffs = samps-means
    cov = (diffs[:, :, None]*diffs[:, None, :]).mean(axis=0)

    avglnl = lnlikes.mean(axis=(1, 2))[::-1]
    avglnl_err = np.sqrt(lnlike_taus/(nsteps*nwalkers) * lnlike_var)[::-1]
    betas = betas[::-1]
    if betas[0] > 0.0:
        betas = np.concatenate(([0.0], betas))
        avglnl = np.concatenate(([avglnl[0]], avglnl))
        avglnl_err = np.concatenate(([avglnl_err[0]], avglnl_err))

    return xmap, means, cov, samps, lnprobs, avglnl, betas, avglnl_err


def samplePtemcee(t, y, ye, mup, sigp, Tmax, nwalkers=100, nsteps=1000,
                  nburn=None, ntemps=21, sampleFile=None, maxTemp=np.inf):

    ndim = len(mup)
    ndata = len(t)

    if nburn is None:
        nburn = nsteps//4

    doTheSampling = True
    betas = None

    if sampleFile is not None:
        with h5.File(sampleFile, "a") as f:
            if ('ptemcee/chain' in f
                    and 'ptemcee/lnprobability' in f
                    and 'ptemcee/lnlikelihood' in f
                    and 'ptemcee/betas' in f):
                chain = f['ptemcee/chain'][...]
                lnprobability = f['ptemcee/lnprobability'][...]
                lnlikelihood = f['ptemcee/lnlikelihood'][...]
                betas = f['ptemcee/betas'][...]
                try:
                    assert chain.shape == (ntemps, nwalkers, nsteps, ndim)
                    assert lnprobability.shape == (ntemps, nwalkers, nsteps)
                    assert lnlikelihood.shape == (ntemps, nwalkers, nsteps)
                    assert betas.shape == (ntemps,)
                    chain = chain
                    samps = chain[0].reshape((-1, ndim))
                    lnprobs = lnprobability[0].reshape((-1, ))
                    lnlikes = lnlikelihood
                    doTheSampling = False
                except AssertionError:
                    pass

    if doTheSampling:

        if betas is None:
            betas = ptemcee.make_ladder(ndim, ntemps, maxTemp)

        sampler = ptemcee.Sampler(nwalkers, ndim, loglike, logprior,
                                  logl_args=(t, y, ye), logp_args=(mup, sigp),
                                  betas=betas, adaptive=True)

        p0 = mup[None, None, :] + np.random.normal(0.0, 1.0e-4,
                                                   (ntemps, nwalkers, ndim))

        if nburn > 0:
            for i, result in enumerate(sampler.sample(p0, iterations=nburn,
                                       storechain=False)):
                print("Burn in {0:d} steps: {1:.1f}%"
                      .format(nburn, 100*(i+1)/nburn), end='\r')
            print('')
            sampler.reset()
        else:
            result = (p0, )

        for i, result in enumerate(sampler.sample(*result, iterations=nsteps,
                                   storechain=True)):
            print("Sampling {0:d} steps: {1:.1f}%".format(nsteps,
                  100*(i+1)/nsteps),
                  end='\r')
        print('')

        chain = sampler.chain
        samps = sampler.flatchain[0]
        lnprobs = sampler.lnprobability[0].reshape((-1, ))
        lnlikes = sampler.lnlikelihood
        betas = sampler.betas

        if sampleFile is not None:
            f = h5.File(sampleFile, 'a')
            if 'ptemcee/chain' in f:
                f['ptemcee/chain'].resize(sampler.chain.shape)
                f['ptemcee/chain'][...] = sampler.chain[...]
            else:
                f.create_dataset('ptemcee/chain', data=sampler.chain,
                                 maxshape=(None, None, None, None))
            
            if 'ptemcee/lnprobability' in f:
                f['ptemcee/lnprobability'].resize(sampler.lnprobability.shape)
                f['ptemcee/lnprobability'][...] = sampler.lnprobability[...]
            else:
                f.create_dataset('ptemcee/lnprobability',
                                 data=sampler.lnprobability,
                                 maxshape=(None, None, None))
            
            if 'ptemcee/lnlikelihood' in f:
                f['ptemcee/lnlikelihood'].resize(sampler.lnlikelihood.shape)
                f['ptemcee/lnlikelihood'][...] = sampler.lnlikelihood[...]
            else:
                f.create_dataset('ptemcee/lnlikelihood',
                                 data=sampler.lnlikelihood,
                                 maxshape=(None, None, None))
            
            if 'ptemcee/betas' in f:
                f['ptemcee/betas'].resize(betas.shape)
                f['ptemcee/betas'][...] = betas[...]
            else:
                f.create_dataset('ptemcee/betas', data=betas,
                                 maxshape=(None,))
            f.close()

    labels = ['C{0:01d}'.format(i) for i in range(ndim)] 

    fig = corner.corner(samps, labels=labels)
    figname = "emceePT_corner.png"
    print("Saving", figname)
    fig.savefig(figname)
    plt.close(fig)

    # for k in range(ntemps):
    for k in [0, ntemps-1]:
        for i in range(ndim):
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            for j in range(nwalkers):
                ax.plot(chain[k, j, :, i], alpha=2.0/nwalkers, color='k')
            ax.set_xlabel('# Iterations')
            ax.set_ylabel(labels[i])
            
            figname = "emceePT_trace_T{0:01d}_{1:s}.png".format(k, labels[i])
            print("Saving", figname)
            fig.savefig(figname)
            plt.close(fig)

    imap = lnprobs.argmax()

    taus = autocorr.integrated_time(chain, timeAxis=2, walkerAxis=1)
    lnlike_taus = autocorr.integrated_time(lnlikes, timeAxis=2, walkerAxis=1)
    print("emceePT AutoCorrTau:", taus)
    print("emceePT AutoCorrTau logLike:", lnlike_taus)

    lnlike_adj = lnlikes - lnlikes.mean(axis=(1, 2), keepdims=True)
    lnlike_var = (lnlike_adj*lnlike_adj).mean(axis=(1, 2))

    xmap = samps[imap]
    means = samps.mean(axis=0)
    diffs = samps-means
    cov = (diffs[:, :, None]*diffs[:, None, :]).mean(axis=0)

    avglnl = lnlikes.mean(axis=(1, 2))[::-1]
    avglnl_err = np.sqrt(lnlike_taus/(nsteps*nwalkers) * lnlike_var)[::-1]
    betas = betas[::-1]
    if betas[0] > 0.0:
        betas = np.concatenate(([0.0], betas))
        avglnl = np.concatenate(([avglnl[0]], avglnl))
        avglnl_err = np.concatenate(([avglnl_err[0]], avglnl_err))

    return xmap, means, cov, samps, lnprobs, avglnl, betas, avglnl_err


def comparePosts(mu_true, mu_exact, cov_exact, mu_num, cov_num,
                 samps_emcee, samps_emceePT, sigLimit=4):

    ndim = len(mu_true)

    icov_num = np.linalg.inv(cov_num)
    detcov_num = np.linalg.det(cov_num)

    sig_exact = np.sqrt(np.diagonal(cov_exact))
    sig_num = np.sqrt(np.diagonal(cov_exact))

    labels = ['C{0:01d}'.format(i) for i in range(ndim)] 

    for i in range(ndim):

        x_ex = np.linspace(mu_exact[i] - sigLimit*sig_exact[i],
                           mu_exact[i] + sigLimit*sig_exact[i], 100)
        x_nu = np.linspace(mu_num[i] - sigLimit*sig_num[i],
                           mu_num[i] + sigLimit*sig_num[i], 100)

        p_ex = math.pow(2*np.pi*cov_exact[i, i], -0.5) * np.exp(
            -0.5 * (x_ex-mu_exact[i])*(x_ex-mu_exact[i]) / (cov_exact[i, i]))
        p_nu = math.pow(2*np.pi*cov_num[i, i], -0.5) * np.exp(
            -0.5 * (x_nu-mu_num[i])*(x_nu-mu_num[i]) / (cov_num[i, i]))
                            
        fig, ax = plt.subplots(1, 1)
        ax.plot(x_ex, p_ex, label='Exact')
        ax.plot(x_nu, p_nu, label='Numerical Optimization')
        ax.hist(samps_emcee[:, i], bins=20, histtype='step', density=True,
                label='emcee Sampling')
        ax.hist(samps_emceePT[:, i], bins=20, histtype='step', density=True,
                label='emcee-pt Sampling')

        ax.set_xlabel(labels[i])
        ax.set_ylabel('p({0:s} | Data)'.format(labels[i]))

        ax.legend()

        figname = "post_1d_{0:s}.png".format(labels[i])
        print("Saving", figname)
        fig.savefig(figname)

        plt.close(fig)


def integrateZbeta_direct(beta, t, y, ye, mup, sigp, Nx=20):

    scalar = False
    if np.isscalar(beta):
        scalar = True

    beta = np.atleast_1d(beta)

    ndim =  mup.shape[0]
    ranges = np.empty((ndim, 2))
    ranges[:, 0] = mup - 3*sigp
    ranges[:, 1] = mup + 3*sigp

    res = np.empty(beta.shape)
    err = np.empty(beta.shape)
    
    for i, be in enumerate(beta):
        print("integrating:", i, be)
        def f(*x):
            return np.exp(logprior(x, mup, sigp) + be * loglike(x, t, y, ye))

        res[i], err[i] = scipy.integrate.nquad(f, ranges, opts={'limit': Nx})

    if scalar:
        return res[0], err[0]
    return res, err


def integrateDlogZDbeta_direct(beta, t, y, ye, mup, sigp, ZBeta, ZBeta_err,
                               Nx=10):

    scalar = False
    if np.isscalar(beta):
        scalar = True

    beta = np.atleast_1d(beta)

    ndim =  mup.shape[0]
    ranges = np.empty((ndim, 2))
    ranges[:, 0] = mup - 3*sigp
    ranges[:, 1] = mup + 3*sigp

    res = np.empty(beta.shape)
    err = np.empty(beta.shape)
    

    for i, be in enumerate(beta):
        print("integrating:", i, be)
        def f(*x):
            logl = loglike(x, t, y, ye)
            return logl * np.exp(logprior(x, mup, sigp) + be * logl)
        res[i], err[i] = scipy.integrate.nquad(f, ranges, opts={'limit': Nx})

    ZBeta = np.atleast_1d(ZBeta)
    ZBeta_err = np.atleast_1d(ZBeta_err)

    dlZdB = res / ZBeta
    dlZdB_err = dlZdB * np.sqrt((ZBeta_err/ZBeta)**2 + (err/res)**2)

    if scalar:
        return dlZdB[0], dlZdB_err[0]
    return dlZdB, dlZdB_err


def integrate_trap_f(x, f):
    dx = x[1:]-x[:-1]
    fc = 0.5*(f[:-1] + f[1:])
    I = (fc * dx).sum()
    return I

def integrate_trap(x, f):

    I1 = integrate_trap_f(x[::2], f[::2])
    I2 = integrate_trap_f(x, f)

    return I2, abs(I2-I1)


def integrate_trap_err(x, f, ferr):

    I, err1 = integrate_trap(x, f)

    fvar = ferr*ferr
    err2 = math.sqrt(integrate_trap_f(x, fvar))

    return I, math.sqrt(err1*err1 + err2*err2)


def integrate_simp(x, f):

    N = len(x)
    if N%2 == 0:
        raise ValueError("Need an odd # of sample points!")

    h1 = x[1:-1:2]-x[:-1:2]
    h2 = x[2::2]-x[1:-1:2]
    h = h1+h2

    I1_terms = 0.5*(f[2::2]+f[:-1:2]) * h
    I2_terms = 0.5*(f[1:-1:2]+f[:-1:2]) * h1 + 0.5*(f[2::2]+f[1:-1:2]) * h2
    
    err_terms = (h*h-3*h1*h2)/(3*h1*h2)*(I2_terms-I1_terms)

    I2 = I2_terms.sum()
    err = err_terms.sum()

    I = I2 + err

    return I, err


def integrate_simp_err(x, f, ferr):

    I, err1 = integrate_simp(x, f)

    fvar = ferr*ferr
    err2 = math.sqrt(integrate_simp(x, fvar)[0])

    return I, math.sqrt(err1*err1 + err2*err2)


def analyzeEfficiency(t, y, ye, mup, sigp):


    nd = len(mup)

    # emcee
    tau_emcee = []
    nt_emcee = []
    nw_emcee = []
    af_emcee = []
    tau_emceePT = []
    nt_emceePT = []
    nw_emceePT = []
    af_emceePT = []
    taf_emceePT = []

    nb = 500
    nt = 5000
    nws = [10, 20, 40, 80, 160]

    for nw in nws:
        print("emcee - nw = {0:d}".format(nw))
        sampler = emcee.EnsembleSampler(nw, nd, logpost,
                                        args=(t, y, ye, mup, sigp))

        p0 = mup[None, :] + np.random.normal(0.0, 1.0e-4, (nw, nd))

        for i, result in enumerate(sampler.sample(p0, iterations=nb,
                                   storechain=False)):
            print("Burn in {0:d} steps: {1:.1f}%"
                  .format(nb, 100*(i+1)/nb), end='\r')
        burnedState = result
        print('')
        sampler.reset()

        for i, result in enumerate(sampler.sample(*burnedState,
                                   iterations=nt, storechain=True)):
            print("Sampling {0:d} steps: {1:.1f}%".format(
                  nt, 100*(i+1)/nt),
                  end='\r')
        print('')

        fig, ax = plt.subplots(nd, 1)
        for i in range(nd):
            for j in range(nw):
                ax[i].plot(sampler.chain[j, :, i], color='k',
                           alpha=2/nw)
            ax[i].set_ylabel('C{0:01d}'.format(i))
        figname = "emcee_tau_trace_nw_{0:03d}.png".format(nw)
        print("Saving", figname)
        fig.savefig(figname)
        plt.close(fig)

        try:
            tau = autocorr.integrated_time(sampler.chain,
                                           timeAxis=1, walkerAxis=0)
        except ValueError:
            tau = np.array([nt]*nd)

        tau_emcee.append(tau.mean())
        nt_emcee.append(sampler.chain.shape[1])
        nw_emcee.append(sampler.chain.shape[0])
        af_emcee.append(sampler.acceptance_fraction.mean())


    tau_emcee = np.array(tau_emcee)
    nt_emcee = np.array(nt_emcee)
    nw_emcee = np.array(nw_emcee)
    af_emcee = np.array(af_emcee)

    fig, ax = plt.subplots(3, 1)
    ax[0].plot(nw_emcee, nt_emcee*nw_emcee/tau_emcee, marker='.')
    ax[1].plot(nw_emcee, 1.0/tau_emcee, marker='.')
    ax[2].plot(nw_emcee, af_emcee, marker='.')
   
    ax[0].set_xscale('log')
    ax[1].set_xscale('log')
    ax[2].set_xscale('log')
    ax[0].set_yscale('log')

    ax[2].set_xlabel('# of walkers')
    ax[0].set_ylabel(r'$N_{eff}$')
    ax[1].set_ylabel(r'$\eta = N_{eff} / N$')
    ax[2].set_ylabel(r'acceptance fraction')

    figname = "emcee_efficiency.png"
    print("Saving", figname)
    fig.savefig(figname)
    plt.close(fig)


if __name__ == "__main__":

    sampleFile = None
    if len(sys.argv) > 1:
        sampleFile = sys.argv[1]

    generateData = True

    if sampleFile is not None:
        try:
            with h5.File(sampleFile, 'a') as f:
                f = h5.File(sampleFile, 'r')
                t = f['data/t'][...]
                y = f['data/y'][...]
                ye = f['data/ye'][...]
                Tmax = f['data/Tmax'][0]
                mup = f['prior/mu'][...]
                sigp = f['prior/sig'][...]
                xtrue = f['true/x'][...]
            Ndata = t.shape[0]
            Ndim = mup.shape[0]
            generateData = False
        except (IOError, KeyError):
            pass

    if generateData:
        Ndim = 5
        Ndata = 100
        Tmax = 1.5

        mup = np.array([1.0, 0.5, 0.0, 0.0, 0.0])[:Ndim]
        sigp = np.array([0.3, 0.5, 0.1, 0.1, 0.1])[:Ndim]

        xtrue = genPriorSample(mup, sigp)
        t, y, ye = genData(Ndata, Tmax, xtrue, 0.1)

        if sampleFile is not None:
            f = h5.File(sampleFile, 'a')
            for name, val in zip(['data/t', 'data/y', 'data/ye', 'data/Tmax',
                                  'prior/mu', 'prior/sig', 'true/x'],
                                 [t, y, ye, Tmax, mup, sigp, xtrue]):
                val = np.atleast_1d(val)
                if name in f:
                    f[name].resize(val.shape)
                    f[name][...] = val[...]
                else:
                    f.create_dataset(name, data=val,
                                     maxshape=[None]*len(val.shape))
            f.close()

    T = np.linspace(0, Tmax, 100)


    """
    fig, ax = plt.subplots(1, 1)
    ax.errorbar(t, y, ye, ls='')
    ax.errorbar(t, evaluatePolyModel(t, xtrue), label='True')
    plt.show()
    """

    detcovD, wD = getDataPars(t, y, ye)
    mup, covp, icovp, detcovp, wp = getPriorPars(mup, sigp)
    muL, covL, icovL, detcovL, wL = getLikePars(Ndim, t, y, ye)
    muP, covP, icovP, detcovP, wP = getBetaPars(1.0, mup, icovp, muL, icovL)

    optp = opt.minimize(logprior_opt, mup, args=(mup, sigp))
    optL = opt.minimize(loglike_opt, mup, args=(t, y, ye))
    optP = opt.minimize(logpost_opt, mup, args=(t, y, ye, mup, sigp))

    mup_opt = optp.x
    muL_opt = optL.x
    muP_opt = optP.x
    covp_opt = optp.hess_inv
    covL_opt = optL.hess_inv
    covP_opt = optP.hess_inv

    if not optp.success:
        print(optp)
    if not optL.success:
        print(optL)
    if not optP.success:
        print(optP)

    print("Exact Prior Opt:    ", mup)    
    print("Numerical Prior Opt:", mup_opt)
    print("Exact Like Opt:    ", muL)    
    print("Numerical Like Opt:", muL_opt)
    print("Exact Post Opt:    ", muP)    
    print("Numerical Post Opt:", muP_opt)

    print("Exact Prior Cov:    ", covp)    
    print("Numerical Prior Cov:", covp_opt)
    print("Exact Like Cov:    ", covL)    
    print("Numerical Like Cov:", covL_opt)
    print("Exact Post Cov:    ", covP)    
    print("Numerical Post Cov:", covP_opt)

    fig, ax = plt.subplots(1, 1)
    ax.errorbar(t, y, ye, ls='', marker='.')
    ax.plot(t, evaluatePolyModel(t, xtrue), label='True')
    ax.plot(t, evaluatePolyModel(t, mup), label='Prior')
    ax.plot(t, evaluatePolyModel(t, muL), label='Max Like')
    ax.plot(t, evaluatePolyModel(t, muP), label='Max Post')
    ax.legend()
    figname = "data+best.png"
    print("Saving", figname)
    fig.savefig("data+best.png")
    plt.close(fig)

    # analyzeEfficiency(t, y, ye, mup, sigp)
    resEmcee = sampleEmcee(t, y, ye, mup, sigp, Tmax, nburn=2000,
                           nsteps=30000, sampleFile=sampleFile)

    muP_MAP_em = resEmcee[0]
    muP_mean_em = resEmcee[1]
    covP_em = resEmcee[2]
    samples_em = resEmcee[3]
    logprob_em = resEmcee[4]

    print("Emcee MAP:      ", muP_MAP_em)
    print("Emcee Post Mean:", muP_mean_em)
    print("Emcee Post Cov: ", covP_em)

    resEmceePT = sampleEmceePT(t, y, ye, mup, sigp, Tmax, nburn=500,
                               nwalkers=40, nsteps=5000, ntemps=21,
                               sampleFile=sampleFile)
    muP_MAP_emPT = resEmceePT[0]
    muP_mean_emPT = resEmceePT[1]
    covP_emPT = resEmceePT[2]
    samples_emPT = resEmceePT[3]
    logprob_emPT = resEmceePT[4]
    avglogL_emPT = resEmceePT[5]
    beta_emPT = resEmceePT[6]
    avglogL_err_emPT = resEmceePT[7]

    comparePosts(xtrue, muP, covP, muP_opt, covP_opt, samples_em, samples_emPT)

    beta = np.linspace(0, 1, 1000)
    muB, covB, icovB, detcovB, wB = getBetaPars(beta, mup, icovp, muL, icovL)
    Zbeta = getZBeta(beta, Ndata, detcovD, detcovp, detcovB, wD, wp, wB)

    dlZdB_ex = getDlogZDBeta(beta, Ndata, detcovD, muL, muB, icovL, covB,
                             wD, wL)

    lZBeta_emPT = np.cumsum((0.5*(avglogL_emPT[:-1]+avglogL_emPT[1:])
                            * np.diff(beta_emPT)))
    lZBeta_emPT = np.concatenate(([0.0], lZBeta_emPT))

    lZBeta_var_emPT = np.cumsum(0.5*(avglogL_err_emPT[:-1]**2
                                     + avglogL_err_emPT[1:]**2)
                                * np.diff(beta_emPT))
    lZBeta_var_emPT = np.concatenate(([0.0], lZBeta_var_emPT))
    lZBeta_err_emPT = np.sqrt(lZBeta_var_emPT)

    lZBeta_int = np.cumsum(0.5*(dlZdB_ex[:-1]+dlZdB_ex[1:])
                           * np.diff(beta))
    lZBeta_int = np.concatenate(([0.0], lZBeta_int))

    dlZdB_diff = np.log(Zbeta[1:]/Zbeta[:-1]) / np.diff(beta)
    beta_c = 0.5*(beta[1:]+beta[:-1])

    print("Integrating")

    """
    beta_dir = beta_emPT[::2]
    if beta_dir[-1] != 1.0:
        beta_dir = np.concatenate((beta_dir, [1.0]))
    ZBeta_dir, ZBeta_err_dir = integrateZbeta_direct(
        beta_dir, t, y, ye, mup, sigp)
    lZBeta_dir = np.log(ZBeta_dir)
    lZBeta_err_dir = ZBeta_err_dir / ZBeta_dir

    dlZdB_dir, dlZdB_err_dir = integrateDlogZDbeta_direct(
        beta_dir, t, y, ye, mup, sigp, ZBeta_dir, ZBeta_err_dir)
    """


    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(beta, dlZdB_ex, label='exact')
    ax.errorbar(beta_emPT, avglogL_emPT, avglogL_err_emPT, marker='.', ls='',
                label='emcee-PT')
    ax.plot(beta_c, dlZdB_diff, label='exact-differentiated')
    # ax.errorbar(beta_dir, dlZdB_dir, dlZdB_err_dir, marker='.', ls='',
    #             label='direct integration')
    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel(r'$d \log Z / d\beta  = \langle \log \ell \rangle_\beta$')
    ax.legend()
    ax.set_xlim(0, 1)

    ax.set_xscale('symlog', linthreshx=1e-3)
    # ax.set_yscale('symlog')
    figname = "dlogZdbeta.png"
    print("Saving", figname)
    fig.savefig(figname)
    plt.close(fig)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(beta, np.log(Zbeta), label='exact')
    ax.errorbar(beta_emPT, lZBeta_emPT, lZBeta_err_emPT, ls='', marker='.',
                label='emcee-PT')
    ax.plot(beta, lZBeta_int, label='exact-integrated')
    # ax.errorbar(beta_dir, lZBeta_dir, lZBeta_err_dir, marker='.', ls='',
    #              label='direct integration')
    ax.set_xlim(0, 1)
    ax.set_xscale('symlog', linthreshx=1e-3)
    # ax.set_yscale('symlog')
    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel(r'$\log Z(\beta)$')
    ax.legend()
    figname = "logZbeta.png"
    print("Saving", figname)
    fig.savefig(figname)
    plt.close(fig)

    lZ_emPT_2, lZ_emPT_err_2 = integrate_trap_err(beta_emPT, avglogL_emPT,
                                                  avglogL_err_emPT)
    lZ_emPT_4, lZ_emPT_err_4 = integrate_simp_err(beta_emPT, avglogL_emPT,
                                                  avglogL_err_emPT)

    Z_emPT_2 = math.exp(lZ_emPT_2)
    Z_emPT_err_2 = Z_emPT_2 * lZ_emPT_err_2
    Z_emPT_4 = math.exp(lZ_emPT_4)
    Z_emPT_err_4 = Z_emPT_2 * lZ_emPT_err_4

    print("Evidence Exact:     ", Zbeta[-1])
    # print("Evidence Direct:     ", ZBeta_dir[-1], ZBeta_err_dir[-1])
    print("Evidence emcee-PT_2:", Z_emPT_2, "+/-", Z_emPT_err_2)
    print("Evidence emcee-PT_4:", Z_emPT_4, "+/-", Z_emPT_err_4)
