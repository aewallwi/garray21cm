
def cov_mat_simulated(
    ndraws=1000,
    compress_by_redundancy=False,
    output_dir="./",
    mode="gsm",
    nside_sky=defaults.nside_sky,
    clobber=True,
    order_by_bl_length=False,
    return_uvdata=False,
    **array_config_kwargs,
):
    """Estimate a bootstrapped gsm covariance matrix using random rotations of GlobalSkyModel.

    Parameters
    ----------
    ndraws: int, optional
        number of realizations of the sky to derive covariance from.
        default is 1000
    compress_by_redundancy: bool, optional
        if True, only compute covariance for one baseline per redundant group.
    output_dir: str, optional
        where to write template container
    mode: str, optional
        'gsm' for gsm or 'eor' for random fluctuations.
    nside_sky: int, optional
        nsides of healpix sky to simulate.
    clobber: bool, optional
        if true, overwrite existing files. If not, dont.
        only applies to templaet data.
    return uvdata: bool, optional
        if true, return uvdata object as well

    Returns
    -------
    cov-mat: array-like
        covariance matrix that is (Nfreqs * Nbls) x (Nfreqs * Nbls)
        derived from randomly drawing a simulated sky.
    if return_uvdata:
        UVData object with all of the metadata / data shape.
    """
    uvdata, beams, beam_ids = visibilities.initialize_uvdata(
        output_dir=output_dir, clobber=clobber, **array_config_kwargs
    )
    if mode == "gsm":
        signalcube = visibilities.initialize_gsm(uvdata.freq_array[0], nside_sky=nside_sky)
    if compress_by_redundancy:
        uvdata_compressed = uvdata.compress_by_redundancy(tol=0.25 * 3e8 / uvdata.freq_array.max(), inplace=False)
        nblfrqs = uvdata_compressed.Nbls * uvdata_compressed.Nfreqs
        data_inds = np.where(uvdata_compressed.time_array == uvdata.time_array[0])[0]
        if order_by_bl_length:
            data_inds = data_inds[np.argsort(np.abs(uvdata_compressed.uvw_array[data_inds, 0]))]
    else:
        data_inds = np.where(uvdata.time_array == uvdata.time_array[0])[0]
        nblfrqs = uvdata.Nbls * uvdata.Nfreqs
        if order_by_bl_length:
            data_inds = data_inds[np.argsort(np.abs(uvdata.uvw_array[data_inds, 0]))]

    cov_mat = np.zeros((nblfrqs, nblfrqs), dtype=complex)
    mean_mat = np.zeros(nblfrqs, dtype=complex)

    for draw in tqdm.tqdm(range(ndraws)):
        # generate random rotation
        if mode == "gsm":
            rot = hp.Rotator(
                rot=(
                    np.random.rand() * 360,
                    (np.random.rand() * 180 - 90),
                    np.random.rand() * 360,
                )
            )
            signalcube = np.asarray(rot.rotate_map_pixel(signalcube))
        else:
            signalcube = visibilities.initialize_eor(uvdata.freq_array[0], nside_sky=nside_sky)
        uvdata.data_array[:] = 0.0
        simulator = vis_cpu.VisCPU(
            uvdata=uvdata,
            sky_freqs=uvdata.freq_array[0],
            beams=beams,
            beam_ids=beam_ids,
            sky_intensity=signalcube,
        )
        simulator.simulate()
        if compress_by_redundancy:
            uvdata_compressed = simulator.uvdata.compress_by_redundancy(
                tol=0.25 * 3e8 / uvdata.freq_array.max(), inplace=False
            )
            data_vec = uvdata_compressed.data_array[data_inds, 0, :, 0].flatten()
        else:
            data_vec = simulator.uvdata.data_array[data_inds, 0, :, 0].flatten()
        mean_mat += data_vec
        cov_mat += np.outer(data_vec, np.conj(data_vec))
    mean_mat = mean_mat / ndraws
    cov_mat = cov_mat / ndraws - np.outer(mean_mat, np.conj(mean_mat))
    if return_uvdata:
        if compress_by_redundancy:
            return uvdata_compressed, cov_mat
        else:
            return cov_mat, uvdata
    else:
        return cov_mat
