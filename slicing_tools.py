import holoviews as hv


def spatial_im(ds, group, dynamic=True):
    return ds.to(
        hv.Image, kdims=['x', 'y'],
        group=group, label='spatial',
        dynamic=dynamic
        )


def spectra(ds, group, dynamic=True):
    return ds.to(
        hv.Curve, kdims='wavelength',
        group=group, label='pixel spectrum',
        dynamic=dynamic
        )


def mean_std(ds, dynamic=True):
    mean = ds.to(
        hv.Curve, kdims='wavelength',
        label='crop mean spectrum', 
        dynamic=dynamic
        )
    spread = ds.to(
        hv.Spread, kdims='wavelength',
        label='crop std',
        dynamic=dynamic
        )
    return mean * spread


def slicer(ds, group, dynamic=True):
    im = spatial_im(ds, group, dynamic=dynamic) 
    s = spectra(ds, group, dynamic=dynamic)
    return im + s


def slicer_with_mean(ds, mean_ds, group, dynamic=True):
    sopts = {
        'Curve': {'color': 'red'}
        }
    pixel = spectra(ds, group, dynamic=dynamic).options(sopts)
    mean = mean_std(mean_ds, dynamic=dynamic)
    im = spatial_im(ds, group, dynamic=dynamic)
    layout = im + pixel * mean
    return layout