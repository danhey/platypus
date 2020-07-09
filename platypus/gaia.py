from __future__ import division, print_function

from astropy.coordinates import SkyCoord, Angle
from astropy.stats import sigma_clip
from astropy.utils.exceptions import AstropyUserWarning
import astropy.units as u
import numpy as np

__all__ = ['get_nearby_gaia']

def get_nearby_gaia(tpf, magnitude_limit=20):
    """Make the Gaia Figure Elements"""
    # Get the positions of the Gaia sources
    c1 = SkyCoord(tpf.ra, tpf.dec, frame='icrs', unit='deg')
    # Use pixel scale for query size
    pix_scale = 4.0  # arcseconds / pixel for Kepler, default
    if tpf.mission == 'TESS':
        pix_scale = 21.0
    # We are querying with a diameter as the radius, overfilling by 2x.
    from astroquery.vizier import Vizier
    Vizier.ROW_LIMIT = -1
    result = Vizier.query_region(c1, catalog=["I/345/gaia2"],
                                 radius=Angle(np.max(tpf.shape[1:]) * pix_scale, "arcsec"))
    no_targets_found_message = ValueError('Either no sources were found in the query region '
                                          'or Vizier is unavailable')
    too_few_found_message = ValueError('No sources found brighter than {:0.1f}'.format(magnitude_limit))
    empty = dict(x=None, y=None, size=None)
    if result is None:
        return empty
    elif len(result) == 0:
        return empty
    result = result["I/345/gaia2"].to_pandas()
    result = result[result.Gmag < magnitude_limit]
    if len(result) == 0:
        return empty

    # Apply correction for proper motion
    year = ((tpf.astropy_time[0].jd - 2457206.375) * u.day).to(u.year)
    pmra = ((np.nan_to_num(np.asarray(result.pmRA)) * u.milliarcsecond/u.year) * year).to(u.deg).value
    pmdec = ((np.nan_to_num(np.asarray(result.pmDE)) * u.milliarcsecond/u.year) * year).to(u.deg).value
    result.RA_ICRS += pmra
    result.DE_ICRS += pmdec

    # Convert to pixel coordinates
    radecs = np.vstack([result['RA_ICRS'], result['DE_ICRS']]).T
    coords = tpf.wcs.all_world2pix(radecs, 0)

    # Gently size the points by their Gaia magnitude
    sizes = 64.0 / 2**(result['Gmag']/5.0)
    one_over_parallax = 1.0 / (result['Plx']/1000.)
    source = dict(ra=result['RA_ICRS'],
                                        dec=result['DE_ICRS'],
                                        pmra=result['pmRA'],
                                        pmde=result['pmDE'],
                                        source=result['Source'].astype(str),
                                        Gmag=result['Gmag'],
                                        plx=result['Plx'],
                                        one_over_plx=one_over_parallax,
                                        x=coords[:, 0] + tpf.column,
                                        y=coords[:, 1] + tpf.row,
                                        size=sizes)

    return source