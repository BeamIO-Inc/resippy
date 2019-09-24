from resippy.photogrammetry.nav.applanix_sbet_nav import ApplanixSBETNav
from resippy.photogrammetry.nav.applanix_eo_nav import ApplanixEONav


class NavFactory:

    @staticmethod
    def from_applanix_sbet_file(filename,   # type: str
                                proj,       # type: str
                                zone,       # type: str
                                ellps,      # type: str
                                datum       # type: str
                                ):          # type: (...) -> ApplanixSBETNav
        applanix_sbet_nav = ApplanixSBETNav()
        applanix_sbet_nav.load_from_file(filename, proj, zone, ellps, datum)
        return applanix_sbet_nav

    @staticmethod
    def from_applanix_eo_file(filename,
                              proj,
                              zone,
                              ellps,
                              datum
                              ):
        applanix_eo_nav = ApplanixEONav()
        applanix_eo_nav.load_from_file(filename, proj, zone, ellps, datum)
        return applanix_eo_nav
