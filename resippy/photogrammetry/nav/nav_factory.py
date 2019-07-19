from resippy.photogrammetry.nav.applanix_sbet_nav import ApplanixSBETNav


class NavFactory:

    @staticmethod
    def from_applanix_sbet_file(filename,       # type: str
                                proj='utm',     # type: str
                                zone=None,      # type: str
                                ellps='WGS84',  # type: str
                                datum='WGS84'   # type: str
                                ):          # type: (...) -> ApplanixSBETNav
        applanix_sbet_nav = ApplanixSBETNav()
        applanix_sbet_nav.load_from_file(filename, proj, zone, ellps, datum)
        return applanix_sbet_nav
