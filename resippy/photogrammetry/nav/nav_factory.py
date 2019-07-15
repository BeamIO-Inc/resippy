from resippy.photogrammetry.nav.applanix_sbet_nav import ApplanixSBETNav


class NavFactory:

    @staticmethod
    def from_applanix_sbet_file(filename    # type: str
                                ):          # type: (...) -> ApplanixSBETNav
        applanix_sbet_nav = ApplanixSBETNav()
        applanix_sbet_nav.load_from_file(filename)
        return applanix_sbet_nav
