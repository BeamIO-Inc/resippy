from resippy.atmospheric_compensation.utils import quick_calculations


def main():
    cloud_height = 30000
    cloud_height_units = 'ft'
    cloud_extent = quick_calculations.arm_extent_at_zero_elevation(cloud_height, cloud_height_units)
    stop = 1


if __name__ == '__main__':
    main()
