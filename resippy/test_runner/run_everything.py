import resippy.test_runner.run_all_timings as run_all_timings
import resippy.test_runner.run_all_demos as run_all_demos


if __name__ == '__main__':
    run_all_timings.main()
    run_all_demos.main()
    # these will get run automatically when imported
    import resippy.test_runner.run_all_tests
    import resippy.test_runner.run_all_demo_tests
