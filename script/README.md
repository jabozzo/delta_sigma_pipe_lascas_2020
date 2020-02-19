# Paper graphs

This is the script that generated the graphs in "A blind calibration scheme for switched-capacitor pipeline analog-to-digital converters". This script requires the `calib` package, use with Python 3.7.3.

Since results take about two days to compute, the original data is included and a switch in the script (`JUST_PLOT`) can be changed in order to use that data and skip computation.


## Installation

    pip3 install -r <path/to/calib/repo/requirements.txt>
    pip3 install <path/to/calib/repo>
    pip3 install -r requirements.txt

## Usage

Sorry, console arguments were not implemented. To customize the graphs edit the file and the following switches can be changed:

`TOGETHER`: (True/FALSE) Plot all the experiments in one graph

`TESTS`: (a tuple with all of some of numbers 0, 1 and 2) Which test to simulate. 0 is for deadband size, 1 is for threshold noise and 2 is for reference noise.

`S_TAU`: (A float >= 0) Amount of static variation in the amplifier loss. This is models the standard deviation of a normal distribution in the time domain. Then this distribution is converted to a lognormal distribution by assuming linear settling. The unit is time - constants (as in exp(-x/tau)).

`PLOT_STD`: (True/False) Include in the plot the standard deviation of the monte-carlo results in the plots.

`DRY_RUN`: (True/False) Useful for debugging and checking parameters, runs the script but skips the estimation part. The result is just the same as no calibration. Use to test the script does not crash.

More fine-graned parameters can be found on the `if __name__ == "__main__":` section at the end of the script.


After tweaking, run the script and wait (or not if pre-processed data is used)

    python3 paper_graphs.py

## Author

This script was made by Juan Andrés Bozzo for his Master of Engineering Sciences. Feel free to contact me at [jabozzo@uc.cl](jabozzo@uc.cl)
