MI Kernels
==========

# Requirements

You will need the following things installed:

 - Python >= 2.7 (2.7 preferred; other versions untested)
 - NumPy
 - SciPy
 - scikit-learn (at least v.0.12)
 - cherrypy
 - sqlite3 (with Python support)
 - possibly other things I'm forgetting...

# Overview

In order to run many experiments across multiple machines, this experimental
setup includes an experiment server and client. The server `./src/server.py`
takes a list a configuration file and a directory to store results. You will
need to set up a results directory first (you can call it whatever you like):

    $ mkdir results

Currently, the `.gitignore` file contains an entry for `results` so that it will
not show up in version control. Since the results can be several gigabytes in
size, you will not want to commit any results to the repository. You can add
entries to the `.gitignore` file so that your results directory does not appear.

Then, you can test the server with:

    $ ./src/server.py config/test.yaml results

The YAML-formatted configuration file (`test.yaml`) contains a list of
experimental settings to run. The server will take a while the first time it is
started, since it is generating the required experiments ("tasks") and building
a database to store the results. After the server is started, you can start up a
client on any machine like so:

    $ ./src/client.py [URL or IP of server]

Assuming the server is accessible from the client machine, the client will begin
making requests to the server. The server will respond with an experiment to
run, and the client will begin running the experiment. When it is finished, it
will submit the results back to the server and request another experiment. You
can start up as many clients as you like in parallel. You can check the status
of the experiments by navigating to `http://url.of.myserver.com:2112/` from a
browser (when the server is running).

When the experiments are finished, you can stop the clients and server. Then,
you can compute the statistics from the results as in the following example:

    $ ./src/statistics.py config/test.yaml results test auc auc_stats.csv

This will compute the test-set AUC (area under ROC curve) for the appropriate
results and store them `auc_stats.csv` file. Another option is to use `train`
for training-set statistics or `accuracy` for accuracy.

The scripts all checks for existing results, so if you add more experiments and
rerun the server, only the new tasks will be performed. Similarly, statistics
are only computed when the corresponding line in the `auc_stats.csv` file is not
already present.

To make the statistics computation even easier, there is a script called
`update_stats.py` that will compute statistics for a list of experiments
(passing the appropriate arguments to the `statistics.py` script). The list of
desired arguments is stored in a file in the config folder (I have created
`config/stats.config` for this purpose). You just have to call it like so:

    $ ./src/update_stats.py config/stats.config

Also note, it is important to run all scripts from the root directory of the
repository, since the data/fold/etc. directories are hard-coded into the
scripts relative to the working directory.

# Relevant Code

For introducing new kernel function, the only code you will need to modify is in
`src/kernels.py`. Each kernel is a class. The `__init__` constructor method
takes a set of parameters and creates class variables using the appropriate
parameters. Then, the `__call__` method is implemented to make the class
instance callable as a function of two arguments. For set kernels, the two
arguments are Python lists of NumPy arrays. Each array is a bag in which the
rows are the individual feature vectors, and each column is a feature.

Typically, a kernel function is defined between two instances. In this case, we
are interested in a kernel function defined between two bags. However, in
practice, we need to compute a kernel matrix, which is a matrix consisting of
pairwise kernel values between two lists of bags. Thus, the kernel matrix is
what the `__call__` method should compute. The entry Kij should contain the
(individual) kernel function between bag i from the first list and j from the
second list. The two lists need not be the same, so Kij need not be a square
matrix. If you like, you can define some other method like so:

    def k(self, X, Y):
        # Returns kernel between individual bags X and Y

Then succinctly implement `__call__` using:

    def __call__(self, B, C):
        return np.array([[self.k(X, Y) for Y in C] for X in B])

However, note that when using NumPy (much like MATLAB), it is much more
efficient to use matrix computations than using loops, since NumPy uses
fast C and Fortran libraries to perform matrix computations, while the Python
loops are relatively slow. But for things like the box-counting kernel, you
might not have a choice.

# Notes

Originally, I ran the miGraph experiments and put the results in:

    /home/gary/shared/mi_kernel_results

However, this version of the code was using *squared* Euclidean distance rather
than simple Euclidean distance to compute the affinities. The code has been
corrected, and the new results are in:

    /home/gary/shared/migraph_results

3/28/14: It turns out that the authors actually were using squared Euclidean
distance, so we will revert back to the original results.

11/5/14: I adjusted the range of the gamma2 parameter for the twolevel kernel;
the results are stored in the twolevel2 files. On average, the performance did
not change significantly, but the select values now fall well within the range.
