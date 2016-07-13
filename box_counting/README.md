Box Counting Kernel
===================

The first step is to create inter-valued scaled versions of the C4.5 datasets:

 - For the musk datasets use the original datasets from the UCI repository.
 - For TRX, `scale_c45.py` is used to scale the original TRX by a factor of 10;
   however, since the dataset in `../data` is already scaled by a factor of 10,
   an overall factor of 100 is used to scale trx as in:

    $ ./scale_c45.py ../data/trx.data 100

  - For the text datasets, `../src/scale_data.py` with a factor of 1000 is
    used to scale the datasets (with features noralized first).
  - For the remaining datasets, `../src/scale_data.py` with a factor of 100 is
    used to scale the datasets (with features noralized first).

After scaling datasets, they are converted to the db/spec format using the
`c45_to_dbspec.py` script.

Then, the kernel server and client scripts are used to compute the "and" and
"or" kernels. The `andor.py` script is used to construct the "and/or" kernel
from these kernels.

Next, the `hardmargin.py` script is used to compute the cross validated scores
using a hard margin SVM (alphas equal to 1/diag).

Finally, the empirical kernel is computed using the normal client/server setup
in the directory above.

For the birdsong datasets, all of the kernel matrices are the same, so after
BRCR is finished, the kernel files are simply copied to create the kernel files
for the other datasets.
