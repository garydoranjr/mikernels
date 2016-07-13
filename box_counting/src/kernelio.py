"""
Handles reading/writing kernel values
"""
import thread

KERNEL_MANAGERS = {}

def get_kernel_manager(kernel_path):
    if kernel_path not in KERNEL_MANAGERS:
        KERNEL_MANAGERS[kernel_path] = KernelManager(kernel_path)
    return KERNEL_MANAGERS[kernel_path]

class KernelManager(object):

    def __init__(self, kernel_path):
        self.kernel_path = kernel_path
        self.connection_thread = None
        self._connection = None
        self.dataset_ids = {}
        self.ktype_ids = {}
        self._init_database()

    def _init_database(self):
        connection = self.get_connection()
        with connection:
            # Create tables and indices
            connection.execute(
                'CREATE TABLE IF NOT EXISTS datasets '
                '(dataset_id integer primary key, '
                'dataset_name text unique)'
            )
            connection.execute(
                'CREATE INDEX IF NOT EXISTS dataset_name_index '
                'ON datasets (dataset_name)'
            )
            connection.execute(
                'CREATE TABLE IF NOT EXISTS ktypes '
                '(ktype_id integer primary key, '
                'ktype_name text unique)'
            )
            connection.execute(
                'CREATE INDEX IF NOT EXISTS ktype_name_index '
                'ON ktypes (ktype_name)'
            )

            connection.execute(
                'CREATE TABLE IF NOT EXISTS kernel '
                '(dataset_id integer, ktype_id integer, '
                'epsilon real, delta real, seed integer, '
                'i integer, j integer, '
                'mantissa real, exponent real, time real)'
            )
            connection.execute(
                'CREATE INDEX IF NOT EXISTS kernel_index '
                'ON kernel (dataset_id, ktype_id, '
                'epsilon, delta, seed, i, j)'
            )

        # Load existing name -> id mappings
        cursor = connection.cursor()

        cursor.execute('SELECT dataset_id, dataset_name FROM datasets')
        for did, dname in cursor.fetchall():
            self.dataset_ids[dname] = did

        cursor.execute('SELECT ktype_id, ktype_name FROM ktypes')
        for tid, tname in cursor.fetchall():
            self.ktype_ids[tname] = tid

    def get_dataset_id(self, dataset_name):
        if dataset_name not in self.dataset_ids:
            connection = self.get_connection()
            with connection:
                connection.execute(
                    'INSERT INTO datasets '
                    'VALUES (NULL, ?)',
                    (dataset_name,)
                )
                cursor = connection.cursor()
                cursor.execute(
                    'SELECT dataset_id '
                    'FROM datasets '
                    'WHERE dataset_name=?',
                    (dataset_name,)
                )
                for did in cursor.fetchall():
                    # Should only ever be 1 result by uniqueness
                    self.dataset_ids[dataset_name] = did[0]

        return self.dataset_ids[dataset_name]

    def get_ktype_id(self, ktype_name):
        if ktype_name not in self.ktype_ids:
            connection = self.get_connection()
            with connection:
                connection.execute(
                    'INSERT INTO ktypes '
                    'VALUES (NULL, ?)',
                    (ktype_name,)
                )
                cursor = connection.cursor()
                cursor.execute(
                    'SELECT ktype_id '
                    'FROM ktypes '
                    'WHERE ktype_name=?',
                    (ktype_name,)
                )
                for tid in cursor.fetchall():
                    # Should only ever be 1 result by uniqueness
                    self.ktype_ids[ktype_name] = tid[0]

        return self.ktype_ids[ktype_name]

    def get_connection(self):
        current_thread = thread.get_ident()
        if (self._connection is None
            or self.connection_thread != current_thread):
            import sqlite3
            self.connection_thread = current_thread
            self._connection = sqlite3.connect(self.kernel_path)
        return self._connection

    def is_finished(self, dataset, ktype, epsilon, delta, seed, i, j):
        if j < i:
            j, i = i, j
        connection = self.get_connection()
        cursor = connection.cursor()
        dataset_id = self.get_dataset_id(dataset)
        ktype_id = self.get_ktype_id(ktype)
        cursor.execute(
            'SELECT * FROM kernel WHERE '
            'dataset_id=? AND ktype_id=? AND '
            'epsilon=? AND delta=? '
            'AND seed=? AND i=? and j=?',
            (dataset_id, ktype_id, epsilon, delta, seed, i, j)
        )
        return (cursor.fetchone() is not None)

    def store_kernel(self, submission, dataset, ktype,
            epsilon, delta, seed, i, j):
        if j < i:
            j, i = i, j

        dataset_id = self.get_dataset_id(dataset)
        ktype_id = self.get_ktype_id(ktype)
        mantissa = submission['mantissa']
        exponent = submission['exponent']
        time = submission['time']

        records = [
            (dataset_id, ktype_id, epsilon, delta, seed, i, j, mantissa, exponent, time)
        ]

        connection = self.get_connection()
        with connection:
                connection.executemany(
                    'INSERT INTO kernel '
                    'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', records
                )

    def get_kernel(self, dataset, ktype,
            epsilon, delta, seed, i, j):
        if j < i:
            j, i = i, j

        dataset_id = self.get_dataset_id(dataset)
        ktype_id = self.get_ktype_id(ktype)

        connection = self.get_connection()
        cursor = connection.cursor()
        cursor.execute(
            'SELECT mantissa, exponent FROM kernel '
            'WHERE dataset_id=? AND ktype_id=? AND '
            'epsilon=? AND delta=? '
            'AND seed=? AND i=? AND j=?',
            (dataset_id, ktype_id, epsilon, delta, seed, i, j)
        )
        return cursor.fetchone()

    def get_time(self, dataset, ktype,
            epsilon, delta, seed, i, j):
        if j < i:
            j, i = i, j

        dataset_id = self.get_dataset_id(dataset)
        ktype_id = self.get_ktype_id(ktype)

        connection = self.get_connection()
        cursor = connection.cursor()
        cursor.execute(
            'SELECT time FROM kernel '
            'WHERE dataset_id=? AND ktype_id=? AND '
            'epsilon=? AND delta=? '
            'AND seed=? AND i=? AND j=?',
            (dataset_id, ktype_id, epsilon, delta, seed, i, j)
        )
        return cursor.fetchone()[0]

    def get_total_time(self, dataset, ktype,
        epsilon, delta, seed):

        dataset_id = self.get_dataset_id(dataset)
        ktype_id = self.get_ktype_id(ktype)

        connection = self.get_connection()
        cursor = connection.cursor()
        cursor.execute(
            'SELECT SUM(time) FROM kernel '
            'WHERE dataset_id=? AND ktype_id=? AND '
            'epsilon=? AND delta=? AND seed=?',
            (dataset_id, ktype_id, epsilon, delta, seed)
        )
        return cursor.fetchone()[0]
