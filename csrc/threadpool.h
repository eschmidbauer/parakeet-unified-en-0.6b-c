/*  threadpool.h – Minimal pthread-based worker pool for parallel-for.
 *
 *  Design:
 *    • N-1 background worker threads + the calling thread participates as
 *      tid=0. No external deps beyond POSIX pthreads (libSystem on macOS,
 *      glibc on Linux).
 *    • Fork-join semantics: pk_parallel() returns only after all workers
 *      finish the current job.
 *    • Pool is a lazy-init singleton — first call to pk_pool() spawns threads.
 *    • Thread count: clamp(PK_THREADS env var, 1, min(NPROCS_ONLN, 8)).
 *      Default is 4 on machines with >=4 cores.
 */
#ifndef PK_THREADPOOL_H
#define PK_THREADPOOL_H

typedef struct ThreadPool ThreadPool;

/* Signature of work functions passed to pk_parallel.
 * `tid` is in [0, nthreads), `nthreads` is the total worker count
 * (including the calling thread). */
typedef void (*ParallelFn)(int tid, int nthreads, void *arg);

/* Return the process-wide singleton pool. Lazy-initialised on first call. */
ThreadPool *pk_pool(void);

/* Number of workers in the pool (including the calling thread). */
int pk_pool_nthreads(ThreadPool *p);

/* Dispatch `fn` to all workers. The calling thread runs as tid=0 in parallel
 * with the pool workers (tid=1..N-1). Returns after all finish. */
void pk_parallel(ThreadPool *p, ParallelFn fn, void *arg);

#endif /* PK_THREADPOOL_H */
