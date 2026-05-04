/*  threadpool.c – Minimal fork-join pthread pool. See threadpool.h. */

#include "threadpool.h"
#include <pthread.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define PK_MAX_THREADS 16

struct ThreadPool {
    int         n_total;     /* total participants: N-1 workers + 1 main */
    pthread_t   workers[PK_MAX_THREADS - 1];

    pthread_mutex_t mtx;
    pthread_cond_t  work_cv;    /* signalled when new work is posted */
    pthread_cond_t  done_cv;    /* signalled when all workers finish */

    pthread_mutex_t dispatch_mtx; /* serialises concurrent pk_parallel() calls */

    ParallelFn  fn;
    void       *arg;
    unsigned    gen;            /* generation counter; increments per job */
    int         n_done;         /* workers finished in current gen */
    int         shutdown;
};

/* Per-thread context passed to worker_main. */
struct WorkerCtx {
    ThreadPool *pool;
    int         tid;            /* 1..N-1 */
};

static void *worker_main(void *arg_)
{
    struct WorkerCtx *ctx = (struct WorkerCtx *)arg_;
    ThreadPool *p = ctx->pool;
    int tid = ctx->tid;
    unsigned last_gen = 0;

    for (;;) {
        pthread_mutex_lock(&p->mtx);
        while (!p->shutdown && p->gen == last_gen)
            pthread_cond_wait(&p->work_cv, &p->mtx);
        if (p->shutdown) {
            pthread_mutex_unlock(&p->mtx);
            free(ctx);
            return NULL;
        }
        last_gen = p->gen;
        ParallelFn fn = p->fn;
        void *arg = p->arg;
        pthread_mutex_unlock(&p->mtx);

        fn(tid, p->n_total, arg);

        pthread_mutex_lock(&p->mtx);
        p->n_done++;
        if (p->n_done == p->n_total - 1)
            pthread_cond_signal(&p->done_cv);
        pthread_mutex_unlock(&p->mtx);
    }
}

/* Determine thread count from env var / cpu count. */
static int choose_n_threads(void)
{
    const char *env = getenv("PK_THREADS");
    if (env) {
        int n = atoi(env);
        if (n < 1) n = 1;
        if (n > PK_MAX_THREADS) n = PK_MAX_THREADS;
        return n;
    }

    long nprocs = sysconf(_SC_NPROCESSORS_ONLN);
    if (nprocs < 1) nprocs = 1;
    if (nprocs > PK_MAX_THREADS) nprocs = PK_MAX_THREADS;
    return (int)nprocs;
}

/* Build a pool of n_total participants (1 main + n_total-1 workers). */
static ThreadPool *make_pool(int n_total)
{
    if (n_total < 1) n_total = 1;
    if (n_total > PK_MAX_THREADS) n_total = PK_MAX_THREADS;

    ThreadPool *p = (ThreadPool *)calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->n_total = n_total;
    pthread_mutex_init(&p->mtx, NULL);
    pthread_mutex_init(&p->dispatch_mtx, NULL);
    pthread_cond_init(&p->work_cv, NULL);
    pthread_cond_init(&p->done_cv, NULL);

    for (int i = 0; i < n_total - 1; i++) {
        struct WorkerCtx *ctx = (struct WorkerCtx *)malloc(sizeof(*ctx));
        if (!ctx) {
            p->n_total = i + 1;
            break;
        }
        ctx->pool = p;
        ctx->tid  = i + 1;
        if (pthread_create(&p->workers[i], NULL, worker_main, ctx) != 0) {
            fprintf(stderr, "pk_pool: pthread_create failed\n");
            free(ctx);
            p->n_total = i + 1;
            break;
        }
    }
    return p;
}

static void teardown_pool(ThreadPool *p)
{
    if (!p) return;
    pthread_mutex_lock(&p->mtx);
    p->shutdown = 1;
    pthread_cond_broadcast(&p->work_cv);
    pthread_mutex_unlock(&p->mtx);

    for (int i = 0; i < p->n_total - 1; i++)
        pthread_join(p->workers[i], NULL);

    pthread_mutex_destroy(&p->mtx);
    pthread_mutex_destroy(&p->dispatch_mtx);
    pthread_cond_destroy(&p->work_cv);
    pthread_cond_destroy(&p->done_cv);
    free(p);
}

/* ── Process-wide singleton ── */

static ThreadPool *g_pool = NULL;
static pthread_once_t g_pool_once = PTHREAD_ONCE_INIT;

static void destroy_global_pool(void)
{
    teardown_pool(g_pool);
    g_pool = NULL;
}

static void init_global_pool(void)
{
    g_pool = make_pool(choose_n_threads());
    atexit(destroy_global_pool);
}

/* ── Thread-local override ──
 * Lets long-running hosts (e.g. servers handling many concurrent sessions)
 * shard sessions across independent pools to bypass the dispatch_mtx that
 * serialises pk_parallel() calls within a single pool. */
static __thread ThreadPool *g_tls_pool = NULL;

ThreadPool *pk_pool(void)
{
    if (g_tls_pool) return g_tls_pool;
    pthread_once(&g_pool_once, init_global_pool);
    return g_pool;
}

void pk_set_pool(ThreadPool *p)
{
    g_tls_pool = p;
}

ThreadPool *pk_pool_create(int n_threads)
{
    return make_pool(n_threads);
}

void pk_pool_destroy(ThreadPool *p)
{
    teardown_pool(p);
}

int pk_pool_nthreads(ThreadPool *p)
{
    return p->n_total;
}

void pk_parallel(ThreadPool *p, ParallelFn fn, void *arg)
{
    if (p->n_total <= 1) {
        fn(0, 1, arg);
        return;
    }

    /* Serialise concurrent callers so only one job is in-flight at a time. */
    pthread_mutex_lock(&p->dispatch_mtx);

    /* Post work to all workers. */
    pthread_mutex_lock(&p->mtx);
    p->fn = fn;
    p->arg = arg;
    p->n_done = 0;
    p->gen++;
    pthread_cond_broadcast(&p->work_cv);
    pthread_mutex_unlock(&p->mtx);

    /* Main thread participates as tid=0. */
    fn(0, p->n_total, arg);

    /* Wait for the N-1 workers to finish. */
    pthread_mutex_lock(&p->mtx);
    while (p->n_done < p->n_total - 1)
        pthread_cond_wait(&p->done_cv, &p->mtx);
    pthread_mutex_unlock(&p->mtx);

    pthread_mutex_unlock(&p->dispatch_mtx);
}
