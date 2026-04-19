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

static ThreadPool *g_pool = NULL;
static pthread_once_t g_pool_once = PTHREAD_ONCE_INIT;

static void destroy_pool(void)
{
    if (!g_pool) return;
    pthread_mutex_lock(&g_pool->mtx);
    g_pool->shutdown = 1;
    pthread_cond_broadcast(&g_pool->work_cv);
    pthread_mutex_unlock(&g_pool->mtx);

    for (int i = 0; i < g_pool->n_total - 1; i++)
        pthread_join(g_pool->workers[i], NULL);

    pthread_mutex_destroy(&g_pool->mtx);
    pthread_mutex_destroy(&g_pool->dispatch_mtx);
    pthread_cond_destroy(&g_pool->work_cv);
    pthread_cond_destroy(&g_pool->done_cv);
    free(g_pool);
    g_pool = NULL;
}

static void init_pool(void)
{
    int n = choose_n_threads();
    g_pool = (ThreadPool *)calloc(1, sizeof(*g_pool));
    g_pool->n_total = n;
    pthread_mutex_init(&g_pool->mtx, NULL);
    pthread_mutex_init(&g_pool->dispatch_mtx, NULL);
    pthread_cond_init(&g_pool->work_cv, NULL);
    pthread_cond_init(&g_pool->done_cv, NULL);

    for (int i = 0; i < n - 1; i++) {
        struct WorkerCtx *ctx = (struct WorkerCtx *)malloc(sizeof(*ctx));
        ctx->pool = g_pool;
        ctx->tid  = i + 1;
        if (pthread_create(&g_pool->workers[i], NULL, worker_main, ctx) != 0) {
            fprintf(stderr, "pk_pool: pthread_create failed\n");
            free(ctx);
            g_pool->n_total = i + 1;
            break;
        }
    }

    atexit(destroy_pool);
}

ThreadPool *pk_pool(void)
{
    pthread_once(&g_pool_once, init_pool);
    return g_pool;
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
