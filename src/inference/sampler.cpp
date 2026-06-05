#include "inference/sampler.h"
#include "core/simd_ops.h"
#include "core/vector_ops.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace xai {

static uint64_t g_rng = 1234;
static std::vector<PI> g_sample_buf;
static std::vector<uint8_t> g_rep_seen;

void rng_seed(uint64_t s) { g_rng = s; }

float rng_f32() {
    g_rng ^= g_rng >> 12;
    g_rng ^= g_rng << 25;
    g_rng ^= g_rng >> 27;
    return (float)((g_rng * 0x2545F4914F6CDD1Dull) >> 40) / 16777216.0f;
}

void quickselect_topk(PI *arr, int n, int k) {
    if (k >= n) return;
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (arr[mid].v > arr[lo].v) std::swap(arr[lo], arr[mid]);
        if (arr[hi].v  > arr[lo].v) std::swap(arr[lo], arr[hi]);
        if (arr[mid].v > arr[hi].v) std::swap(arr[mid], arr[hi]);
        float pivot = arr[hi].v;
        int i = lo, j = hi - 1;
        while (true) {
            while (i <= j && arr[i].v >  pivot) i++;
            while (i <= j && arr[j].v <= pivot) j--;
            if (i > j) break;
            std::swap(arr[i++], arr[j--]);
        }
        std::swap(arr[i], arr[hi]);
        if (i == k) break;
        else if (i < k) lo = i + 1;
        else hi = i - 1;
    }
}

int sample_token(float *logits, int vs, float temp, int top_k, float top_p) {
    if (temp < 1e-6f) {
        int best = 0;
        for (int i = 1; i < vs; i++) if (logits[i] > logits[best]) best = i;
        return best;
    }
    simd::vec_scale(logits, 1.0f / temp, vs);
    softmax(logits, vs);

    int k = (top_k <= 0 || top_k > vs) ? vs : top_k;
    if ((int)g_sample_buf.size() < vs) g_sample_buf.resize(vs);
    PI *pi = g_sample_buf.data();
    for (int i = 0; i < vs; i++) { pi[i].v = logits[i]; pi[i].i = i; }

    if (k < vs) {
        quickselect_topk(pi, vs, k);
        std::sort(pi, pi+k, [](const PI &a, const PI &b){ return a.v > b.v; });
    } else {
        std::sort(pi, pi+vs, [](const PI &a, const PI &b){ return a.v > b.v; });
    }

    int cutoff = k;
    if (top_p < 1.0f) {
        float cum = 0.0f;
        for (int i = 0; i < k; i++) {
            cum += pi[i].v;
            if (cum >= top_p) { cutoff = i + 1; break; }
        }
    }
    float sum = 0.0f;
    for (int i = 0; i < cutoff; i++) sum += pi[i].v;
    float r = rng_f32() * sum, cdf = 0.0f;
    int chosen = pi[0].i;
    for (int i = 0; i < cutoff; i++) {
        cdf += pi[i].v;
        if (r <= cdf) { chosen = pi[i].i; break; }
    }
    return chosen;
}

void apply_repetition_penalty(float *logits, int vocab_size,
                              const int *history, int hist_len,
                              float rep_p) {
    if (rep_p == 1.0f || hist_len == 0) return;
    if ((int)g_rep_seen.size() < vocab_size) g_rep_seen.resize(vocab_size, 0);
    for (int j = 0; j < hist_len; j++) {
        int tid = history[j];
        if (tid >= 0 && tid < vocab_size) g_rep_seen[tid] = 1;
    }
    for (int j = 0; j < vocab_size; j++) {
        if (g_rep_seen[j]) {
            if (logits[j] > 0) logits[j] /= rep_p;
            else               logits[j] *= rep_p;
        }
    }
    for (int j = 0; j < hist_len; j++) {
        int tid = history[j];
        if (tid >= 0 && tid < vocab_size) g_rep_seen[tid] = 0;
    }
}

} // namespace xai