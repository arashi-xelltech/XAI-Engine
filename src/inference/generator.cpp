#include "inference/generator.h"
#include "inference/forward.h"
#include "model/model_loader.h"    // <-- ДОБАВЛЕНО для полного определения Model
#include "model/tensor_ops.h"
#include "inference/sampler.h"
#include "model/tokenizer.h"
#include <cstdio>
#include <cmath>
#include <ctime>
#include <vector>

namespace xai {

double time_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

std::string json_escape(const std::string &s) {
    std::string out; out.reserve(s.size() * 2);
    for (char c : s) {
        if      (c == '"' || c == '\\') { out += '\\'; out += c; }
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else if ((unsigned char)c >= 0x20) out += c;
    }
    return out;
}

void json_print_escaped(FILE *f, const char *s) {
    for (int i = 0; s[i]; i++) {
        char c = s[i];
        if      (c == '"' || c == '\\') { fputc('\\', f); fputc(c, f); }
        else if (c == '\n') { fputc('\\', f); fputc('n', f); }
        else if (c == '\r') { fputc('\\', f); fputc('r', f); }
        else if (c == '\t') { fputc('\\', f); fputc('t', f); }
        else if ((unsigned char)c >= 0x20) fputc(c, f);
    }
}

GenerateResult generate(Model *m, RunState *s,
                        const int *prompt, int plen,
                        int max_new, float temp, int top_k,
                        float top_p, float rep_p,
                        bool ignore_eos, bool streaming) {
    GenerateResult result;
    result.prompt_tokens = plen;

    std::vector<int> history(prompt, prompt + plen);
    history.reserve(plen + max_new);

    bool is_greedy = (temp < 1e-6f);
    bool no_rep    = (fabsf(rep_p - 1.0f) < 1e-6f);

    double t0 = time_sec();
    forward_batch(m, s, prompt, 0, plen);
    result.prefill_time = time_sec() - t0;

    int pos = plen;

    if (!no_rep)
        apply_repetition_penalty(s->logits, m->cfg.vocab_size,
                                 history.data(), (int)history.size(), rep_p);

    int next;
    if (is_greedy) {
        next = 0;
        for (int i = 1; i < m->cfg.vocab_size; i++)
            if (s->logits[i] > s->logits[next]) next = i;
    } else {
        next = sample_token(s->logits, m->cfg.vocab_size, temp, top_k, top_p);
    }

    history.push_back(next);
    char buf[256];
    tok_decode_one(&m->tok, next, buf, sizeof(buf));
    result.text += buf;
    if (streaming) { printf("%s", buf); fflush(stdout); }
    result.generated_tokens++;

    double t_gen_start = time_sec();

    for (int i = 1; i < max_new; i++) {
        if (!ignore_eos && next == m->tok.eos_id) { result.finish_reason = "eos"; break; }
        if (pos >= m->cfg.max_seq_len - 1)        { result.finish_reason = "max_context"; break; }

        if (is_greedy && no_rep) {
            forward_transformer(m, s, next, pos++);
            next = wt_matmul_argmax(&m->lm_head, s->x);
        } else {
            forward(m, s, next, pos++);
            if (!no_rep)
                apply_repetition_penalty(s->logits, m->cfg.vocab_size,
                                         history.data(), (int)history.size(), rep_p);
            if (is_greedy) {
                next = 0;
                for (int j = 1; j < m->cfg.vocab_size; j++)
                    if (s->logits[j] > s->logits[next]) next = j;
            } else {
                next = sample_token(s->logits, m->cfg.vocab_size, temp, top_k, top_p);
            }
        }

        history.push_back(next);
        tok_decode_one(&m->tok, next, buf, sizeof(buf));
        result.text += buf;
        if (streaming) { printf("%s", buf); fflush(stdout); }
        result.generated_tokens++;
    }

    result.gen_time = time_sec() - t_gen_start;
    return result;
}

void print_result_json(const GenerateResult &r, const char *prompt) {
    printf("{\n");
    printf("  \"prompt\": \""); json_print_escaped(stdout, prompt); printf("\",\n");
    printf("  \"prompt_tokens\": %d,\n", r.prompt_tokens);
    printf("  \"generated_text\": \""); json_print_escaped(stdout, r.text.c_str()); printf("\",\n");
    printf("  \"generated_tokens\": %d,\n", r.generated_tokens);
    printf("  \"prefill_time_sec\": %.4f,\n", r.prefill_time);
    printf("  \"prefill_tokens_per_sec\": %.2f,\n",
           r.prefill_time > 0 ? r.prompt_tokens / r.prefill_time : 0.0);
    printf("  \"generation_time_sec\": %.4f,\n", r.gen_time);
    printf("  \"generation_tokens_per_sec\": %.2f,\n",
           r.gen_time > 0 ? r.generated_tokens / r.gen_time : 0.0);
    printf("  \"finish_reason\": \"%s\"\n", r.finish_reason.c_str());
    printf("}\n");
}

void print_result_text(const GenerateResult &r) {
    printf("\n\n[prefill: %d tok in %.2fs (%.1f tok/s)"
           " | gen: %d tok in %.2fs (%.1f tok/s)]\n",
           r.prompt_tokens, r.prefill_time,
           r.prefill_time > 0 ? r.prompt_tokens / r.prefill_time : 0.0,
           r.generated_tokens, r.gen_time,
           r.gen_time > 0 ? r.generated_tokens / r.gen_time : 0.0);
}

std::string build_json_response(const GenerateResult &r,
                                const std::string &prompt) {
    std::string ep = json_escape(prompt), et = json_escape(r.text);
    char buf[4096];
    snprintf(buf, sizeof(buf),
             "{\"prompt\":\"%s\",\"prompt_tokens\":%d,"
             "\"generated_text\":\"%s\",\"generated_tokens\":%d,"
             "\"prefill_time_sec\":%.4f,\"prefill_tokens_per_sec\":%.2f,"
             "\"generation_time_sec\":%.4f,\"generation_tokens_per_sec\":%.2f,"
             "\"finish_reason\":\"%s\"}",
             ep.c_str(), r.prompt_tokens, et.c_str(), r.generated_tokens,
             r.prefill_time,
             r.prefill_time > 0 ? r.prompt_tokens / r.prefill_time : 0.0,
             r.gen_time,
             r.gen_time > 0 ? r.generated_tokens / r.gen_time : 0.0,
             r.finish_reason.c_str());
    return buf;
}

} // namespace xai