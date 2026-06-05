#pragma once
#include <string>

namespace xai {

struct Model;
struct RunState;

struct GenerateResult {
    std::string text;
    int    prompt_tokens    = 0;
    int    generated_tokens = 0;
    double prefill_time     = 0;
    double gen_time         = 0;
    std::string finish_reason = "length";
};

GenerateResult generate(Model *m, RunState *s,
                        const int *prompt, int plen,
                        int max_new, float temp, int top_k,
                        float top_p, float rep_p,
                        bool ignore_eos, bool streaming);

// Timing helper
double time_sec();

// JSON escape
std::string json_escape(const std::string &s);
void json_print_escaped(FILE *f, const char *s);

// Output formatting
void print_result_json(const GenerateResult &r, const char *prompt);
void print_result_text(const GenerateResult &r);
std::string build_json_response(const GenerateResult &r, const std::string &prompt);

} // namespace xai