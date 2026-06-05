#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <memory>
#include <string>
#include <thread>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <vector>
#include <csignal>

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/select.h>
#include <errno.h>
#include <strings.h>

#include "core/config.h"
#include "core/thread_pool.h"
#include "model/model_loader.h"
#include "model/tensor_ops.h"      // <-- ДОБАВЛЕНО для wt_matmul_argmax
#include "model/tokenizer.h"
#include "inference/forward.h"
#include "inference/generator.h"
#include "inference/sampler.h"

using namespace xai;

int xai::g_num_threads    = 1;
int xai::g_sliding_window = 0;

/* ================================================================
 * JSON PARSER FOR SERVER REQUESTS
 * ================================================================ */

struct ServerRequest {
    std::string prompt;
    int   max_tokens  = 256;
    float temperature = 0.8f;
    int   top_k       = 50;
    float top_p       = 0.9f;
    float rep_p       = 1.1f;
    bool  ignore_eos  = false;
    uint64_t seed     = 42;
    bool  seed_set    = false;
};

struct JP { const char *d; int p, len; };

static void jp_ws(JP *j) {
    while (j->p < j->len &&
           (j->d[j->p]==' '||j->d[j->p]=='\t'||
            j->d[j->p]=='\n'||j->d[j->p]=='\r')) j->p++;
}
static char jp_peek(JP *j) { jp_ws(j); return j->p < j->len ? j->d[j->p] : 0; }

static char *jp_str(JP *j) {
    jp_ws(j);
    if (j->d[j->p] != '"') return nullptr;
    j->p++;
    int start = j->p;
    while (j->p < j->len && j->d[j->p] != '"') {
        if (j->d[j->p] == '\\') j->p++;
        j->p++;
    }
    int slen = j->p - start;
    char *s  = (char*)malloc(slen + 1);
    memcpy(s, j->d + start, slen);
    s[slen] = 0;
    j->p++;
    return s;
}

static double jp_num(JP *j) {
    jp_ws(j);
    char buf[64]; int i = 0;
    while (j->p < j->len && i < 63) {
        char c = j->d[j->p];
        if ((c>='0'&&c<='9')||c=='-'||c=='+'||c=='.'||c=='e'||c=='E')
            buf[i++] = c, j->p++;
        else break;
    }
    buf[i] = 0;
    return atof(buf);
}

static void jp_skip(JP *j) {
    jp_ws(j);
    char c = j->d[j->p];
    if (c == '"') { free(jp_str(j)); }
    else if (c == '{') {
        j->p++;
        if (jp_peek(j) != '}') {
            while (1) {
                free(jp_str(j)); jp_ws(j); j->p++;
                jp_skip(j); jp_ws(j);
                if (j->d[j->p] == ',') j->p++; else break;
            }
        }
        jp_ws(j); j->p++;
    }
    else if (c == '[') {
        j->p++;
        if (jp_peek(j) != ']') {
            while (1) {
                jp_skip(j); jp_ws(j);
                if (j->d[j->p] == ',') j->p++; else break;
            }
        }
        jp_ws(j); j->p++;
    }
    else jp_num(j);
}

static ServerRequest parse_server_request(const char *json, int len) {
    ServerRequest req;
    JP j = {json, 0, len};
    if (jp_peek(&j) != '{') return req;
    j.p++;
    while (jp_peek(&j) != '}' && j.p < j.len) {
        char *k = jp_str(&j); if (!k) break;
        jp_ws(&j); if (j.p < j.len && j.d[j.p] == ':') j.p++;

        if      (!strcmp(k,"prompt"))      { char *s=jp_str(&j); if(s){req.prompt=s;free(s);} }
        else if (!strcmp(k,"max_tokens"))  req.max_tokens  = (int)jp_num(&j);
        else if (!strcmp(k,"temperature")) req.temperature = (float)jp_num(&j);
        else if (!strcmp(k,"top_k"))       req.top_k       = (int)jp_num(&j);
        else if (!strcmp(k,"top_p"))       req.top_p       = (float)jp_num(&j);
        else if (!strcmp(k,"rep_p"))       req.rep_p       = (float)jp_num(&j);
        else if (!strcmp(k,"ignore_eos")) {
            jp_ws(&j);
            if (j.d[j.p]=='t'||j.d[j.p]=='T') {
                req.ignore_eos = true;
                while (j.p<j.len && j.d[j.p]!=','&&j.d[j.p]!='}') j.p++;
            } else if (j.d[j.p]=='f'||j.d[j.p]=='F') {
                req.ignore_eos = false;
                while (j.p<j.len && j.d[j.p]!=','&&j.d[j.p]!='}') j.p++;
            } else req.ignore_eos = (int)jp_num(&j) != 0;
        }
        else if (!strcmp(k,"seed")) { req.seed=(uint64_t)jp_num(&j); req.seed_set=true; }
        else jp_skip(&j);

        free(k); jp_ws(&j);
        if (j.p < j.len && j.d[j.p] == ',') j.p++;
    }
    return req;
}

/* ================================================================
 * HTTP SERVER WITH WORKER POOL
 * ================================================================ */

static volatile sig_atomic_t g_server_running = 1;
static void server_sigint(int) { g_server_running = 0; }

struct ServerWorkItem {
    int client_fd;
    std::string client_ip;
    ServerRequest request;
};

class ServerWorkerPool {
public:
    ServerWorkerPool(Model *m, int num_workers)
        : model_(m), stop_(false) {
        for (int i = 0; i < num_workers; i++) {
            workers_.emplace_back([this] { worker_loop(); });
        }
    }

    ~ServerWorkerPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto &w : workers_) w.join();
    }

    void enqueue(ServerWorkItem item) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push(std::move(item));
        }
        cv_.notify_one();
    }

private:
    void worker_loop() {
        while (true) {
            ServerWorkItem item;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this] { return stop_ || !queue_.empty(); });
                if (stop_ && queue_.empty()) return;
                item = std::move(queue_.front());
                queue_.pop();
            }

            // Выполняем генерацию
            RunState s;
            alloc_state(&s, &model_->cfg);

            if (item.request.seed_set) rng_seed(item.request.seed);
            reset_state(&s, &model_->cfg);

            int ids[MAX_SEQ_LEN];
            int n = tok_encode(&model_->tok, item.request.prompt.c_str(),
                               ids, MAX_SEQ_LEN, 0);
            double ts0 = time_sec();
            auto r = generate(model_, &s, ids, n, item.request.max_tokens,
                              item.request.temperature, item.request.top_k,
                              item.request.top_p, item.request.rep_p,
                              item.request.ignore_eos, false);

            std::string resp = build_json_response(r, item.request.prompt);
            free_state(&s);

            // Отправляем ответ
            char header[512];
            int blen = (int)resp.size();
            int hlen = snprintf(header, sizeof(header),
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: application/json\r\n"
                "Content-Length: %d\r\n"
                "Connection: close\r\n"
                "Access-Control-Allow-Origin: *\r\n"
                "\r\n", blen);
            write(item.client_fd, header, hlen);
            if (blen > 0) write(item.client_fd, resp.c_str(), blen);
            close(item.client_fd);

            printf("[%s] POST /generate → 200 (%.2fs)\n",
                   item.client_ip.c_str(), time_sec() - ts0);
            fflush(stdout);
        }
    }

    Model *model_;
    std::vector<std::thread> workers_;
    std::queue<ServerWorkItem> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool stop_;
};

static char *http_read_request(int fd, char *buf, int buf_size, int *body_len) {
    int total = 0;
    while (total < buf_size - 1) {
        int r = (int)read(fd, buf + total, buf_size - 1 - total);
        if (r <= 0) return nullptr;
        total += r; buf[total] = '\0';
        char *hdr_end = strstr(buf, "\r\n\r\n");
        if (hdr_end) {
            hdr_end += 4;
            int content_len = 0;
            char *cl = strcasestr(buf, "Content-Length:");
            if (cl) content_len = atoi(cl + 15);
            int body_received = total - (int)(hdr_end - buf);
            while (body_received < content_len && total < buf_size - 1) {
                int r2 = (int)read(fd, buf + total, buf_size - 1 - total);
                if (r2 <= 0) break;
                total += r2; body_received += r2;
            }
            buf[total] = '\0';
            *body_len  = body_received;
            return hdr_end;
        }
    }
    return nullptr;
}

static void http_respond(int fd, int code, const char *status,
                         const char *ctype, const std::string &body) {
    char header[512];
    int blen = (int)body.size();
    int hlen = snprintf(header, sizeof(header),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %d\r\n"
        "Connection: close\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "\r\n", code, status, ctype, blen);
    write(fd, header, hlen);
    if (blen > 0) write(fd, body.c_str(), blen);
}

static void run_server(Model *m, int port, int default_max,
                       float default_temp, int default_top_k,
                       float default_top_p, float default_rep_p,
                       int server_threads) {

    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) { fprintf(stderr,"socket failed\n"); return; }
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr; memset(&addr,0,sizeof(addr));
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons((uint16_t)port);

    if (bind(server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        fprintf(stderr,"bind %d: %s\n", port, strerror(errno));
        close(server_fd); return;
    }
    if (listen(server_fd, SOMAXCONN) < 0) {
        fprintf(stderr,"listen: %s\n", strerror(errno));
        close(server_fd); return;
    }

    signal(SIGINT,  server_sigint);
    signal(SIGPIPE, SIG_IGN);

    ServerWorkerPool worker_pool(m, server_threads);

    printf("\n=== XAI Server on http://0.0.0.0:%d ===\n", port);
    printf("Concurrent workers: %d\n", server_threads);
    printf("POST /generate  GET /health\nCtrl+C to stop.\n\n");

    std::vector<char> req_buf(65536);

    while (g_server_running) {
        fd_set fds; struct timeval tv;
        FD_ZERO(&fds); FD_SET(server_fd, &fds);
        tv.tv_sec = 1; tv.tv_usec = 0;
        if (select(server_fd+1, &fds, nullptr, nullptr, &tv) <= 0) continue;

        struct sockaddr_in cli; socklen_t cli_len = sizeof(cli);
        int cli_fd = accept(server_fd, (struct sockaddr*)&cli, &cli_len);
        if (cli_fd < 0) continue;

        int body_len = 0;
        char *body = http_read_request(cli_fd, req_buf.data(),
                                       (int)req_buf.size(), &body_len);
        if (!body) { close(cli_fd); continue; }

        char method[16]={}, path_buf[256]={};
        sscanf(req_buf.data(), "%15s %255s", method, path_buf);
        std::string client_ip = inet_ntoa(cli.sin_addr);
        printf("[%s] %s %s", client_ip.c_str(), method, path_buf);
        fflush(stdout);

        if (!strcmp(method,"OPTIONS")) {
            const char *cors =
                "HTTP/1.1 204 No Content\r\n"
                "Access-Control-Allow-Origin: *\r\n"
                "Access-Control-Allow-Methods: POST, GET, OPTIONS\r\n"
                "Access-Control-Allow-Headers: Content-Type\r\n"
                "Content-Length: 0\r\nConnection: close\r\n\r\n";
            write(cli_fd, cors, strlen(cors));
            printf(" → 204\n");
            close(cli_fd);
        }
        else if (!strcmp(path_buf,"/health") && !strcmp(method,"GET")) {
            char hbuf[512];
            snprintf(hbuf, sizeof(hbuf),
                "{\"status\":\"ok\",\"threads\":%d,\"server_workers\":%d,\"weight_format\":\"%s\"}",
                g_num_threads, server_threads,
                m->layers[0].q_proj.fmt==WF_F16?"f16":
                m->layers[0].q_proj.fmt==WF_Q8?"q8":"f32");
            http_respond(cli_fd, 200, "OK", "application/json", hbuf);
            printf(" → 200\n");
            close(cli_fd);
        }
        else if (!strcmp(path_buf,"/generate") && !strcmp(method,"POST")) {
            auto sreq = parse_server_request(body, body_len);
            if (sreq.prompt.empty()) {
                http_respond(cli_fd, 400, "Bad Request", "application/json",
                             "{\"error\":\"Missing prompt\"}");
                printf(" → 400\n");
                close(cli_fd);
            } else {
                if (sreq.max_tokens <= 0) sreq.max_tokens = default_max;
                ServerWorkItem item;
                item.client_fd = cli_fd;
                item.client_ip = client_ip;
                item.request   = std::move(sreq);
                worker_pool.enqueue(std::move(item));
                printf(" → queued\n");
            }
        }
        else {
            http_respond(cli_fd, 404, "Not Found", "application/json",
                         "{\"error\":\"Not found\"}");
            printf(" → 404\n");
            close(cli_fd);
        }
    }

    printf("\nShutting down.\n");
    close(server_fd);
}

/* ================================================================
 * CHAT MODE
 * ================================================================ */

static void chat(Model *m, int max_new, float temp, int top_k,
                 float top_p, float rep_p, bool ignore_eos) {
    RunState s; alloc_state(&s, &m->cfg);
    char line[4096];
    int  ids[MAX_SEQ_LEN];
    std::vector<int> history;  // хранит всю историю токенов
    bool first_turn = true;

    printf("\n╔══════════════════════════════════════════╗\n");
    printf("║           XAI Chat Mode                  ║\n");
    printf("╠══════════════════════════════════════════╣\n");
    printf("║ Commands:                                ║\n");
    printf("║   /quit, /q      - Exit                  ║\n");
    printf("║   /clear         - Clear history         ║\n");
    printf("║   /temp N        - Set temperature       ║\n");
    printf("║   /topk N        - Set top_k             ║\n");
    printf("║   /topp N        - Set top_p             ║\n");
    printf("║   /repp N        - Set repetition penalty║\n");
    printf("║   /max N         - Set max new tokens    ║\n");
    printf("║   /eos           - Toggle ignore EOS     ║\n");
    printf("║   /status        - Show current settings ║\n");
    printf("║   /help, /?      - Show this help        ║\n");
    printf("╚══════════════════════════════════════════╝\n");

    auto print_status = [&]() {
        printf("\n┌── Current Settings ──────────────────────┐\n");
        printf("│ temp=%.2f  top_k=%-4d top_p=%.2f  rep_p=%.2f │\n", temp, top_k, top_p, rep_p);
        printf("│ max_new=%-4d  ignore_eos=%-5s  threads=%-3d   │\n",
               max_new, ignore_eos ? "true" : "false", g_num_threads);
        printf("│ history: %d tokens  window=%d              │\n",
               (int)history.size(), g_sliding_window);
        printf("└──────────────────────────────────────────┘\n\n");
    };

    print_status();
    printf("Start chatting! Type your message and press Enter.\n\n");

    while (true) {
        printf("You> "); fflush(stdout);
        if (!fgets(line, sizeof(line), stdin)) break;
        int len = (int)strlen(line);
        while (len > 0 && (line[len-1]=='\n'||line[len-1]=='\r')) line[--len]=0;
        if (!len) continue;

        // Commands
        if (line[0] == '/') {
            char cmd[32], arg[128] = "";
            sscanf(line, "/%31s %127[^\n]", cmd, arg);

            if (!strcmp(cmd,"quit")||!strcmp(cmd,"q")) break;
            if (!strcmp(cmd,"clear")) {
                history.clear();
                reset_state(&s, &m->cfg);
                printf("  ✓ History cleared\n");
                continue;
            }
            if (!strcmp(cmd,"temp")) { temp  = (float)atof(arg); printf("  ✓ temp=%.2f\n",temp); continue; }
            if (!strcmp(cmd,"topk")) { top_k = atoi(arg); printf("  ✓ top_k=%d\n",top_k); continue; }
            if (!strcmp(cmd,"topp")) { top_p = (float)atof(arg); printf("  ✓ top_p=%.2f\n",top_p); continue; }
            if (!strcmp(cmd,"repp")) { rep_p = (float)atof(arg); printf("  ✓ rep_p=%.2f\n",rep_p); continue; }
            if (!strcmp(cmd,"max"))  { max_new = atoi(arg); printf("  ✓ max_new=%d\n",max_new); continue; }
            if (!strcmp(cmd,"eos"))  { ignore_eos=!ignore_eos; printf("  ✓ ignore_eos=%s\n",ignore_eos?"true":"false"); continue; }
            if (!strcmp(cmd,"status")) { print_status(); continue; }
            if (!strcmp(cmd,"help")||!strcmp(cmd,"?")) {
                printf("Commands: /quit /clear /temp /topk /topp /repp /max /eos /status /help\n");
                continue;
            }
            printf("  ✗ Unknown command: /%s\n", cmd);
            continue;
        }

        // Tokenize user message
        int n = tok_encode(&m->tok, line, ids, MAX_SEQ_LEN, 0);
        if (n == 0) { printf("  (empty tokenization)\n"); continue; }

        // Append to history
        history.insert(history.end(), ids, ids + n);

        // Reset state and prefill with history
        reset_state(&s, &m->cfg);

        double t_start = time_sec();
        printf("AI> "); fflush(stdout);

        // Use generate with empty prompt but pre-filled history
        bool is_greedy = (temp < 1e-6f);
        bool no_rep    = (fabsf(rep_p - 1.0f) < 1e-6f);

        forward_batch(m, &s, history.data(), 0, (int)history.size());
        int pos = (int)history.size();

        if (!no_rep)
            apply_repetition_penalty(s.logits, m->cfg.vocab_size,
                                     history.data(), (int)history.size(), rep_p);

        int next;
        if (is_greedy) {
            next = 0;
            for (int i = 1; i < m->cfg.vocab_size; i++)
                if (s.logits[i] > s.logits[next]) next = i;
        } else {
            next = sample_token(s.logits, m->cfg.vocab_size, temp, top_k, top_p);
        }

        history.push_back(next);
        char buf[256];
        tok_decode_one(&m->tok, next, buf, sizeof(buf));
        printf("%s", buf); fflush(stdout);
        int gen_count = 1;

        for (int i = 1; i < max_new; i++) {
            if (!ignore_eos && next == m->tok.eos_id) break;
            if (pos >= m->cfg.max_seq_len - 1) break;

            if (is_greedy && no_rep) {
                forward_transformer(m, &s, next, pos++);
                next = wt_matmul_argmax(&m->lm_head, s.x);
            } else {
                forward(m, &s, next, pos++);
                if (!no_rep)
                    apply_repetition_penalty(s.logits, m->cfg.vocab_size,
                                             history.data(), (int)history.size(), rep_p);
                if (is_greedy) {
                    next = 0;
                    for (int j = 1; j < m->cfg.vocab_size; j++)
                        if (s.logits[j] > s.logits[next]) next = j;
                } else {
                    next = sample_token(s.logits, m->cfg.vocab_size, temp, top_k, top_p);
                }
            }

            history.push_back(next);
            tok_decode_one(&m->tok, next, buf, sizeof(buf));
            printf("%s", buf); fflush(stdout);
            gen_count++;
        }

        double elapsed = time_sec() - t_start;
        printf("\n  [%d tokens, %.2fs, %.1f tok/s]\n\n",
               gen_count, elapsed, gen_count / elapsed);
    }

    printf("\nGoodbye!\n");
    free_state(&s);
}

/* ================================================================
 * MAIN
 * ================================================================ */

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
            "Usage: %s model.xai --mode [json|server|chat|info] [options]\n"
            "  --prompt      | default: nothing\n"
            "  --max_new     | default: 128\n"
            "  --temperature | default: 0.4\n"
            "  --top_k       | default: 30\n"
            "  --top_p       | default: 0.7\n"
            "  --rep_p       | default: 1.2\n"
            "  --seed        | default: 56\n"
            "  --threads     | default: 1\n"
            "  --server_threads | default: 4 (server mode)\n"
            "  --ignore_eos  |\n"
            "  --port        | default: 7171\n"
            "  --window      | default: full\n", argv[0]);
        return 1;
    }

    const char *model_path = argv[1];
    const char *prompt = nullptr, *mode = nullptr;
    int   max_new       = 128;
    float temp          = 0.4f;
    int   top_k         = 30;
    float top_p         = 0.7f;
    float rep_p         = 1.2f;
    uint64_t seed       = 56;
    bool  ignore_eos    = true;
    int   port          = 7171;
    int   server_threads = 4;

    for (int i = 2; i < argc; i++) {
        if      (!strcmp(argv[i],"--mode")   && i+1<argc) mode    = argv[++i];
        else if (!strcmp(argv[i],"--prompt")       && i+1<argc) prompt  = argv[++i];
        else if (!strcmp(argv[i],"--max_new")       && i+1<argc) max_new = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--temperature")       && i+1<argc) temp    = (float)atof(argv[++i]);
        else if (!strcmp(argv[i],"--top_k")       && i+1<argc) top_k   = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--top_p")  && i+1<argc) top_p   = (float)atof(argv[++i]);
        else if (!strcmp(argv[i],"--rep_p")  && i+1<argc) rep_p   = (float)atof(argv[++i]);
        else if (!strcmp(argv[i],"--seed")       && i+1<argc) seed    = (uint64_t)atoll(argv[++i]);
        else if (!strcmp(argv[i],"--threads")       && i+1<argc) g_num_threads = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--server_threads") && i+1<argc) server_threads = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--ignore_eos") || !strcmp(argv[i],"--no-eos")) ignore_eos = true;
        else if (!strcmp(argv[i],"--port")   && i+1<argc) port    = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--window") && i+1<argc) g_sliding_window = atoi(argv[++i]);
    }

    if (!mode) { fprintf(stderr,"Error: --mode required\n"); return 1; }

    bool is_json   = !strcasecmp(mode,"json");
    bool is_server = !strcasecmp(mode,"server");
    bool is_chat   = !strcasecmp(mode,"chat");
    bool info      = !strcasecmp(mode,"info");
    bool his_mode  = !strcasecmp(mode,"56");
    bool k_mode    = !strcasecmp(mode,"17");

    if (!is_json && !is_server && !is_chat && !his_mode && !k_mode && !info) {
        fprintf(stderr,"Unknown mode '%s'\n", mode); return 1;
    }

    if (g_num_threads < 1) g_num_threads = 1;
    if (g_num_threads > MAX_THREADS) g_num_threads = MAX_THREADS;
    if (server_threads < 1) server_threads = 1;
    if (server_threads > 32) server_threads = 32;

    rng_seed(seed);

    if (g_num_threads > 1)
        g_pool = std::make_unique<BarrierPool>(g_num_threads);

    Model model;

    if (his_mode) {
        const char* text = "why you did this to me?";
        while (true) {
            system("clear");
            int spaces = rand() % 5;
            for (int i = 0; i < spaces; i++) printf(" ");
            printf("%s\n", text);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    } else if (k_mode) {
        const char* text = "i never forgive you for what you did to me";
        while (true) {
            system("clear");
            int spaces = rand() % 5;
            for (int i = 0; i < spaces; i++) printf(" ");
            printf("%s\n", text);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    } else if (info) {
        fprintf(stderr,"{\"XAI-Engine ver 2.0 made by XellTech\"}\n"); return 1;
    } else {
        if (load_model(model_path, &model, is_json) != 0) return 1;
    }

    if (is_json) {
        if (!prompt) { fprintf(stderr,"{\"error\":\"json mode requires --prompt\"}\n"); return 1; }
        RunState s; alloc_state(&s, &model.cfg);
        int ids[MAX_SEQ_LEN];
        int n = tok_encode(&model.tok, prompt, ids, MAX_SEQ_LEN, 0);
        if (n == 0) { fprintf(stderr,"{\"error\":\"Empty prompt\"}\n"); free_state(&s); return 1; }
        auto r = generate(&model, &s, ids, n, max_new, temp, top_k, top_p,
                          rep_p, ignore_eos, false);
        print_result_json(r, prompt);
        free_state(&s);
    } else if (is_chat) {
        chat(&model, max_new, temp, top_k, top_p, rep_p, ignore_eos);
    } else {
        run_server(&model, port, max_new, temp, top_k, top_p, rep_p, server_threads);
    }

    g_pool.reset();
    free_model(&model);
    return 0;
}