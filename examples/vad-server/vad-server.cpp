// Standalone Silero VAD HTTP service (C++ port of vad_service.py).
//
// Endpoints:
//   POST /vad    body = raw little-endian float32 PCM, exactly 512 samples
//                      (2048 bytes) at 16 kHz. Query: ?threshold=0.5 (optional).
//                      Response: {"prob": float, "speech": bool}
//   POST /reset  clears the VAD context's internal state
//   GET  /health liveness probe

#include "httplib.h"
#include "json.hpp"
#include "whisper.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>

static void cb_log_disable(enum ggml_log_level, const char *, void *) {}

using json = nlohmann::ordered_json;

static constexpr int kSampleRate    = WHISPER_SAMPLE_RATE; // 16000
static constexpr int kWindowSamples = 512;                 // silero v5 window
static constexpr int kSampleBytes   = (int) sizeof(float);

struct cli_params {
    std::string host      = "127.0.0.1";
    int         port      = 8090;
    std::string vad_model = "models/ggml-silero-v5.1.2.bin";
    float       threshold = 0.5f;
    int         n_threads = std::min(4, (int) std::thread::hardware_concurrency());
    bool        use_gpu   = false;
};

static void print_usage(char ** argv, const cli_params & p) {
    fprintf(stderr, "\nusage: %s [options]\n\n", argv[0]);
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h,        --help              show this help message and exit\n");
    fprintf(stderr, "  --host HOST                    [%-15s] bind address\n",       p.host.c_str());
    fprintf(stderr, "  --port N                       [%-15d] bind port\n",          p.port);
    fprintf(stderr, "  -vm FNAME, --vad-model FNAME   [%s] Silero VAD model path\n", p.vad_model.c_str());
    fprintf(stderr, "  --threshold N                  [%-15.2f] default speech threshold\n", p.threshold);
    fprintf(stderr, "  -t N,      --threads N         [%-15d] CPU threads\n",        p.n_threads);
    fprintf(stderr, "  -ug,       --use-gpu           [%-15s] use GPU for VAD\n",    p.use_gpu ? "true" : "false");
    fprintf(stderr, "\n");
}

static bool parse_params(int argc, char ** argv, cli_params & p) {
    auto next = [&](int & i, const std::string & arg) -> const char * {
        if (i + 1 >= argc) {
            fprintf(stderr, "error: argument %s requires a value\n", arg.c_str());
            std::exit(1);
        }
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if      (a == "-h"  || a == "--help")      { print_usage(argv, p); std::exit(0); }
        else if (a ==        "--host")             { p.host      = next(i, a); }
        else if (a ==        "--port")             { p.port      = std::stoi(next(i, a)); }
        else if (a == "-vm" || a == "--vad-model") { p.vad_model = next(i, a); }
        else if (a ==        "--threshold")        { p.threshold = std::stof(next(i, a)); }
        else if (a == "-t"  || a == "--threads")   { p.n_threads = std::stoi(next(i, a)); }
        else if (a == "-ug" || a == "--use-gpu")   { p.use_gpu   = true; }
        else { fprintf(stderr, "error: unknown argument: %s\n", a.c_str()); print_usage(argv, p); return false; }
    }
    return true;
}

static std::string now_hms() {
    using namespace std::chrono;
    auto t  = system_clock::now();
    auto tt = system_clock::to_time_t(t);
    std::tm tm{};
    localtime_r(&tt, &tm);
    char buf[16];
    snprintf(buf, sizeof(buf), "%02d:%02d:%02d", tm.tm_hour, tm.tm_min, tm.tm_sec);
    return std::string(buf);
}

// Silero v5.1 uses 32 ms windows (512 samples at 16 kHz). whisper's VAD op
// returns one probability per window, so a chunk of N*512 samples produces N
// probabilities. The server reports the max over the chunk plus the derived
// speech bool.
int main(int argc, char ** argv) {
    cli_params params;
    if (!parse_params(argc, argv, params)) {
        return 1;
    }

    ggml_backend_load_all();
    whisper_log_set(cb_log_disable, nullptr);

    fprintf(stderr, "[%s] loading VAD model: %s\n", now_hms().c_str(), params.vad_model.c_str());
    whisper_vad_context_params cparams = whisper_vad_default_context_params();
    cparams.n_threads = params.n_threads;
    cparams.use_gpu   = params.use_gpu;

    whisper_vad_context * vctx = whisper_vad_init_from_file_with_params(params.vad_model.c_str(), cparams);
    if (vctx == nullptr) {
        fprintf(stderr, "error: failed to initialize VAD model from %s\n", params.vad_model.c_str());
        return 2;
    }
    fprintf(stderr, "[%s] model loaded\n", now_hms().c_str());

    std::mutex model_mu;
    std::atomic<float> default_threshold{params.threshold};
    std::mutex state_mu;
    int last_speech = -1; // -1 unknown, 0 silence, 1 speech

    httplib::Server svr;
    svr.set_default_headers({{"Server", "whisper-vad-server"}});

    svr.Get("/health", [](const httplib::Request &, httplib::Response & res) {
        res.set_content(json{{"ok", true}}.dump(), "application/json");
    });

    svr.Post("/reset", [&](const httplib::Request &, httplib::Response & res) {
        {
            std::lock_guard<std::mutex> lk(model_mu);
            whisper_vad_free(vctx);
            vctx = whisper_vad_init_from_file_with_params(params.vad_model.c_str(), cparams);
        }
        {
            std::lock_guard<std::mutex> lk(state_mu);
            last_speech = -1;
        }
        fprintf(stderr, "[%s] reset\n", now_hms().c_str());
        if (vctx == nullptr) {
            res.status = 500;
            res.set_content(json{{"ok", false}, {"error", "reload failed"}}.dump(), "application/json");
            return;
        }
        res.set_content(json{{"ok", true}}.dump(), "application/json");
    });

    svr.Post("/vad", [&](const httplib::Request & req, httplib::Response & res) {
        const int nbytes = (int) req.body.size();
        if (nbytes == 0 || nbytes % (kWindowSamples * kSampleBytes) != 0) {
            res.status = 400;
            res.set_content(json{
                {"error", "bad chunk size: must be a positive multiple of "
                          "512 float32 samples (2048 bytes)"},
                {"got_bytes", nbytes},
            }.dump(), "application/json");
            return;
        }
        const int n_samples = nbytes / kSampleBytes;
        const float * samples = reinterpret_cast<const float *>(req.body.data());

        float prob = 0.0f;
        {
            std::lock_guard<std::mutex> lk(model_mu);
            if (!whisper_vad_detect_speech(vctx, samples, n_samples)) {
                res.status = 500;
                res.set_content(json{{"error", "VAD inference failed"}}.dump(), "application/json");
                return;
            }
            const int n = whisper_vad_n_probs(vctx);
            const float * probs = whisper_vad_probs(vctx);
            for (int i = 0; i < n; ++i) prob = std::max(prob, probs[i]);
        }

        float thr = default_threshold.load();
        auto it = req.params.find("threshold");
        if (it != req.params.end()) {
            try { thr = std::stof(it->second); } catch (...) {}
        }
        const bool speech = prob >= thr;

        {
            std::lock_guard<std::mutex> lk(state_mu);
            const int now_state = speech ? 1 : 0;
            if (now_state != last_speech) {
                fprintf(stderr, "[%s] %s (prob=%.3f)\n",
                        now_hms().c_str(),
                        speech ? "speech" : "silence",
                        prob);
                last_speech = now_state;
            }
        }

        res.set_content(json{{"prob", prob}, {"speech", speech}}.dump(),
                        "application/json");
    });

    fprintf(stderr, "[%s] listening on http://%s:%d (threshold=%.2f)\n",
            now_hms().c_str(), params.host.c_str(), params.port, params.threshold);
    if (!svr.listen(params.host, params.port)) {
        fprintf(stderr, "error: failed to bind %s:%d\n", params.host.c_str(), params.port);
        whisper_vad_free(vctx);
        return 3;
    }

    whisper_vad_free(vctx);
    return 0;
}
