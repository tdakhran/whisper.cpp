#include "common-whisper.h"

#include "whisper.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

struct cli_params {
    int32_t     n_threads         = std::min(4, (int32_t) std::thread::hardware_concurrency());
    std::string vad_model         = "models/ggml-silero-v5.1.2.bin";
    std::string fname_inp         = {};
    int32_t     chunk_ms          = 100;
    float       vad_threshold     = 0.5f;
    float       vad_threshold_off = 0.25f;
    bool        use_gpu           = false;
    bool        no_sleep          = false;
};

static void cb_log_disable(enum ggml_log_level, const char *, void *) {}

static std::string timestamp_from_ms(int64_t ms) {
    const int64_t hours   = ms / (1000 * 60 * 60); ms -= hours   * (1000 * 60 * 60);
    const int64_t minutes = ms / (1000 * 60);      ms -= minutes * (1000 * 60);
    const int64_t seconds = ms / 1000;             ms -= seconds * 1000;
    char buf[32];
    snprintf(buf, sizeof(buf), "%02d:%02d:%02d.%03d",
            (int) hours, (int) minutes, (int) seconds, (int) ms);
    return std::string(buf);
}

static void print_usage(char ** argv, const cli_params & params) {
    fprintf(stderr, "\nusage: %s [options]\n\n", argv[0]);
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h,        --help                 show this help message and exit\n");
    fprintf(stderr, "  -f FNAME,  --file FNAME           [%-7s] input WAV file\n",                       params.fname_inp.c_str());
    fprintf(stderr, "  -vm FNAME, --vad-model FNAME      [%-7s] Silero VAD model path\n",                params.vad_model.c_str());
    fprintf(stderr, "  -c N,      --chunk-ms N           [%-7d] chunk size in milliseconds\n",           params.chunk_ms);
    fprintf(stderr, "  -vt N,     --vad-threshold N      [%-7.2f] threshold to enter speak\n",           params.vad_threshold);
    fprintf(stderr, "  -vto N,    --vad-threshold-off N  [%-7.2f] threshold to return to no_speak\n",    params.vad_threshold_off);
    fprintf(stderr, "  -t N,      --threads N            [%-7d] number of CPU threads\n",                params.n_threads);
    fprintf(stderr, "  -ug,       --use-gpu              [%-7s] use GPU for the VAD model\n",            params.use_gpu ? "true" : "false");
    fprintf(stderr, "  -ns,       --no-sleep             [%-7s] process as fast as possible\n",          params.no_sleep ? "true" : "false");
    fprintf(stderr, "\n");
}

static bool parse_params(int argc, char ** argv, cli_params & params) {
    auto next = [&](int & i, const std::string & arg) -> const char * {
        if (i + 1 >= argc) {
            fprintf(stderr, "error: argument %s requires a value\n", arg.c_str());
            std::exit(1);
        }
        return argv[++i];
    };

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if      (arg == "-h"   || arg == "--help")              { print_usage(argv, params); std::exit(0); }
        else if (arg == "-f"   || arg == "--file")              { params.fname_inp         = next(i, arg); }
        else if (arg == "-vm"  || arg == "--vad-model")         { params.vad_model         = next(i, arg); }
        else if (arg == "-c"   || arg == "--chunk-ms")          { params.chunk_ms          = std::stoi(next(i, arg)); }
        else if (arg == "-vt"  || arg == "--vad-threshold")     { params.vad_threshold     = std::stof(next(i, arg)); }
        else if (arg == "-vto" || arg == "--vad-threshold-off") { params.vad_threshold_off = std::stof(next(i, arg)); }
        else if (arg == "-t"   || arg == "--threads")           { params.n_threads         = std::stoi(next(i, arg)); }
        else if (arg == "-ug"  || arg == "--use-gpu")           { params.use_gpu           = true; }
        else if (arg == "-ns"  || arg == "--no-sleep")          { params.no_sleep          = true; }
        else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            print_usage(argv, params);
            return false;
        }
    }

    if (params.fname_inp.empty()) {
        fprintf(stderr, "error: missing input file (-f)\n");
        return false;
    }
    if (params.chunk_ms <= 0) {
        fprintf(stderr, "error: chunk size must be positive\n");
        return false;
    }
    if (params.vad_threshold_off > params.vad_threshold) {
        fprintf(stderr, "error: vad-threshold-off must be <= vad-threshold\n");
        return false;
    }
    return true;
}

int main(int argc, char ** argv) {
    ggml_backend_load_all();
    whisper_log_set(cb_log_disable, nullptr);

    cli_params params;
    if (!parse_params(argc, argv, params)) {
        print_usage(argv, params);
        return 1;
    }

    std::vector<float> pcmf32;
    std::vector<std::vector<float>> pcmf32s;
    if (!read_audio_data(params.fname_inp, pcmf32, pcmf32s, false)) {
        fprintf(stderr, "error: failed to read audio data from %s\n", params.fname_inp.c_str());
        return 2;
    }

    whisper_vad_context_params ctx_params = whisper_vad_default_context_params();
    ctx_params.n_threads = params.n_threads;
    ctx_params.use_gpu   = params.use_gpu;

    whisper_vad_context * vctx = whisper_vad_init_from_file_with_params(params.vad_model.c_str(), ctx_params);
    if (vctx == nullptr) {
        fprintf(stderr, "error: failed to initialize VAD model from %s\n", params.vad_model.c_str());
        return 3;
    }

    const int64_t samples_per_chunk = (int64_t) params.chunk_ms * WHISPER_SAMPLE_RATE / 1000;
    if (samples_per_chunk <= 0) {
        fprintf(stderr, "error: chunk size too small for sample rate %d\n", WHISPER_SAMPLE_RATE);
        whisper_vad_free(vctx);
        return 4;
    }

    bool is_speaking = false;

    const auto t_wall_start = std::chrono::steady_clock::now();

    for (int64_t offset = 0; offset < (int64_t) pcmf32.size(); offset += samples_per_chunk) {
        const int64_t chunk_end = std::min<int64_t>(offset + samples_per_chunk, pcmf32.size());
        const int     chunk_len = (int) (chunk_end - offset);

        if (!whisper_vad_detect_speech(vctx, pcmf32.data() + offset, chunk_len)) {
            fprintf(stderr, "error: VAD inference failed at sample %lld\n", (long long) offset);
            whisper_vad_free(vctx);
            return 5;
        }

        const int n_probs = whisper_vad_n_probs(vctx);
        float max_prob = 0.0f;
        if (n_probs > 0) {
            float * probs = whisper_vad_probs(vctx);
            for (int i = 0; i < n_probs; ++i) {
                max_prob = std::max(max_prob, probs[i]);
            }
        }

        if (is_speaking) {
            if (max_prob < params.vad_threshold_off)      is_speaking = false;
        } else {
            if (max_prob >= params.vad_threshold)         is_speaking = true;
        }

        const int64_t start_ms = offset    * 1000 / WHISPER_SAMPLE_RATE;
        const int64_t end_ms   = chunk_end * 1000 / WHISPER_SAMPLE_RATE;

        printf("\r[%s - %s]  p=%.3f  %-8s",
                timestamp_from_ms(start_ms).c_str(),
                timestamp_from_ms(end_ms).c_str(),
                max_prob,
                is_speaking ? "speak" : "no speak");
        fflush(stdout);

        if (!params.no_sleep) {
            std::this_thread::sleep_until(t_wall_start + std::chrono::milliseconds(end_ms));
        }
    }

    whisper_vad_free(vctx);
    printf("\n");
    return 0;
}
