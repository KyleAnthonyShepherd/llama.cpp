#pragma once

// loop guard monitor, see PLAN-loop-guard.md
// no llama dependency on purpose: a scratch program can replay a trace file through it

#include <cctype>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct server_loop_monitor {
    int32_t n_gram       = 8;
    int32_t n_window     = 256;
    int32_t n_min        = 256;
    int32_t n_line_min   = 12;
    int32_t n_words      = 300; // novelty is taken over this many last words
    int32_t n_words_min  = 600; // no novelty trip before this many words in a phase

    int32_t n_tokens = 0; // tokens seen since reset()
    int32_t n_fresh  = 0; // tokens seen since reset() or rearm()

    // token n-grams
    std::vector<int32_t>         last; // ring of the last n_gram tokens
    std::unordered_set<uint64_t> seen;
    std::vector<uint8_t>         hits; // ring of the last n_window hit flags
    int32_t                      n_hits = 0;

    // words. A loop goes over the same things again, so it stops bringing new words
    // (numbers, names, row patterns). Work through a list or a table keeps bringing them
    std::string                     word_cur;
    std::unordered_set<std::string> words_seen;
    std::vector<uint8_t>            word_new;   // ring of the last n_words "first time seen" flags
    int32_t                         n_word_new  = 0;
    int32_t                         n_word      = 0; // words since reset()
    int32_t                         n_word_fresh = 0; // words since reset() or rearm()

    // lines
    std::string line_cur;
    int32_t     line_cur_tokens = 0;
    std::unordered_map<std::string, int32_t> lines;
    std::string line_top;         // the most repeated line
    int32_t     line_repeats = 0; // how often it was produced
    int32_t     line_extra   = 0; // raised by rearm(), a new trip needs this many more

    // logprob and entropy over the last n_stat tokens
    static constexpr int32_t n_stat = 64;
    float  stat_logprob[n_stat] = {};
    float  stat_entropy[n_stat] = {};
    double sum_logprob = 0.0;
    double sum_entropy = 0.0;

    void configure(int32_t n_gram_new, int32_t n_window_new, int32_t n_min_new, int32_t n_line_min_new, int32_t n_words_new, int32_t n_words_min_new) {
        n_gram      = n_gram_new   > 0 ? n_gram_new   : 1;
        n_window    = n_window_new > 0 ? n_window_new : 1;
        n_min       = n_min_new;
        n_line_min  = n_line_min_new;
        n_words     = n_words_new  > 0 ? n_words_new  : 1;
        n_words_min = n_words_min_new;

        reset();
    }

    void reset() {
        n_tokens = 0;
        n_fresh  = 0;

        last.assign(n_gram, -1);
        seen.clear();
        hits.assign(n_window, 0);
        n_hits = 0;

        word_cur.clear();
        words_seen.clear();
        word_new.assign(n_words, 0);
        n_word_new   = 0;
        n_word       = 0;
        n_word_fresh = 0;

        line_cur.clear();
        line_cur_tokens = 0;
        lines.clear();
        line_top.clear();
        line_repeats = 0;
        line_extra   = 0;

        for (int32_t i = 0; i < n_stat; ++i) {
            stat_logprob[i] = 0.0f;
            stat_entropy[i] = 0.0f;
        }
        sum_logprob = 0.0;
        sum_entropy = 0.0;
    }

    // after a trip: keep what was seen, but ask for new evidence before the next trip
    void rearm(int32_t line_thold) {
        n_fresh      = 0;
        n_word_fresh = 0;

        hits.assign(n_window, 0);
        n_hits = 0;

        line_extra = line_repeats - line_thold + 2;
        if (line_extra < 0) {
            line_extra = 0;
        }
    }

    void push(int32_t token, const std::string & piece, float logprob, float entropy) {
        // n-gram
        {
            last[n_tokens % n_gram] = token;

            bool hit = false;
            if (n_tokens + 1 >= n_gram) {
                // FNV-1a over the last n_gram tokens, oldest first
                uint64_t h = 1469598103934665603ull;
                for (int32_t i = 1; i <= n_gram; ++i) {
                    h ^= (uint64_t) (uint32_t) last[(n_tokens + i) % n_gram];
                    h *= 1099511628211ull;
                }
                hit = !seen.insert(h).second;
            }

            uint8_t & slot = hits[n_tokens % n_window];
            n_hits += (hit ? 1 : 0) - slot;
            slot = hit ? 1 : 0;
        }

        // words and lines
        {
            bool counted = false;
            for (const char c : piece) {
                if (std::isspace((unsigned char) c)) {
                    word_end();
                } else if (word_cur.size() < 64) {
                    word_cur += (char) std::tolower((unsigned char) c);
                }

                if (c == '\n') {
                    line_end();
                    counted = false;
                    continue;
                }
                if (!counted) {
                    line_cur_tokens++;
                    counted = true;
                }
                if (line_cur.size() < 512) {
                    line_cur += c;
                }
            }
        }

        // stats
        {
            const int32_t i = n_tokens % n_stat;
            sum_logprob += (double) logprob - stat_logprob[i];
            sum_entropy += (double) entropy - stat_entropy[i];
            stat_logprob[i] = logprob;
            stat_entropy[i] = entropy;
        }

        n_tokens++;
        n_fresh++;
    }

    // share of the last n_window token n-grams that were produced before
    float rate() const {
        return (float) n_hits / (float) n_window;
    }

    // share of the last n_words words that were new. 1 until the window is full
    float novelty() const {
        return n_word >= n_words ? (float) n_word_new / (float) n_words : 1.0f;
    }

    float mean_logprob() const {
        const int32_t n = n_tokens < n_stat ? n_tokens : n_stat;
        return n > 0 ? (float) (sum_logprob / n) : 0.0f;
    }

    float mean_entropy() const {
        const int32_t n = n_tokens < n_stat ? n_tokens : n_stat;
        return n > 0 ? (float) (sum_entropy / n) : 0.0f;
    }

    // a threshold of 0 turns its rule off
    bool tripped(float rate_thold, float novelty_thold, int32_t line_thold) const {
        if (line_thold > 0 && line_repeats >= line_thold + line_extra) {
            return true;
        }
        // the window must be full of what came after the last rearm
        if (novelty_thold > 0.0f && n_word >= n_words_min && n_word_fresh >= n_words && novelty() <= novelty_thold) {
            return true;
        }
        return rate_thold > 0.0f && n_tokens >= n_min && n_fresh >= n_window && rate() >= rate_thold;
    }

private:
    void word_end() {
        // the punctuation around a word is not part of it
        size_t b = 0;
        size_t e = word_cur.size();
        while (b < e && std::ispunct((unsigned char) word_cur[b])) {
            b++;
        }
        while (e > b && std::ispunct((unsigned char) word_cur[e - 1])) {
            e--;
        }

        if (e > b) {
            const bool is_new = words_seen.insert(word_cur.substr(b, e - b)).second;

            uint8_t & slot = word_new[n_word % n_words];
            n_word_new += (is_new ? 1 : 0) - slot;
            slot = is_new ? 1 : 0;

            n_word++;
            n_word_fresh++;
        }

        word_cur.clear();
    }

    void line_end() {
        size_t b = 0;
        size_t e = line_cur.size();
        while (b < e && (line_cur[b] == ' ' || line_cur[b] == '\t' || line_cur[b] == '\r')) {
            b++;
        }
        while (e > b && (line_cur[e - 1] == ' ' || line_cur[e - 1] == '\t' || line_cur[e - 1] == '\r')) {
            e--;
        }

        if (line_cur_tokens >= n_line_min && e > b) {
            const std::string line = line_cur.substr(b, e - b);
            const int32_t n = ++lines[line];
            if (n > line_repeats) {
                line_repeats = n;
                line_top     = line;
            }
        }

        line_cur.clear();
        line_cur_tokens = 0;
    }
};
