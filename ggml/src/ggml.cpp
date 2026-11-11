#include "ggml-impl.h"

#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <exception>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

static std::terminate_handler previous_terminate_handler;

// set once a handler has printed the backtrace, so a later abort() does not print it again
static std::atomic<bool> ggml_crash_reported{false};

GGML_NORETURN static void ggml_uncaught_exception() {
    if (std::exception_ptr e = std::current_exception()) {
        try {
            std::rethrow_exception(e);
        } catch (const std::exception & ex) {
            fprintf(stderr, "uncaught exception: %s\n", ex.what());
        } catch (...) {
            fprintf(stderr, "uncaught exception of an unknown type\n");
        }
    } else {
        fprintf(stderr, "std::terminate called without an exception\n");
    }
    ggml_print_backtrace();
    ggml_crash_reported = true;
    if (previous_terminate_handler) {
        previous_terminate_handler();
    }
    abort(); // unreachable unless previous_terminate_handler was nullptr
}

#if defined(_WIN32)
// MSVC release builds exit on abort() and on an access violation without a word, so say what happened
static void ggml_on_sigabrt(int) {
    if (!ggml_crash_reported.exchange(true)) {
        fprintf(stderr, "abort() called\n");
        ggml_print_backtrace();
    }
    fflush(stderr);
}

static LONG WINAPI ggml_on_seh(EXCEPTION_POINTERS * info) {
    if (!ggml_crash_reported.exchange(true)) {
        const EXCEPTION_RECORD * rec = info->ExceptionRecord;
        fprintf(stderr, "unhandled exception 0x%08lx at %p\n", (unsigned long) rec->ExceptionCode, rec->ExceptionAddress);
        if (rec->ExceptionCode == EXCEPTION_ACCESS_VIOLATION && rec->NumberParameters >= 2) {
            fprintf(stderr, "  %s address 0x%llx\n", rec->ExceptionInformation[0] ? "writing" : "reading",
                    (unsigned long long) rec->ExceptionInformation[1]);
        }
        ggml_print_backtrace();
    }
    fflush(stderr);
    return EXCEPTION_CONTINUE_SEARCH;
}

// the CRT ends the process with __fastfail on these, which skips every handler above
static void ggml_on_invalid_parameter(const wchar_t * expr, const wchar_t * func, const wchar_t * file, unsigned int line, uintptr_t) {
    fprintf(stderr, "CRT invalid parameter: %ls in %ls (%ls:%u)\n", expr ? expr : L"?", func ? func : L"?", file ? file : L"?", line);
    ggml_print_backtrace();
    ggml_crash_reported = true;
    fflush(stderr);
}

static void ggml_on_purecall() {
    fprintf(stderr, "pure virtual function call\n");
    ggml_print_backtrace();
    ggml_crash_reported = true;
    fflush(stderr);
}

// GGML_BACKTRACE_VEH=1: report the fatal codes first, on the faulting thread. Heap corruption (0xC0000374) and a
// stack buffer overrun end the process before any handler above runs. Off by default: the codes are first-chance,
// so a library that handles one of them itself (ASan commits its shadow memory this way) is reported too
static LONG WINAPI ggml_on_veh(EXCEPTION_POINTERS * info) {
    const DWORD code = info->ExceptionRecord->ExceptionCode;
    if (code == EXCEPTION_STACK_OVERFLOW) {
        // too little stack left for more than a note
        static const char msg[] = "stack overflow\n";
        DWORD n = 0;
        WriteFile(GetStdHandle(STD_ERROR_HANDLE), msg, sizeof(msg) - 1, &n, NULL);
    } else if (code == EXCEPTION_ACCESS_VIOLATION || code == 0xC0000409 || code == EXCEPTION_ILLEGAL_INSTRUCTION ||
               code == EXCEPTION_INT_DIVIDE_BY_ZERO || code == 0xC0000374) {
        fprintf(stderr, "exception 0x%08lx at %p\n", (unsigned long) code, info->ExceptionRecord->ExceptionAddress);
        ggml_print_backtrace();
        fflush(stderr);
    }
    return EXCEPTION_CONTINUE_SEARCH;
}
#endif

static bool ggml_uncaught_exception_init = []{
    const char * GGML_NO_BACKTRACE = getenv("GGML_NO_BACKTRACE");
    if (GGML_NO_BACKTRACE) {
        return false;
    }
    const auto prev{std::get_terminate()};
    GGML_ASSERT(prev != ggml_uncaught_exception);
    previous_terminate_handler = prev;
    std::set_terminate(ggml_uncaught_exception);
#if defined(_WIN32)
    signal(SIGABRT, ggml_on_sigabrt);
    SetUnhandledExceptionFilter(ggml_on_seh);
    _set_invalid_parameter_handler(ggml_on_invalid_parameter);
    _set_purecall_handler(ggml_on_purecall);
    if (getenv("GGML_BACKTRACE_VEH")) {
        AddVectoredExceptionHandler(1, ggml_on_veh);
    }
#endif
    return true;
}();
