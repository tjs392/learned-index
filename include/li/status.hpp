#pragma once
/* 
* The error handling foundation
*/

#include <new>
#include <type_traits>
#include <utility>

namespace li {

enum class Status {
    ok,
    not_found,
    out_of_bounds,
    capacity,
    invariant_violated,
};

template <class T>
class [[nodiscard]] Result {
public:
    
    // On success, carry a value
    Result(T value) noexcept(std::is_nothrow_move_constructible_v<T>) : has_value_(true) {
        new (&storage_.value) T(std::move(value));
    }

    Result(Status status) noexcept : status_(status), has_value_(false) {}

    Result(const Result& other) noexcept(std::is_nothrow_copy_constructible_v<T>) {
        copy_from(other);
    }

    Result(Result&& other) noexcept(std::is_nothrow_move_constructible_v<T>) {
        move_from(std::move(other));
    }

    Result& operator=(const Result& other) noexcept(std::is_nothrow_copy_constructible_v<T>) {
        if (this != &other) {
            destroy();
            copy_from(other);
        }
        return *this;
    }

    Result& operator=(Result&& other) noexcept(std::is_nothrow_move_constructible_v<T>) {
        if (this != &other) {
            destroy();
            move_from(std::move(other));
        }
        return *this;
    }

    ~Result() { destroy(); }

    [[nodiscard]] bool ok() const noexcept {
        return has_value_;
    }

    [[nodiscard]] Status status() const noexcept {
        return has_value_ ? Status::ok : status_;
    }

    [[nodiscard]] const T& value() const& noexcept { return storage_.value; }
    [[nodiscard]] T& value() & noexcept { return storage_.value; }
    [[nodiscard]] T&& value() && noexcept { return std::move(storage_.value); }

private:
    union Storage {
        Storage() noexcept {}
        ~Storage() noexcept {}
        T value;
    } storage_;

    Status status_ = Status::ok;
    bool has_value_ = false;

    void copy_from(const Result& other) {
        has_value_ = other.has_value_;
        if (has_value_) {
            new (&storage_.value) T(other.storage_.value);
        } else {
            status_ = other.status_;
        }
    }

    void move_from(Result&& other) {
        has_value_ = other.has_value_;
        if (has_value_) {
            new (&storage_.value) T(std::move(other.storage_.value));
        } else {
            status_ = other.status_;
        }
    }

    void destroy() noexcept {
        if (has_value_) storage_.value.~T();
    }
};

}

#if defined(LI_INVARIANT_CHECKS)
#include <cstdio>
#include <cstdlib>
#define LI_ASSERT(cond) \
    do { \
        if (!(cond)) { \
            std::fprintf(stderr, "LI_ASSERT failed: %s\n at %s:%d\n", \
                #cond, __FILE__, __LINE__); \
            std::abort(); \
        } \
    } while (0)
#else
#define LI_ASSERT(cond) ((void)0)
#endif



