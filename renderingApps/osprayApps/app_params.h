#pragma once

template <typename T>
struct AppParam {
    bool changed = false;
    T param;

    AppParam() = default;
    AppParam(const AppParam<T> &p) = default;
    AppParam &operator=(const AppParam<T> &p) = default;

    AppParam(const T &p) : param(p) {}
    AppParam &operator=(const T &p)
    {
        changed = true;
        param = p;
        return *this;
    }

    explicit operator T() const
    {
        return param;
    }
};

