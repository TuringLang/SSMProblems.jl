using DifferentiationInterface: DifferentiationInterface
using ADTypes: AutoMooncake

"""
    central_diff(f, x; h=1e-6)

Central finite-difference gradient of `f` at `x`, used as the reference for gradient checks.
"""
function central_diff(f, x; h=1e-6)
    g = zeros(length(x))
    for i in eachindex(x)
        xp = collect(float.(x))
        xm = collect(float.(x))
        xp[i] += h
        xm[i] -= h
        g[i] = (f(xp) - f(xm)) / (2h)
    end
    return g
end

"""
    check_gradients(f, θ; rtol=1e-6, backend=AutoMooncake(; config=nothing))

Compare the reverse-mode gradient of `f` at `θ` against central finite differences. Returns a
`NamedTuple` `(; ad, fd, max_abs_error, agrees)` so callers can both assert `agrees` and report
the deviation. Requires the AD backend's package to be loaded (`Mooncake` for the default).
"""
function check_gradients(f, θ; rtol=1e-6, backend=AutoMooncake(; config=nothing))
    ad = DifferentiationInterface.gradient(f, backend, θ)
    fd = central_diff(f, θ)
    return (; ad, fd, max_abs_error=maximum(abs, ad .- fd), agrees=isapprox(ad, fd; rtol))
end
