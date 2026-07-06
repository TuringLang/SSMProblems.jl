export with_activity

# Activity flags record which fields of an inner atom depend on the differentiated
# parameters. They are carried by a `WithFlags` wrapper, independent of the Kalman kernels,
# and drive the pullback skipping in the analytic reverse pass. A model never passed through
# `with_activity` is all-active, so its gradient is correct by default.

"Pairs a component with its per-field activity flags (a type-domain Boolean tuple)."
struct WithFlags{C,flags}
    component::C
end
function WithFlags(component, ::Val{flags}) where {flags}
    return WithFlags{typeof(component),flags}(component)
end

"A component of type `T`, either plain or wrapped in `WithFlags`."
const MaybeWithFlags{T} = Union{T,WithFlags{<:T}}

# Unwrap a (possibly flagged) component back to the plain component the forward pass sees.
_component(w::WithFlags) = w.component
_component(c) = c

# The flags a component was wrapped with, or all-active for a plain component.
_field_flags(::WithFlags{C,flags}) where {C,flags} = flags
_field_flags(component) = ntuple(Returns(true), Val(fieldcount(typeof(component))))

"Per-call wrapper that resolves its component and stamps the activity flags."
struct Activated{F,flags}
    f::F
end
(a::Activated{F,flags})(ctx) where {F,flags} = WithFlags(resolve(a.f, ctx), Val(flags))
resolve(a::Activated, ctx) = a(ctx)

"""
    with_activity(model, ::Val{flags})

Stamp activity flags `(dyn=..., obs=...)` into a hierarchical model's inner components so the
analytic reverse pass can skip the adjoints of parameter-independent fields. `flags` must be a
type-domain constant built outside the differentiated region. A model not passed through
`with_activity` is all-active.
"""
function with_activity(m::HierarchicalSSM, ::Val{flags}) where {flags}
    return StateSpaceModel(
        m.prior,
        HierarchicalDynamics(
            m.dyn.outer, Activated{typeof(m.dyn.inner),flags.dyn}(m.dyn.inner)
        ),
        HierarchicalObservation(Activated{typeof(m.obs.inner),flags.obs}(m.obs.inner)),
    )
end
