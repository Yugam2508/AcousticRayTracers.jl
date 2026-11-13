### exact eigenrays for layered environments (method = :arcs)

# In a layer where the soundspeed is linear in depth, rays are circular arcs, so
# in a range-independent environment with flat boundaries and a constant or
# piecewise-linear depth-only soundspeed profile, the range X, travel time T and
# path length L of a ray have closed forms in the ray parameter p = cos(θ)/c
# [COA ch. 3]. A ray with n turning events (boundary reflections or refraction
# turns) is n repeats of one closed-form cycle, so tracing it costs the same at
# any range. Eigenrays are the roots of X(p) = R for each ray family (launch
# direction, n), bracketed on a launch-angle grid and refined by bisection.
#
# Depths in this file are positive down (d = -z).

struct ArcProfile{T}
  d::Vector{T}    # node depths, surface to seabed, strictly increasing
  c::Vector{T}    # soundspeed at nodes
end

# reason why the environment/options do not support exact arc tracing, or nothing
function _arcs_unsupported(env, backscatter, scatterers)
  backscatter && return "backscatter is not supported"
  scatterers === nothing || return "scatterers are not supported"
  is_range_dependent(env) && return "environment must be range independent"
  is_constant(env.bathymetry) || return "bathymetry must be flat"
  is_constant(env.altimetry) || return "altimetry must be flat"
  ss = env.soundspeed
  is_constant(ss) || (ss isa SampledFieldZ && ss.interp isa UnderwaterAcoustics.Linear) ||
    return "soundspeed must be constant, or a SampledField in z with linear interpolation"
  nothing
end

_use_arcs(pm) = pm.method === :arcs ||
  (pm.method === :auto && _arcs_unsupported(pm.env, pm.backscatter, pm.scatterers) === nothing)

# strip (possibly nested) dual numbers; the eigenray search runs on plain floats
_primal(x::ForwardDiff.Dual) = _primal(ForwardDiff.value(x))
_primal(x::Real) = Float64(x)

# linear interpolation with flat extrapolation (as SampledField with Linear())
function _arc_interp(dk, ck, d)
  d ≤ dk[1] && return ck[1] + zero(d)
  d ≥ dk[end] && return ck[end] + zero(d)
  i = searchsortedlast(dk, d)
  ck[i] + (ck[i+1] - ck[i]) * (d - dk[i]) / (dk[i+1] - dk[i])
end

# soundspeed profile from the surface to the seabed; the soundspeed field is only
# evaluated at its own (plain float) knots, so that dual-number boundary depths
# never index the interpolant (Interpolations.jl #645)
function _arc_profile(env)
  a = value(env.altimetry)
  D = value(env.bathymetry)
  ss = env.soundspeed
  if is_constant(ss)
    c = value(ss)
    T = promote_type(typeof(a), typeof(D), typeof(c))
    return ArcProfile(T[-a, D], T[c, c])
  end
  dk = sort!([-Float64(z) for z ∈ ss.zrange])
  ck = [value(ss, -d) for d ∈ dk]
  T = promote_type(typeof(a), typeof(D), eltype(ck))
  d = T[-a; [x for x ∈ dk if -a < x < D]; D]
  ArcProfile(d, T[_arc_interp(dk, ck, x) for x ∈ d])
end

@inline _arc_layer(P, d) = clamp(searchsortedlast(P.d, d), 1, length(P.d) - 1)

function _arc_c(P, d)
  i = _arc_layer(P, d)
  P.c[i] + (P.c[i+1] - P.c[i]) * (d - P.d[i]) / (P.d[i+1] - P.d[i])
end

# (X, T, L) along a monotone segment between depths a < b (p c ≤ 1 on [a, b]);
# aturn/bturn: that endpoint is a refraction turning point (c = 1/p there)
function _arc_xtl(P, p, a, b; aturn=false, bturn=false)
  x = zero(promote_type(typeof(p), typeof(a), typeof(b), eltype(P.c)))
  t = zero(x)
  l = zero(x)
  a < b || return x, t, l
  for i ∈ _arc_layer(P, a):length(P.d)-1
    P.d[i] ≥ b && break
    d1 = max(a, P.d[i])
    d2 = min(b, P.d[i+1])
    d1 < d2 || continue
    g = (P.c[i+1] - P.c[i]) / (P.d[i+1] - P.d[i])
    c1 = P.c[i] + g * (d1 - P.d[i])
    c2 = P.c[i] + g * (d2 - P.d[i])
    if (bturn && d2 == b) || (aturn && d1 == a)
      cn = bturn && d2 == b ? c1 : c2           # non-turning end of the arc
      sn = sqrt(1 - (p * cn)^2)
      x += sn / (p * abs(g))
      t += log((1 + sn) / (p * cn)) / abs(g)
      l += (π/2 - asin(p * cn)) / (p * abs(g))
      continue
    end
    s1 = sqrt(max(1 - (p * c1)^2, zero(x)))
    s2 = sqrt(max(1 - (p * c2)^2, zero(x)))
    x += p * (d2 - d1) * (c1 + c2) / (s1 + s2)
    if abs(c2 - c1) < 1e-10 * c1                # isovelocity layer
      t += (d2 - d1) / (c1 * s1)
      l += (d2 - d1) / s1
    else
      t += log(c2 * (1 + s1) / (c1 * (1 + s2))) / g
      l += (asin(p * c2) - asin(p * c1)) / (p * g)
    end
  end
  x, t, l
end

# upper turning depth of a ray with parameter p through depth d0: (depth, is_surface)
function _arc_upper(P, p, d0)
  T = promote_type(typeof(p), typeof(d0), eltype(P.c))
  i = _arc_layer(P, d0)
  while true
    if p * P.c[i] ≥ 1
      g = (P.c[i+1] - P.c[i]) / (P.d[i+1] - P.d[i])
      return T(P.d[i] + (1 / p - P.c[i]) / g), false
    end
    i == 1 && return T(P.d[1]), true
    i -= 1
  end
end

# lower turning depth of a ray with parameter p through depth d0: (depth, is_seabed)
function _arc_lower(P, p, d0)
  T = promote_type(typeof(p), typeof(d0), eltype(P.c))
  i = _arc_layer(P, d0)
  n = length(P.d)
  while true
    if p * P.c[i+1] ≥ 1
      g = (P.c[i+1] - P.c[i]) / (P.d[i+1] - P.d[i])
      return T(P.d[i] + (1 / p - P.c[i]) / g), false
    end
    i == n - 1 && return T(P.d[n]), true
    i += 1
  end
end

# leg quantities for one p: turning depths and (X, T, L) from the source (s) and
# receiver (r) depths up (u) and down (d) to them; nothing if the receiver depth
# is not reached by a ray with this p from the source
function _arc_legs(P, p, zs, zr)
  zu, us = _arc_upper(P, p, zs)
  zd, db = _arc_lower(P, p, zs)
  zu ≤ zr ≤ zd || return nothing
  su = _arc_xtl(P, p, zu, zs; aturn=!us)
  sd = _arc_xtl(P, p, zs, zd; bturn=!db)
  ru = _arc_xtl(P, p, zu, zr; aturn=!us)
  rd = _arc_xtl(P, p, zr, zd; bturn=!db)
  (; zu, us, zd, db, su, sd, ru, rd)
end

# ray family (dir = +1 launched up / -1 down, n turning events) ->
# ((X, T, L), upper events, lower events, final direction), or nothing
function _arc_family(L, dir, n, zs, zr)
  if n == 0
    if dir > 0
      zr ≤ zs || return nothing
      return (L.su .- L.ru, 0, 0, 1)
    else
      zr ≥ zs || return nothing
      return (L.sd .- L.rd, 0, 0, -1)
    end
  end
  ud = L.su .+ L.sd
  if dir > 0
    v = L.su .+ (n - 1) .* ud .+ (isodd(n) ? L.ru : L.rd)
    nu, nd = cld(n, 2), fld(n, 2)
  else
    v = L.sd .+ (n - 1) .* ud .+ (isodd(n) ? L.rd : L.ru)
    nu, nd = fld(n, 2), cld(n, 2)
  end
  (v, nu, nd, isodd(n) ? -dir : dir)
end

function _arc_X(L, dir, n, zs, zr)
  L === nothing && return NaN
  fam = _arc_family(L, dir, n, zs, zr)
  fam === nothing ? NaN : fam[1][1]
end

_arc_X(P::ArcProfile, p, dir, n, zs, zr) = _arc_X(_arc_legs(P, p, zs, zr), dir, n, zs, zr)

function _arc_bisect(f, a, b, fa; tol=1e-15, maxiters=200)
  for _ ∈ 1:maxiters
    m = (a + b) / 2
    fm = f(m)
    isnan(fm) && return NaN
    if sign(fm) == sign(fa)
      a, fa = m, fm
    else
      b = m
    end
    abs(b - a) ≤ tol * abs(m) && break
  end
  (a + b) / 2
end

# roots of F on a grid of ray parameters pg with values Fg (NaN where the ray
# family does not reach the receiver)
function _arc_scan(F, pg, Fg)
  roots = Float64[]
  for k ∈ 1:length(pg)-1
    a, b, fa, fb = pg[k], pg[k+1], Fg[k], Fg[k+1]
    isnan(fa) && isnan(fb) && continue
    if isnan(fa) != isnan(fb)
      # one end is invalid (e.g. the ray turns before reaching the receiver);
      # near the validity boundary X(p) can run off to infinity (a ray grazing
      # a soundspeed maximum), so a root can sit between the boundary and the
      # valid grid point: locate the boundary by bisection and test that
      # sub-interval
      lo, hi, fv = isnan(fa) ? (a, b, fb) : (b, a, fa)    # lo invalid, hi valid
      for _ ∈ 1:80
        m = (lo + hi) / 2
        isnan(F(m)) ? (lo = m) : (hi = m)
      end
      fe = F(hi)
      if !isnan(fe) && fe != 0 && sign(fe) != sign(fv)
        push!(roots, _arc_bisect(F, hi, isnan(fa) ? b : a, fe))
      end
    elseif sign(fa) != sign(fb)
      push!(roots, _arc_bisect(F, a, b, fa))
    elseif k > 1 && !isnan(Fg[k-1]) && abs(fa) < abs(Fg[k-1]) && abs(fa) < abs(fb)
      # local minimum of |F| without a sign change: look for a hidden root
      # pair by golden-section search on |F|
      lo, hi = minmax(pg[k-1], pg[k+1])
      for _ ∈ 1:100
        m1 = hi - 0.618 * (hi - lo)
        m2 = lo + 0.618 * (hi - lo)
        abs(F(m1)) < abs(F(m2)) ? (hi = m2) : (lo = m1)
      end
      pm = (lo + hi) / 2
      fm = F(pm)
      if !isnan(fm) && sign(fm) != sign(fa)
        push!(roots, _arc_bisect(F, pg[k-1], pm, Fg[k-1]))
        push!(roots, _arc_bisect(F, pm, pg[k+1], fm))
      end
    end
  end
  sort!(filter!(!isnan, roots))
  [roots[k] for k ∈ eachindex(roots) if k == 1 || roots[k] - roots[k-1] > 1e-11 * roots[k]]
end

# eigenray ray parameters as (p, dir, n) on plain floats; the launch-angle grid
# spans [min_angle, max_angle], and the number of turning events is bounded by
# the shortest turning-cycle range of any ray on the grid that reaches the
# receiver depth (capped to guard rays grazing a soundspeed minimum)
function _arc_roots(P, zs, zr, R, pm)
  cs = _arc_c(P, zs)
  ngrid = pm.nbeams > 0 ? max(pm.nbeams ÷ 2, 16) : 4000
  roots = Tuple{Float64,Int,Int}[]
  for dir ∈ (1, -1)
    θlo, θhi = dir > 0 ? (max(pm.min_angle, 0.0), pm.max_angle) : (max(-pm.max_angle, 0.0), -pm.min_angle)
    θhi > 0 || continue
    pg = cos.(range(max(θlo, 1e-6), θhi; length=ngrid)) ./ cs
    Lg = [_arc_legs(P, p, zs, zr) for p ∈ pg]
    nmax = 0
    for L ∈ Lg
      L === nothing && continue
      ud = L.su[1] + L.sd[1]
      ud > 0 && (nmax = max(nmax, floor(Int, min(R / ud, 1e4)) + 2))
    end
    for n ∈ 0:nmax
      Fg = [_arc_X(L, dir, n, zs, zr) - R for L ∈ Lg]
      all(isnan, Fg) && continue
      for p ∈ _arc_scan(p -> _arc_X(P, p, dir, n, zs, zr) - R, pg, Fg)
        push!(roots, (p, dir, n))
      end
    end
  end
  roots
end

# complex ForwardDiff-friendly x^n
_arc_pow(x, n) = n == 0 ? one(x) : prod(x for _ ∈ 1:n)

# eigenray (p₀, dir, n) found on plain floats -> RayArrival of type T2, or nothing
function _arc_arrival(::Type{T1}, ::Type{T2}, pm, P, P₀, p₀, dir, n, zs, zr, R, f, paths) where {T1,T2}
  env = pm.env
  p = p₀
  if !(T1 <: AbstractFloat)
    # restore derivatives of the root by the implicit function theorem
    dX = ForwardDiff.derivative(q -> _arc_X(P₀, q, dir, n, _primal(zs), _primal(zr)), p₀)
    p = p₀ - (_arc_X(P, p₀, dir, n, zs, zr) - R) / dX
  end
  L = _arc_legs(P, p, zs, zr)
  L === nothing && return nothing
  fam = _arc_family(L, dir, n, zs, zr)
  fam === nothing && return nothing
  (X, t, len), nu, nd, fdir = fam
  abs(X - R) < 1e-6 * max(R, 1) || return nothing
  ns = L.us ? nu : 0
  nb = L.db ? nd : 0
  nturn = n - ns - nb                             # refraction turning points
  cs = _arc_c(P, zs)
  cr = _arc_c(P, zr)
  θₛ = dir * acos(clamp(p * cs, -1, 1))
  θᵣ = -fdir * acos(clamp(p * cr, -1, 1))        # minus the propagation elevation
  dXdp = ForwardDiff.derivative(q -> _arc_X(P, q, dir, n, zs, zr), p)
  q = abs(dXdp * sin(θₛ) * sin(θᵣ)) / cs         # ray-tube spreading at the receiver
  Γ = one(Complex{T1})                            # boundary reflections
  if ns > 0
    c₁ = P.c[1]
    ρ = value(env.density, (0.0, 0.0, -P.d[1]))
    Γ *= _arc_pow(reflection_coef(env.surface, f, asin(clamp(p * c₁, -1, 1)), ρ, c₁), ns)
  end
  if nb > 0
    c₁ = P.c[end]
    ρ = value(env.density, (0.0, 0.0, -P.d[end]))
    Γ *= _arc_pow(reflection_coef(env.seabed, f, asin(clamp(p * c₁, -1, 1)), ρ, c₁), nb)
  end
  abs(Γ) / q < pm.min_amplitude && return nothing
  # caustic phase: -π/2 per refraction turning point, +π/2 when dX/dp < 0
  k = nturn - (dXdp < 0 ? 1 : 0)
  mid_temp = (minimum(env.temperature) + maximum(env.temperature)) / 2
  A = Γ * cis(-π/2 * k) * √abs(cr * cos(θₛ) / (R * cs * q))     # [COA (3.65)]
  A *= absorption(f, len, env.salinity, mid_temp, maximum(env.bathymetry) / 2)
  path = Array{@NamedTuple{x::T1,y::T1,z::T1}}(undef, 0)
  if paths
    ds = pm.ds ≤ 0 ? _primal(minimum(env.bathymetry)) / 10 : pm.ds
    _arc_path!(path, P, p, dir, n, L, zs, zr, ds)
    path[end] = (x=T1(R), y=zero(T1), z=T1(-zr))
  end
  RayArrival(T1(t), Complex{T1}(A), ns, nb, T1(θₛ), T1(θᵣ), path)::T2
end

# ray path as points at most ~ds apart in range: in a linear layer, sin θ is
# linear in range, so points along the arc follow in closed form
function _arc_path!(path, P, p, dir, n, L, zs, zr, ds)
  push!(path, (x=0.0, y=0.0, z=-zs))
  segs = []
  d = zs
  up = dir > 0
  for _ ∈ 1:n
    dt = up ? L.zu : L.zd
    push!(segs, (d, dt))
    d = dt
    up = !up
  end
  push!(segs, (d, zr))
  r = zero(p)
  for (a, b) ∈ segs
    lo, hi = minmax(a, b)
    nodes = [lo; [x for x ∈ P.d if lo < x < hi]; hi]
    b < a && reverse!(nodes)
    for j ∈ 1:length(nodes)-1
      d1, d2 = nodes[j], nodes[j+1]
      i = _arc_layer(P, (d1 + d2) / 2)
      g = (P.c[i+1] - P.c[i]) / (P.d[i+1] - P.d[i])
      c1 = P.c[i] + g * (d1 - P.d[i])
      c2 = P.c[i] + g * (d2 - P.d[i])
      s1 = sqrt(max(1 - (p * c1)^2, zero(p)))
      s2 = sqrt(max(1 - (p * c2)^2, zero(p)))
      if abs(c2 - c1) < 1e-10 * c1
        r += abs(d2 - d1) * p * c1 / s1             # straight segment
        push!(path, (x=r, y=0.0, z=-d2))
        continue
      end
      Δx = abs((s1 - s2) / (p * g))
      m = max(1, ceil(Int, _primal(Δx) / ds))
      for k ∈ 1:m
        s = s1 + (s2 - s1) * k / m
        c = sqrt(max(1 - s^2, zero(s))) / p
        push!(path, (x=r + Δx * k / m, y=0.0, z=-(P.d[i] + (c - P.c[i]) / g)))
      end
      r += Δx
    end
  end
  path
end

# the direct ray between a source and receiver at the same depth is horizontal
# (p = 1/c), the end point of the ray-parameter range that the scan cannot
# bracket; it exists when the soundspeed is constant around that depth, and then
# spreads spherically (q = R)
function _arc_horizontal(::Type{T1}, ::Type{T2}, pm, P, zs, zr, R, f, paths) where {T1,T2}
  zs == zr && pm.min_angle ≤ 0 ≤ pm.max_angle && R > 0 || return nothing
  n = length(P.d)
  i = _arc_layer(P, zs)
  iso(j) = abs(P.c[j+1] - P.c[j]) < 1e-10 * P.c[j]
  iso(i) || return nothing
  zs == P.d[i] && i > 1 && !iso(i - 1) && return nothing
  zs == P.d[i+1] && i + 1 < n && !iso(i + 1) && return nothing
  env = pm.env
  cs = _arc_c(P, zs)
  mid_temp = (minimum(env.temperature) + maximum(env.temperature)) / 2
  A = absorption(f, R, env.salinity, mid_temp, maximum(env.bathymetry) / 2) / R
  path = Array{@NamedTuple{x::T1,y::T1,z::T1}}(undef, 0)
  paths && append!(path, [(x=zero(T1), y=zero(T1), z=T1(-zs)), (x=T1(R), y=zero(T1), z=T1(-zr))])
  RayArrival(T1(R / cs), Complex{T1}(A), 0, 0, zero(T1), zero(T1), path)::T2
end

function _arcs_arrivals(::Type{T1}, ::Type{T2}, pm, tx, rx, paths) where {T1,T2}
  P = _arc_profile(pm.env)
  P₀ = ArcProfile(_primal.(P.d), _primal.(P.c))
  zs = T1(-location(tx).z)
  zr = T1(-location(rx).z)
  R = T1(location(rx).x - location(tx).x)
  P.d[1] ≤ zs ≤ P.d[end] && P.d[1] ≤ zr ≤ P.d[end] || error("Source and receiver must be in the water")
  f = frequency(tx)
  roots = _arc_roots(P₀, _primal(zs), _primal(zr), _primal(R), pm)
  erays = T2[]
  a = _arc_horizontal(T1, T2, pm, P, zs, zr, R, f, paths)
  a === nothing || push!(erays, a)
  for (p₀, dir, n) ∈ roots
    a = _arc_arrival(T1, T2, pm, P, P₀, p₀, dir, n, zs, zr, R, f, paths)
    a === nothing || push!(erays, a)
  end
  sort!(erays; by=Base.Fix2(getfield, :t))
end
