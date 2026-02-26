#=
House Elevation Model for VOI Analysis

Ported from S24 HouseElevation package with simplifications:
- Dropped Unitful (all units in feet / USD)
- Trapezoidal integration for EAD (deterministic, no Monte Carlo)
- Standard discount formula: 1/(1+r)^t
- Positive cost convention (lower = better)
- Wired into SimOptDecisions framework

SLR scenarios are provided as time series vectors (e.g., from BRICK ensemble).

References:
- Zarekarizi et al. (2020): House elevation cost model
- Ruckert et al. (2019): BRICK sea-level projections
=#

using DataFrames
using Distributions
using Interpolations
using Random
using SimOptDecisions

# ============================================================
# Depth-Damage Function
# ============================================================

struct DepthDamageFunction
    itp::Interpolations.AbstractExtrapolation
end

function DepthDamageFunction(depths_ft::Vector{Float64}, damages_pct::Vector{Float64})
    order = sortperm(depths_ft)
    return DepthDamageFunction(
        LinearInterpolation(depths_ft[order], damages_pct[order]; extrapolation_bc=Flat())
    )
end

"""Return damage as a percentage of house value for a given flood depth (ft)."""
(ddf::DepthDamageFunction)(depth_ft::Real) = ddf.itp(Float64(depth_ft))

# ============================================================
# House
# ============================================================

struct House
    value_usd::Float64
    area_ft2::Float64
    height_above_gauge_ft::Float64
    ddf::DepthDamageFunction
end

"""Construct a House from a HAZUS depth-damage row and physical parameters."""
function House(row::DataFrameRow; value_usd, area_ft2, height_above_gauge_ft)
    depths_ft = Float64[]
    damages_pct = Float64[]
    for (col, val) in pairs(row)
        col_str = string(col)
        startswith(col_str, "ft") || continue
        string(val) == "NA" && continue
        depth_str = col_str[3:end]
        is_neg = endswith(depth_str, "m")
        depth_str = is_neg ? depth_str[1:end-1] : depth_str
        depth_str = replace(depth_str, "_" => ".")
        d = parse(Float64, depth_str)
        push!(depths_ft, is_neg ? -d : d)
        push!(damages_pct, parse(Float64, string(val)))
    end
    ddf = DepthDamageFunction(depths_ft, damages_pct)
    return House(Float64(value_usd), Float64(area_ft2), Float64(height_above_gauge_ft), ddf)
end

# ============================================================
# Elevation Cost (Zarekarizi et al. 2020)
# ============================================================

const ELEVATION_COST_ITP = let
    thresholds = [0.0, 5.0, 8.5, 12.0, 14.0]
    rates_per_sqft = [80.36, 82.5, 86.25, 103.75, 113.75]
    LinearInterpolation(thresholds, rates_per_sqft)
end

"""Cost (USD) to elevate a house by Δh_ft feet. Returns 0 for no elevation."""
function elevation_cost(house::House, Δh_ft::Real)
    Δh_ft ≈ 0.0 && return 0.0
    (Δh_ft < 0 || Δh_ft > 14) && error("Elevation must be between 0 and 14 ft, got $Δh_ft")
    base_cost = 10_000 + 300 + 470 + 4_300 + 2_175 + 3_500  # $20,745
    return base_cost + house.area_ft2 * ELEVATION_COST_ITP(Δh_ft)
end

# ============================================================
# Expected Annual Damage (trapezoidal integration)
# ============================================================

"""
Compute EAD via trapezoidal integration over the exceedance-probability curve.
Deterministic — no Monte Carlo sampling needed.
"""
function expected_annual_damage(house::House, Δh_ft::Real, slr_ft::Real, surge_dist)
    n = 1000
    p_exceed = range(0.0001, 0.9999; length=n)
    flood_depths = quantile.(Ref(surge_dist), 1.0 .- collect(p_exceed))
    net_depths = flood_depths .+ slr_ft .- house.height_above_gauge_ft .- Δh_ft
    damages_usd = (house.ddf.(net_depths) ./ 100.0) .* house.value_usd
    ead = sum(
        (damages_usd[i] + damages_usd[i + 1]) / 2 * (p_exceed[i + 1] - p_exceed[i])
        for i in 1:(n - 1)
    )
    return ead
end

# ============================================================
# SimOptDecisions Integration
# ============================================================

struct HouseConfig <: SimOptDecisions.AbstractConfig
    house::House
    years::Vector{Int}
    surge_dist::UnivariateDistribution
end

"""
SLR scenario with a pre-computed time series of sea-level rise values.
`slr_ft` must have the same length as `config.years`.
"""
struct SLRScenario <: SimOptDecisions.AbstractScenario
    slr_ft::Vector{Float64}
    discount_rate::Float64
end

SimOptDecisions.@policydef ElevationPolicy begin
    @continuous height_ft 0.0 14.0
end

function SimOptDecisions.simulate(
    config::HouseConfig, scenario::SLRScenario, policy::ElevationPolicy, rng::AbstractRNG
)
    h = SimOptDecisions.value(policy.height_ft)
    construction = elevation_cost(config.house, h)
    t0 = minimum(config.years)
    ead_npv = sum(enumerate(config.years)) do (i, year)
        slr_ft = scenario.slr_ft[i]
        ead = expected_annual_damage(config.house, h, slr_ft, config.surge_dist)
        ead / (1.0 + scenario.discount_rate)^(year - t0)
    end
    return (investment=construction, expected_damage=ead_npv, total_cost=construction + ead_npv)
end

"""Compute total cost (construction + NPV of expected damages) for a given elevation height."""
function compute_total_cost(config::HouseConfig, scenario::SLRScenario, height_ft::Real)
    policy = ElevationPolicy(; height_ft)
    result = SimOptDecisions.simulate(config, scenario, policy, Random.default_rng())
    return result.total_cost
end
