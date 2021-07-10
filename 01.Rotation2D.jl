# -*- coding: utf-8 -*-
# # Who am I ?
#
# - My name is *Pierre Navaro*
# - Ph.D in Computational Aeroacoustics, 1998-2002 (Université du Havre) (Fortran 77+PVM)
# - Scientific Software Engineer in Strasbourg (2003-2015) (Fortran 90-2003 + OpenMP-MPI)
# - Moved to Rennes in 2015 (Numpy + Cython, R + Rcpp)
# - Julia user since July 2018 (Julia v1.0)
#     * Simulation numérique en physique des plasmas
#     * Assimilation de données météo

# # Advection equation for a rotation in two dimensional domain
#
# $$
# \frac{d f}{dt} +  (y \frac{d f}{dx} - x \frac{d f}{dy}) = 0
# $$
#
# $$ 
# x \in [-\pi, \pi],\qquad y \in [-\pi, \pi] \qquad \mbox{ and } \qquad t \in [0, 200\pi] 
# $$
#
#
# https://github.com/JuliaVlasov/FourierAdvections.jl in `notebooks` directory
#
# You can open these julia files as notebooks with [jupytext](https://jupytext.readthedocs.io)

using FFTW, LinearAlgebra, Plots, ProgressMeter
using BenchmarkTools

"""
    Mesh( xmin, xmax, nx, ymin, ymax, ny)

mesh information
"""
struct Mesh
    
    nx   :: Int64
    ny   :: Int64
    x    :: Vector{Float64}
    y    :: Vector{Float64}
    kx   :: Vector{Float64}
    ky   :: Vector{Float64}
    
    function Mesh( xmin, xmax, nx, ymin, ymax, ny)
        
        x = range(xmin, stop=xmax, length=nx+1)[1:end-1]  ## we remove the end point
        y = range(ymin, stop=ymax, length=ny+1)[1:end-1]  ## periodic boundary condition
        kx = 2π/(xmax-xmin)*[0:nx÷2-1;nx÷2-nx:-1]
        ky = 2π/(ymax-ymin)*[0:ny÷2-1;ny÷2-ny:-1]

        new( nx, ny, x, y, kx, ky)
    end
end

# ### Function to compute exact solution

function exact(time, mesh :: Mesh; shift=1)
   
    f = zeros(Float64,(mesh.nx, mesh.ny))
    for (i, x) in enumerate(mesh.x), (j, y) in enumerate(mesh.y)  # two loops
        xn = cos(time)*x - sin(time)*y
        yn = sin(time)*x + cos(time)*y
        f[i,j] = exp(-(xn-shift)*(xn-shift)/0.1)*exp(-(yn-shift)*(yn-shift)/0.1)
    end

    f
end

mesh = Mesh(-π, π, 128, -π, π, 128)
f = exact(0.0, mesh)
contour(mesh.x, mesh.y, f; aspect_ratio=:equal, clims=(0.,1.))

# ## Create the gif to show what we are computing

# +
function animation( tf, nt)
    
    nx, ny = 64, 64
    xmin, xmax, nx = -π, π, nx
    ymin, ymax, ny = -π, π, ny
    mesh = Mesh(xmin, xmax, nx, ymin, ymax, ny)
    f  = zeros(Float64,(nx,ny))
    dt = tf / nt
    bar = Progress(nt,1) ## progress bar
    t = 0
    @gif for n=1:nt
       
       f .= exact(t, mesh)
       t += dt
       p = contour(mesh.x, mesh.y, f)
       plot!(p[1]; clims=(0.,1.), aspect_ratio=:equal)
       plot!(sqrt(2) .* cos.(-pi:0.1:pi+0.1), sqrt(2) .* sin.(-pi:0.1:pi+0.1))
       next!(bar) ## increment the progress bar
        
    end
    
end

animation( 2π, 100)
# -

# ###  Vectorized version
#
# - We store the 2d arrays `exky` and `ekxy` to compute the derivatives.
# - We save cpu time by computing them before the loop over time

function vectorized(tf, nt, mesh::Mesh)

    dt = tf/nt

    
    f = exact(0.0, mesh)

    exky = exp.( 1im*tan(dt/2) .* mesh.x  .* mesh.ky')
    ekxy = exp.(-1im*sin(dt)   .* mesh.y' .* mesh.kx )
    
    for n = 1:nt
        f .= real(ifft(exky .* fft(f, 2), 2)) # df / dt = -x * df / dy  in [t,t+dt/2]
        f .= real(ifft(ekxy .* fft(f, 1), 1)) # df / dt = y * df / dx in [t,t+dt]
        f .= real(ifft(exky .* fft(f, 2), 2)) # df / dt = -x * df / dy in [t+dt/2,t+dt]
    end

    f
end

nt, tf = 1000, 200π
mesh = Mesh(-π, π, 256, -π, π, 512)
f = vectorized(0.1, 1, mesh) # trigger build
@time norm(vectorized(tf, nt, mesh) .- exact(tf, mesh))

# ```
# julia --check-bounds=no -O3 --track-allocation=user program.jl
# ```

# ```julia
#         - function vectorized(tf, nt, mesh::Mesh)
#         - 
# 234240560     dt = tf/nt
#         - 
#         0     f = exact(0.0, mesh)
#         - 
#   1048656     exky = exp.( 1im*tan(dt/2) .* mesh.x  .* mesh.ky')
#   1048656     ekxy = exp.(-1im*sin(dt)   .* mesh.y' .* mesh.kx )
#         -     
#         0     for n = 1:nt
# 2621712000         f .= real(ifft(exky .* fft(f, 2), 2))
# 2621712000         f .= real(ifft(ekxy .* fft(f, 1), 1))
# 2621712000         f .= real(ifft(exky .* fft(f, 2), 2))
#         -     end
#         - 
#         0     f
#         - end
# ```

# ## Inplace computation 
# - We remove the Float64-Complex128 conversion by allocating the distribution function `f` as a Complex array
# - Note that we need to use the inplace assignement operator ".="  to initialize the `f` array.
# - We use inplace computation for fft with the "bang" operator `!`

function inplace(tf, nt, mesh::Mesh)

    dt = tf/nt

    f  = zeros(Complex{Float64},(mesh.nx,mesh.ny))
    f .= exact(0.0, mesh)

    exky = exp.( 1im*tan(dt/2) .* mesh.x  .* mesh.ky')
    ekxy = exp.(-1im*sin(dt)   .* mesh.y' .* mesh.kx )
    
    for n = 1:nt
        
        fft!(f, 2)
        f .= exky .* f
        ifft!(f,2)
        
        fft!(f, 1)
        f .= ekxy .* f
        ifft!(f, 1)
        
        fft!(f, 2)
        f .= exky .* f
        ifft!(f,2)
        
    end

    real(f)
end

f = inplace(0.1, 1, mesh) # trigger build
@time norm(inplace(tf, nt, mesh) .- exact(tf, mesh))

# ## Inplace computation and fft plans
#
# To apply fft plan to an array A, we use a preallocated output array Â by calling `mul!(Â, plan, A)`. 
# The input array A must be a complex floating-point array like the output Â.
# The inverse-transform is computed inplace by applying `inv(P)` with `ldiv!(A, P, Â)`.

function with_fft_plans_inplace(tf, nt, mesh::Mesh)

    dt = tf/nt

    f  = zeros(Complex{Float64},(mesh.nx,mesh.ny))
    f .= exact(0.0, mesh)
    f̂  = similar(f)

    exky = exp.( 1im*tan(dt/2) .* mesh.x  .* mesh.ky')
    ekxy = exp.(-1im*sin(dt)   .* mesh.y' .* mesh.kx )

    Px = plan_fft(f, 1)    
    Py = plan_fft(f, 2)
        
    for n = 1:nt
        
        mul!(f̂, Py, f)
        f̂ .= f̂ .* exky
        ldiv!(f, Py, f̂)
        
        mul!(f̂, Px, f)
        f̂ .= f̂ .* ekxy 
        ldiv!(f, Px, f̂)
        
        mul!(f̂, Py, f)
        f̂ .= f̂ .* exky
        ldiv!(f, Py, f̂)
        
    end

    real(f)
end
#----------------------------------------------------------------------------

f = with_fft_plans_inplace(0.1, 1, mesh) # trigger build
@time norm(with_fft_plans_inplace(tf, nt, mesh) .- exact(tf, mesh))

# ## Explicit transpose of `f`
#
# - Multidimensional arrays in Julia are stored in column-major order.
# - FFTs along y are slower than FFTs along x
# - We can speed-up the computation by allocating the transposed `f` 
# and transpose f for each advection along y.

function with_fft_transposed(tf, nt, mesh::Mesh)

    dt = tf/nt

    f  = zeros(Complex{Float64},(mesh.nx,mesh.ny))
    f̂  = similar(f)
    fᵗ = zeros(Complex{Float64},(mesh.ny,mesh.nx))
    f̂ᵗ = similar(fᵗ)

    exky = exp.( 1im*tan(dt/2) .* mesh.x' .* mesh.ky )
    ekxy = exp.(-1im*sin(dt)   .* mesh.y' .* mesh.kx )
    
    FFTW.set_num_threads(4)
    Px = plan_fft(f,  1, flags=FFTW.PATIENT)    
    Py = plan_fft(fᵗ, 1, flags=FFTW.PATIENT)
    
    f .= exact(0.0, mesh)
    
    for n = 1:nt
        transpose!(fᵗ,f)
        mul!(f̂ᵗ, Py, fᵗ)
        f̂ᵗ .= f̂ᵗ .* exky
        ldiv!(fᵗ, Py, f̂ᵗ)
        transpose!(f,fᵗ)
        
        mul!(f̂, Px, f)
        f̂ .= f̂ .* ekxy 
        ldiv!(f, Px, f̂)
        
        transpose!(fᵗ,f)
        mul!(f̂ᵗ, Py, fᵗ)
        f̂ᵗ .= f̂ᵗ .* exky
        ldiv!(fᵗ, Py, f̂ᵗ)
        transpose!(f,fᵗ)
    end
    real(f)
end

f = with_fft_transposed(0.1, 1, mesh) # trigger build
@time norm(with_fft_transposed(tf, nt, mesh) .- exact(tf, mesh))

vectorized_bench = @benchmark vectorized(tf, nt, mesh)
inplace_bench = @benchmark inplace(tf, nt, mesh)
with_fft_plans_inplace_bench = @benchmark with_fft_plans_inplace(tf, nt, mesh)
with_fft_transposed_bench = @benchmark with_fft_transposed(tf, nt, mesh)

d = Dict() 
d["inplace"] = minimum(inplace_bench.times) / 1e6
d["vectorized"] = minimum(vectorized_bench.times) / 1e6
d["with_fft_plans_inplace"] = minimum(with_fft_plans_inplace_bench.times) / 1e6
d["with_fft_transposed"] = minimum(with_fft_transposed_bench.times) / 1e6;

for (key, value) in sort(collect(d), by=last)
    println(rpad(key, 25, "."), lpad(round(value, digits=1), 6, "."))
end

# ## Conclusion
# - Use pre-allocations of memory and inplace computation are very important
# - Try to always do computation on data contiguous in memory
# - In this notebook, use btime to not taking account of time consumed in compilation.
