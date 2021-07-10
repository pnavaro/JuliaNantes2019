# -*- coding: utf-8 -*-
using  FFTW, LinearAlgebra

struct Mesh
    
    nx   :: Int
    ny   :: Int
    x    :: Vector{Float64}
    y    :: Vector{Float64}
    kx   :: Vector{Float64}
    ky   :: Vector{Float64}
    
    function Mesh( xmin, xmax, nx, ymin, ymax, ny)
        x = range(xmin, stop=xmax, length=nx+1)[1:end-1]  ## we remove the end point
        y = range(ymin, stop=ymax, length=ny+1)[1:end-1]  ## periodic boundary condition
        new( nx, ny, x, y, kx, ky)
    end
end

function exact(time, mesh; shift=1)
   
    f = zeros(Float64,(mesh.nx, mesh.ny))
    for (i, x) in enumerate(mesh.x), (j, y) in enumerate(mesh.y)
        xn = cos(time)*x - sin(time)*y
        yn = sin(time)*x + cos(time)*y
        f[i,j] = exp(-(xn-shift)*(xn-shift)/0.1)*exp(-(yn-shift)*(yn-shift)/0.1)
    end

    f
end

function vectorized(tf, nt, mesh::Mesh)

    dt = tf/nt

    kx = 2π/(mesh.xmax-mesh.xmin)*[0:mesh.nx÷2-1;mesh.nx÷2-mesh.nx:-1]
    ky = 2π/(mesh.ymax-mesh.ymin)*[0:mesh.ny÷2-1;mesh.ny÷2-mesh.ny:-1]

    f = exact(0.0, mesh)

    exky = exp.( 1im*tan(dt/2) .* mesh.x  .* ky')
    ekxy = exp.(-1im*sin(dt)   .* mesh.y' .* kx )
    
    for n = 1:nt
        f .= real(ifft(exky .* fft(f, 2), 2))
        f .= real(ifft(ekxy .* fft(f, 1), 1))
        f .= real(ifft(exky .* fft(f, 2), 2))
    end

    f
end

mesh = Mesh(-π, π, 256, -π, π, 256)
nt, tf = 1000, 200.
println( " error = ", norm(vectorized(tf, nt, mesh) .- exact(tf, mesh)))
