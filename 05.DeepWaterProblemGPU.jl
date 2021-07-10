# -*- coding: utf-8 -*-
include("04.DeepWaterProblem.jl")

using CUDAdrv
CUDAdrv.name(CuDevice(0))

using CuArrays
using CuArrays.CUFFT
CuArrays.allowscalar(false)

using BenchmarkTools

# +
function matsuno!(h, u, p, pinv, Γ, ϵ, Dx, H, Π⅔, unew, hnew, I₀, I₁, I₂, I₃)
    
    @. unew = Γ * h
    pinv * unew
    
    @. hnew = Dx * h
    @. I₁   = hnew
    pinv * I₁
    
    @. I₁ = unew * I₁
    p * I₁
    
    @. I₁ = I₁ * ϵ * Π⅔ - hnew
    @. hnew = h
    pinv * hnew
    
    @. unew = u
    pinv * unew
    
    @. I₃ = hnew * unew
    p * I₃
    
    @. h = H * u
    @. I₂ = Γ * u 
    pinv * I₂
    
    @. hnew = I₂ * hnew 
    p * hnew
    
    @. h = h - (I₃ * Dx + hnew * H) * ϵ * Π⅔
    @. unew = unew^2  
    p * unew
    
    @. u = I₁ - unew * Dx * ϵ/2 * Π⅔
    
end
# -

function loop_over_time( h, u, N, Nt, dt, ϵ, k, kmax )
    
    Uhat = CuArray{ComplexF64,2}(undef, (N,2))
    dU   = CuArray{ComplexF64,2}(undef, (N,2))

    data = zeros(ComplexF64,(N,2,Nt))
    
    Γ   = CuArray(abs.(k))
    Dx  = CuArray(1im * k)        # Differentiation
    H   = CuArray(-1im * sign.(k)) # Hilbert transform
    Π⅔  = CuArray(abs.(k) .< kmax * 2/3) # Dealiasing low-pass filter

    hnew = CuArray{ComplexF64}(undef, N)
    unew = CuArray{ComplexF64}(undef, N)
    
    I₀ = CuArray{ComplexF64}(undef,N)
    I₁ = CuArray{ComplexF64}(undef,N)
    I₂ = CuArray{ComplexF64}(undef,N)
    I₃ = CuArray{ComplexF64}(undef,N)
    
    p    = plan_fft!(h)
    pinv = plan_ifft!(h)
    
    p * h
    p * u
    
    h .= Π⅔ .* h 
    u .= Π⅔ .* u
    
    U = CuArray(hcat(h,u))
    
    for j in 1:Nt
        
        Uhat .= U
        matsuno!( view(Uhat,:,1), view(Uhat,:,2), p, pinv, 
                  Γ, ϵ, Dx, H, Π⅔, unew, hnew, I₀, I₁, I₂, I₃ )
    
        dU   .= Uhat
        Uhat .= U .+ dt/2 .* Uhat
    
        matsuno!(view(Uhat,:,1), view(Uhat,:,2), p, pinv,
                 Γ, ϵ, Dx, H, Π⅔, unew, hnew, I₀, I₁, I₂, I₃  )
    
        dU   .+= 2 .* Uhat
        Uhat .= U .+ dt/2 .* Uhat
    
        matsuno!( view(Uhat,:,1), view(Uhat,:,2), p, pinv, 
                 Γ, ϵ, Dx, H, Π⅔, unew, hnew, I₀, I₁, I₂, I₃ )
    
        dU   .+= 2 .* Uhat
        Uhat .= U .+ dt .* Uhat
        
        matsuno!( view(Uhat,:,1), view(Uhat,:,2), p, pinv, 
                  Γ, ϵ, Dx, H, Π⅔, unew, hnew, I₀, I₁, I₂, I₃ )
    
        dU .+= Uhat
        U  .+= dt/6 .* dU
             
        data[:,:,j] .= collect(U)
        
    end
    data
end

# +
function main_gpu(N :: Int64; animation=true)

    param = ( ϵ  = 1/2,
              N  = N,
              L  = 10.,
              T  = 5.,
              dt = 0.001 )
    
    mesh    = Mesh(param)
    times   = Times(param.dt, param.T)
    init    = BellCurve(param,2.5)
    model   = Matsuno(param)
    mesh    = Mesh(param)
    
    h  = CuArray(init.h)
    u  = CuArray(init.u)
    
    data = loop_over_time( h, u, mesh.N, times.Nt, cu(param.dt), cu(param.ϵ), mesh.k, mesh.kmax )

    if animation
        create_animation( mesh, times, model, data )
    end

end
# -

@time main_gpu( 2^10; animation=true)

@time main_gpu( 2^14; animation=false)

@time main_cpu( 2^14; animation=false)

# ##### 
