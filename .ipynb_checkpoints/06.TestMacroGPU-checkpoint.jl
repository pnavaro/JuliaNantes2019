# -*- coding: utf-8 -*-
include("04.DeepWaterProblem.jl")

using CUDAdrv
CUDAdrv.name(CuDevice(0))
using CuArrays
using CuArrays.CUFFT

macro matsuno(h, u)
    return esc( quote
    
        hnew   .= Γ .* $h
        unew   .= pinv * hnew
        hnew   .= Dx .* $h    
        I₁     .= pinv * hnew
        unew  .*= I₁    
        I₁     .= p * unew
        I₁     .= I₁ .* ϵ .* Π⅔ .- hnew    
        hnew   .= pinv * $h
        unew   .= pinv * $u
        I₂     .= hnew .* unew    
        I₃     .= p * I₂
        $h     .= H .* $u
        I₀     .= Γ .* $u   
        I₂     .= pinv * I₀
        I₂    .*= hnew    
        hnew   .= p * I₂
        $h    .-= (I₃ .* Dx .+ hnew .* H) .* ϵ .* Π⅔
        I₃     .= unew.^2    
        unew   .= p * I₃
        $u     .= I₁ .- unew  .* Dx .* ϵ/2 .* Π⅔
            
    end)
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
    ϵ       = param.ϵ
    mesh    = Mesh(param)
    
    Γ       = abs.(mesh.k)
    Dx      = 1im * mesh.k          # Differentiation
    H       = -1im * sign.(mesh.k)  # Hilbert transform
    Π⅔      = Γ .< mesh.kmax * 2/3  # Dealiasing low-pass filter

    hnew = zeros(ComplexF64, mesh.N)
    unew = zeros(ComplexF64, mesh.N)
    
    p    = plan_fft(hnew)
    pinv = plan_ifft(hnew)

    I₀ = zeros(ComplexF64, mesh.N)
    I₁ = zeros(ComplexF64, mesh.N)
    I₂ = zeros(ComplexF64, mesh.N)
    I₃ = zeros(ComplexF64, mesh.N)
    
    h  = init.h
    u  = init.u
    
    h .= p * h
    u .= p * u
    
    h .= Π⅔ .* h 
    u .= Π⅔ .* u

    dt   = times.dt
    data = zeros(ComplexF64,(mesh.N,2,times.Nt))
    n    = param.N
    
    hhat = zeros(ComplexF64, mesh.N)
    uhat = zeros(ComplexF64, mesh.N)

    dh   = zeros(ComplexF64, mesh.N)
    du   = zeros(ComplexF64, mesh.N)

    @showprogress 1 for j in 1:times.Nt
        
        hhat .= h
        uhat .= u    
        @matsuno( hhat, uhat )
    
        dh   .= hhat
        hhat .= h .+ dt/2 .* hhat
        
        du   .= uhat
        uhat .= u .+ dt/2 .* uhat
    
        @matsuno( hhat, uhat )
    
        dh   .+= 2 .* hhat
        hhat .= h .+ dt/2 .* hhat
        du   .+= 2 .* uhat
        uhat .= u .+ dt/2 .* uhat
    
        @matsuno( hhat, uhat )
    
        dh   .+= 2 .* hhat
        hhat .= h .+ dt .* hhat
        du   .+= 2 .* uhat
        uhat .= u .+ dt .* uhat
        
        @matsuno( hhat, uhat )
    
        dh .+= hhat
        h  .+= dt/6 .* dh
        du .+= uhat
        u  .+= dt/6 .* du
                
        data[:,:,j] .= hcat(collect(h),collect(u))
        
    end

    if animation
        create_animation( mesh, times, model, data )
    end

end
# -

@time main_gpu( 2^12; animation=true)


